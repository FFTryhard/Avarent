#include "engine.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>
#include <deque>

// ==========================
// CONFIG
// ==========================
static const double INITIAL_CAPITAL = 1'000'000.0;
static const double TRANSACTION_COST_BPS = 10.0;
static const int REBALANCE_DAYS = 5;       // wired up this time
static const int SHUFFLE_ITERATIONS = 500; // for p-value harness

struct BacktestMetrics
{
  double finalValue;
  double sharpe;
  double maxDrawdown;
};

struct rollingSharpe
{
private:
  std::deque<double> rets;
  double last_nav = -1.0;
  int window = 126; // 6-month rolling window for adaptability

public:
  void push(double nav)
  {
    if (last_nav > 0)
    {
      rets.push_back((nav - last_nav) / last_nav);
      if (rets.size() > window)
        rets.pop_front();
    }
    last_nav = nav;
  }

  const double getSharpe()
  {
    if (rets.size() < 40)
    {
      return std::numeric_limits<double>::quiet_NaN();
    }

    double mean = std::accumulate(rets.begin(), rets.end(), 0.0) / rets.size();

    double sq_sum = 0.0;
    for (double x : rets)
      sq_sum += (x - mean) * (x - mean);

    double sd = std::sqrt(sq_sum / rets.size());

    if (sd <= 1e-12)
      return 0.0;

    return std::sqrt(252.0) * (mean / sd); // Annualized Sharpe
  }
};

void runBacktest(Engine &engine, const std::vector<std::string> &headers,
                 const std::vector<std::vector<double>> &allReturns,
                 const std::vector<std::string> &dates);

void runShuffleHarness(const std::vector<std::string> &headers,
                       const std::vector<std::vector<double>> &allReturns,
                       int numIterations);

std::vector<double>
riskParitySignal(const std::vector<std::vector<double>> &histBeforeToday,
                 int window);

std::vector<double>
rollingVol(const std::vector<std::vector<double>> &returnHistory, int window);

BacktestMetrics calcMetrics(const std::vector<double> &portfolioHistory);

double dotSum(const std::vector<double> &pos,
              const std::vector<double> &prices);

double absSum(const std::vector<double> &v);

void divideBy(std::vector<double> &v, double d);

void clampLow(std::vector<double> &v, double lo);

void clipUpper(std::vector<double> &v, double hi);

int main()
{
  std::cout << "A: entering main\n";
  
  init_python();

  std::cout << "B: python initialized\n";

  Engine engine(profile_defaults(EngineProfile::LONG_SHORT_EQUITY));

  std::cout << "C: engine created\n";

  std::ifstream file(
      "C:/Users/H/Documents/projects/Avarent/data/etf_daily_log_returns_new_time_period2.csv");
  if (!file.is_open())
    throw std::runtime_error("couldn't open file");

  std::string line;
  std::getline(file, line);

  // Parse headers (skip first column which is Date)
  std::stringstream ss_h(line);
  std::string token_h;
  std::vector<std::string> headers;
  int counter = 0;
  while (std::getline(ss_h, token_h, ','))
    if (counter++ != 0)
      headers.push_back(token_h);

  engine.setHeaders(headers);

  // Load all rows upfront
  std::vector<std::vector<double>> allReturns;
  std::vector<std::string> dates;

  while (std::getline(file, line))
  {
    if (line.empty())
      continue;
    std::stringstream ss(line);
    std::string token;
    std::vector<double> row;
    bool first = true;
    std::string dateStr;

    while (std::getline(ss, token, ','))
    {
      if (first)
      {
        dateStr = token;
        first = false;
        continue;
      }
      if (token.empty())
        continue;
      try
      {
        row.push_back(std::stod(token));
      }
      catch (...)
      {
      }
    }

    dates.push_back(dateStr);
    allReturns.push_back(row);
  }

  runBacktest(engine, headers, allReturns, dates);

  // Shuffle harness for p-value computation (comment out for normal runs)
  runShuffleHarness(headers, allReturns, SHUFFLE_ITERATIONS);

  // std::cerr << "HAHAHAHAHAHAHA\n";
  return 0;
}

// ==========================
// HELPERS
// ==========================

// Element-wise multiply then sum  (dot product / portfolio value)
double dotSum(const std::vector<double> &pos,
              const std::vector<double> &prices)
{
  double s = 0.0;
  for (size_t i = 0; i < pos.size(); ++i)
    s = s + (pos[i] * prices[i]);
  return s;
}

/*
pos is shares held per asset, prices is price per asset. Multiplying them gives
dollar value per asset, summing gives total invested value. Add cash on top and
you have portfolio NAV.
*/

// Absolute sum of a vector
double absSum(const std::vector<double> &v)
{
  double s = 0.0;
  for (double x : v)
    s = s + std::abs(x);
  return s;
}

// Element-wise divide (in-place)
void divideBy(std::vector<double> &v, double d)
{
  if (std::abs(d) < 1e-12)
  {
    std::fill(v.begin(), v.end(), 0.0);
    return;
  }
  for (double &x : v)
    x = x / d; // norming, d comes from absSum
}

void clampLow(std::vector<double> &v, double lo)
{
  for (double &x : v)
    x = std::max(x, lo);
}

void clipUpper(std::vector<double> &v, double hi)
{
  for (double &x : v)
    x = std::min(x, hi);
}

struct MarketFeatures
{
  double vol;
  double momentum;

  double vol_z;
  double mom_z;
};

MarketFeatures computeFeatures(const std::deque<std::vector<double>> &window)
{
  MarketFeatures f;

  int T = window.size();
  int N = window.front().size();

  // 1. collapse multivariate returns → single series
  std::vector<double> series;
  series.reserve(T);

  for (const auto &day : window)
  {
    double avg = 0.0;
    for (double r : day)
      avg += r;
    avg /= N;
    series.push_back(avg);
  }

  // 2. compute mean
  double mean = 0.0;
  for (double x : series)
    mean += x;
  mean /= T;

  // 3. compute volatility
  double var = 0.0;
  for (double x : series)
  {
    double d = x - mean;
    var += d * d;
  }
  var /= T;

  f.vol = std::sqrt(var * 252.0);

  // 4. compute momentum (start vs end)
  double start = series.front();
  double end = series.back();

  double sum = 0.0;
  for (double r : series)
    sum += r;
  f.momentum = sum;

  return f;
}

double z_score(const std::deque<double> &v, double x)
{
  double mean = 0.0;
  for (double a : v)
    mean += a;
  mean /= v.size();

  double var = 0.0;
  for (double a : v)
  {
    double d = a - mean;
    var += d * d;
  }
  var /= v.size();

  double std = std::sqrt(var);

  if (std < 1e-8)
    return 0.0;
  return (x - mean) / std;
};

// ==========================
// METRICS
// ==========================

BacktestMetrics calcMetrics(const std::vector<double> &portfolioHistory)
{
  // Daily returns
  std::vector<double> rets;
  rets.reserve(portfolioHistory.size());
  for (size_t i = 1; i < portfolioHistory.size(); ++i)
  {
    double prev = portfolioHistory[i - 1];
    rets.push_back(prev > 1e-12 ? (portfolioHistory[i] - prev) / prev : 0.0);
  } // nieuw - oud / oud lol. gwn de portfolio NAV's naar percentage change

  double mean = 0.0;
  for (double r : rets)
    mean = mean + r;
  mean = mean / static_cast<double>(rets.size());

  double var = 0.0;
  for (double r : rets)
    var = var + ((r - mean) * (r - mean));
  var = var / static_cast<double>(rets.size());
  double std = std::sqrt(var);

  // mean en variance calculated van de percentage change returns

  double sharpe = std > 1e-8 ? std::sqrt(252.0) * mean / std
                             : 0.0; // 252 is amound of trading days in a year

  // Max drawdown
  double peak = portfolioHistory[0];
  double maxDD = 0.0;
  for (double v : portfolioHistory)
  {
    peak = std::max(peak, v);
    double dd = (v - peak) / peak;
    maxDD = std::min(maxDD, dd);
  }

  return {portfolioHistory.back(), sharpe, maxDD};
}

std::vector<double> blendPortfolios(
    const std::vector<std::vector<double>> &strategyPortfolios,
    const std::vector<double> &weights)
{
  int n = strategyPortfolios.size();
  int m = strategyPortfolios[0].size();

  std::vector<double> result(m, 0.0);

  for (int i = 0; i < n; i++)
  {
    double w = weights[i];

    const std::vector<double> &p = strategyPortfolios[i];

    for (int j = 0; j < m; j++)
    {
      result[j] += w * p[j];
    }
  }

  return result;
}

// ==========================
// ROLLING VOL  (20-day std)
// ==========================
// returnHistory[asset][day], returns vol for each asset on the current day
std::vector<double>
rollingVol(const std::vector<std::vector<double>> &returnHistory,
           int window = 20)
{
  int nAssets = static_cast<int>(returnHistory.size());
  std::vector<double> vol(nAssets, 1e-6);

  for (int a = 0; a < nAssets; ++a)
  {
    const std::vector<double> &h = returnHistory[a];
    int n = static_cast<int>(h.size());
    if (n < window)
      continue; // not enough history yet

    int start = n - window;
    double sum = 0.0, sumSq = 0.0;
    for (int k = start; k < n; ++k)
    {
      sum += h[k];
      sumSq += h[k] * h[k];
    }
    double mean = sum / window;
    double variance = sumSq / window - mean * mean;
    vol[a] = variance > 0.0 ? std::sqrt(variance) : 1e-6;
  }
  return vol;
} // 20 day window std

// Inverse-vol weights, normalised to sum to 1  (shift(1) = use previous day's
// vol) We call this AFTER pushing today's return, so the last entry is "today"
// — we pass the history *before* today's push to replicate the .shift(1).
std::vector<double>
riskParitySignal(const std::vector<std::vector<double>> &histBeforeToday,
                 int window = 20)
{
  std::vector<double> vol = rollingVol(histBeforeToday, window);
  int n = static_cast<int>(vol.size());
  std::vector<double> invVol(n);
  for (int i = 0; i < n; ++i)
    invVol[i] = 1.0 / std::max(vol[i], 1e-6);

  double total = 0.0;
  for (double v : invVol)
    total += v;
  if (total > 1e-12)
    for (double &v : invVol)
      v /= total;
  else
    std::fill(invVol.begin(), invVol.end(), 0.0);

  return invVol;
}

// ==========================
// REGIME-SPECIFIC STRATEGIES
// ==========================

std::vector<double>
longVolatilitySignal(const std::vector<std::vector<double>> &histBeforeToday,
                     int window = 20)
{
  std::vector<double> vol = rollingVol(histBeforeToday, window);
  int n = static_cast<int>(vol.size());
  std::vector<double> sig(n);
  double total = 0.0;
  for (int i = 0; i < n; ++i)
  {
    sig[i] = vol[i];
    total += sig[i];
  }
  if (total > 1e-12)
    for (double &v : sig)
      v /= total;
  else
    std::fill(sig.begin(), sig.end(), 0.0);
  return sig;
}

std::vector<double> equalWeightBlend(
    const std::vector<std::vector<double>> &baseSignals)
{
  if (baseSignals.empty())
    return {};

  const std::size_t numSignals = baseSignals.size();
  const std::size_t signalSize = baseSignals[0].size();

  // Ensure all signals have the same length
  for (const auto &signal : baseSignals)
  {
    if (signal.size() != signalSize)
      throw std::runtime_error("Signal size mismatch");
  }

  std::vector<double> blended(signalSize, 0.0);

  for (const auto &signal : baseSignals)
  {
    for (std::size_t i = 0; i < signalSize; ++i)
    {
      blended[i] += signal[i];
    }
  }

  const double weight = 1.0 / static_cast<double>(numSignals);

  for (double &value : blended)
  {
    value *= weight;
  }

  return blended;
}

std::vector<double>
reducedRiskRiskParitySignal(const std::vector<std::vector<double>> &histBeforeToday,
                            int window = 20)
{
  std::vector<double> sig = riskParitySignal(histBeforeToday, window);
  for (double &v : sig)
    v *= 0.5; // Scale exposure by 0.5
  return sig;
}

std::vector<double>
crossSectionalMomentumSignal(const std::vector<std::vector<double>> &histBeforeToday,
                             int window = 20)
{
  int nAssets = static_cast<int>(histBeforeToday.size());
  std::vector<double> sig(nAssets, 0.0);
  double total = 0.0;

  for (int a = 0; a < nAssets; ++a)
  {
    const auto &h = histBeforeToday[a];
    int n = static_cast<int>(h.size());
    if (n < window)
      continue;

    double cumRet = 1.0;
    for (int k = n - window; k < n; ++k)
      cumRet *= (1.0 + h[k]);
    cumRet -= 1.0;

    sig[a] = std::max(0.0, cumRet);
    total += sig[a];
  }

  if (total > 1e-12)
    for (double &v : sig)
      v /= total;
  else
    std::fill(sig.begin(), sig.end(), 0.0);

  return sig;
}

std::vector<double>
trendFollowingSignal(const std::vector<std::vector<double>> &histBeforeToday,
                     int window = 60)
{
  int nAssets = static_cast<int>(histBeforeToday.size());
  std::vector<double> sig(nAssets, 0.0);
  std::vector<double> vol = rollingVol(histBeforeToday, 20);
  double total = 0.0;

  for (int a = 0; a < nAssets; ++a)
  {
    const auto &h = histBeforeToday[a];
    int n = static_cast<int>(h.size());
    if (n < window)
      continue;

    double cumRet = 1.0;
    for (int k = n - window; k < n; ++k)
      cumRet *= (1.0 + h[k]);
    cumRet -= 1.0;

    if (cumRet > 0.0)
      sig[a] = 1.0 / std::max(vol[a], 1e-6);
    else
      sig[a] = 0.0;
    total += sig[a];
  }

  if (total > 1e-12)
    for (double &v : sig)
      v /= total;
  else
    std::fill(sig.begin(), sig.end(), 0.0);

  return sig;
}

std::vector<double>
meanReversionSignal(const std::vector<std::vector<double>> &histBeforeToday,
                    int window = 5)
{
  int nAssets = static_cast<int>(histBeforeToday.size());
  std::vector<double> sig(nAssets, 0.0);
  double total = 0.0;

  for (int a = 0; a < nAssets; ++a)
  {
    const auto &h = histBeforeToday[a];
    int n = static_cast<int>(h.size());
    if (n < window)
      continue;

    double cumRet = 1.0;
    for (int k = n - window; k < n; ++k)
      cumRet *= (1.0 + h[k]);
    cumRet -= 1.0;

    sig[a] = std::max(0.0, -cumRet);
    total += sig[a];
  }

  if (total > 1e-12)
    for (double &v : sig)
      v /= total;
  else
    std::fill(sig.begin(), sig.end(), 0.0);

  return sig;
}

// BTRTRLT
std::string DumbRegimes(double z_vol, double z_mom)
{
  // Neutral guard (true inactivity zone)
  if (fabs(z_vol) < 0.5 && fabs(z_mom) < 0.5)
    return "Neutral";

  if (z_vol > 1.0)
    return "Stressed";

  if (z_vol < -1.0)
    return "Compressed";

  if (z_mom > 1.0)
    return "Trending";

  if (z_mom < -1.0)
    return "Transitioning";

  // fallback = structure unclear
  return "Dispersed";
}

std::vector<double> BaselineExposure(
    const std::vector<double> &baseW,
    double z_vol,
    double z_mom)
{
  double vol_scale = 1.0 / (1.0 + std::max(0.0, z_vol));
  double trend_bias = 1.0 + 0.5 * z_mom;

  double scale = vol_scale * trend_bias;

  std::vector<double> w = baseW;

  for (auto &x : w)
    x *= scale;

  return w;
}

// ==========================
// MAIN BACKTEST FUNCTION
// ==========================
void runBacktest(Engine &engine, const std::vector<std::string> &headers,
                 const std::vector<std::vector<double>> &allReturns,
                 const std::vector<std::string> &dates)
{
  int nAssets = static_cast<int>(headers.size());
  int nDays = static_cast<int>(allReturns.size());

  // returnHistory[asset][day up to yesterday] — for shift(1)
  std::vector<std::vector<double>> returnHistory(nAssets);

  // Synthetic prices — cumulative product starting at 100
  std::vector<double> prices(nAssets, 100.0);

  // Portfolio state
  double cashBase0 = INITIAL_CAPITAL;
  double cashBase1 = INITIAL_CAPITAL;
  double cashBase2 = INITIAL_CAPITAL;
  double cashBase3 = INITIAL_CAPITAL;
  double cashBase4 = INITIAL_CAPITAL;
  double cashBase5 = INITIAL_CAPITAL;

  double cashBase = INITIAL_CAPITAL;
  double cashOverlay = INITIAL_CAPITAL;

  // Turnover tracking
  double overlayTurnoverSum = 0.0;
  double baseTurnoverSum = 0.0;
  int numRebalances = 0;

  std::vector<double> posBase0(nAssets, 0.0);
  std::vector<double> posBase1(nAssets, 0.0);
  std::vector<double> posBase2(nAssets, 0.0);
  std::vector<double> posBase3(nAssets, 0.0);
  std::vector<double> posBase4(nAssets, 0.0);
  std::vector<double> posBase5(nAssets, 0.0);

  std::vector<double> posOverlay(nAssets, 0.0);
  std::vector<double> posBase(nAssets, 0.0);

  std::vector<double> histBase0, histBase1, histBase2, histBase3, histBase4, histBase5, histBase, histOverlay;
  histBase0.reserve(nDays);
  histBase1.reserve(nDays);
  histBase2.reserve(nDays);
  histBase3.reserve(nDays);
  histBase4.reserve(nDays);
  histBase5.reserve(nDays);

  histBase.reserve(nDays);
  histOverlay.reserve(nDays);

  bool snapshotReady = false;
  sysOutput snapshot{}; // last valid snapshot, persisted across days

  rollingSharpe rollingsharpe0, rollingsharpe1, rollingsharpe2, rollingsharpe3, rollingsharpe4, rollingsharpe5;

  double sharpe0 = std::numeric_limits<double>::quiet_NaN();
  double sharpe1 = std::numeric_limits<double>::quiet_NaN();
  double sharpe2 = std::numeric_limits<double>::quiet_NaN();
  double sharpe3 = std::numeric_limits<double>::quiet_NaN();
  double sharpe4 = std::numeric_limits<double>::quiet_NaN();
  double sharpe5 = std::numeric_limits<double>::quiet_NaN();

  std::vector<double> sharpes = {sharpe0, sharpe1, sharpe2, sharpe3, sharpe4, sharpe5};
  std::vector<double> stratprobs{1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0};

  std::vector<std::string> regime_labels = {"Stressed", "Transitioning", "Dispersed", "Trending", "Compressed", "Neutral"};
  std::vector<int> indices = {0, 1, 2, 3, 4, 5};
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> dist(0, 5);

  // BTRTRLT
  std::deque<std::vector<double>> windowReturns;
  std::deque<double> volHistory;
  std::deque<double> momHistory;

  std::unordered_map<std::string, int> regimeDays;
  std::unordered_map<std::string, int> regimeTransitions;
  std::string previousRegime = "";

  for (int i = 0; i < nDays; ++i)
  {

    const std::vector<double> &todayReturns = allReturns[i];
    const std::string &ts = dates[i];

    // BASELINE THAT REPLACES THE REGIME LOGIC ITSELF (TEMP)

    /*

    windowReturns.push_back(todayReturns);

    if (windowReturns.size() > 100) //was 20
    {
      windowReturns.pop_front();
    }

    if (windowReturns.size() < 100) //135 it started degrading
      continue;

    bool z_ready = (volHistory.size() >= 250);

    MarketFeatures f = computeFeatures(windowReturns);

    volHistory.push_back(f.vol);
    momHistory.push_back(f.momentum);

    if (volHistory.size() > 250)
      volHistory.pop_front();

    if (momHistory.size() > 250)
      momHistory.pop_front();

    if (z_ready)
    {
      f.vol_z = z_score(volHistory, f.vol);
      f.mom_z = z_score(momHistory, f.momentum);
    }
    else
    {
      f.vol_z = 0.0;
      f.mom_z = 0.0;
    }

    auto dumb_regime = DumbRegimes(f.vol_z, f.mom_z);

    */
    //================================================

    bool isRebalanceDay = (i % REBALANCE_DAYS == 0);

    // --- Update synthetic prices ---
    for (int a = 0; a < nAssets; ++a)
      prices[a] = prices[a] * (1.0 + todayReturns[a]);

    // --- Regime-dependent Base Signal (shift-1: use history BEFORE today) ---
    std::vector<double> baseSig;

    std::string currentRegime = snapshotReady ? snapshot.decision_regime : "Neutral";

    regimeDays[currentRegime]++;
    if (currentRegime != previousRegime)
    {
      regimeTransitions[currentRegime]++;
      previousRegime = currentRegime;
    }

    // std::string currentRegime = dumb_regime;
    // std::string currentRegime = regime_labels[dist(gen)];

    std::vector<std::vector<double>> baseSignals(6);

    baseSignals[0] = longVolatilitySignal(returnHistory);

    baseSignals[1] = reducedRiskRiskParitySignal(returnHistory);

    baseSignals[2] = crossSectionalMomentumSignal(returnHistory);

    baseSignals[3] = trendFollowingSignal(returnHistory);

    baseSignals[4] = meanReversionSignal(returnHistory);

    baseSignals[5] = riskParitySignal(returnHistory);

    /*
    for (const auto &x : indices)
    {
      LOG(x);
    }
    LOG("");

    // Shuffle the indices vector
    std::shuffle(indices.begin(), indices.end(), gen);
    */

    if (currentRegime == regime_labels[0])
      baseSig = longVolatilitySignal(returnHistory);
    else if (currentRegime == regime_labels[1])
      baseSig = reducedRiskRiskParitySignal(returnHistory);
    else if (currentRegime == regime_labels[2])
      baseSig = crossSectionalMomentumSignal(returnHistory);
    else if (currentRegime == regime_labels[3])
      baseSig = trendFollowingSignal(returnHistory);
    else if (currentRegime == regime_labels[4])
      baseSig = meanReversionSignal(returnHistory);
    else if (currentRegime == regime_labels[5])
      baseSig = riskParitySignal(returnHistory);

    /*
    if (currentRegime == regime_labels[indices[0]])
      baseSig = longVolatilitySignal(returnHistory);
    else if (currentRegime == regime_labels[indices[1]])
      baseSig = reducedRiskRiskParitySignal(returnHistory);
    else if (currentRegime == regime_labels[indices[2]])
      baseSig = crossSectionalMomentumSignal(returnHistory);
    else if (currentRegime == regime_labels[indices[3]])
      baseSig = trendFollowingSignal(returnHistory);
    else if (currentRegime == regime_labels[indices[4]])
      baseSig = meanReversionSignal(returnHistory);
    else if (currentRegime == regime_labels[indices[5]])
      baseSig = riskParitySignal(returnHistory);
    */

    // --- Push today into history AFTER computing signal ---
    for (int a = 0; a < nAssets; ++a)
      returnHistory[a].push_back(todayReturns[a]);

    // --- Engine ingestion ---
    engine.ingest(todayReturns);
    if (engine.ready())
    {
      snapshot = engine.getOutput();
      snapshotReady = true;
    }

    // Base helper NAV('s)
    double valBase0 = cashBase0 + dotSum(posBase0, prices);
    double valBase1 = cashBase1 + dotSum(posBase1, prices);
    double valBase2 = cashBase2 + dotSum(posBase2, prices);
    double valBase3 = cashBase3 + dotSum(posBase3, prices);
    double valBase4 = cashBase4 + dotSum(posBase4, prices);
    double valBase5 = cashBase5 + dotSum(posBase5, prices);

    double sumExp = 0.0;
    for (const auto &x : sharpes)
    {
      if (!std::isnan(x))
      {
        sumExp += std::exp(x);
      }
      else
      {
        sumExp = std::numeric_limits<double>::quiet_NaN();
        break; // If any is NaN, we can't compute Softmax
      }
    }

    if (!std::isnan(sumExp) && sumExp > 0)
    {
      for (size_t i = 0; i < sharpes.size(); i++)
      {
        stratprobs[i] = std::exp(sharpes[i]) / sumExp;
      }
    }

    // Overlay NAV
    double valOverlay = cashOverlay + dotSum(posOverlay, prices);

    // Adaptive Base overlay
    double valBase = cashBase + dotSum(posBase, prices);

    //----------------------------------------------------------

    // std::vector<double> targetWBase = blendPortfolios(baseSignals, stratprobs);
    std::vector<double> targetWBase = equalWeightBlend(baseSignals);
    std::vector<double> targetWOverlay = baseSig; // Start from baseSig (overlay will re-normalize itself)

    // bool isEmergency = false;

    std::cout << i << " " << valBase << " " << i << " " << valOverlay << std::endl;

    // ---- Execute Overlay ----
    if (isRebalanceDay /*|| isEmergency*/)
    {
      double overlayTradeVol = 0.0;
      for (int a = 0; a < nAssets; ++a)
      {
        double targetDollars = targetWOverlay[a] * valOverlay;
        double currentDollars = posOverlay[a] * prices[a];
        double trade = targetDollars - currentDollars;
        overlayTradeVol += std::abs(trade);
        double cost = std::abs(trade) * (TRANSACTION_COST_BPS / 10000.0);
        cashOverlay = cashOverlay - cost;
        posOverlay[a] = posOverlay[a] + trade / prices[a];
        cashOverlay = cashOverlay - trade;
      }
      if (valOverlay > 1e-6)
        overlayTurnoverSum += overlayTradeVol / valOverlay;
    }

    if (isRebalanceDay /*|| isEmergency*/)
    {
      double baseTradeVol = 0.0;
      for (int a = 0; a < nAssets; ++a)
      {
        double targetDollars = targetWBase[a] * valBase;
        double currentDollars = posBase[a] * prices[a];
        double trade = targetDollars - currentDollars;
        baseTradeVol += std::abs(trade);
        double cost = std::abs(trade) * (TRANSACTION_COST_BPS / 10000.0);
        cashBase = cashBase - cost;
        posBase[a] = posBase[a] + trade / prices[a];
        cashBase = cashBase - trade;
      }
      if (valBase > 1e-6)
        baseTurnoverSum += baseTradeVol / valBase;
      numRebalances++;
    }

    // ---- Execute Base Strat(s) ----
    if (isRebalanceDay)
    {
      for (int a = 0; a < nAssets; ++a)
      {
        double targetDollars = baseSignals[0][a] * valBase0;
        double currentDollars = posBase0[a] * prices[a];
        double trade = targetDollars - currentDollars;
        double cost = std::abs(trade) * (TRANSACTION_COST_BPS / 10000.0);
        cashBase0 -= cost;
        posBase0[a] += trade / prices[a];
        cashBase0 -= trade;
      }
    }

    if (isRebalanceDay)
    {
      for (int a = 0; a < nAssets; ++a)
      {
        double targetDollars = baseSignals[1][a] * valBase1;
        double currentDollars = posBase1[a] * prices[a];
        double trade = targetDollars - currentDollars;
        double cost = std::abs(trade) * (TRANSACTION_COST_BPS / 10000.0);
        cashBase1 -= cost;
        posBase1[a] += trade / prices[a];
        cashBase1 -= trade;
      }
    }

    if (isRebalanceDay)
    {
      for (int a = 0; a < nAssets; ++a)
      {
        double targetDollars = baseSignals[2][a] * valBase2;
        double currentDollars = posBase2[a] * prices[a];
        double trade = targetDollars - currentDollars;
        double cost = std::abs(trade) * (TRANSACTION_COST_BPS / 10000.0);
        cashBase2 -= cost;
        posBase2[a] += trade / prices[a];
        cashBase2 -= trade;
      }
    }

    if (isRebalanceDay)
    {
      for (int a = 0; a < nAssets; ++a)
      {
        double targetDollars = baseSignals[3][a] * valBase3;
        double currentDollars = posBase3[a] * prices[a];
        double trade = targetDollars - currentDollars;
        double cost = std::abs(trade) * (TRANSACTION_COST_BPS / 10000.0);
        cashBase3 -= cost;
        posBase3[a] += trade / prices[a];
        cashBase3 -= trade;
      }
    }

    if (isRebalanceDay)
    {
      for (int a = 0; a < nAssets; ++a)
      {
        double targetDollars = baseSignals[4][a] * valBase4;
        double currentDollars = posBase4[a] * prices[a];
        double trade = targetDollars - currentDollars;
        double cost = std::abs(trade) * (TRANSACTION_COST_BPS / 10000.0);
        cashBase4 -= cost;
        posBase4[a] += trade / prices[a];
        cashBase4 -= trade;
      }
    }

    if (isRebalanceDay)
    {
      for (int a = 0; a < nAssets; ++a)
      {
        double targetDollars = baseSignals[5][a] * valBase5;
        double currentDollars = posBase5[a] * prices[a];
        double trade = targetDollars - currentDollars;
        double cost = std::abs(trade) * (TRANSACTION_COST_BPS / 10000.0);
        cashBase5 -= cost;
        posBase5[a] += trade / prices[a];
        cashBase5 -= trade;
      }
    }

    // Base strat(s)
    valBase0 = cashBase0 + dotSum(posBase0, prices);
    valBase1 = cashBase1 + dotSum(posBase1, prices);
    valBase2 = cashBase2 + dotSum(posBase2, prices);
    valBase3 = cashBase3 + dotSum(posBase3, prices);
    valBase4 = cashBase4 + dotSum(posBase4, prices);
    valBase5 = cashBase5 + dotSum(posBase5, prices);

    rollingsharpe0.push(valBase0);
    rollingsharpe1.push(valBase1);
    rollingsharpe2.push(valBase2);
    rollingsharpe3.push(valBase3);
    rollingsharpe4.push(valBase4);
    rollingsharpe5.push(valBase5);

    sharpes[0] = rollingsharpe0.getSharpe();
    sharpes[1] = rollingsharpe1.getSharpe();
    sharpes[2] = rollingsharpe2.getSharpe();
    sharpes[3] = rollingsharpe3.getSharpe();
    sharpes[4] = rollingsharpe4.getSharpe();
    sharpes[5] = rollingsharpe5.getSharpe();

    // Overlay
    valOverlay = cashOverlay + dotSum(posOverlay, prices);
    // Adaptive Base
    valBase = cashBase + dotSum(posBase, prices);

    // history
    histBase0.push_back(valBase0);
    histBase1.push_back(valBase1);
    histBase2.push_back(valBase2);
    histBase3.push_back(valBase3);
    histBase4.push_back(valBase4);
    histBase5.push_back(valBase5);

    histBase.push_back(valBase);
    histOverlay.push_back(valOverlay);
  }

  // ==========================
  // FINAL RESULTS (NON ROLLING)
  // ==========================
  auto [finalB, sharpeB, ddB] = calcMetrics(histBase);
  auto [finalO, sharpeO, ddO] = calcMetrics(histOverlay);

  std::string sep(40, '=');
  std::cerr << "\n"
            << sep << "\n";
  std::cerr << std::setw(30) << "BACKTEST RESULTS" << "\n";
  std::cerr << sep << "\n";
  std::cerr << std::left << std::setw(15) << "Metric"
            << " | " << std::setw(10) << "Smart Base"
            << " | " << "Avarent Overlay\n";
  std::cerr << std::string(40, '-') << "\n";
  std::cerr << std::setw(15) << "Final Value"
            << " | $" << std::fixed << std::setprecision(0) << std::setw(9)
            << finalB << " | $" << finalO << "\n";
  std::cerr << std::setw(15) << "Sharpe Ratio"
            << " | " << std::setprecision(2) << std::setw(10) << sharpeB
            << " | " << sharpeO << "\n";
  std::cerr << std::setw(15) << "Max Drawdown"
            << " | " << std::setprecision(2) << ddB * 100.0 << "%"
            << "      | " << ddO * 100.0 << "%\n";
  std::cerr << sep << "\n";

  double rebalPerYear = 252.0 / REBALANCE_DAYS;
  double annTurnoverOverlay = numRebalances > 0 ? (overlayTurnoverSum / numRebalances) * rebalPerYear : 0.0;
  double annTurnoverBase = numRebalances > 0 ? (baseTurnoverSum / numRebalances) * rebalPerYear : 0.0;

  std::cerr << "\n";
  std::cerr << std::string(40, '=') << "\n";
  std::cerr << std::setw(30) << "TURNOVER METRICS" << "\n";
  std::cerr << std::string(40, '=') << "\n";
  std::cerr << std::left << std::setw(20) << "Overlay Ann. TO"
            << " | " << std::fixed << std::setprecision(2) << annTurnoverOverlay * 100.0 << "%\n";
  std::cerr << std::setw(20) << "Base Ann. TO"
            << " | " << annTurnoverBase * 100.0 << "%\n";
  std::cerr << std::setw(20) << "Rebalances"
            << " | " << numRebalances << "\n";
  std::cerr << std::string(40, '=') << "\n";

  std::string sep2(70, '=');
  std::cerr << "\n"
            << sep2 << "\n";
  std::cerr << std::setw(45) << "REGIME DURATION METRICS" << "\n";
  std::cerr << sep2 << "\n";
  std::cerr << std::left << std::setw(20) << "Regime"
            << " | " << std::setw(19) << "Avg Duration (days)"
            << " | " << std::setw(11) << "Transitions"
            << " | " << "% of Sample\n";
  std::cerr << std::string(70, '-') << "\n";

  for (const auto &pair : regimeDays)
  {
    std::string r = pair.first;
    int days = pair.second;
    int trans = regimeTransitions[r];
    double avgDur = trans > 0 ? (double)days / trans : 0.0;
    double pct = (double)days / nDays * 100.0;

    std::cerr << std::left << std::setw(20) << r
              << " | " << std::setw(19) << std::fixed << std::setprecision(1) << avgDur
              << " | " << std::setw(11) << trans
              << " | " << std::setprecision(1) << pct << "%\n";
  }
  std::cerr << sep2 << "\n";

  /*

  for (int i = 0; i < engine.htts.size(); i++)
  {
    std::cerr << "(" << engine.htts[i][0] << ", " << engine.htts[i][1] << ", " << engine.htts[i][2] << ")" << "\n";
  }

  */

  std::ofstream out("data.bin", std::ios::binary);

  uint64_t rows = engine.htts.size();
  uint64_t cols = engine.htts[0].size();

  out.write(reinterpret_cast<char *>(&rows), sizeof(rows));
  out.write(reinterpret_cast<char *>(&cols), sizeof(cols));

  for (const auto &row : engine.htts)
    out.write(reinterpret_cast<const char *>(row.data()),
              cols * sizeof(double));
}

// ==========================
// SHUFFLE HARNESS (P-VALUE)
// ==========================
// Pre-computes prices and all 6 signals once, then replays
// the overlay simulation N times with uniformly random regime
// labels to build a null distribution for Sharpe.

void runShuffleHarness(const std::vector<std::string> &headers,
                       const std::vector<std::vector<double>> &allReturns,
                       int numIterations)
{
  int nAssets = static_cast<int>(headers.size());
  int nDays = static_cast<int>(allReturns.size());

  // --- Phase 1: Pre-compute deterministic data ---
  std::vector<std::vector<double>> returnHistory(nAssets);
  std::vector<double> prices(nAssets, 100.0);

  // Store prices and all 6 signal vectors for each day
  std::vector<std::vector<double>> pricesPerDay(nDays);
  // signals[day][strategy] = weight vector
  std::vector<std::vector<std::vector<double>>> signals(nDays,
                                                        std::vector<std::vector<double>>(6));

  for (int i = 0; i < nDays; ++i)
  {
    const auto &todayReturns = allReturns[i];

    // Update synthetic prices (same order as runBacktest)
    for (int a = 0; a < nAssets; ++a)
      prices[a] *= (1.0 + todayReturns[a]);
    pricesPerDay[i] = prices;

    // Compute all 6 signals from returnHistory BEFORE today's push
    signals[i][0] = longVolatilitySignal(returnHistory);
    signals[i][1] = reducedRiskRiskParitySignal(returnHistory);
    signals[i][2] = crossSectionalMomentumSignal(returnHistory);
    signals[i][3] = trendFollowingSignal(returnHistory);
    signals[i][4] = meanReversionSignal(returnHistory);
    signals[i][5] = riskParitySignal(returnHistory);

    // Push today into history
    for (int a = 0; a < nAssets; ++a)
      returnHistory[a].push_back(todayReturns[a]);
  }

  std::cerr << "\n[SHUFFLE HARNESS] Pre-computation done. "
            << nDays << " days, " << nAssets << " assets.\n";
  std::cerr << "[SHUFFLE HARNESS] Running " << numIterations
            << " random-regime iterations...\n";

  // --- Phase 2: Run N overlay simulations with random regimes ---
  std::mt19937 gen(42); // fixed seed for reproducibility
  std::uniform_int_distribution<> regimeDist(0, 5);

  std::vector<double> shuffledSharpes(numIterations);
  std::vector<double> shuffledReturns(numIterations);
  std::vector<double> shuffledMDDs(numIterations);

  for (int iter = 0; iter < numIterations; ++iter)
  {
    double cash = INITIAL_CAPITAL;
    std::vector<double> pos(nAssets, 0.0);
    std::vector<double> navHist;
    navHist.reserve(nDays);

    for (int i = 0; i < nDays; ++i)
    {
      const auto &p = pricesPerDay[i];

      // Random regime label
      int regimeIdx = regimeDist(gen);
      const auto &baseSig = signals[i][regimeIdx];

      double nav = cash + dotSum(pos, p);

      bool isRebalanceDay = (i % REBALANCE_DAYS == 0);

      if (isRebalanceDay && !baseSig.empty())
      {
        for (int a = 0; a < nAssets; ++a)
        {
          double targetDollars = baseSig[a] * nav;
          double currentDollars = pos[a] * p[a];
          double trade = targetDollars - currentDollars;
          double cost = std::abs(trade) * (TRANSACTION_COST_BPS / 10000.0);
          cash -= cost;
          pos[a] += trade / p[a];
          cash -= trade;
        }
      }

      nav = cash + dotSum(pos, p);
      navHist.push_back(nav);
    }

    auto metrics = calcMetrics(navHist);
    shuffledSharpes[iter] = metrics.sharpe;
    shuffledReturns[iter] = metrics.finalValue;
    shuffledMDDs[iter] = metrics.maxDrawdown;

    if ((iter + 1) % 100 == 0)
      std::cerr << "  ... completed " << (iter + 1) << "/" << numIterations << "\n";
  }

  // --- Phase 3: Compute and print distribution statistics ---
  std::sort(shuffledSharpes.begin(), shuffledSharpes.end());

  double mean = 0.0;
  for (double s : shuffledSharpes)
    mean += s;
  mean /= numIterations;

  double variance = 0.0;
  for (double s : shuffledSharpes)
    variance += (s - mean) * (s - mean);
  variance /= numIterations;
  double stddev = std::sqrt(variance);

  double minS = shuffledSharpes.front();
  double maxS = shuffledSharpes.back();
  double p5 = shuffledSharpes[static_cast<int>(0.05 * numIterations)];
  double p25 = shuffledSharpes[static_cast<int>(0.25 * numIterations)];
  double p50 = shuffledSharpes[static_cast<int>(0.50 * numIterations)];
  double p75 = shuffledSharpes[static_cast<int>(0.75 * numIterations)];
  double p95 = shuffledSharpes[static_cast<int>(0.95 * numIterations)];

  // Mean return and MDD
  double meanRet = 0.0, meanMDD = 0.0;
  for (int i = 0; i < numIterations; i++)
  {
    meanRet += shuffledReturns[i];
    meanMDD += shuffledMDDs[i];
  }
  meanRet /= numIterations;
  meanMDD /= numIterations;

  std::string sep(70, '=');
  std::cerr << "\n"
            << sep << "\n";
  std::cerr << std::setw(45) << "SHUFFLED LABEL TEST RESULTS" << "\n";
  std::cerr << sep << "\n";
  std::cerr << std::left << std::fixed;
  std::cerr << std::setw(30) << "  Iterations"
            << " : " << numIterations << "\n";
  std::cerr << std::setw(30) << "  Mean Sharpe"
            << " : " << std::setprecision(4) << mean << "\n";
  std::cerr << std::setw(30) << "  Std Sharpe"
            << " : " << stddev << "\n";
  std::cerr << std::setw(30) << "  Min Sharpe"
            << " : " << minS << "\n";
  std::cerr << std::setw(30) << "  Max Sharpe"
            << " : " << maxS << "\n";
  std::cerr << std::setw(30) << "  5th Percentile"
            << " : " << p5 << "\n";
  std::cerr << std::setw(30) << "  25th Percentile"
            << " : " << p25 << "\n";
  std::cerr << std::setw(30) << "  Median"
            << " : " << p50 << "\n";
  std::cerr << std::setw(30) << "  75th Percentile"
            << " : " << p75 << "\n";
  std::cerr << std::setw(30) << "  95th Percentile"
            << " : " << p95 << "\n";
  std::cerr << std::setw(30) << "  Mean Final NAV"
            << " : $" << std::setprecision(0) << meanRet << "\n";
  std::cerr << std::setw(30) << "  Mean Max Drawdown"
            << " : " << std::setprecision(2) << meanMDD * 100.0 << "%\n";
  std::cerr << sep << "\n";

  // Print all individual Sharpes for the record (one per line, easy to parse)
  std::cerr << "\n[RAW SHARPE VALUES - sorted]\n";
  for (int i = 0; i < numIterations; i++)
  {
    std::cerr << std::setprecision(4) << shuffledSharpes[i];
    if (i + 1 < numIterations)
      std::cerr << ", ";
    if ((i + 1) % 20 == 0)
      std::cerr << "\n";
  }
  std::cerr << "\n";

  // P-value helper: count how many exceed each threshold
  std::cerr << "\n[P-VALUE HELPER]\n";
  std::cerr << "To compute empirical p-value: count shuffled Sharpes >= your true Sharpe.\n";
  std::cerr << "  # >= 0.30 : ";
  int cnt30 = 0;
  for (double s : shuffledSharpes)
    if (s >= 0.30)
      cnt30++;
  std::cerr << cnt30 << " / " << numIterations << " (p = " << std::setprecision(4)
            << (double)cnt30 / numIterations << ")\n";
  std::cerr << "  # >= 0.69 : ";
  int cnt69 = 0;
  for (double s : shuffledSharpes)
    if (s >= 0.69)
      cnt69++;
  std::cerr << cnt69 << " / " << numIterations << " (p = "
            << (double)cnt69 / numIterations << ")\n";
  std::cerr << "  # >= 0.72 : ";
  int cnt72 = 0;
  for (double s : shuffledSharpes)
    if (s >= 0.72)
      cnt72++;
  std::cerr << cnt72 << " / " << numIterations << " (p = "
            << (double)cnt72 / numIterations << ")\n";
  std::cerr << sep << "\n";
}
