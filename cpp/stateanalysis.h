#pragma once

#include <iostream>
#include <vector>
#include <deque>
#include <limits>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <set>
#include "utils.h"

// MAKE THEM INLINE YOU IDIOT

// what i need to do:
/*

- define regimes first
    - GEC combinations classify certain regimes
- seperate interpretation from classification, just a simple enum to string.
- THEN use patterns, bring in eigenstuff. Top contributors.

- final report combines everything with what regime, the state, what causes it, why its important, and what to do next (implication)

*/

struct GECStates
{

    GECStates(int rollinghistsize, int stride) : rhs(rollinghistsize), stride_rhs(stride) {}

private:
    int rhs;
    int stride_rhs;

    int G = 0;
    int E = 0;
    int C = 0;

    int G_transition_state = 0; // 1 soft, 2 hard
    int E_transition_state = 0; // 0 = undecided yet
    int C_transition_state = 0;

    int G_spike = 0;
    int E_spike = 0;
    int C_spike = 0;

    bool ready = false;

    std::deque<int> rollinghistory_G;
    std::deque<int> rollinghistory_E;
    std::deque<int> rollinghistory_C;

public:
    void update()
    {
        if (rollinghistory_G.size() >= rhs) // assuming E and C are the same size too
        {
            ready = true;

            auto [G_t, G_spike] = setTransitionType(rollinghistory_G);
            auto [E_t, E_spike] = setTransitionType(rollinghistory_E);
            auto [C_t, C_spike] = setTransitionType(rollinghistory_C);

            setG_t(G_t);
            setE_t(E_t);
            setC_t(C_t);

            setG_spike(G_spike);
            setE_spike(E_spike);
            setC_spike(C_spike);

            for (size_t i = 0; i < stride_rhs; ++i)
            {
                rollinghistory_G.pop_front();
                rollinghistory_G.pop_front();
                rollinghistory_G.pop_front();
            }
        }
        else
        {
            rollinghistory_G.push_back(G);
            rollinghistory_E.push_back(E);
            rollinghistory_C.push_back(C);
        }
    }

    std::tuple<int, int> setTransitionType(const std::deque<int> &buffer)
    {
        if (buffer.size() < 3)
            throw std::invalid_argument("not enough data in buffer, (set transition type)"); // not enough data for spike

        int maxDelta = 0;
        bool hasRise = false;
        bool hasDrop = false;
        int spikeStrength = 0;

        for (size_t i = 1; i < buffer.size(); ++i)
        {
            int prev = buffer[i - 1];
            int curr = buffer[i];

            int delta = curr - prev;
            int absDelta = abs(delta);

            // Track largest step change
            if (absDelta > maxDelta)
                maxDelta = absDelta;

            // Track direction
            if (delta > 0)
                hasRise = true;
            else if (delta < 0)
                hasDrop = true;

            // Detect spike (sign flip)
            if (i >= 2)
            {
                int prevDelta = buffer[i - 1] - buffer[i - 2];

                // sign change = spike
                if ((prevDelta > 0 && delta < 0) || (prevDelta < 0 && delta > 0))
                {
                    spikeStrength = abs(prevDelta) + abs(delta);
                }
            }
        }

        // ---- Classification ----

        int spike_present = 0;

        // Case: spike (rise + drop in one motion)
        if (hasRise && hasDrop && spikeStrength > 0)
        {
            if (spikeStrength >= 4)
                spike_present = 1;
        }

        // Case: normal transitions (only rise OR only drop)
        if (maxDelta <= 2)
            return {1, spike_present}; // soft
        else
            return {2, spike_present}; // hard
    }
    // setters

    //  add some guarding later

    void setG(int x)
    {
        G = x;
    }
    void setE(int x)
    {
        E = x;
    }

    void setC(int x)
    {
        C = x;
    }

    void setG_t(int x)
    {
        G_transition_state = x;
    }

    void setE_t(int x)
    {
        E_transition_state = x;
    }

    void setC_t(int x)
    {
        E_transition_state = x;
    }

    void setG_spike(int x)
    {
        G_spike = x;
    }

    void setE_spike(int x)
    {
        E_spike = x;
    }

    void setC_spike(int x)
    {
        C_spike = x;
    }

    // getters

    int getG()
    {
        return G;
    }

    int getE()
    {
        return E;
    }

    int getC()
    {
        return C;
    }

    int getG_t()
    {
        return G_transition_state;
    }

    int getE_t()
    {
        return E_transition_state;
    }

    int getC_t()
    {
        return C_transition_state;
    }

    int getG_spike()
    {
        return G_spike;
    }

    int getE_spike()
    {
        return E_spike;
    }

    int getC_spike()
    {
        return C_spike;
    }

    bool isReady()
    {
        return ready;
    }
};

struct Patterns
{
    std::vector<std::string> headers;

    Eigen::MatrixXd eigenVectors; // columns = components
    Eigen::VectorXd eigenValues;

    int eff; // effective rank
};

struct domEigenvectors
{

    Eigen::VectorXd ep1;
    Eigen::VectorXd ep2;
    Eigen::VectorXd ep3;

    bool init = false;
};

struct FactorTracker
{
    Eigen::VectorXd smoothed;
    std::vector<int> last_top;
    int persistence_count = 0;
    bool initialized = false;

    double alpha = 0.2;           // smoothing
    int required_persistence = 3; // hysteresis

    void reset()
    {
        initialized = false;
        persistence_count = 0;
        last_top.clear();
    }
};

enum class CurRegime
{
    // Core structure
    SingleFactor,
    StrongDominance,
    MultiFactor,
    WeakStructure,

    // Dynamics
    StableTrend,
    StableDispersion,
    Exploratory,
    Transitioning,
    HardSwitch,

    // Risk states
    Compression,
    Expansion,
    Chaotic,
    VolatilitySpike,

    // Edge cases
    Recovery,
    Exhaustion,

    Unknown
};

inline double temporalConsistency(const std::deque<CurRegime>& h)
{
    int count = 0;
    for (auto r : h)
        if (r == h.back()) count++;

    return (double)count / h.size();
}


inline std::string enum_to_string_regime(CurRegime regime)
{
    switch (regime)
    {
    case CurRegime::SingleFactor:
        return "SingleFactor";
    case CurRegime::StrongDominance:
        return "StrongDominance";
    case CurRegime::MultiFactor:
        return "MultiFactor";
    case CurRegime::WeakStructure:
        return "WeakStructure";

    case CurRegime::StableTrend:
        return "StableTrend";
    case CurRegime::StableDispersion:
        return "StableDispersion";
    case CurRegime::Exploratory:
        return "Exploratory";
    case CurRegime::Transitioning:
        return "Transitioning";
    case CurRegime::HardSwitch:
        return "HardSwitch";

    case CurRegime::Compression:
        return "Compression";
    case CurRegime::Expansion:
        return "Expansion";
    case CurRegime::Chaotic:
        return "Chaotic";
    case CurRegime::VolatilitySpike:
        return "VolatilitySpike";

    case CurRegime::Recovery:
        return "Recovery";
    case CurRegime::Exhaustion:
        return "Exhaustion";

    case CurRegime::Unknown:
        return "Unknown";
    default:
        return "Unknown";
    }
}

inline std::string interpret(CurRegime r)
{
    switch (r)
    {
    case CurRegime::SingleFactor:
        return "Market is driven by a single dominant factor. Correlations are high and price action is strongly directional.";

    case CurRegime::StrongDominance:
        return "One combination of factors overwhelmingly controls market behavior. Extreme concentration and reduced diversification benefits.";

    case CurRegime::MultiFactor:
        return "Multiple independent drivers are active. Cross-sectional dispersion is elevated and correlations are lower.";

    case CurRegime::WeakStructure:
        return "No clear structure or dominance. Signals are weak, noisy, and prone to false positives.";

    case CurRegime::StableTrend:
        return "Persistent and stable directional movement. Trends are orderly with low structural disruption.";

    case CurRegime::StableDispersion:
        return "Consistent multi-factor regime. Relative relationships between assets are stable and exploitable.";

    case CurRegime::Exploratory:
        return "Market is searching for a new structure. Behavior is tentative with shifting correlations and leadership.";

    case CurRegime::Transitioning:
        return "Existing structure is breaking down. Increasing uncertainty as the market moves between regimes.";

    case CurRegime::HardSwitch:
        return "Abrupt and decisive structural break. Previous relationships no longer hold; rapid repricing underway.";

    case CurRegime::Compression:
        return "Market dynamics are tightening. Volatility and dispersion are decreasing, often preceding a breakout.";

    case CurRegime::Expansion:
        return "Market dynamics are broadening. Increasing dispersion and opportunity set across multiple factors.";

    case CurRegime::Chaotic:
        return "Highly unstable regime. Erratic price action, breakdown of structure, and unreliable signals.";

    case CurRegime::VolatilitySpike:
        return "Sudden surge in volatility. Short-term instability and dislocations dominate behavior.";

    case CurRegime::Recovery:
        return "Stability is returning after a period of disruption. Structure is reforming and signals improving.";

    case CurRegime::Exhaustion:
        return "Prevailing trend or structure is weakening. Momentum fading and risk of reversal increasing.";

    default:
        return "No clear regime detected. Market structure is ambiguous.";
    }
}

inline CurRegime classifyRegime(GECStates s)
{
    // HARD SWITCH
    if (s.getG_spike() == 1)
        return CurRegime::HardSwitch;

    if (s.getE_spike() == 1)
        return CurRegime::VolatilitySpike;

    if (s.getE() == 2)
        return CurRegime::Chaotic;

    // DOMINANCE / CONFINEMENT LOGIC
    if (s.getC() == -2 && s.getG() == -2)
        return CurRegime::StrongDominance;

    if (s.getC() <= -1 && s.getG() <= -1)
        return CurRegime::SingleFactor;

    if (s.getC() >= 1 && s.getG() >= 1)
        return CurRegime::MultiFactor;

    // STABILITY / EXPLORATION
    if (s.getG() <= -1 && s.getE() <= 0)
        return CurRegime::StableTrend;

    if (s.getG() >= 1 && s.getE() <= 0)
        return CurRegime::StableDispersion;

    if (s.getG() >= 1 && s.getE() >= 1)
        return CurRegime::Exploratory;

    // STRUCTURAL TRANSITIONS
    if (s.getC() >= 1 && s.getE() >= 1)
        return CurRegime::Transitioning;

    if (s.getC_t() == 2)
        return CurRegime::Compression;

    if (s.getE_t() == 2 && s.getG_t() == 1) //added g_t //1 was before
        return CurRegime::Expansion;

    // EDGE CONDITIONS
    if (s.getE() == -1 && s.getG() == -1)
        return CurRegime::Recovery;

    if (s.getG() == 0 && s.getC() == 0)
        return CurRegime::WeakStructure;

    if (s.getG() == -1 && s.getE() >= 1)
        return CurRegime::Exhaustion;

    return CurRegime::Unknown;
}

inline void normalize(Eigen::VectorXd &v)
{
    double s = v.sum();
    if (s > 1e-12)
        v /= s;
}

inline std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
getTop3EigenVectors_Managed(const Patterns &p, domEigenvectors &r)
{
    // --- First iteration: initialize reference vectors and return early ---
    if (!r.init)
    {
        r.ep1 = p.eigenVectors.col(0);
        r.ep2 = p.eigenVectors.col(1);
        r.ep3 = p.eigenVectors.col(2);
        r.init = true;
        return {r.ep1, r.ep2, r.ep3};
    }

    // --- Compute similarity of every current eigenvector against each reference ---
    const int n = p.eigenVectors.cols();

    std::vector<double> ep1_sim(n), ep2_sim(n), ep3_sim(n);
    for (int i = 0; i < n; i++)
    {
        auto col = p.eigenVectors.col(i);
        ep1_sim[i] = std::abs(cosineSimilarity(col, r.ep1));
        ep2_sim[i] = std::abs(cosineSimilarity(col, r.ep2));
        ep3_sim[i] = std::abs(cosineSimilarity(col, r.ep3));
    }

    // --- Greedy best-match with conflict resolution (no two axes share a column) ---
    std::set<int> used;

    auto getBestMatch = [&](const std::vector<double> &sim) -> int
    {
        int best = -1;
        double bestVal = -1.0;
        for (int i = 0; i < (int)sim.size(); i++)
        {
            if (!used.count(i) && sim[i] > bestVal)
            {
                bestVal = sim[i];
                best = i;
            }
        }
        used.insert(best);
        return best;
    };

    int idx1 = getBestMatch(ep1_sim);
    int idx2 = getBestMatch(ep2_sim);
    int idx3 = getBestMatch(ep3_sim);

    // --- Extract matched eigenvectors ---
    Eigen::VectorXd e1 = p.eigenVectors.col(idx1);
    Eigen::VectorXd e2 = p.eigenVectors.col(idx2); // So auto silently gave me a read-only proxy, not a vector.
    Eigen::VectorXd e3 = p.eigenVectors.col(idx3);

    // --- Fix sign ambiguity ---
    if (r.ep1.dot(e1) < 0)
        e1 = -e1;
    if (r.ep2.dot(e2) < 0)
        e2 = -e2;
    if (r.ep3.dot(e3) < 0)
        e3 = -e3;

    // --- Update references for next iteration ---
    r.ep1 = e1;
    r.ep2 = e2;
    r.ep3 = e3;

    return {e1, e2, e3};
}

inline std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
getTop3EigenVectors(const Patterns &p)
{
    return {p.eigenVectors.col(0), p.eigenVectors.col(1),p.eigenVectors.col(2)};
}

inline std::vector<double> eigenStateProjections3dim(const Eigen::VectorXd &e1, const Eigen::VectorXd &e2, const Eigen::VectorXd &e3, const Eigen::VectorXd &s)
{
    /*
    float proj1 = pow(e1.dot(s), 2) / s.dot(s);
    float proj2 = pow(e2.dot(s), 2) / s.dot(s);
    float proj3 = pow(e3.dot(s), 2) / s.dot(s);

    // return {proj1, proj2, proj3};

    float alpha = 4.0f; // tune this

    float p1 = exp(alpha * proj1);
    float p2 = exp(alpha * proj2);
    float p3 = exp(alpha * proj3);

    float sum = p1 + p2 + p3;
    */

    //return {p1 / sum, p2 / sum, p3 / sum};
    return {e1.dot(s) * e1.dot(s), e2.dot(s) * e2.dot(s), e3.dot(s) * e3.dot(s)}; //squared. dominance profile
}

inline std::vector<std::pair<int, double>>
getTopK(const Eigen::VectorXd &eig, FactorTracker &tracker, int topN)
{
    Eigen::VectorXd importance = eig;
    normalize(importance);

    // --- EMA smoothing ---
    if (!tracker.initialized)
    {
        tracker.smoothed = importance;
        tracker.initialized = true;
    }
    else
    {
        tracker.smoothed =
            tracker.alpha * importance +
            (1.0 - tracker.alpha) * tracker.smoothed;
    }

    // --- build pairs ---
    std::vector<std::pair<int, double>> pairs;
    for (int i = 0; i < tracker.smoothed.size(); ++i)
        pairs.emplace_back(i, tracker.smoothed(i));

    std::sort(pairs.begin(), pairs.end(),
              [](auto &a, auto &b)
              { return a.second > b.second; });

    // --- candidate ---
    std::vector<int> candidate;
    for (int i = 0; i < std::min(topN, (int)pairs.size()); ++i)
        candidate.push_back(pairs[i].first);

    // --- INIT ---
    if (tracker.last_top.empty())
    {
        tracker.last_top = candidate;
        tracker.persistence_count = 0;
    }
    else
    {
        // --- OVERLAP CHECK ---
        int overlap = 0;
        for (int c : candidate)
        {
            for (int l : tracker.last_top)
            {
                if (c == l)
                {
                    overlap++;
                    break;
                }
            }
        }

        int required_overlap = std::max(1, topN / 2);

        if (overlap >= required_overlap)
        {
            tracker.persistence_count++;
        }
        else
        {
            tracker.persistence_count = 1;
        }

        if (tracker.persistence_count >= tracker.required_persistence)
        {
            tracker.last_top = candidate;
            tracker.persistence_count = 0;
        }
    }

    // --- output ---
    std::vector<std::pair<int, double>> out;
    for (int idx : tracker.last_top)
        out.emplace_back(idx, tracker.smoothed(idx));

    return out;
}

inline std::vector<std::pair<int, double>>
getTopK1(const Patterns &p, FactorTracker &tracker, int topN)
{
    Eigen::VectorXd importance = p.eigenVectors.col(0).cwiseAbs();
    normalize(importance);

    // --- EMA smoothing ---
    if (!tracker.initialized)
    {
        tracker.smoothed = importance;
        tracker.initialized = true;
    }
    else
    {
        tracker.smoothed =
            tracker.alpha * importance +
            (1.0 - tracker.alpha) * tracker.smoothed;
    }

    // --- build pairs ---
    std::vector<std::pair<int, double>> pairs;
    for (int i = 0; i < tracker.smoothed.size(); ++i)
        pairs.emplace_back(i, tracker.smoothed(i));

    std::sort(pairs.begin(), pairs.end(),
              [](auto &a, auto &b)
              { return a.second > b.second; });

    // --- candidate ---
    std::vector<int> candidate;
    for (int i = 0; i < std::min(topN, (int)pairs.size()); ++i)
        candidate.push_back(pairs[i].first);

    // --- INIT ---
    if (tracker.last_top.empty())
    {
        tracker.last_top = candidate;
        tracker.persistence_count = 0;
    }
    else
    {
        // --- OVERLAP CHECK ---
        int overlap = 0;
        for (int c : candidate)
        {
            for (int l : tracker.last_top)
            {
                if (c == l)
                {
                    overlap++;
                    break;
                }
            }
        }

        int required_overlap = std::max(1, topN / 2);

        if (overlap >= required_overlap)
        {
            tracker.persistence_count++;
        }
        else
        {
            tracker.persistence_count = 1;
        }

        if (tracker.persistence_count >= tracker.required_persistence)
        {
            tracker.last_top = candidate;
            tracker.persistence_count = 0;
        }
    }

    // --- output ---
    std::vector<std::pair<int, double>> out;
    for (int idx : tracker.last_top)
        out.emplace_back(idx, tracker.smoothed(idx));

    return out;
}

inline std::tuple<std::vector<std::string>, std::vector<double>> getAxisInfo(std::vector<std::pair<int, double>> top, std::vector<std::string> headers)
{

    std::vector<std::string> AXIS_INFO;
    std::vector<double> AXIS_WEIGHTS;

    for (auto [idx, w] : top)
    {
        AXIS_INFO.push_back(headers[idx]);
        AXIS_WEIGHTS.push_back(w);
    }

    return {AXIS_INFO, AXIS_WEIGHTS};
}

/*
inline std::vector<std::pair<int,double>> getTopSubspace(const Patterns& p, int topN = 4) //n = what weights to show
{
    int k = p.eff; //how many patterns matter

    Eigen::VectorXd importance = Eigen::VectorXd::Zero(p.eigenVectors.rows());

    for (int j = 0; j < k; ++j)
    {
        auto v = p.eigenVectors.col(j);
        importance += v.array().square().matrix(); //sum the squared vectors. so we get the most important ones piling up above the rest
    }

    std::vector<std::pair<int,double>> pairs;

    for (int i = 0; i < importance.size(); ++i)
        pairs.emplace_back(i, importance(i)); //keep the indexing

    std::sort(pairs.begin(), pairs.end(),
        [](auto& a, auto& b){ return a.second > b.second; }); //sort it so the most important ones come out

    pairs.resize(topN);
    return pairs;
}
*/

inline std::vector<std::pair<int, double>>
getTopSubspace(const Patterns &p, FactorTracker &tracker, int topN)
{
    int k = p.eff;

    Eigen::VectorXd importance =
        Eigen::VectorXd::Zero(p.eigenVectors.rows());

    for (int j = 0; j < k; ++j)
    {
        auto v = p.eigenVectors.col(j);
        importance += v.array().square().matrix();
    }

    normalize(importance);

    // --- EMA smoothing ---
    if (!tracker.initialized)
    {
        tracker.smoothed = importance;
        tracker.initialized = true;
    }
    else
    {
        tracker.smoothed =
            tracker.alpha * importance +
            (1.0 - tracker.alpha) * tracker.smoothed;
    }

    // --- build pairs ---
    std::vector<std::pair<int, double>> pairs;
    for (int i = 0; i < tracker.smoothed.size(); ++i)
        pairs.emplace_back(i, tracker.smoothed(i));

    std::sort(pairs.begin(), pairs.end(),
              [](auto &a, auto &b)
              { return a.second > b.second; });

    // --- candidate ---
    std::vector<int> candidate;
    for (int i = 0; i < std::min(topN, (int)pairs.size()); ++i)
        candidate.push_back(pairs[i].first);

    // --- INIT ---
    if (tracker.last_top.empty())
    {
        tracker.last_top = candidate;
        tracker.persistence_count = 0;
    }
    else
    {
        // --- OVERLAP CHECK ---
        int overlap = 0;
        for (int c : candidate)
        {
            for (int l : tracker.last_top)
            {
                if (c == l)
                {
                    overlap++;
                    break;
                }
            }
        }

        int required_overlap = std::max(1, topN / 2);

        if (overlap >= required_overlap)
        {
            tracker.persistence_count++;
        }
        else
        {
            tracker.persistence_count = 1;
        }

        if (tracker.persistence_count >= tracker.required_persistence)
        {
            tracker.last_top = candidate;
            tracker.persistence_count = 0;
        }
    }

    // --- output ---
    std::vector<std::pair<int, double>> out;
    for (int idx : tracker.last_top)
        out.emplace_back(idx, tracker.smoothed(idx));

    return out;
}

inline std::string describeDrivers(const Patterns &p, FactorTracker &tracker)
{
    std::ostringstream out;

    if (p.eff == 1)
    {
        auto top = getTopK1(p, tracker, 3); // indexes matter

        for (auto &[idx, w] : top)
            out << p.headers[idx] << " "; // output: indexes of headers
                                          // std::cerr << p.headers[idx] << " ";
    }
    else
    {
        auto top = getTopSubspace(p, tracker, 4);

        for (auto &[idx, w] : top)
            out << p.headers[idx] << " "; // same but summed vectors
                                          // std::cerr << p.headers[idx] << " ";
    }

    return out.str();
}

inline std::string buildExplanation(CurRegime r, const Patterns &p, FactorTracker &tracker)
{
    std::ostringstream out;

    out << "Regime: " << interpret(r) << "\n";                            // returns string of what regime
    out << "Dominant driver(s) :" << describeDrivers(p, tracker) << "\n"; // dominant drivers (headers)

    if (p.eff == 1)
        out << "Structure concentrated in a single dominant direction.\n";
    else
        out << "Structure distributed across multiple factors.\n";

    return out.str();
}

inline std::string action(CurRegime r)
{
    switch (r)
    {
    case CurRegime::SingleFactor:
        return "Lean into directional exposure. Concentrate risk in the dominant driver; trend-following and momentum strategies are favored.";

    case CurRegime::StrongDominance:
        return "Aggressively align with the dominant factor. High conviction directional trades; avoid diversification as it dilutes edge.";

    case CurRegime::MultiFactor:
        return "Shift to relative value and cross-sectional strategies. Diversify exposures; exploit dispersion between factors.";

    case CurRegime::WeakStructure:
        return "Reduce position sizes. Avoid strong directional bets; focus on short-term, low-conviction trades or stay neutral.";

    case CurRegime::StableTrend:
        return "Maintain and ride trends. Use pullbacks to add exposure; trailing risk management is effective.";

    case CurRegime::StableDispersion:
        return "Systematically harvest dispersion. Pair trades, stat arb, and market-neutral strategies are well suited.";

    case CurRegime::Exploratory:
        return "Probe lightly with small positions. Prioritize optionality and flexibility; avoid committing to a single narrative.";

    case CurRegime::Transitioning:
        return "Reduce conviction and de-risk. Gradually unwind prior regime exposures; prepare for regime shift.";

    case CurRegime::HardSwitch:
        return "Rapidly exit outdated positions. Reassess models; consider reversing bias if new structure is confirmed.";

    case CurRegime::Compression:
        return "Tighten risk and prepare for breakout. Consider volatility expansion strategies (e.g., options). Avoid overtrading.";

    case CurRegime::Expansion:
        return "Increase diversification. Capture multiple emerging opportunities; favor broad exposure across factors.";

    case CurRegime::Chaotic:
        return "Significantly reduce exposure. Prioritize capital preservation; avoid leverage and directional bets.";

    case CurRegime::VolatilitySpike:
        return "Cut risk immediately. Short-term hedging or volatility strategies may be appropriate; expect dislocations.";

    case CurRegime::Recovery:
        return "Gradually reintroduce risk. Focus on high-quality setups as stability returns; avoid rushing back to full exposure.";

    case CurRegime::Exhaustion:
        return "Take profits on existing trends. Avoid initiating new trend trades; look for reversal or mean-reversion setups.";

    default:
        return "No clear action. Maintain neutral positioning and wait for clarity.";
    }
}

inline std::string finalReport(CurRegime r, const Patterns &p, FactorTracker &tracker)
{
    std::ostringstream out;

    out << buildExplanation(r, p, tracker) << "\n";
    out << action(r) << "\n";

    return out.str();
}