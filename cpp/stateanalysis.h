#pragma once

#include "stateanalysis.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>   
#include <utility>
#include <vector>

enum class CurRegime
{
  Trending,
  Stressed,
  //Compressed,
  //Transitioning,
  Dispersed,
  Neutral
};

struct Prototype
{
  CurRegime regime;
  double G, E, C;
};

inline Eigen::MatrixXd transform(const std::vector<Prototype>& ps)
{
    Eigen::MatrixXd means(ps.size(), 3);

    for (size_t i = 0; i < ps.size(); ++i)
    {
        means(i, 0) = ps[i].G;
        means(i, 1) = ps[i].E;
        means(i, 2) = ps[i].C;
    }

    return means;
}

inline std::vector<Prototype> reversetransform(const Eigen::MatrixXd& means)
{
    if (means.cols() != 3)
    {
        throw std::invalid_argument("Matrix must have exactly 3 columns.");
    }

    std::vector<Prototype> ps(means.rows());

    for (Eigen::Index i = 0; i < means.rows(); ++i)
    {
        ps[i].G = means(i, 0);
        ps[i].E = means(i, 1);
        ps[i].C = means(i, 2);
    }

    return ps;
}

const std::vector<Prototype> default_prototypes = {
    // Regime                G      E      C
    {CurRegime::Trending,    0.50,  0.20, -1.00},  // stable factor, orderly, concentrated
    {CurRegime::Stressed,    1.50,  1.90, -0.50},  // locked-in panic factor, extreme E, concentrated
    {CurRegime::Dispersed,   1.80,  0.20,  2.00},  // flat spectrum -> eigengap noise -> high G, calm, dispersed
    {CurRegime::Neutral,     0.20,  0.10,  0.20},  // near unconditional mean
};

inline CurRegime classifyRegime(double G, double E, double C, const std::vector<Prototype> &current_protos)
{
  // E weighted 1.0x because it's the primary separating axis
  auto dist = [](double g, double e, double c,
                 const Prototype &p)
  {
    return std::sqrt(
        1.0 * std::pow(g - p.G, 2) +
        1.0 * std::pow(e - p.E, 2) + // E matters moren, so it was 2.0
        1.0 * std::pow(c - p.C, 2));
  };

  double min_dist = std::numeric_limits<double>::max();
  CurRegime best = CurRegime::Neutral;

  for (const auto &p : current_protos)
  {
    double d = dist(G, E, C, p);
    if (d < min_dist)
    {
      min_dist = d;
      best = p.regime;
    }
  }
  return best;
}

inline size_t argmax(const std::vector<double> &v)
{
  return std::distance(v.begin(),
                       std::max_element(v.begin(), v.end()));
}

inline size_t second_argmax(const std::vector<double> &v)
{
  size_t best = argmax(v);

  size_t second = (best == 0) ? 1 : 0;
  for (size_t i = 0; i < v.size(); ++i)
  {
    if (i == best)
      continue;
    if (v[i] > v[second])
      second = i;
  }
  return second;
}

inline std::vector<double> softmax(const std::vector<double> &v)
{
  std::vector<double> result = v;

  const double max_val = *std::max_element(result.begin(), result.end());

  double sum = 0.0;
  for (double &x : result)
  {
    x = std::exp(x - max_val);
    sum += x;
  }

  for (double &x : result)
    x /= sum;

  return result;
}

// SOFTMAX PROBABLY BETTER

// MAKE THEM INLINE YOU IDIOT

// what i need to do:
/*

- define regimes first
    - GEC combinations classify certain regimes
- seperate interpretation from classification, just a simple enum to string.
- THEN use patterns, bring in eigenstuff. Top contributors.

- final report combines everything with what regime, the state, what causes it,
why its important, and what to do next (implication)

*/

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

inline double temporalConsistency(const std::deque<CurRegime> &h)
{
  int count = 0;
  for (auto r : h)
    if (r == h.back())
      count++;

  return (double)count / h.size();
}

inline std::string enum_to_string_regime(CurRegime regime)
{
  switch (regime)
  {
  case CurRegime::Trending:
    return "Trending";
  case CurRegime::Stressed:
    return "Stressed";
  //case CurRegime::Compressed:
  //  return "Compressed";
  //case CurRegime::Transitioning:
  // return "Transitioning";
  case CurRegime::Dispersed:
    return "Dispersed";
  case CurRegime::Neutral:
    return "Neutral";
  default:
    return "Unknown";
  }
}

inline std::string interpret(CurRegime r)
{
  switch (r)
  {
  case CurRegime::Trending:
    return "Market exhibits persistent directional movement. Trends are "
           "well-defined and price action is driven by sustained momentum.";

  case CurRegime::Stressed:
    return "Market is under stress. Volatility, correlations, and risk "
           "aversion are elevated, often leading to disorderly price action.";

  /*

  case CurRegime::Compressed:
    return "Volatility and dispersion are subdued. Market activity is "
           "constrained, often preceding a significant breakout or regime "
           "change.";

  case CurRegime::Transitioning:
    return "Existing market structure is evolving. Leadership, correlations, "
           "and factor relationships are shifting as the market searches for "
           "a new equilibrium.";
  
  */

  case CurRegime::Dispersed:
    return "Multiple independent drivers are influencing markets. "
           "Cross-sectional dispersion is elevated and relative-value "
           "opportunities are more abundant.";

  case CurRegime::Neutral:
    return "No strong directional or structural bias is present. Market "
           "conditions are balanced and signals are mixed.";

  default:
    return "Unknown market regime.";
  }
}

inline void normalize(Eigen::VectorXd &v)
{
  double s = v.sum();
  if (s > 1e-12)
    v /= s;
}

inline std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
getTop3EigenVectors(const Patterns &p)
{
  return {p.eigenVectors.col(0), p.eigenVectors.col(1), p.eigenVectors.col(2)};
}

inline std::vector<double> eigenStateProjections3dim(const Eigen::VectorXd &e1,
                                                     const Eigen::VectorXd &e2,
                                                     const Eigen::VectorXd &e3,
                                                     const Eigen::VectorXd &s)
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

  // return {p1 / sum, p2 / sum, p3 / sum};
  return {e1.dot(s) * e1.dot(s), e2.dot(s) * e2.dot(s),
          e3.dot(s) * e3.dot(s)}; // squared. dominance profile
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
        tracker.alpha * importance + (1.0 - tracker.alpha) * tracker.smoothed;
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

inline std::tuple<std::vector<std::string>, std::vector<double>>
getAxisInfo(std::vector<std::pair<int, double>> top,
            std::vector<std::string> headers)
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
