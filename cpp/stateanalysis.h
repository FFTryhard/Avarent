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
//=======================================
struct PrototypeGlobalCov
{
    CurRegime regime;

    Eigen::Matrix<double,6,1> mean;

    double prior = 0.25;
};
const std::vector<PrototypeGlobalCov> defaultPrototypesGlobalCov =
{
    {
        CurRegime::Trending,
        (Eigen::Matrix<double,6,1>() <<
            -1.0,
            -1.0,
            -1.0,
             0.0,
             0.0,
             0.0).finished(),
        0.25
    },

    {
        CurRegime::Stressed,
        (Eigen::Matrix<double,6,1>() <<
             1.0,
             1.8,
             1.0,
             1.0,
             2.0,
             1.0).finished(),
        0.25
    },

    {
        CurRegime::Dispersed,
        (Eigen::Matrix<double,6,1>() <<
             0.8,
             0.2,
             2.0,
             0.6,
             0.0,
             1.5).finished(),
        0.25
    },

    {
        CurRegime::Neutral,
        (Eigen::Matrix<double,6,1>() <<
             0.0,
             0.0,
             0.0,
             0.0,
             0.0,
             0.0).finished(),
        0.25
    }
};
inline double mahalanobisDistanceSquared(
    const Eigen::Matrix<double, 6, 1>& x,
    const PrototypeGlobalCov& prototype,
    const Eigen::Matrix<double, 6, 6>& invCov)
{
    const Eigen::Matrix<double, 6, 1> d = x - prototype.mean;
    return (d.transpose() * invCov * d)(0,0);
}
struct RegimeProbability
{
    CurRegime regime;
    double probability;
};
inline std::vector<RegimeProbability> classifyRegimesGlobalCov(
    const Eigen::Matrix<double, 6, 1>& state,
    const Eigen::Matrix<double, 6, 6>& invMarketCov,
    const std::vector<PrototypeGlobalCov>& prototypes,
    double temperature = 1.0)
{
    std::vector<double> logits;
    logits.reserve(prototypes.size());

    for (const auto& p : prototypes)
    {
        const double d2 =
            mahalanobisDistanceSquared(
                state,
                p,
                invMarketCov);

        logits.push_back(
            -d2 / temperature +
            std::log(p.prior));
    }

    const double maxLogit =
        *std::max_element(logits.begin(), logits.end());

    double sum = 0.0;

    for (double& l : logits)
    {
        l = std::exp(l - maxLogit);
        sum += l;
    }

    std::vector<RegimeProbability> result;
    result.reserve(prototypes.size());

    for (size_t i = 0; i < prototypes.size(); ++i)
    {
        result.push_back({
            prototypes[i].regime,
            logits[i] / sum
        });
    }

    return result;
}
//=======================================
struct Prototype
{
    CurRegime regime;

    Eigen::VectorXd mean;     // size 6
    Eigen::MatrixXd invCov;   // 6x6

    double prior;             // optional
};

inline Eigen::MatrixXd diagonalInvCov(
    double g,
    double e,
    double c,
    double dg,
    double de,
    double dc)
{
    Eigen::MatrixXd inv = Eigen::MatrixXd::Zero(6,6);

    inv(0,0)=1.0/(g*g);
    inv(1,1)=1.0/(e*e);
    inv(2,2)=1.0/(c*c);

    inv(3,3)=1.0/(dg*dg);
    inv(4,4)=1.0/(de*de);
    inv(5,5)=1.0/(dc*dc);

    return inv;
}

inline Eigen::VectorXd makeVector(std::initializer_list<double> values)
{
    Eigen::VectorXd v(values.size());

    std::size_t i = 0;
    for (double value : values)
        v(static_cast<Eigen::Index>(i++)) = value;

    return v;
}

const std::vector<Prototype> default_prototypes3 =
{

// =====================================================
// TRENDING
// Locked geometry.
// Low instability.
// Low complexity.
// Small velocity.
// =====================================================

{
    CurRegime::Trending,

    makeVector({
        -1.0,   // G
        -1.0,   // E
        -1.0,   // C

         0.0,   // dG
         0.0,   // dE
         0.0    // dC
    }),

    diagonalInvCov(
        1.0,
        1.0,
        1.0,

        0.5,
        0.5,
        0.5
    ),

    0.25
},


// =====================================================
// STRESSED
// Expanding unstable state.
// Velocity matters.
// E dominates.
// =====================================================

{
    CurRegime::Stressed,

    makeVector({
         1.0,   // G
         1.8,   // E
         1.0,   // C

         1.0,   // dG
         2.0,   // dE
         1.0    // dC
    }),

    diagonalInvCov(
        1.0,
        0.7,
        1.0,

        0.8,
        0.5,
        0.8
    ),

    0.25
},


// =====================================================
// DISPERSED
// Complexity explosion.
// Geometry changes.
// Lyapunov not necessarily extreme.
// =====================================================

{
    CurRegime::Dispersed,

    makeVector({
         0.8,   // G
         0.2,   // E
         2.0,   // C

         0.6,   // dG
         0.0,   // dE
         1.5    // dC
    }),

    diagonalInvCov(
        1.0,
        1.5,
        0.5,

        1.0,
        1.5,
        0.5
    ),

    0.25
},


// =====================================================
// NEUTRAL
// No strong structure.
// Broad basin.
// =====================================================

{
    CurRegime::Neutral,

    makeVector({
         0.0,
         0.0,
         0.0,

         0.0,
         0.0,
         0.0
    }),

    diagonalInvCov(
        2.0,
        2.0,
        2.0,

        2.0,
        2.0,
        2.0
    ),

    0.25
}

};

inline double distance2(
    const Eigen::VectorXd& x,
    const Prototype& p) 
{
    Eigen::VectorXd d=x-p.mean;

    return d.transpose()
           *
           p.invCov
           *
           d;
}

inline std::vector<double> softmaxRegimes(
    const Eigen::VectorXd& x,
    const std::vector<Prototype>& prototypes,
    double temperature)
{
    std::vector<double> logits;

    for(auto& p:prototypes)
    {
        double d2 =
            distance2(x,p);

        logits.push_back(
            -d2/temperature
        );
    }


    double maxL =
        *std::max_element(
            logits.begin(),
            logits.end()
        );


    double sum=0;

    for(auto& l:logits)
    {
        l=std::exp(l-maxL);
        sum+=l;
    }


    for(auto& l:logits)
        l/=sum;


    return logits;
}

//---------------------------



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

inline double pref(double x, double mu, double sigma)
{
    const double d = (x - mu) / sigma;
    return std::exp(-0.5 * d * d);
}

// Trending:
// Low geometry drift
// Low Lyapunov exponent
// Low complexity
inline double scoreTrending(double G, double E, double C)
{
    return
        0.40 * pref(G, -1.2, 0.8) +
        0.35 * pref(E, -1.2, 0.8) +
        0.25 * pref(C, -1.0, 0.9);
}

// Stressed:
// Geometry becoming unstable
// Positive Lyapunov
// Moderate-high complexity
inline double scoreStressed(double G, double E, double C)
{
    return
        0.30 * pref(G, 1.0, 0.9) +
        0.45 * pref(E, 1.5, 0.8) +
        0.25 * pref(C, 0.8, 0.9);
}

// Dispersed:
// Complexity dominates
// Geometry somewhat elevated
// Lyapunov not necessarily large
inline double scoreDispersed(double G, double E, double C)
{
    return
        0.25 * pref(G, 0.8, 0.9) +
        0.15 * pref(E, 0.0, 1.2) +
        0.60 * pref(C, 2.0, 0.8);
}

// Neutral:
// Everything around the center
inline double scoreNeutral(double G, double E, double C)
{
    return
        0.33 * pref(G, 0.0, 0.8) +
        0.33 * pref(E, 0.0, 0.8) +
        0.34 * pref(C, 0.0, 0.8);
}

inline CurRegime classifyRegime(double G, double E, double C)
{
    std::vector<double> scores = {
        scoreTrending(G, E, C),
        scoreStressed(G, E, C),
        scoreDispersed(G, E, C),
        scoreNeutral(G, E, C)
    };

    int best = argmax(scores);
    int second = second_argmax(scores);

    if (scores[best] - scores[second] < 0.10)
        return CurRegime::Neutral;

    return static_cast<CurRegime>(best);
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
