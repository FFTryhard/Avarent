#pragma once

#include "client_config.h"
#include "lle.h"
#include "rawprocessing.h"
#include "robustnormalizer.h"
#include "rawprocessing2.h"
#include "stateanalysis.h"
#include "utils.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#define DATA(x) std::cout << x << '\n';
#define LOG(x) std::cerr << x << std::endl;

struct sysOutput {
  // Decision outputs
  std::string decision_regime;
  std::string decision_regime_rationale;
  std::string decision_action;
  double regime_confidence;
  int regime_age; // ticks since last regime switch
  std::vector<std::string> model_input_variables;
  std::vector<double> model_raw_input;
  std::vector<double> model_pc1_weights_smoothed;
};

class Engine {
public:
  // Construct from a validated ClientConfig
  explicit Engine(const ClientConfig &cfg);

  void ingest(const std::vector<double> &row); // feed new data

  bool ready();

  void setHeaders(std::vector<std::string> x);

  std::vector<std::string> getHeaders();

  CurRegime updateRegime(CurRegime detected_regime);

  sysOutput getOutput();

  //tempo
  std::vector<std::vector<double>> htts;

private:
  // Config — declared first so subsequent members can reference it
  ClientConfig config;

  // FIXED constants — not client-configurable
  static constexpr int SCORE_MIN = 0;

  // Fixed strides — internal implementation detail
  static constexpr int LLE_REGIME_STRIDE = 10;
  static constexpr int REGIME_HISTORY_STRIDE = 1;
  static constexpr int STABILITY_STRIDE = 1;
  static constexpr int SWIRL_STEP = 50;
  static constexpr int SWIRL_DT = 1;

  // counters:
  int iteration = 0;
  int window_index = 0;
  int regime_age_counter = 0;

  // scores
  int regime_score = 0;

  // inits
  bool activated = false;
  bool regime_tracking_init = false;
  bool pc1_smoothed_init = false;
  bool output_ready = false;
  bool headers_are_set = false;
  bool regime_init = false;

  CurRegime confirmed_regime;
  CurRegime regime_candidate;

  std::vector<std::string> headers;

  std::deque<double> lle_regime;

  std::deque<std::vector<double>> WINDOW;

  // Normalizer — initialized from config
  AdaptiveOnlineNormalizer in{
      config.normalizer_base_alpha, config.normalizer_min_alpha,
      config.normalizer_decay, config.normalizer_update_interval};

  AdaptiveOnlineNormalizer2 in2;

  // History buffer for macroscopic G and C deltas
  std::deque<NormalizerState> state_history; //use this

  MatrixBuffer U_buff, Uvec_buff;

  // Test RobustScores for G2, E2, C2
  RobustScore g2_score{config.robust_history, config.robust_warmup}, e2_score{config.robust_history, config.robust_warmup}, c2_score{config.robust_history, config.robust_warmup};
  RobustScore e_score{config.robust_history, config.robust_warmup};
  
  EMA g2_ema{config.gec_ema_alpha}, e2_ema{config.gec_ema_alpha}, c2_ema{config.gec_ema_alpha};
  EMA e_ema{config.gec_ema_alpha};

  // aux deltas
  ScalarBuffer_ ddelta_buff, dde_delta_buff;

  // Analysis
  FactorTracker tracker;
  FactorTracker tracker_axis1;
  FactorTracker tracker_axis2;
  FactorTracker tracker_axis3;

  domEigenvectors eigenvecTracker;

  std::deque<CurRegime> regimehistory;
  CurRegime previous_regime{CurRegime::Neutral};

  std::deque<double> t_raw_smoothed;

  // Regime age tracking
  CurRegime last_tracked_regime{};

  std::unordered_map<int, std::vector<int>> regime_durations;

  // PC1 weight smoothing (uses EMA to prevent instrument classification
  // flicker)
  Eigen::VectorXd pc1_smoothed;

  sysOutput current_output;
};

