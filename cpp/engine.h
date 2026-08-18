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
#include <array>
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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "python_runtime.h"
#include "rollingcov.h"

#define DATA(x) std::cout << x << '\n';
#define LOG(x) std::cerr << x << std::endl;

namespace py = pybind11;

struct sysOutput
{
  // Decision outputs
  std::string decision_regime;
  std::string decision_regime_rationale;
  std::string decision_action;
  double regime_confidence;
  int regime_age; // ticks since last regime switch
  std::vector<std::string> model_input_variables;
  std::vector<double> model_raw_input;
  std::vector<double> model_pc1_weights_smoothed;
  std::vector<double> regimeWeights;
};

class Engine
{
public:
  // Construct from a validated ClientConfig
  // explicit Engine(const ClientConfig &cfg);
  /*This is only a declaration. It tells the compiler that Engine has a constructor taking a const ClientConfig&, but it doesn't define what the constructor does. The actual implementation must appear elsewhere, for example:*/
  /*
  Engine(const ClientConfig &cfg)
      : config(cfg)
  {

    if (!Py_IsInitialized())
    {
      throw std::runtime_error(
          "Python interpreter is not initialized");
    }

    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("append")("C:/Users/H/Documents/projects/Avarent/avarent"); // folder containing hmm_fitter.py

    py::module_ hmm = py::module_::import("hmm_fitter");
    fit_and_match = hmm.attr("fit_and_match");

    previous_means = transform(default_prototypes); // example
  }
  */
  Engine(const ClientConfig &cfg)
      : config(cfg)
  {
    std::cout << "Engine: start" << std::endl;

    /*
    py::module_ sys = py::module_::import("sys");

    std::cout << "Engine: imported sys" << std::endl;

    sys.attr("path").attr("append")(
        "C:/Users/H/Documents/projects/Avarent/avarent");

    std::cout << "Engine: path added" << std::endl;

    py::module_ hmm;

    try
    {
      std::cout << "importing numpy..." << std::endl;
      py::module_ np = py::module_::import("numpy");
      std::cout << "numpy OK" << std::endl;

      std::cout << "importing scipy..." << std::endl;
      py::module_ scipy = py::module_::import("scipy");
      std::cout << "scipy OK" << std::endl;

      std::cout << "importing hmmlearn..." << std::endl;
      py::module_ hmmlearn = py::module_::import("hmmlearn");
      std::cout << "hmmlearn OK" << std::endl;

      std::cout << "importing hmm_fitter..." << std::endl;
      hmm = py::module_::import("hmm_fitter");
      std::cout << "hmm_fitter OK" << std::endl;
    }
    catch (const py::error_already_set &e)
    {
      std::cerr << e.what() << std::endl;
    }

    std::cout << "Engine: imported hmm_fitter" << std::endl;

    fit_and_match = hmm.attr("fit_and_match");

    std::cout << "Engine: got function" << std::endl;

    previous_means = transform(default_prototypes);
    */

    std::cout << "Engine: finished" << std::endl;
  }

  void ingest(const std::vector<double> &row); // feed new data

  bool ready();

  void setHeaders(std::vector<std::string> x);

  std::vector<std::string> getHeaders();

  CurRegime updateRegime(CurRegime detected_regime);

  sysOutput getOutput();

  // tempo
  std::deque<std::vector<double>> htts;

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

  RollingCovariance rolcov{200, 6}; //stateanalysis.h mahalanobis global

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
  std::deque<NormalizerState> state_history; // use this

  MatrixBuffer U_buff, Uvec_buff;

  // Test RobustScores for G2, E2, C2
  RobustScore g2_score{config.robust_history, config.robust_warmup}, e2_score{config.robust_history, config.robust_warmup}, c2_score{config.robust_history, config.robust_warmup};
  RobustScore e_score{config.robust_history, config.robust_warmup};

  EMA g2_ema{config.gec_ema_alpha}, e2_ema{config.gec_ema_alpha}, c2_ema{config.gec_ema_alpha};
  EMA e_ema{config.gec_ema_alpha};
  EMA dg_ema{config.gec_ema_alpha}, de_ema{config.gec_ema_alpha}, dc_ema{config.gec_ema_alpha};

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

  /*

  py::object fit_and_match;
  Eigen::MatrixXd previous_means;              // K x 3
  std::vector<std::vector<double>> gec_window; // N x 3

  */ 

  std::deque<std::array<double, 3>> gec_history; // for k-spacing GEC deltas

  // void runHMM(const std::vector<std::array<double, 3>>& data);
};
