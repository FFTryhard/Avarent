#pragma once

#include <string>
#include <vector>
#include "utils.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Dense>
#include <iomanip>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <limits>
#include "rawprocessing.h"
#include "utils.h"
#include <filesystem>
#include <deque>
#include <cstdio>
#include "lle.h"
#include "robustnormalizer.h"
#include "stateanalysis.h"
#include "rollingbinbuff.h"
#include "MarkovRegimeModel.h"

#define DATA(x) std::cout << x << '\n';
#define LOG(x) std::cerr << x << std::endl;

struct sysOutput
{
    std::string regime;
    std::string description;
    std::string action;
    double G, E, C;
    std::vector<double> variable_k1_weights;
    std::vector<std::string> variables;
    std::vector<std::string> eigen_axis1_contributors, eigen_axis2_contributors, eigen_axis3_contributors;
    std::vector<double> eigen_axis1_weights, eigen_axis2_weights, eigen_axis3_weights;
    std::vector<double> dominant_eig_projection_coordinates_3dim; // xyz
    std::vector<double> variable_weights;
    double structural_stability;
    double confidence; 
    std::vector<std::string> top_next_regimes;
    std::vector<double> top_next_probs;
    double momentum_score;
    double carry_score;
    double vol_score;
    double directional_bias;
    double transition_risk;
    double risk_posture;
    double system_state_score;
};

class Engine
{
public:
    // Engine(); // constructor, ill do some nice to have things later
    //  Tunable parameters
    const int SCORE_MAX = 3;  // threshold to confirm switch
    const int SCORE_MIN = 0;  // lower bound //switching is too slow → lower SCORE_MAX || still a bit twitchy → increase SCORE_MAX or lower DECAY_STEP
    const int DECAY_STEP = 1; // how fast score decreases

    // State (keep these persistent across calls)
    bool regime_init = false;
    CurRegime confirmed_regime;
    CurRegime regime_candidate;
    int regime_score = 0;

    void ingest(const std::vector<double> &row); // feed new data

    bool ready();

    void setHeaders(std::vector<std::string> x);
    std::vector<std::string> getHeaders();
    bool headers_are_set = false;

    CurRegime updateRegime(CurRegime detected_regime);

    sysOutput getOutput();

private:
    std::vector<std::string> headers;
    // counters:
    int iteration = 0; // keeps getting updated. one object lives in mem

    int window_index = 0;

    std::deque<double> lle_regime;
    int N = 100;
    int lle_regime_stride = 10;

    std::deque<std::vector<double>> WINDOW;

    AdaptiveOnlineNormalizer in;

    int EFFECTIVE_BUFFER = 250, BUFFER_STRIDE = 20;

    MatrixBuffer U_buff, Uvec_buff;

    bool activated = false;

    // G(t)
    RobustScore ds_score{100, 30}, mos_score{100, 30}, ddelta_score{100, 30};

    // E(t)
    RobustScore lle_level_score{100, 30}, srs_score{100, 30}, lle_shift_score{100, 30};

    // C(T)
    RobustScore deff_delta_score{100, 30};

    // aux deltas
    ScalarBuffer_ ddelta_buff, dde_delta_buff;

    // GEC rolling buffers
    RollingBinBuffer G_bin{6}, E_bin{6}, C_bin{6};

    EMA g{0.2}, e{0.2}, c{0.2}, t{0.2};

    // Analysis
    GECStates gec_states{6, 1};

    FactorTracker tracker;

    FactorTracker tracker_axis1;
    FactorTracker tracker_axis2;
    FactorTracker tracker_axis3;

    domEigenvectors eigenvecTracker;

    std::deque<CurRegime> regimehistory;
    int regimehistory_N = 8;
    int regimehistory_stride = 1;

    MarkovRegimeModel markovmodel{16}; // num regimes
    std::deque<CurRegime> markovregimehistory;
    int markovregimehistory_N = 100;
    int markovregimehistory_stride = 1;

    std::deque<double> t_raw_smoothed;
    int t_raw_smoothed_N = 30; // fast is 20, slow is 40
    int t_raw_smoothed_stride = 1;

    sysOutput current_output;
    bool output_ready = false;
};
