#pragma once

#include <array>
#include <string>
#include <vector>
#include <cmath>

// ============================================================
//  Client configuration for the Avarent engine
// ============================================================

struct ConfigViolation {
    std::string field;
    std::string message;
    double value;
    double bound;
};

enum class EngineProfile {
    LONG_SHORT_EQUITY,
    GLOBAL_MACRO,
    SYSTEMATIC_TREND,
    RISK_PARITY
};

struct ClientConfig {
    // ---- TIME_HORIZON ----
    int    window_size;
    int    window_stride;
    int    robust_history;
    int    robust_warmup;
    int    lle_history_depth;
    int    regime_confidence_depth;
    int    stability_window;
    int    swirl_window;

    int hmmwindow;
    int hmmwindowstride;

    // ---- SENSITIVITY ----
    int    regime_confirm_count;
    int    regime_decay_step;
    double gec_ema_alpha;
    double normalizer_base_alpha;
    double normalizer_min_alpha;
    double normalizer_decay;
    int    normalizer_update_interval;
    // ---- THRESHOLD ----
    double pca_variance_retained;
    double eigengap_min_ratio;
    double lle_shift_cap;
    double stability_decay_k;
    int    min_duration_episodes;
    double swirl_speed_floor;
    double pc1_significance;
    double lle_r2_threshold;

    // ---- WEIGHT ----
    std::array<double, 3> g_weights; // {DS, MOS, DD} — must sum to 1.0
    std::array<double, 3> e_weights; // {LLE, LLE_SHIFT, SRS} — must sum to 1.0

    std::array<double, 3> t_weights; // {G, E, C} — must sum to 1.0
};

// ============================================================
//  Named profile defaults
// ============================================================

inline ClientConfig profile_defaults(EngineProfile profile) {
    // Baseline: LONG_SHORT_EQUITY
    ClientConfig cfg{};

    // TIME_HORIZON
    cfg.window_size             = 250;
    cfg.window_stride           = 20;
    cfg.hmmwindow               = 200;
    cfg.hmmwindowstride         = 50;
    cfg.robust_history          = 100;
    cfg.robust_warmup           = 30;
    cfg.lle_history_depth       = 100;
    cfg.regime_confidence_depth = 10;
    cfg.stability_window        = 30;
    cfg.swirl_window            = 100;

    // SENSITIVITY
    cfg.regime_confirm_count      = 3;
    cfg.regime_decay_step         = 1;
    cfg.gec_ema_alpha             = 0.10;
    cfg.normalizer_base_alpha     = 0.20;
    cfg.normalizer_min_alpha      = 0.01;
    cfg.normalizer_decay          = 0.001;
    cfg.normalizer_update_interval = 10;

    // THRESHOLD
    cfg.pca_variance_retained  = 0.90;
    cfg.eigengap_min_ratio     = 1e-3;
    cfg.lle_shift_cap          = 8.0;
    cfg.stability_decay_k      = 5.0;
    cfg.min_duration_episodes  = 4;
    cfg.swirl_speed_floor      = 0.10;
    cfg.pc1_significance       = 1.5;
    cfg.lle_r2_threshold       = 0.85;

    // WEIGHT
    cfg.g_weights = {0.25, 0.60, 0.15};
    cfg.e_weights = {0.40, 0.35, 0.25};

    cfg.t_weights = {0.35, 0.40, 0.25};

    // Apply profile-specific deltas
    switch (profile) {

    case EngineProfile::GLOBAL_MACRO:
        cfg.window_size             = 350;
        cfg.window_stride           = 25;
        cfg.robust_history          = 150;
        cfg.robust_warmup           = 40;
        cfg.lle_history_depth       = 150;
        cfg.regime_confidence_depth = 12;
        cfg.stability_window        = 40;
        cfg.swirl_window            = 280;

        cfg.regime_confirm_count      = 4;
        cfg.gec_ema_alpha             = 0.15;
        cfg.normalizer_base_alpha     = 0.15;
        cfg.normalizer_min_alpha      = 0.008;
        cfg.normalizer_decay          = 0.0008;
        cfg.normalizer_update_interval = 12;

        cfg.pca_variance_retained = 0.92;
        cfg.stability_decay_k     = 4.0;
        cfg.pc1_significance      = 1.3;
        break;

    case EngineProfile::SYSTEMATIC_TREND:
        cfg.window_size             = 400;
        cfg.window_stride           = 30;
        cfg.robust_history          = 200;
        cfg.robust_warmup           = 50;
        cfg.lle_history_depth       = 200;
        cfg.regime_confidence_depth = 16;
        cfg.stability_window        = 60;
        cfg.swirl_window            = 320;

        cfg.regime_confirm_count      = 5;
        cfg.gec_ema_alpha             = 0.12;
        cfg.normalizer_base_alpha     = 0.12;
        cfg.normalizer_min_alpha      = 0.005;
        cfg.normalizer_decay          = 0.0005;
        cfg.normalizer_update_interval = 15;

        cfg.pca_variance_retained = 0.88;
        cfg.stability_decay_k     = 3.5;

        cfg.g_weights = {0.50, 0.35, 0.15};
        break;

    case EngineProfile::RISK_PARITY:
        cfg.window_stride           = 15;
        cfg.robust_history          = 80;
        cfg.robust_warmup           = 25;
        cfg.lle_history_depth       = 80;
        cfg.regime_confidence_depth = 8;
        cfg.stability_window        = 20;

        cfg.regime_confirm_count      = 2;
        cfg.gec_ema_alpha             = 0.25;
        cfg.normalizer_base_alpha     = 0.25;
        cfg.normalizer_min_alpha      = 0.015;
        cfg.normalizer_decay          = 0.0015;
        cfg.normalizer_update_interval = 8;

        cfg.pca_variance_retained = 0.92;
        cfg.stability_decay_k     = 6.0;

        cfg.t_weights = {0.25, 0.35, 0.40};
        break;

    case EngineProfile::LONG_SHORT_EQUITY:
    default:
        break; // baseline already set
    }

    return cfg;
}

// ============================================================
//  Validation — returns ALL violations, never early-exits
// ============================================================

inline std::vector<ConfigViolation> validate(const ClientConfig& cfg) {
    std::vector<ConfigViolation> v;

    auto check_int = [&](const char* name, int val, int lo, int hi) {
        if (val < lo) v.push_back({name, "below minimum", (double)val, (double)lo});
        if (val > hi) v.push_back({name, "above maximum", (double)val, (double)hi});
    };

    auto check_dbl = [&](const char* name, double val, double lo, double hi) {
        if (val < lo) v.push_back({name, "below minimum", val, lo});
        if (val > hi) v.push_back({name, "above maximum", val, hi});
    };

    auto check_sum = [&](const char* name, const std::array<double, 3>& w) {
        double s = w[0] + w[1] + w[2];
        if (std::abs(s - 1.0) > 1e-6)
            v.push_back({name, "weights must sum to 1.0", s, 1.0});
    };

    // --- Bounds ---

    // TIME_HORIZON
    check_int("window_size",             cfg.window_size,             200, 600);
    check_int("window_stride",           cfg.window_stride,           5,   100);
    check_int("robust_history",          cfg.robust_history,          50,  300);
    check_int("robust_warmup",           cfg.robust_warmup,           15,  100);
    check_int("lle_history_depth",       cfg.lle_history_depth,       50,  300);
    check_int("regime_confidence_depth", cfg.regime_confidence_depth, 5,   50);
    check_int("stability_window",        cfg.stability_window,        10,  100);
    check_int("swirl_window",            cfg.swirl_window,            100, 500);

    // SENSITIVITY
    check_int("regime_confirm_count",      cfg.regime_confirm_count,      2,     8);
    check_int("regime_decay_step",         cfg.regime_decay_step,         1,     3);
    check_dbl("gec_ema_alpha",             cfg.gec_ema_alpha,             0.05,  0.50);
    check_dbl("normalizer_base_alpha",     cfg.normalizer_base_alpha,     0.05,  0.40);
    check_dbl("normalizer_min_alpha",      cfg.normalizer_min_alpha,      0.005, 0.05);
    check_dbl("normalizer_decay",          cfg.normalizer_decay,          0.0005,0.005);
    check_int("normalizer_update_interval",cfg.normalizer_update_interval,5,   30);

    // THRESHOLD
    check_dbl("pca_variance_retained",  cfg.pca_variance_retained,  0.80, 0.99);
    check_dbl("eigengap_min_ratio",     cfg.eigengap_min_ratio,     1e-5, 1e-1);
    check_dbl("lle_shift_cap",          cfg.lle_shift_cap,          4.0,  15.0);
    check_dbl("stability_decay_k",      cfg.stability_decay_k,      2.0,  10.0);
    check_int("min_duration_episodes",  cfg.min_duration_episodes,  3,    15);
    check_dbl("swirl_speed_floor",      cfg.swirl_speed_floor,      0.05, 0.25);
    check_dbl("pc1_significance",       cfg.pc1_significance,       0.5,  3.0);
    check_dbl("lle_r2_threshold",       cfg.lle_r2_threshold,       0.70, 0.98);

    // WEIGHT — individual element bounds
    for (int i = 0; i < 3; ++i) {
        check_dbl("g_weights[]", cfg.g_weights[i], 0.05, 0.70);
        check_dbl("e_weights[]", cfg.e_weights[i], 0.05, 0.60);
        check_dbl("t_weights[]", cfg.t_weights[i], 0.10, 0.60);
    }

    // --- Coupling constraints ---
    if (cfg.robust_warmup >= cfg.robust_history)
        v.push_back({"robust_warmup", "must be < robust_history", (double)cfg.robust_warmup, (double)cfg.robust_history});
    if (cfg.swirl_window >= cfg.window_size)
        v.push_back({"swirl_window", "must be < window_size", (double)cfg.swirl_window, (double)cfg.window_size});
    if (cfg.window_stride >= cfg.window_size)
        v.push_back({"window_stride", "must be < window_size", (double)cfg.window_stride, (double)cfg.window_size});
    if (cfg.regime_decay_step > cfg.regime_confirm_count)
        v.push_back({"regime_decay_step", "must be <= regime_confirm_count", (double)cfg.regime_decay_step, (double)cfg.regime_confirm_count});
    if (cfg.normalizer_min_alpha >= cfg.normalizer_base_alpha)
        v.push_back({"normalizer_min_alpha", "must be < normalizer_base_alpha", cfg.normalizer_min_alpha, cfg.normalizer_base_alpha});

    // --- Weight sums ---
    check_sum("g_weights", cfg.g_weights);
    check_sum("e_weights", cfg.e_weights);
    check_sum("t_weights", cfg.t_weights);

    return v;
}
