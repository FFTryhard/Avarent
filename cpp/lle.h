#pragma once

#include "utils.h"
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <stdexcept>

struct LLE_Result {
    double lle;                       // estimated largest Lyapunov exponent (per unit time)
    std::vector<double> t;            // time axis for curve
    std::vector<double> mean_log_d;   // <log distance> curve used for fit
    int fit_start_k;
    int fit_end_k;
};

inline double splitMedianMadChange(const std::vector<double> &data){

    int m_idx = data.size() / 2;
    double eps = 1e-6;

    std::vector <double> m1;
    std::vector <double> m2;

    for(int i = 0; i < m_idx; i++){
        m1.push_back(data[i]);
    }
    for(int i = m_idx; i < data.size(); i++){
        m2.push_back(data[i]);
    }

    double m1_m = median(m1);
    double m2_m = median(m2);
    double mad = computeMAD(data);

    double z = std::abs(m2_m - m1_m) / (1.4826 * mad);

    return z + eps;
}

// simple linear regression y = a + b x on [k0, k1] inclusive
inline double slopeOLS_(const std::vector<double>& x, const std::vector<double>& y, int k0, int k1) {
    double sx=0, sy=0, sxx=0, sxy=0;
    int n = 0;
    for (int k = k0; k <= k1; ++k) {
        sx += x[k];
        sy += y[k];
        sxx += x[k]*x[k];
        sxy += x[k]*y[k];
        ++n;
    }
    double denom = (n*sxx - sx*sx);
    if (std::abs(denom) < 1e-12) return 0.0;
    return (n*sxy - sx*sy) / denom;
}

// Minimal Rosenstein LLE from state vectors
inline LLE_Result largest_lyapunov_rosenstein(
    const Eigen::MatrixXd& X,   // N x d
    double dt                  // sampling step in seconds (or your time units)
) {
    if (dt <= 0.0) throw std::invalid_argument("dt must be > 0.");
    const int N = (int)X.rows();
    const int d = (int)X.cols();
    if (N < 200) throw std::invalid_argument("Need more data. Give me >= 200 samples (more is better).");
    if (d <= 0) throw std::invalid_argument("X must have d>0 columns.");

    // Hard-coded defaults chosen to be sane without babysitting
    const int theiler = std::max(10, (int)std::round(1.0 / dt)); // exclude ~1 time unit if dt small, else >=10 samples
    const int k_max   = std::min(300, (N - 1) / 5);              // track divergence up to ~N/5 or 300 steps
    const double eps  = 1e-12;

    // For each i, find nearest neighbor j (excluding Theiler window and requiring room for k_max steps)
    std::vector<int> nbr(N, -1);
    std::vector<double> d0(N, 0.0);

    const int i_min = 0;
    const int i_max = N - 1 - k_max;  // need i+k_max < N

    for (int i = i_min; i <= i_max; ++i) {
        double best = std::numeric_limits<double>::infinity();
        int best_j = -1;

        // brute force nearest neighbor (O(N^2)); fine for N up to maybe ~50k.
        // If N is huge, you’ll want a KD-tree. Don’t overengineer unless needed.
        for (int j = i_min; j <= i_max; ++j) {
            if (std::abs(j - i) <= theiler) continue;
            double dist2 = (X.row(i) - X.row(j)).squaredNorm();
            if (dist2 < best) {
                best = dist2;
                best_j = j;
            }
        }

        if (best_j >= 0) {
            nbr[i] = best_j;
            d0[i] = std::sqrt(std::max(best, eps));
        }
    }

    // Build mean log-distance curve
    std::vector<double> t(k_max + 1), mean_log_d(k_max + 1, 0.0);
    for (int k = 0; k <= k_max; ++k) t[k] = k * dt;

    for (int k = 0; k <= k_max; ++k) {
        double sum = 0.0;
        int cnt = 0;
        for (int i = i_min; i <= i_max; ++i) {
            int j = nbr[i];
            if (j < 0) continue;
            double dist = (X.row(i + k) - X.row(j + k)).norm();
            dist = std::max(dist, eps);
            sum += std::log(dist);
            ++cnt;
        }
        mean_log_d[k] = (cnt > 0) ? (sum / cnt) : 0.0;
    }

    // Automatically pick a fit region:
    // - avoid k=0 (noise / neighbor selection artifacts)
    // - use early segment before saturation
    // Minimal heuristic:
    //   fit over k in [k0, k1] where curve is increasing and roughly linear
    int k0 = 5;
    int k1 = std::max(k0 + 20, k_max / 3);

    // Robustify: remove obvious non-growth / saturation by scanning for a stable slope window
    // We choose the window with the largest positive slope among a few candidates.
    double best_slope = -1e9;
    int best_k0 = k0, best_k1 = k1;

    const int win = std::min(80, std::max(30, k_max / 4)); // window size in steps
    for (int start = 5; start + win <= k_max; start += 5) {
        int end = start + win;

        // Reject windows that already saturate: slope too small or negative
        double s = slopeOLS_(t, mean_log_d, start, end);
        if (s > best_slope) {
            best_slope = s;
            best_k0 = start;
            best_k1 = end;
        }
    }

    // One more sanity: if slope is not positive, you probably don't have chaos or data is junk
    // Still return it; user can interpret.
    double lle = best_slope;

    // Extra robust check: if the curve is extremely noisy, median-filter slopes a bit
    // (Keep minimal: no extra knobs.)
    // We'll compute per-step increments in the fit window and drop outliers via MAD.
    {
        std::vector<double> inc;
        inc.reserve(best_k1 - best_k0);
        for (int k = best_k0; k < best_k1; ++k)
            inc.push_back((mean_log_d[k+1] - mean_log_d[k]) / dt);

        double med = median(inc);
        std::vector<double> absdev;
        absdev.reserve(inc.size());
        for (double v : inc) absdev.push_back(std::abs(v - med));
        double mad = median(absdev);
        double thresh = med + 6.0 * std::max(mad, 1e-12);

        // recompute slope using only increments not absurdly large
        // (equivalent to trimming points)
        std::vector<double> xt, yt;
        xt.reserve(best_k1 - best_k0 + 1);
        yt.reserve(best_k1 - best_k0 + 1);

        xt.push_back(t[best_k0]);
        yt.push_back(mean_log_d[best_k0]);
        for (int k = best_k0; k < best_k1; ++k) {
            double step_slope = (mean_log_d[k+1] - mean_log_d[k]) / dt;
            if (std::abs(step_slope) <= std::abs(thresh)) {
                xt.push_back(t[k+1]);
                yt.push_back(mean_log_d[k+1]);
            }
        }

        if ((int)xt.size() >= 10) {
            // OLS on filtered points
            double sx=0, sy=0, sxx=0, sxy=0;
            int n = (int)xt.size();
            for (int i = 0; i < n; ++i) {
                sx += xt[i]; sy += yt[i];
                sxx += xt[i]*xt[i];
                sxy += xt[i]*yt[i];
            }
            double denom = (n*sxx - sx*sx);
            if (std::abs(denom) > 1e-12)
                lle = (n*sxy - sx*sy) / denom;
        }
    }

    LLE_Result out;
    out.lle = lle;
    out.t = std::move(t);
    out.mean_log_d = std::move(mean_log_d);
    out.fit_start_k = best_k0;
    out.fit_end_k = best_k1;
    return out;
}

inline std::tuple<bool, double> stabilityCheckLLE(const LLE_Result &out){

    std::vector<double> t_target;
    std::vector<double> mean_log_d_target;

    int window_size = out.fit_end_k - out.fit_start_k;

    for(int i = out.fit_start_k; i < out.fit_end_k; i++){
        double mldt = out.mean_log_d[i];
        double tt = out.t[i];

        t_target.push_back(tt);
        mean_log_d_target.push_back(mldt);
    }

    //run r squared, slope check and min window check on target windows: R2=∑(xi​−xˉ)2∑(yi​−yˉ​)2(∑(xi​−xˉ)(yi​−yˉ​))2​

    double R2 = computeR2(t_target, mean_log_d_target); //should be > 0.97
    //IF I ONLY WANT CHAOS, INSINT SLOPE > 0. BUT FUCK THAT
    //MIN WINDOW ALREADY CHECKED I THINK

    //to do: 2 mini windows outside, size 20. slope consistency
    //to do: each iteration exponent should be stable looking, maybe track that instead of mean_log_d/t

    std::cerr << "r2: " << R2 << std::endl;

    bool stable = R2 > 0.93 && window_size > 20;
    return {stable, R2};
}