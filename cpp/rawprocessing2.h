#pragma once

#include "rawprocessing.h"
#include <cmath>
#include <stdexcept>
#include <vector>
#include "utils.h"

// ─────────────────────────────────────────────────────────────────────────────
// NormalizerState
//
// Returned every tick. GEC reads directly from this instead of recomputing
// its own eigendecomposition.
//
//   eigenvalues   → C  (participation ratio delta)
//   eigenvectors  → G  (subspace rotation via principal angles)
//   maha          → E  (instability proxy, raw anomaly magnitude)
//   z             → normalized output (keep for spike detection etc.)
//   ready         → false during warmup, GEC should return Neutral
// ─────────────────────────────────────────────────────────────────────────────
struct NormalizerState
{
    Eigen::VectorXd eigenvalues;  // ascending, size = dim
    Eigen::MatrixXd eigenvectors; // columns are eigenvectors, size = dim×dim
    double maha = 0.0;
    std::vector<double> z; // whitened residual (original output)
    bool ready = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// GEC helper functions
//
// Call these with consecutive NormalizerState pairs to produce G, E, C scalars
// that feed directly into your continuous regime scoring.
// ─────────────────────────────────────────────────────────────────────────────

// G — subspace rotation magnitude between two eigenvector bases.
// Uses principal angles between column spaces (Wedin bound proxy).
// Returns the Frobenius norm of the principal angle vector.
inline double computeG(const Eigen::MatrixXd &V_prev,
                       const Eigen::MatrixXd &V_curr,
                       const Eigen::VectorXd &eigenvalues)
{
    // Fix: use largest eigenvalue ratio, not smallest
    int n = eigenvalues.size(); 
    double confidence = (eigenvalues[n-1] - eigenvalues[n-2]) + 1e-12 / eigenvalues[n-1];

    Eigen::MatrixXd U_overlap = V_prev.transpose() * V_curr;
    double sin_normMOS = wedin_theoremKGT1(U_overlap);
    double sin_normDS  = wedin_theoremK1(
        Eigen::VectorXd(V_curr.col(n-1)),   // dominant eigenvector
        Eigen::VectorXd(V_prev.col(n-1))
    );
    //if λ1 and λ2 are almost identical, the data forms a perfect circle, meaning there is no point in tracking locking
    return confidence * (0.5 * sin_normMOS + 0.5 * sin_normDS);
}
// C — rate of change of eigenvalue participation ratio (complexity delta).
// Participation ratio PR = (Σλ)² / Σλ²  ∈ [1, dim].
// High PR → many factors contributing equally (dispersed/complex).
// Low PR  → one factor dominates (compressed/trending).
// C = PR(t) - PR(t-1):  positive = complexity expanding, negative = collapsing.
inline double computeC(const Eigen::VectorXd &eig_prev,
                       const Eigen::VectorXd &eig_curr)
{
    auto participationRatio = [](const Eigen::VectorXd &e) -> double
    {
        double sum_e = e.sum();
        if (sum_e < 1e-10)
            return 1.0;
        double sum_e2 = e.array().square().sum();
        if (sum_e2 < 1e-10)
            return 1.0;
        return (sum_e * sum_e) / sum_e2;
    };

    return participationRatio(eig_curr); /* - participationRatio(eig_prev); */
}

// E — instability / anomaly magnitude.
// Directly use maha from NormalizerState — it is already the Mahalanobis
// distance of the current observation from the running mean, normalized by
// the current covariance. This is exactly what E should measure and it is
// computed before the normalizer absorbs the signal.
//
// Optionally z-score E against its own rolling history to make it comparable
// in scale to G and C before feeding the regime scorer.
inline double computeE(double maha)
{
    return maha; // pass-through; z-score externally if needed
}

// ─────────────────────────────────────────────────────────────────────────────
// AdaptiveOnlineNormalizer
// ─────────────────────────────────────────────────────────────────────────────
class AdaptiveOnlineNormalizer2
{
public:
    AdaptiveOnlineNormalizer2() = default;

    AdaptiveOnlineNormalizer2(double base_alpha_,
                              double min_alpha_,
                              double decay_,
                              int warmup_steps_ = 150)
        : base_alpha(base_alpha_), min_alpha(min_alpha_), decay(decay_), warmup_steps(warmup_steps_)
    {
    }

    // ── Main entry point ────────────────────────────────────────────────────
    // Returns a NormalizerState every tick.
    // GEC reads .eigenvalues, .eigenvectors, .maha directly.
    // .ready is false during warmup — return Neutral in the regime classifier.
    NormalizerState process(const std::vector<double> &x_std)
    {
        NormalizerState state;

        if (x_std.empty())
            return state;

        if (!initialized)
        {
            init(x_std.size());
            mean = Eigen::Map<const Vec>(x_std.data(), dim);
            state.z = std::vector<double>(dim, 0.0);
            state.ready = false;
            state.eigenvalues = eigenvalues_cache;
            state.eigenvectors = eigenvectors_cache;
            state.maha = 0.0;
            ++t;
            return state;
        }

        if (x_std.size() != static_cast<size_t>(dim))
            throw std::runtime_error("AdaptiveOnlineNormalizer: dimension mismatch");

        Vec x = Eigen::Map<const Vec>(x_std.data(), dim);

        // ── diff BEFORE mean update (used for both maha and whitened output) ──
        Vec diff = x - mean;

        // ── Mahalanobis distance (E signal — read BEFORE normalizer absorbs it) ──
        Vec z_prev = whitening * diff;
        double maha = std::sqrt(std::max(z_prev.squaredNorm(), eps));

        // ── Adaptive alpha ──────────────────────────────────────────────────
        double alpha = compute_alpha(maha);

        // ── Mean update ─────────────────────────────────────────────────────
        Vec prev_mean = mean;
        mean = alpha * x + (1.0 - alpha) * mean;

        // ── Covariance update (EMA) ──────────────────────────────────────────
        Vec d = x - prev_mean;
        cov = alpha * (d * d.transpose()) + (1.0 - alpha) * cov;
        cov.diagonal().array() += eps; // regularize

        // ── Eigendecomposition every tick (interval = 1) ────────────────────
        // 9×9 SelfAdjointEigenSolver is microseconds — no reason to skip.
        // This is the source of G and C; staleness here = staleness in GEC.
        update_eigen();

        // ── Whitened output (keep for spike detection / downstream use) ──────
        Vec z = whitening * diff;

        // ── Populate state ───────────────────────────────────────────────────
        state.eigenvalues = eigenvalues_cache;
        state.eigenvectors = eigenvectors_cache;
        state.maha = maha;
        state.z = std::vector<double>(z.data(), z.data() + dim);
        state.ready = (t >= static_cast<size_t>(warmup_steps));

        ++t;
        return state;
    }

    // Convenience: returns the previous tick's eigenvalues/eigenvectors so the
    // caller can compute G and C deltas without storing state externally.
    const Eigen::VectorXd &prevEigenvalues() const { return eigenvalues_prev; }
    const Eigen::MatrixXd &prevEigenvectors() const { return eigenvectors_prev; }

    bool isReady() const { return t >= static_cast<size_t>(warmup_steps); }
    size_t tick() const { return t; }

private:
    using Vec = Eigen::VectorXd;
    using Mat = Eigen::MatrixXd;

    size_t dim = 0;
    size_t t = 0;
    bool initialized = false;

    Vec mean;
    Mat cov;
    Mat whitening;

    // Cached eigen state — updated every tick
    Vec eigenvalues_cache;
    Mat eigenvectors_cache;

    // Previous tick — exposed for delta computation (G, C)
    Vec eigenvalues_prev;
    Mat eigenvectors_prev;

    // ── Hyperparams ──────────────────────────────────────────────────────────
    double base_alpha = 0.2;
    double min_alpha = 0.01;
    double decay = 0.001;
    int warmup_steps = 150; // replaces isReady() discrete buffer gate

    static constexpr double eps = 1e-6;

    void init(size_t d)
    {
        dim = d;
        mean = Vec::Zero(d);
        cov = Mat::Identity(d, d);
        whitening = Mat::Identity(d, d);
        eigenvalues_cache = Vec::Ones(d);
        eigenvectors_cache = Mat::Identity(d, d);
        eigenvalues_prev = Vec::Ones(d);
        eigenvectors_prev = Mat::Identity(d, d);
        initialized = true;
    }

    double compute_alpha(double maha) const
    {
        double alpha_time = base_alpha / (1.0 + decay * static_cast<double>(t));
        double boost = std::min(3.0, 1.0 + 0.15 * maha);
        double alpha = alpha_time * boost;
        return std::clamp(alpha, min_alpha, 0.5);
    }

    void update_eigen()
    {
        // Rotate cache → prev before overwriting
        eigenvalues_prev = eigenvalues_cache;
        eigenvectors_prev = eigenvectors_cache;

        Eigen::SelfAdjointEigenSolver<Mat> solver(cov);
        if (solver.info() != Eigen::Success)
            return; // keep stale cache rather than exploding

        eigenvalues_cache = solver.eigenvalues();
        eigenvectors_cache = solver.eigenvectors();

        // Clamp eigenvalues for numerical stability
        for (int i = 0; i < eigenvalues_cache.size(); ++i)
            eigenvalues_cache[i] = std::max(eigenvalues_cache[i], eps);

        // Rebuild whitening matrix Σ^(-1/2)
        Vec inv_sqrt = eigenvalues_cache.array().rsqrt();
        whitening = eigenvectors_cache * inv_sqrt.asDiagonal() * eigenvectors_cache.transpose();
    }
};