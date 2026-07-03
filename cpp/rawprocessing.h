#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <vector>


class AdaptiveOnlineNormalizer {
public:
  AdaptiveOnlineNormalizer() = default;
  AdaptiveOnlineNormalizer(double base_alpha_, double min_alpha_, double decay_,
                           int eig_interval)
      : base_alpha(base_alpha_), min_alpha(min_alpha_), decay(decay_),
        eig_update_interval(eig_interval) {}

  std::vector<double> process(const std::vector<double> &x_std) {
    if (x_std.empty())
      return {};

    if (!initialized) {
      init(x_std.size());
      mean = Eigen::Map<const Vec>(x_std.data(), dim);
      return std::vector<double>(dim, 0.0);
    }

    if (x_std.size() != dim)
      throw std::runtime_error("Dimension mismatch");

    Vec x = Eigen::Map<const Vec>(x_std.data(), dim);

    // ---- diff BEFORE update ----
    Vec diff = x - mean;

    // ---- Mahalanobis (using cached whitening) ----
    Vec z_prev = whitening * diff;
    double maha = std::sqrt(std::max(z_prev.squaredNorm(), eps));

    // ---- Adaptive alpha ----
    double alpha = compute_alpha(maha);

    // ---- Mean update ----
    Vec prev_mean = mean;
    mean = alpha * x + (1.0 - alpha) * mean;

    // ---- Covariance update (EMA) ----
    Vec d = x - prev_mean;
    cov = alpha * (d * d.transpose()) + (1.0 - alpha) * cov;

    // ---- Regularization ----
    cov.diagonal().array() += eps;

    // ---- Update whitening occasionally ----
    if (t % eig_update_interval == 0)
      update_whitening();

    // ---- Final normalized output ----
    Vec z = whitening * diff;

    ++t;

    return std::vector<double>(z.data(), z.data() + dim);
  }

private:
  using Vec = Eigen::VectorXd;
  using Mat = Eigen::MatrixXd;

  size_t dim = 0;
  size_t t = 1;
  bool initialized = false;

  Vec mean;
  Mat cov;

  // Whitening matrix Σ^(-1/2)
  Mat whitening;

  // ---- Hyperparams ----
  const double base_alpha = 0.2;
  const double min_alpha = 0.01;
  const double decay = 0.001;

  const double eps = 1e-6;

  const size_t eig_update_interval = 10; // tradeoff: cost vs freshness

  void init(size_t d) {
    dim = d;
    mean = Vec::Zero(dim);
    cov = Mat::Identity(dim, dim);
    whitening = Mat::Identity(dim, dim);

    initialized = true;
  }

  double compute_alpha(double maha) const {
    double alpha_time = base_alpha / (1.0 + decay * t);

    // Boost when distribution shifts
    double boost = std::min(3.0, 1.0 + 0.15 * maha);

    double alpha = alpha_time * boost;

    return std::clamp(alpha, min_alpha, 0.5);
  }

  void update_whitening() {
    Eigen::SelfAdjointEigenSolver<Mat> solver(cov);

    if (solver.info() != Eigen::Success)
      return; // keep old whitening instead of exploding

    Vec eigenvalues = solver.eigenvalues();
    Mat eigenvectors = solver.eigenvectors();

    // Clamp eigenvalues (numerical stability)
    for (int i = 0; i < eigenvalues.size(); ++i)
      eigenvalues[i] = std::max(eigenvalues[i], eps);

    // Σ^(-1/2)
    Vec inv_sqrt = eigenvalues.array().rsqrt();

    whitening = eigenvectors * inv_sqrt.asDiagonal() * eigenvectors.transpose();
  }
};

/*

you can see where i injected noise,
and we can see the intrinsic turbulence of the data.
a lot values repeat but thats just because the resolution of my data is very low
and alot of values fall into the same bin. once i get more complicated data that
ranges from 1 - 100 instead of 20 to 30. we can see more. but this looks good


*/
