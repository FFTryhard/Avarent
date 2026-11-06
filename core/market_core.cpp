#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

double bounded_exp(double value) {
    return std::exp(std::clamp(value, -30.0, 30.0));
}

bool cholesky_positive(const std::vector<double>& matrix, int n) {
    std::vector<double> lower(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double value = matrix[static_cast<size_t>(i) * n + j];
            for (int k = 0; k < j; ++k) {
                value -= lower[static_cast<size_t>(i) * n + k] * lower[static_cast<size_t>(j) * n + k];
            }
            if (i == j) {
                if (!std::isfinite(value) || value <= 1e-14) return false;
                lower[static_cast<size_t>(i) * n + j] = std::sqrt(value);
            } else {
                lower[static_cast<size_t>(i) * n + j] = value / lower[static_cast<size_t>(j) * n + j];
            }
        }
    }
    return true;
}

double ukf_kf_error(const std::vector<double>& observations, double mu, double phi, double q, double observation_variance, double alpha, double beta, double kappa) {
    const int n = 1;
    const double scale = alpha * alpha * (n + kappa);
    const double lambda = scale - n;
    const double wm0 = lambda / (n + lambda);
    const double wc0 = wm0 + (1.0 - alpha * alpha + beta);
    const double wi = 1.0 / (2.0 * (n + lambda));
    double ukf_mean = mu;
    double kf_mean = mu;
    double ukf_var = q / (1.0 - phi * phi);
    double kf_var = ukf_var;
    double largest = 0.0;
    const double measurement_mean = -1.2704;
    for (double obs : observations) {
        const double spread = std::sqrt((n + lambda) * ukf_var);
        const double points[3] = {ukf_mean, ukf_mean + spread, ukf_mean - spread};
        const double propagated[3] = {mu + phi * (points[0] - mu), mu + phi * (points[1] - mu), mu + phi * (points[2] - mu)};
        const double predicted_mean = wm0 * propagated[0] + wi * propagated[1] + wi * propagated[2];
        double predicted_var = wc0 * (propagated[0] - predicted_mean) * (propagated[0] - predicted_mean);
        predicted_var += wi * (propagated[1] - predicted_mean) * (propagated[1] - predicted_mean);
        predicted_var += wi * (propagated[2] - predicted_mean) * (propagated[2] - predicted_mean) + q;
        const double predicted_observation = predicted_mean + measurement_mean;
        const double innovation_variance = predicted_var + observation_variance;
        const double gain = predicted_var / innovation_variance;
        const double innovation = obs - predicted_observation;
        ukf_mean = predicted_mean + gain * innovation;
        ukf_var = (1.0 - gain) * (1.0 - gain) * predicted_var + gain * gain * observation_variance;
        const double kf_predicted_mean = mu + phi * (kf_mean - mu);
        const double kf_predicted_var = phi * phi * kf_var + q;
        const double kf_gain = kf_predicted_var / (kf_predicted_var + observation_variance);
        kf_mean = kf_predicted_mean + kf_gain * (obs - kf_predicted_mean - measurement_mean);
        kf_var = (1.0 - kf_gain) * (1.0 - kf_gain) * kf_predicted_var + kf_gain * kf_gain * observation_variance;
        largest = std::max(largest, std::abs(ukf_mean - kf_mean));
        largest = std::max(largest, std::abs(ukf_var - kf_var));
    }
    return largest;
}

py::dict run_core(py::array_t<double, py::array::c_style | py::array::forcecast> returns,
                  py::array_t<double, py::array::c_style | py::array::forcecast> mu_array,
                  py::array_t<double, py::array::c_style | py::array::forcecast> phi_array,
                  py::array_t<double, py::array::c_style | py::array::forcecast> q_array,
                  double forgetting,
                  double variance_floor,
                  double observation_variance,
                  double measurement_mean,
                  double state_bound_sd,
                  double ridge_prior,
                  double standardization_floor,
                  double standardization_clip,
                  int rls_burn_in_days,
                  bool smooth_before_rls,
                  double ema_half_life,
                  bool zero_as_missing) {
    auto r = returns.unchecked<2>();
    auto mu_view = mu_array.unchecked<1>();
    auto phi_view = phi_array.unchecked<1>();
    auto q_view = q_array.unchecked<1>();
    const int rows = static_cast<int>(r.shape(0));
    const int dim = static_cast<int>(r.shape(1));
    if (dim < 1 || mu_view.shape(0) != dim || phi_view.shape(0) != dim || q_view.shape(0) != dim) throw std::runtime_error("return columns and filter parameters must have the same positive dimension");
    if (!(forgetting > 0.0 && forgetting <= 1.0)) throw std::runtime_error("RLS forgetting factor must be in (0, 1]");
    const int p_dim = dim + 1;
    py::array_t<double> h_out({rows, dim});
    py::array_t<double> z_out({rows, dim});
    py::array_t<double> variance_out({rows, dim});
    py::array_t<double> jacobian_out({rows, dim, dim});
    auto h = h_out.mutable_unchecked<2>();
    auto z = z_out.mutable_unchecked<2>();
    auto variance = variance_out.mutable_unchecked<2>();
    auto jacobian = jacobian_out.mutable_unchecked<3>();
    std::vector<double> means(dim, 0.0);
    std::vector<double> mean_count(dim, 0.0);
    std::vector<double> m2(dim, 0.0);
    std::vector<double> state(dim);
    std::vector<double> state_variance(dim);
    std::vector<double> state_mean(dim, 0.0);
    std::vector<double> state_m2(dim, 0.0);
    std::vector<double> state_count(dim, 0.0);
    for (int j = 0; j < dim; ++j) {
        if (!(std::abs(phi_view(j)) < 1.0) || !(q_view(j) > 0.0) || !std::isfinite(mu_view(j))) throw std::runtime_error("invalid frozen filter parameter");
        state[j] = mu_view(j);
        state_variance[j] = q_view(j) / (1.0 - phi_view(j) * phi_view(j));
    }
    std::vector<double> smoothed(static_cast<size_t>(rows) * dim, 0.0);
    const double ema_weight = 1.0 - std::pow(0.5, 1.0 / ema_half_life);
    for (int t = 0; t < rows; ++t) {
        for (int j = 0; j < dim; ++j) {
            const double value = r(t, j);
            if (!std::isfinite(value)) throw std::runtime_error("non-finite return at row " + std::to_string(t));
            const bool missing_observation = zero_as_missing && value == 0.0;
            const double predicted_mean = t == 0 ? mu_view(j) : mu_view(j) + phi_view(j) * (state[j] - mu_view(j));
            const double predicted_variance = t == 0 ? q_view(j) / (1.0 - phi_view(j) * phi_view(j)) : phi_view(j) * phi_view(j) * state_variance[j] + q_view(j);
            double updated_state = predicted_mean;
            double updated_variance = predicted_variance;
            if (!missing_observation) {
                const double residual = value - means[j];
                const double observation = std::log(residual * residual + variance_floor);
                const double innovation_variance = predicted_variance + observation_variance;
                const double gain = predicted_variance / innovation_variance;
                const double innovation = observation - predicted_mean - measurement_mean;
                updated_state = predicted_mean + gain * innovation;
                updated_variance = (1.0 - gain) * (1.0 - gain) * predicted_variance + gain * gain * observation_variance;
            }
            const double stationary_sd = std::sqrt(q_view(j) / (1.0 - phi_view(j) * phi_view(j)));
            if (!std::isfinite(updated_state) || !std::isfinite(updated_variance) || updated_variance <= 0.0) throw std::runtime_error("filter produced a non-finite state or invalid covariance at row " + std::to_string(t));
            if (std::abs(updated_state - mu_view(j)) > state_bound_sd * stationary_sd) throw std::runtime_error("filter state exceeded its declared stationary bound at row " + std::to_string(t));
            state[j] = updated_state;
            state_variance[j] = updated_variance;
            h(t, j) = updated_state;
            variance(t, j) = bounded_exp(updated_state);
            const double prior_smooth = t == 0 ? updated_state : smoothed[static_cast<size_t>(t - 1) * dim + j];
            smoothed[static_cast<size_t>(t) * dim + j] = smooth_before_rls ? prior_smooth + ema_weight * (updated_state - prior_smooth) : updated_state;
            if (!missing_observation) {
                means[j] += (value - means[j]) / (mean_count[j] + 1.0);
                mean_count[j] += 1.0;
            }
        }
    }
    for (int t = 0; t < rows; ++t) {
        for (int j = 0; j < dim; ++j) {
            const double count = state_count[j];
            const double sd = count > 1.0 ? std::sqrt(state_m2[j] / (count - 1.0)) : 0.0;
            const double raw = (smoothed[static_cast<size_t>(t) * dim + j] - state_mean[j]) / std::max(sd, standardization_floor);
            z(t, j) = std::clamp(raw, -standardization_clip, standardization_clip);
            state_count[j] += 1.0;
            const double delta = smoothed[static_cast<size_t>(t) * dim + j] - state_mean[j];
            state_mean[j] += delta / state_count[j];
            state_m2[j] += delta * (smoothed[static_cast<size_t>(t) * dim + j] - state_mean[j]);
        }
    }
    std::vector<double> theta(static_cast<size_t>(dim) * p_dim, 0.0);
    std::vector<double> covariance(static_cast<size_t>(p_dim) * p_dim, 0.0);
    std::vector<double> x(p_dim, 0.0);
    std::vector<double> px(p_dim, 0.0);
    std::vector<double> updated(static_cast<size_t>(p_dim) * p_dim, 0.0);
    for (int i = 0; i < p_dim; ++i) covariance[static_cast<size_t>(i) * p_dim + i] = ridge_prior;
    for (int t = 0; t < rows; ++t) {
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) jacobian(t, i, j) = theta[static_cast<size_t>(i) * p_dim + j + 1];
        }
        if (t + 1 >= rows) break;
        x[0] = 1.0;
        for (int j = 0; j < dim; ++j) x[j + 1] = z(t, j);
        for (int i = 0; i < p_dim; ++i) {
            px[i] = 0.0;
            for (int j = 0; j < p_dim; ++j) px[i] += covariance[static_cast<size_t>(i) * p_dim + j] * x[j];
        }
        double denominator = forgetting;
        for (int i = 0; i < p_dim; ++i) denominator += x[i] * px[i];
        if (!std::isfinite(denominator) || denominator <= 0.0) throw std::runtime_error("RLS covariance update has an invalid denominator at row " + std::to_string(t));
        for (int i = 0; i < dim; ++i) {
            double prediction = theta[static_cast<size_t>(i) * p_dim];
            for (int j = 0; j < dim; ++j) prediction += theta[static_cast<size_t>(i) * p_dim + j + 1] * z(t, j);
            const double error = z(t + 1, i) - prediction;
            for (int j = 0; j < p_dim; ++j) theta[static_cast<size_t>(i) * p_dim + j] += (px[j] / denominator) * error;
        }
        for (int i = 0; i < p_dim; ++i) {
            for (int j = 0; j < p_dim; ++j) updated[static_cast<size_t>(i) * p_dim + j] = (covariance[static_cast<size_t>(i) * p_dim + j] - px[i] * px[j] / denominator) / forgetting;
        }
        for (int i = 0; i < p_dim; ++i) {
            for (int j = i; j < p_dim; ++j) {
                const double sym = 0.5 * (updated[static_cast<size_t>(i) * p_dim + j] + updated[static_cast<size_t>(j) * p_dim + i]);
                covariance[static_cast<size_t>(i) * p_dim + j] = sym;
                covariance[static_cast<size_t>(j) * p_dim + i] = sym;
            }
        }
        if (t >= rls_burn_in_days && !cholesky_positive(covariance, p_dim)) throw std::runtime_error("RLS covariance lost positive definiteness at row " + std::to_string(t));
        for (double value : theta) if (!std::isfinite(value)) throw std::runtime_error("RLS produced a non-finite coefficient at row " + std::to_string(t));
    }
    py::dict result;
    result["h"] = std::move(h_out);
    result["z"] = std::move(z_out);
    result["variance"] = std::move(variance_out);
    result["J"] = std::move(jacobian_out);
    return result;
}

PYBIND11_MODULE(market_core, module) {
    module.def("run_core", &run_core, py::arg("returns"), py::arg("mu"), py::arg("phi"), py::arg("q"), py::arg("forgetting"), py::arg("variance_floor"), py::arg("observation_variance"), py::arg("measurement_mean"), py::arg("state_bound_sd"), py::arg("ridge_prior"), py::arg("standardization_floor"), py::arg("standardization_clip"), py::arg("rls_burn_in_days"), py::arg("smooth_before_rls") = false, py::arg("ema_half_life") = 10.0, py::arg("zero_as_missing") = false);
    module.def("ukf_kf_error", &ukf_kf_error);
    module.def("bounded_exp", &bounded_exp);
}
