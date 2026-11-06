import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def principal_angle(left, right):
    singular = np.linalg.svd(left.T @ right, compute_uv=False)
    return float(np.mean(np.arccos(np.clip(singular, 0.0, 1.0))))


def dynamic_metrics(jacobians, horizon, lag=20, rank=3):
    count, dim, _ = jacobians.shape
    rotation = np.zeros(count)
    _, singular, right = np.linalg.svd(jacobians, full_matrices=False)
    right_vectors = np.swapaxes(right, 1, 2)
    power = np.linalg.matrix_power(jacobians, horizon)
    fragility = np.linalg.svd(power, compute_uv=False)[:, 0]
    energy = singular * singular
    probabilities = energy / np.maximum(energy.sum(axis=1, keepdims=True), 1e-300)
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0.0
    terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    entropy = np.exp(-terms.sum(axis=1))
    persistence = np.max(np.abs(np.linalg.eigvals(jacobians)), axis=1) ** horizon
    if count > lag:
        left_spaces = right_vectors[lag:, :, :rank]
        right_spaces = right_vectors[:-lag, :, :rank]
        cross = np.einsum("nki,nkj->nij", left_spaces, right_spaces)
        rotation[lag:] = np.mean(np.arccos(np.clip(np.linalg.svd(cross, compute_uv=False), 0.0, 1.0)), axis=1)
    return {"R": rotation, "H": entropy, "F": fragility, "rho_power": persistence, "right_vectors": right_vectors}


def rolling_covariances(values, window):
    data = np.asarray(values, dtype=np.float64)
    n, dim = data.shape
    sums = np.vstack([np.zeros(dim), np.cumsum(data, axis=0)])
    outer = np.einsum("ni,nj->nij", data, data)
    outer_sums = np.concatenate([np.zeros((1, dim, dim)), np.cumsum(outer, axis=0)], axis=0)
    output = np.full((n, dim, dim), np.nan)
    for end in range(window - 1, n):
        start = end + 1 - window
        sum_x = sums[end + 1] - sums[start]
        sum_xx = outer_sums[end + 1] - outer_sums[start]
        centered = sum_xx - np.outer(sum_x, sum_x) / window
        output[end] = centered / (window - 1)
    return output


def stress_metrics(dates, log_returns, equal_weight, config):
    n = len(dates)
    ew = np.asarray(equal_weight, dtype=np.float64)
    vol_window = config["volatility_window_days"]
    vol = pd.Series(ew).rolling(vol_window, min_periods=vol_window).std(ddof=1).to_numpy() * np.sqrt(config["annualization_days"])
    wealth = np.cumprod(1.0 + ew)
    peak = pd.Series(wealth).rolling(config["drawdown_peak_window_days"], min_periods=1).max().to_numpy()
    drawdown = wealth / peak - 1.0
    cov60 = rolling_covariances(log_returns, config["correlation_window_days"])
    corr_mean = np.full(n, np.nan)
    for t in range(config["correlation_window_days"] - 1, n):
        covariance = cov60[t]
        scales = np.sqrt(np.maximum(np.diag(covariance), 1e-20))
        correlation = covariance / np.outer(scales, scales)
        corr_mean[t] = (correlation.sum() - np.trace(correlation)) / (log_returns.shape[1] * (log_returns.shape[1] - 1))
    corr_tilde = np.zeros(n)
    for t in range(n):
        if np.isfinite(corr_mean[t]):
            prior = corr_mean[:t]
            prior = prior[np.isfinite(prior)]
            corr_tilde[t] = corr_mean[t] - (float(np.median(prior)) if prior.size else corr_mean[t])
    cov250 = rolling_covariances(log_returns, config["covariance_window_days"])
    absorption = np.full(n, np.nan)
    for t in range(config["covariance_window_days"] - 1, n):
        eigen = np.linalg.eigvalsh(cov250[t])
        absorption[t] = float(np.sum(eigen[-2:]) / max(np.sum(eigen), 1e-300))
    log_abs = np.log(np.abs(ew) + 1e-8)
    ar1 = np.full(n, np.nan)
    for t in range(config["covariance_window_days"] - 1, n):
        sample = log_abs[t - config["covariance_window_days"] + 1:t + 1]
        x = sample[:-1]
        y = sample[1:]
        centered = x - x.mean()
        denominator = float(centered @ centered)
        ar1[t] = float(centered @ (y - y.mean()) / denominator) if denominator > 0.0 else 0.0
    return {"vol": vol, "drawdown": drawdown, "rho_tilde": corr_tilde, "corr_mean": corr_mean, "absorption": absorption, "ar1": ar1, "cov250": cov250}


def direction_analysis(covariances, right_vectors, dates, test_start, seed, draws=1000, rank=3):
    test_mask = np.asarray(dates) >= np.datetime64(test_start)
    test_positions = np.flatnonzero(test_mask)
    angles = []
    for t in test_positions:
        if not np.all(np.isfinite(covariances[t])):
            continue
        _, eigenvectors = np.linalg.eigh(covariances[t])
        covariance_space = eigenvectors[:, -rank:]
        susceptibility_space = right_vectors[t, :, :rank]
        angles.append(principal_angle(covariance_space, susceptibility_space))
    generator = np.random.default_rng(seed)
    reference = np.eye(right_vectors.shape[1])[:, :rank]
    random_angles = np.empty(draws)
    for index in range(draws):
        gaussian = generator.normal(size=(right_vectors.shape[1], rank))
        random_space, _ = np.linalg.qr(gaussian)
        random_angles[index] = principal_angle(reference, random_space[:, :rank])
    percentiles = np.asarray([np.mean(random_angles <= angle) for angle in angles])
    return {"mean_angle_radians": float(np.mean(angles)), "median_haar_percentile": float(np.median(percentiles)), "date_count": int(len(angles)), "haar_draws": int(draws), "haar_mean_angle_radians": float(np.mean(random_angles)), "angles": np.asarray(angles), "haar_angles": random_angles}


def correlation_screen(metrics, features, test_mask):
    fragility = np.asarray(metrics["F"])[test_mask]
    names = ["vol", "drawdown", "rho_tilde", "ar1", "absorption"]
    correlations = {}
    for name in names:
        values = np.asarray(features[name])[test_mask]
        mask = np.isfinite(fragility) & np.isfinite(values)
        correlations[name] = float(spearmanr(fragility[mask], values[mask]).statistic)
    return correlations


def pca_components_90(matrix, variance_share=0.9):
    values = np.asarray(matrix, dtype=np.float64)
    values = values[np.all(np.isfinite(values), axis=1)]
    standardized = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0, ddof=1), 1e-12)
    singular = np.linalg.svd(standardized, compute_uv=False)
    variance = singular * singular
    cumulative = np.cumsum(variance / variance.sum())
    return int(np.searchsorted(cumulative, variance_share) + 1), cumulative


def empirical_two_sided(observed, null_values):
    null = np.asarray(null_values)
    center = float(np.median(null))
    extreme = np.count_nonzero(np.abs(null - center) >= abs(observed - center))
    return float((extreme + 1) / (len(null) + 1))


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[position]))
        adjusted[position] = running
    return adjusted


def holm_intervals(bootstrap_values, alpha=0.05):
    values = np.asarray(bootstrap_values, dtype=np.float64)
    candidates = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 401)))
    intervals = []
    for item in range(values.shape[0]):
        accepted = []
        for null_value in candidates:
            pvalues = []
            for other in range(values.shape[0]):
                left = np.mean(values[other] <= null_value)
                right = np.mean(values[other] >= null_value)
                pvalues.append(min(1.0, 2.0 * min(left, right)))
            adjusted = holm_adjust(pvalues)
            if adjusted[item] >= alpha:
                accepted.append(float(null_value))
        intervals.append([min(accepted), max(accepted)] if accepted else [float(candidates[0]), float(candidates[-1])])
    return intervals
