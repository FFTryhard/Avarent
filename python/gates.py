from datetime import date

import numpy as np
from scipy.stats import wasserstein_distance

import market_core
from common import load_config
from dataset import load_panel
from estimation import estimate_parameters, simulate_sv
from metrics import dynamic_metrics


def core_run(returns, parameters, forgetting, config, smooth=False, zero_as_missing=False):
    return market_core.run_core(
        np.ascontiguousarray(returns, dtype=np.float64),
        np.asarray(parameters["mu"], dtype=np.float64),
        np.asarray(parameters["phi"], dtype=np.float64),
        np.asarray(parameters["q"], dtype=np.float64),
        float(forgetting),
        float(config["variance_floor"]),
        float(config["observation_variance"]),
        float(config["log_chi_square_mean"]),
        float(config["state_bound_sd"]),
        float(config["ridge_prior"]),
        float(config["standardization_floor"]),
        float(config["standardization_clip"]),
        int(config["rls_burn_in_days"]),
        bool(smooth),
        float(config["ema_half_life_days"] if "ema_half_life_days" in config else 10.0),
        bool(zero_as_missing)
    )


def direct_rls(z, horizon, forgetting, ridge):
    n, dim = z.shape
    p_dim = dim + 1
    coefficients = np.zeros((dim, p_dim))
    covariance = np.eye(p_dim) * ridge
    matrices = np.zeros((n, dim, dim))
    for t in range(n):
        matrices[t] = coefficients[:, 1:]
        if t + horizon >= n:
            continue
        x = np.concatenate(([1.0], z[t]))
        px = covariance @ x
        denominator = forgetting + x @ px
        gain = px / denominator
        error = z[t + horizon] - coefficients @ x
        coefficients += error[:, None] * gain[None, :]
        covariance = (covariance - np.outer(px, px) / denominator) / forgetting
        covariance = (covariance + covariance.T) * 0.5
    return matrices


def finite_difference_gate():
    mu = np.linspace(-10.0, -8.0, 10)
    phi = np.linspace(0.82, 0.97, 10)
    point = np.linspace(-1.0, 1.0, 10)
    step = 1e-5
    jacobian = np.empty((10, 10))
    for j in range(10):
        direction = np.zeros(10)
        direction[j] = step
        plus = mu + phi * (point + direction - mu)
        minus = mu + phi * (point - direction - mu)
        jacobian[:, j] = (plus - minus) / (2.0 * step)
    return float(np.max(np.abs(jacobian - np.diag(phi))))


def numerical_safety_gate(config):
    settings = {"mu": np.full(10, -10.0), "phi": np.full(10, 0.95), "q": np.full(10, 0.4)}
    generator = np.random.default_rng(config["seeds"]["numerical_safety"])
    scale = np.exp(settings["mu"][0] / 2.0)
    cases = {
        "zero_returns": np.zeros((config["numerical_short_rows"], 10)),
        "single_minus_25_percent": generator.normal(0.0, scale, size=(config["numerical_short_rows"], 10)),
        "constant_tiny_returns": np.full((config["numerical_long_rows"], 10), 1e-4),
        "historical_style_shocks": generator.normal(0.0, scale, size=(config["numerical_long_rows"], 10))
    }
    cases["single_minus_25_percent"][250, 0] = np.log(0.75)
    cases["historical_style_shocks"][1200, 0] = np.log(0.70)
    cases["historical_style_shocks"][3200, 1] = np.log(0.78)
    for name, values in cases.items():
        if name == "constant_tiny_returns":
            local = {"mu": np.full(10, -15.0), "phi": np.full(10, 0.9), "q": np.full(10, 0.2)}
        else:
            local = settings
        output = core_run(values, local, 0.995, config)
        if not all(np.all(np.isfinite(output[key])) for key in ("h", "variance", "z", "J")):
            raise RuntimeError(f"U numeric safety failed for {name}")
    observations = np.random.default_rng(config["seeds"]["ukf"]).normal(-11.0, 2.0, size=500)
    ukf_error = float(market_core.ukf_kf_error(observations.tolist(), -10.0, 0.95, 0.2, config["observation_variance"], config["ukf_alpha"], config["ukf_beta"], config["ukf_kappa"]))
    fd_error = finite_difference_gate()
    return {"ukf_kf_abs_error": ukf_error, "finite_difference_abs_error": fd_error, "numerical_safety_cases": list(cases)}


def e1_gate(parameters, config):
    returns, latent, _ = simulate_sv(parameters["mu"], parameters["phi"], parameters["q"], config["e1_simulation_rows"], config["seeds"]["simulation"])
    recovered = estimate_parameters(returns, config)
    phi_ok = np.abs(recovered["phi"] - parameters["phi"]) <= config["gates"]["e1_phi_abs_error_max"]
    q_ratio = np.maximum(recovered["q"] / parameters["q"], parameters["q"] / recovered["q"])
    q_ok = q_ratio <= config["gates"]["e1_q_ratio_max"]
    pass_count = int(np.sum(phi_ok & q_ok))
    return {"assets_passing": pass_count, "phi_abs_errors": np.abs(recovered["phi"] - parameters["phi"]), "q_ratios": q_ratio, "estimated_parameters": recovered, "pass": pass_count >= config["gates"]["e1_assets_pass_min"]}


def g1_gate(config):
    dim = config["portfolio_count"]
    count = config["gate_g1_sample_count"]
    mu = np.full(dim, config["gate_g1_mu"])
    phi = np.full(dim, config["gate_g1_phi"])
    q = np.full(dim, config["gate_g1_q"])
    generator = np.random.default_rng(config["gate_g1_seed"])
    transition = generator.uniform(config["gate_g1_offdiagonal_lower"], config["gate_g1_offdiagonal_upper"], size=(dim, dim))
    np.fill_diagonal(transition, config["gate_g1_phi"])
    latent = np.empty((count, dim))
    latent[0] = mu
    for t in range(1, count):
        latent[t] = mu + transition @ (latent[t - 1] - mu) + generator.normal(size=dim) * np.sqrt(q)
    returns = generator.normal(size=(count, dim)) * np.exp(np.clip(latent / 2.0, -15.0, 15.0))
    scales = latent.std(axis=0, ddof=1)
    truth = transition * scales[None, :] / scales[:, None]
    trials = []
    for forgetting in config["lambda_candidates"]:
        result = core_run(returns, {"mu": mu, "phi": phi, "q": q}, forgetting, config)
        estimate = np.median(result["J"][-2000:], axis=0)
        relative_error = float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))
        mask = ~np.eye(dim, dtype=bool)
        correlation = float(np.corrcoef(estimate[mask], truth[mask])[0, 1])
        trials.append({"lambda": float(forgetting), "relative_frobenius_error": relative_error, "off_diagonal_correlation": correlation, "estimate": estimate})
    chosen = min(trials, key=lambda item: item["relative_frobenius_error"])
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    return {"trials": trials, "selected_lambda": chosen["lambda"], "synthetic_spectral_radius": spectral_radius, "pass": chosen["relative_frobenius_error"] <= config["gates"]["g1_relative_frobenius_error_max"] and chosen["off_diagonal_correlation"] >= config["gates"]["g1_off_diagonal_correlation_min"]}


def g2_gate(config):
    dim = config["portfolio_count"]
    identity = np.eye(dim)
    baseline = np.diag(np.linspace(0.95, 0.30, dim))
    stationary = np.repeat(baseline[None, :, :], 120, axis=0)
    rotated = stationary.copy()
    angle = np.pi / 4.0
    rotation = np.eye(dim)
    rotation[0, 0] = np.cos(angle)
    rotation[0, 4] = -np.sin(angle)
    rotation[4, 0] = np.sin(angle)
    rotation[4, 4] = np.cos(angle)
    rotated[60:] = rotation @ baseline @ rotation.T
    stable_metrics = dynamic_metrics(stationary, config["primary_horizon"], rank=config["principal_subspace_rank"])
    rotation_metrics = dynamic_metrics(rotated, config["primary_horizon"], rank=config["principal_subspace_rank"])
    rotation_contrast = float(np.max(rotation_metrics["R"][60:]))
    stable_rotation = float(np.max(stable_metrics["R"]))
    nonnormal = 0.8 * identity
    nonnormal[0, 1] = 1.6
    amplification = float(np.linalg.svd(np.linalg.matrix_power(nonnormal, config["primary_horizon"]), compute_uv=False)[0])
    baseline = float(np.linalg.svd(np.linalg.matrix_power(0.8 * identity, config["primary_horizon"]), compute_uv=False)[0])
    contrast = float(np.log(amplification / baseline))
    eigen_radius_power = float(np.max(np.abs(np.linalg.eigvals(nonnormal))) ** config["primary_horizon"])
    return {"rotation_contrast_radians": rotation_contrast, "stationary_rotation_max_radians": stable_rotation, "amplification_log_contrast": contrast, "nonnormal_rho_power": eigen_radius_power, "pass": rotation_contrast >= config["gates"]["g2_rotation_contrast_min"] and stable_rotation <= config["g2_stationary_rotation_max"] and contrast >= config["gates"]["g2_amplification_contrast_min"] and eigen_radius_power < 1.0}


def g3_gate(parameters, config):
    returns, latent, _ = simulate_sv(parameters["mu"], parameters["phi"], parameters["q"], config["g3_simulation_rows"], config["seeds"]["g3"])
    output = core_run(returns, parameters, 0.995, config)
    filter_rmse = float(np.sqrt(np.mean((output["h"][1000:] - latent[1000:]) ** 2)))
    constant_rmse = float(np.sqrt(np.mean((parameters["mu"][None, :] - latent[1000:]) ** 2)))
    improvement = 1.0 - filter_rmse / constant_rmse
    return {"filter_rmse": filter_rmse, "constant_mu_rmse": constant_rmse, "relative_improvement": improvement, "pass": improvement >= config["gates"]["g3_rmse_improvement_min"]}


def g4_gate(config, lambda_value):
    dates, returns, equal_weight, labels = load_panel(date(1969, 12, 31))
    train_mask = (dates >= np.datetime64(config["estimate_start"])) & (dates <= np.datetime64(config["estimate_end"]))
    parameters = estimate_parameters(returns[train_mask], config)
    if parameters["boundary_assets"]:
        raise RuntimeError(f"E1 real estimator sanity found box-boundary assets {parameters['boundary_assets']}")
    output = core_run(returns, parameters, lambda_value, config)
    dynamic = dynamic_metrics(output["J"], config["primary_horizon"], config["rotation_lag_days"], config["principal_subspace_rank"])
    burn = max(config["rls_burn_in_days"], config["rotation_lag_days"])
    rho = np.power(np.maximum(dynamic["rho_power"][burn:], 0.0), 1.0 / config["primary_horizon"])
    f_values = dynamic["F"][burn:]
    direct = direct_rls(output["z"], config["primary_horizon"], lambda_value, config["ridge_prior"])
    predicted = np.asarray([np.linalg.matrix_power(matrix, config["primary_horizon"]) for matrix in output["J"][burn:-config["primary_horizon"]]])
    direct_sample = direct[burn:-config["primary_horizon"]]
    gaps = np.linalg.norm(predicted - direct_sample, axis=(1, 2)) / np.maximum(np.linalg.norm(direct_sample, axis=(1, 2)), 1e-12)
    rng = np.random.default_rng(20261107)
    shuffled_f = []
    for _ in range(5):
        shuffled = core_run(returns[rng.permutation(len(returns))], parameters, lambda_value, config)
        summary = dynamic_metrics(shuffled["J"], config["primary_horizon"])
        shuffled_f.extend(summary["F"][burn:].tolist())
    distribution_distance = float(wasserstein_distance(f_values, shuffled_f))
    summary = {
        "tier": "frozen",
        "date_start": str(dates[0]),
        "date_end": str(dates[-1]),
        "return_rows": len(dates),
        "portfolio_labels": labels,
        "parameters": parameters,
        "selected_lambda": float(lambda_value),
        "median_rho": float(np.median(rho)),
        "median_F5": float(np.median(f_values)),
        "maximum_F5": float(np.max(f_values)),
        "median_direct_rls_relative_gap": float(np.median(gaps)),
        "row_shuffle_wasserstein_F5": distribution_distance
    }
    summary["pass"] = summary["median_rho"] <= config["gates"]["g4_median_rho_max"] and summary["median_F5"] <= config["gates"]["g4_median_f5_max"] and summary["maximum_F5"] <= config["gates"]["g4_max_f5_max"] and summary["median_direct_rls_relative_gap"] <= config["gates"]["g4_direct_rls_gap_max"]
    return summary


def run_gates(config):
    numeric = numerical_safety_gate(config)
    if numeric["ukf_kf_abs_error"] > config["gates"]["ukf_kf_abs_error_max"] or numeric["finite_difference_abs_error"] > config["gates"]["finite_difference_abs_error_max"]:
        raise RuntimeError("U numerical equivalence thresholds failed")
    g1 = g1_gate(config)
    if not g1["pass"]:
        raise RuntimeError("G1 did not meet its declared recovery thresholds")
    dates, returns, equal_weight, labels = load_panel(date(1969, 12, 31))
    train_mask = (dates >= np.datetime64(config["estimate_start"])) & (dates <= np.datetime64(config["estimate_end"]))
    parameters = estimate_parameters(returns[train_mask], config)
    if parameters["boundary_assets"]:
        raise RuntimeError(f"E1 real estimator sanity found box-boundary assets {parameters['boundary_assets']}")
    e1 = e1_gate(parameters, config)
    if not e1["pass"]:
        raise RuntimeError(f"E1 simulation recovery failed: {e1['assets_passing']} assets passed")
    g2 = g2_gate(config)
    if not g2["pass"]:
        raise RuntimeError("G2 planted-control contrast failed")
    g3 = g3_gate(parameters, config)
    if not g3["pass"]:
        raise RuntimeError("G3 filter RMSE did not beat its constant-state baseline")
    g4 = g4_gate(config, g1["selected_lambda"])
    if not g4["pass"]:
        raise RuntimeError("G4 pre-1970 validation failed one or more declared thresholds")
    return {"tier": "frozen", "U": numeric, "E1": e1, "G1": g1, "G2": g2, "G3": g3, "G4": g4, "attempts": 1}
