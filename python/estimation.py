import numpy as np
from scipy.optimize import minimize

from common import load_config


def causal_residuals(values):
    result = np.empty_like(values)
    mean = 0.0
    for index, value in enumerate(values):
        result[index] = value - mean
        mean += (value - mean) / (index + 1)
    return result


def scalar_likelihood(parameters, observations, config):
    mu, phi, q = parameters
    if q <= 0.0 or abs(phi) >= 1.0:
        return np.inf
    state = mu
    state_var = q / (1.0 - phi * phi)
    total = 0.0
    for index, value in enumerate(observations):
        if index:
            state = mu + phi * (state - mu)
            state_var = phi * phi * state_var + q
        if not np.isfinite(value):
            continue
        innovation_var = state_var + config["observation_variance"]
        innovation = value - state - config["log_chi_square_mean"]
        total += 0.5 * (np.log(2.0 * np.pi * innovation_var) + innovation * innovation / innovation_var)
        gain = state_var / innovation_var
        state += gain * innovation
        state_var = (1.0 - gain) ** 2 * state_var + gain * gain * config["observation_variance"]
    return float(total)


def estimate_parameters(log_returns, config=None, zero_as_missing=False):
    settings = config or load_config()
    rows, dim = log_returns.shape
    residual = np.empty_like(log_returns)
    means = np.zeros(dim)
    counts = np.zeros(dim)
    for t in range(rows):
        residual[t] = log_returns[t] - means
        observed = np.ones(dim, dtype=bool) if not zero_as_missing else (log_returns[t] != 0.0)
        means[observed] += (log_returns[t, observed] - means[observed]) / (counts[observed] + 1.0)
        counts[observed] += 1.0
    observations = np.log(residual * residual + settings["variance_floor"])
    if zero_as_missing:
        observations[log_returns == 0.0] = np.nan
    estimates = []
    for j in range(dim):
        # In the post-freeze zero-as-missing variant, missing measurements are
        # represented by NaN.  Initialize from observed dates only.
        initial_mu = float(np.nanmedian(observations[:, j]) - settings["log_chi_square_mean"])
        result = minimize(
            scalar_likelihood,
            np.asarray([initial_mu, 0.97, 0.02]),
            args=(observations[:, j], settings),
            method="L-BFGS-B",
            bounds=[tuple(settings["mu_bounds"]), tuple(settings["phi_bounds"]), tuple(settings["q_bounds"])],
            options={"ftol": 1e-10, "maxiter": 2000}
        )
        if not result.success and not np.isfinite(result.fun):
            raise RuntimeError(f"quasi-likelihood optimization failed on asset {j}: {result.message}")
        estimates.append(result.x)
    estimates = np.asarray(estimates)
    lower = np.asarray([settings["mu_bounds"][0], settings["phi_bounds"][0], settings["q_bounds"][0]])
    upper = np.asarray([settings["mu_bounds"][1], settings["phi_bounds"][1], settings["q_bounds"][1]])
    boundary = np.any(np.isclose(estimates, lower, atol=1e-7) | np.isclose(estimates, upper, atol=1e-7), axis=1)
    return {"mu": estimates[:, 0], "phi": estimates[:, 1], "q": estimates[:, 2], "objective": float(sum(scalar_likelihood(row, observations[:, i], settings) for i, row in enumerate(estimates))), "boundary_assets": np.flatnonzero(boundary).tolist(), "observations": rows}


def simulate_sv(mu, phi, q, count, seed, coupling=0.0):
    generator = np.random.default_rng(seed)
    dim = len(mu)
    transition = np.diag(phi)
    if coupling:
        transition += np.full((dim, dim), coupling / dim)
        transition[np.diag_indices(dim)] -= coupling / dim
    latent = np.empty((count, dim))
    latent[0] = mu + generator.normal(size=dim) * np.sqrt(q / np.maximum(1e-5, 1.0 - phi * phi))
    for t in range(1, count):
        latent[t] = mu + transition @ (latent[t - 1] - mu) + generator.normal(size=dim) * np.sqrt(q)
    returns = generator.normal(size=(count, dim)) * np.exp(np.clip(latent / 2.0, -15.0, 15.0))
    return returns, latent, transition
