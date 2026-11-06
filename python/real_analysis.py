import json
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from common import ROOT, hash_verified, load_config, sha256_file, write_json
from dataset import create_derived_file, load_derived
from gates import core_run, direct_rls
from metrics import correlation_screen, direction_analysis, dynamic_metrics, empirical_two_sided, holm_adjust, holm_intervals, pca_components_90, rolling_covariances, stress_metrics


def verify_protocol():
    protocol_path = ROOT / "results" / "protocol.json"
    hash_path = ROOT / "results" / "protocol.sha256"
    if not protocol_path.exists() or not hash_path.exists():
        raise RuntimeError("protocol and its SHA-256 must exist before test-period analysis")
    expected = hash_path.read_text(encoding="ascii").strip()
    hash_verified(protocol_path, expected)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    hash_record = json.loads((ROOT / "results" / "protocol_hash.json").read_text(encoding="utf-8"))
    if hash_record["protocol_sha256"] != expected:
        raise RuntimeError("protocol hash record does not match the frozen protocol")
    protocol["protocol_sha256"] = expected
    if protocol["config_sha256"] != sha256_file(ROOT / "config.json"):
        raise RuntimeError("config changed after protocol freeze")
    if protocol["analysis_plan_sha256"] != sha256_file(ROOT / "results" / "analysis_plan.json"):
        raise RuntimeError("analysis plan changed after protocol freeze")
    return protocol


def make_events(drawdown, threshold, merge_days, forecast_days):
    onsets = [index for index in range(1, len(drawdown)) if drawdown[index] <= threshold and drawdown[index - 1] > threshold]
    clusters = []
    for onset in onsets:
        if clusters and onset - clusters[-1]["last_onset"] < merge_days:
            clusters[-1]["last_onset"] = onset
            clusters[-1]["onsets"].append(onset)
        else:
            clusters.append({"first_onset": onset, "last_onset": onset, "onsets": [onset]})
    labels = np.zeros(len(drawdown))
    at_risk = np.ones(len(drawdown), dtype=bool)
    for cluster in clusters:
        start = cluster["first_onset"]
        end = min(len(drawdown), cluster["last_onset"] + merge_days + 1)
        at_risk[start:end] = False
        for onset in cluster["onsets"]:
            low = max(0, onset - forecast_days)
            labels[low:onset] = 1.0
    labels[max(0, len(drawdown) - forecast_days + 1):] = np.nan
    return clusters, labels, at_risk


def fit_expanding(features, labels, at_risk, dates, clusters, config):
    n = len(dates)
    predictions = {"M0": np.full(n, np.nan), "M0_F": np.full(n, np.nan), "M1": np.full(n, np.nan), "M1_F": np.full(n, np.nan)}
    names = {
        "M0": ["vol", "drawdown", "rho_tilde"],
        "M0_F": ["vol", "drawdown", "rho_tilde", "F"],
        "M1": ["vol", "drawdown", "rho_tilde", "ar1", "absorption"],
        "M1_F": ["vol", "drawdown", "rho_tilde", "ar1", "absorption", "F"]
    }
    test_start = int(np.searchsorted(dates, np.datetime64(config["test_start"])))
    calibration_start = np.datetime64(config["calibration_start"])
    for refit_start in range(test_start, n, config["refit_days"]):
        cutoff = refit_start - config["purge_days"]
        matured = sum(1 for item in clusters if item["last_onset"] + config["event_forecast_days"] < cutoff)
        if matured < config["minimum_mature_clusters"]:
            continue
        train = np.arange(n) < cutoff
        train &= dates >= calibration_start
        train &= at_risk
        train &= np.isfinite(labels)
        for name, columns in names.items():
            matrix = np.column_stack([features[key] for key in columns])
            train &= np.all(np.isfinite(matrix), axis=1)
        current_end = min(n, refit_start + config["refit_days"])
        for name, columns in names.items():
            matrix = np.column_stack([features[key] for key in columns])
            local_train = train & np.all(np.isfinite(matrix), axis=1)
            train_index = np.flatnonzero(local_train)
            test_index = np.arange(refit_start, current_end)
            test_index = test_index[at_risk[test_index] & np.all(np.isfinite(matrix[test_index]), axis=1)]
            if len(np.unique(labels[train_index])) < 2 or test_index.size == 0:
                continue
            mean = matrix[train_index].mean(axis=0)
            scale = matrix[train_index].std(axis=0, ddof=1)
            scale = np.maximum(scale, 1e-8)
            model = LogisticRegression(C=config["logistic_C"], penalty="l2", solver="lbfgs", max_iter=1000)
            model.fit((matrix[train_index] - mean) / scale, labels[train_index].astype(int))
            predictions[name][test_index] = model.predict_proba((matrix[test_index] - mean) / scale)[:, 1]
    return predictions


def circular_block_indices(length, block_length, generator):
    pieces = []
    total = 0
    while total < length:
        start = int(generator.integers(0, length))
        piece = (start + np.arange(block_length)) % length
        pieces.append(piece)
        total += block_length
    return np.concatenate(pieces)[:length]


def bootstrap_deltas(labels, predictions, mask, config):
    truth = labels[mask].astype(int)
    pairs = [predictions["M0"][mask], predictions["M0_F"][mask], predictions["M1"][mask], predictions["M1_F"][mask]]
    point = np.asarray([
        roc_auc_score(truth, pairs[1]) - roc_auc_score(truth, pairs[0]),
        roc_auc_score(truth, pairs[3]) - roc_auc_score(truth, pairs[2])
    ])
    generator = np.random.default_rng(config["seeds"]["bootstrap"])
    samples = np.empty((2, config["bootstrap_draws"]))
    redraws = 0
    for draw in range(config["bootstrap_draws"]):
        while True:
            indices = circular_block_indices(len(truth), config["bootstrap_block_days"], generator)
            sampled = truth[indices]
            if np.any(sampled == 0) and np.any(sampled == 1):
                break
            redraws += 1
        auc0 = roc_auc_score(sampled, pairs[0][indices])
        auc0f = roc_auc_score(sampled, pairs[1][indices])
        auc1 = roc_auc_score(sampled, pairs[2][indices])
        auc1f = roc_auc_score(sampled, pairs[3][indices])
        samples[:, draw] = [auc0f - auc0, auc1f - auc1]
    intervals = holm_intervals(samples, alpha=1.0 - config["h2_confidence_level"])
    raw_p = []
    for row in samples:
        left = np.mean(row <= 0.0)
        right = np.mean(row >= 0.0)
        raw_p.append(min(1.0, 2.0 * min(left, right)))
    return {
        "point_estimates": point,
        "bootstrap": samples,
        "holm_adjusted_p_at_zero": holm_adjust(raw_p),
        "holm_adjusted_95_intervals": intervals,
        "redraw_count": redraws,
        "auc": {
            "M0": roc_auc_score(truth, pairs[0]),
            "M0_F": roc_auc_score(truth, pairs[1]),
            "M1": roc_auc_score(truth, pairs[2]),
            "M1_F": roc_auc_score(truth, pairs[3])
        },
        "log_loss": {
            name: float(log_loss(truth, values, labels=[0, 1])) for name, values in zip(("M0", "M0_F", "M1", "M1_F"), pairs)
        },
        "anchor_count": int(np.sum(mask)),
        "event_day_count": int(np.sum(truth == 1))
    }


def block_shuffle(test_returns, block_length, generator):
    count = len(test_returns)
    offset = int(generator.integers(0, block_length))
    rotated = np.roll(test_returns, -offset, axis=0)
    blocks = [rotated[start:start + block_length] for start in range(0, count, block_length)]
    order = generator.permutation(len(blocks))
    shuffled = np.concatenate([blocks[index] for index in order], axis=0)[:count]
    return np.roll(shuffled, offset, axis=0)


def event_placebos(series, clusters, count, spacing, seed, window, start_index):
    generator = np.random.default_rng(seed)
    valid = np.arange(max(window, start_index), len(series))
    placebos = np.empty(count)
    for draw in range(count):
        order = generator.permutation(valid)
        selected = []
        for candidate in order:
            if all(abs(int(candidate) - chosen) >= spacing for chosen in selected):
                selected.append(int(candidate))
                if len(selected) == len(clusters):
                    break
        if len(selected) != len(clusters):
            raise RuntimeError("date-only placebo construction could not meet the frozen spacing rule")
        placebos[draw] = np.mean([np.mean(series[max(0, onset - window):onset]) for onset in selected])
    observed = float(np.mean([np.mean(series[max(0, cluster["first_onset"] - window):cluster["first_onset"]]) for cluster in clusters]))
    center = float(np.median(placebos))
    p_value = float((np.count_nonzero(np.abs(placebos - center) >= abs(observed - center)) + 1) / (count + 1))
    aligned = np.asarray([[series[cluster["first_onset"] + offset] if 0 <= cluster["first_onset"] + offset < len(series) else np.nan for offset in range(-window, window + 1)] for cluster in clusters])
    return {"observed_pre_onset_mean": observed, "placebo_median": center, "p_value": p_value, "placebo_values": placebos, "aligned_median": np.nanmedian(aligned, axis=0), "aligned_offsets": np.arange(-window, window + 1)}


def wedin_diagnostic(jacobians, dates, test_start, lag=20, rank=3):
    singular = np.linalg.svd(jacobians, compute_uv=False)
    errors = []
    gaps = []
    bounds = []
    test_indices = np.flatnonzero(np.asarray(dates) >= np.datetime64(test_start))
    for index in test_indices:
        if index < lag:
            continue
        error = float(np.linalg.norm(jacobians[index] - jacobians[index - lag], ord=2))
        gap = float(min(singular[index, rank - 1], singular[index - lag, rank - 1]) - max(singular[index, rank], singular[index - lag, rank]))
        errors.append(error)
        gaps.append(gap)
        bounds.append(error / gap if gap > error else np.inf)
    finite = np.asarray(bounds)[np.isfinite(bounds)]
    return {"median_gap": float(np.median(gaps)), "median_perturbation_norm": float(np.median(errors)), "informative_share": float(np.mean(np.asarray(bounds) < 1.0)), "finite_bound_median": float(np.median(finite)) if finite.size else None}


def run_real_analysis():
    protocol = verify_protocol()
    config = load_config()
    started = time.perf_counter()
    derived = create_derived_file()
    dates, returns, equal_weight = load_derived()
    validation = json.loads((ROOT / "results" / "validation.json").read_text(encoding="utf-8"))
    parameters = validation["G4"]["parameters"]
    parameters = {key: np.asarray(parameters[key], dtype=np.float64) for key in ("mu", "phi", "q")}
    forgetting = float(protocol["selected_lambda"])
    split = int(np.searchsorted(dates, np.datetime64(config["test_start"])))
    output = core_run(returns, parameters, forgetting, config)
    dynamic5 = dynamic_metrics(output["J"], config["primary_horizon"], config["rotation_lag_days"], config["principal_subspace_rank"])
    dynamic20 = dynamic_metrics(output["J"], config["secondary_horizon"], config["rotation_lag_days"], config["principal_subspace_rank"])
    stress = stress_metrics(dates, returns, equal_weight, config)
    features = {"R": dynamic5["R"], "H": dynamic5["H"], "F": dynamic5["F"], **stress}
    primary_clusters, labels, at_risk = make_events(stress["drawdown"], config["event_drawdown"], config["event_merge_days"], config["event_forecast_days"])
    test_clusters = [item for item in primary_clusters if item["first_onset"] >= split]
    features["F"] = dynamic5["F"]
    predictions = fit_expanding(features, labels, at_risk, dates, primary_clusters, config)
    common_mask = (dates >= np.datetime64(config["test_start"])) & at_risk & np.isfinite(labels)
    for values in predictions.values():
        common_mask &= np.isfinite(values)
    h2_bootstrap = bootstrap_deltas(labels, predictions, common_mask, config)
    h2_bootstrap["event_clusters"] = len(test_clusters)
    delta_primary = float(h2_bootstrap["point_estimates"][1])
    low_primary, high_primary = [float(value) for value in h2_bootstrap["holm_adjusted_95_intervals"][1]]
    if low_primary > 0.0 and delta_primary >= config["h2_support_point_threshold"]:
        h2_verdict = "supported"
    elif high_primary < config["h2_support_point_threshold"]:
        h2_verdict = "unsupported"
    else:
        h2_verdict = "inconclusive"
    test_mask = dates >= np.datetime64(config["test_start"])
    m_vector = np.column_stack([dynamic5["R"], dynamic5["H"], stress["vol"], stress["drawdown"], stress["rho_tilde"], dynamic5["F"]])
    pca_components, pca_cumulative = pca_components_90(m_vector[test_mask], config["pca_variance_share"])
    spearman = correlation_screen(dynamic5, stress, test_mask)
    max_spearman = float(max(abs(value) for value in spearman.values()))
    observed_h1 = {key: float(np.median(dynamic5[key][test_mask])) for key in ("R", "H", "F")}
    surrogate_values = np.empty((config["surrogate_draws"], 3))
    generator = np.random.default_rng(config["seeds"]["surrogate"])
    for draw in range(config["surrogate_draws"]):
        shuffled = returns.copy()
        shuffled[split:] = block_shuffle(returns[split:], config["surrogate_block_days"], generator)
        surrogate_core = core_run(shuffled, parameters, forgetting, config)
        surrogate_start = split - config["rotation_lag_days"]
        surrogate_dynamic = dynamic_metrics(surrogate_core["J"][surrogate_start:], config["primary_horizon"], config["rotation_lag_days"], config["principal_subspace_rank"])
        surrogate_values[draw] = [float(np.median(surrogate_dynamic[key][config["rotation_lag_days"]:])) for key in ("R", "H", "F")]
    raw_surrogate_p = [empirical_two_sided(observed_h1[key], surrogate_values[:, index]) for index, key in enumerate(("R", "H", "F"))]
    adjusted_surrogate_p = holm_adjust(raw_surrogate_p)
    row_shuffle_values = []
    for _ in range(config["row_shuffle_draws"]):
        row_order = generator.permutation(len(returns))
        shuffled_core = core_run(returns[row_order], parameters, forgetting, config)
        surrogate_start = split - config["rotation_lag_days"]
        shuffled_dynamic = dynamic_metrics(shuffled_core["J"][surrogate_start:], config["primary_horizon"], config["rotation_lag_days"], config["principal_subspace_rank"])
        row_shuffle_values.append([float(np.median(shuffled_dynamic[key][config["rotation_lag_days"]:])) for key in ("R", "H", "F")])
    placebo_results = {}
    placebo_p = []
    for index, key in enumerate(("R", "H", "F")):
        placebo_results[key] = event_placebos(dynamic5[key], test_clusters, config["placebo_draws"], config["event_merge_days"], config["seeds"]["placebo"] + index, config["event_forecast_days"], split)
        placebo_p.append(placebo_results[key]["p_value"])
    placebo_adjusted = holm_adjust(placebo_p)
    condition_a = bool(np.all(adjusted_surrogate_p <= config["minimum_surrogate_p"]))
    condition_b = bool(np.any(placebo_adjusted < config["event_placebo_p_threshold"]))
    condition_c = max_spearman < config["h1_max_abs_spearman"]
    condition_d = pca_components >= config["principal_subspace_rank"]
    if condition_a and condition_b and condition_c and condition_d:
        h1_verdict = "supported"
    elif np.all(adjusted_surrogate_p >= config["event_placebo_p_threshold"]):
        h1_verdict = "unsupported"
    else:
        h1_verdict = "inconclusive"
    direction = direction_analysis(stress["cov250"], dynamic5["right_vectors"], dates, config["test_start"], config["seeds"]["haar"], config["direction_surrogate_draws"], config["principal_subspace_rank"])
    wedin = wedin_diagnostic(output["J"], dates, config["test_start"], config["rotation_lag_days"], config["principal_subspace_rank"])
    robustness = {}
    alternatives = [value for value in config["lambda_candidates"] if not np.isclose(value, forgetting, rtol=0.0, atol=1e-12)]
    alt_lambda = alternatives[0] if alternatives else forgetting
    alternative_core = core_run(returns, parameters, alt_lambda, config)
    alternative_dynamic = dynamic_metrics(alternative_core["J"], config["primary_horizon"], config["rotation_lag_days"], config["principal_subspace_rank"])
    robustness["alternative_lambda"] = {"lambda": alt_lambda, "median_F5": float(np.median(alternative_dynamic["F"][test_mask])), "median_R": float(np.median(alternative_dynamic["R"][test_mask])), "median_H": float(np.median(alternative_dynamic["H"][test_mask]))}
    ema_core = core_run(returns, parameters, forgetting, config, smooth=True)
    ema_dynamic = dynamic_metrics(ema_core["J"], config["primary_horizon"], config["rotation_lag_days"], config["principal_subspace_rank"])
    robustness["ema_before_rls"] = {"half_life_days": config["ema_half_life_days"], "median_F5": float(np.median(ema_dynamic["F"][test_mask])), "median_R": float(np.median(ema_dynamic["R"][test_mask])), "median_H": float(np.median(ema_dynamic["H"][test_mask]))}
    robustness["horizon_20"] = {"median_F20": float(np.median(dynamic20["F"][test_mask])), "median_rho_power_20": float(np.median(dynamic20["rho_power"][test_mask]))}
    alternative_thresholds = {}
    for threshold in config["event_threshold_robustness"]:
        clusters, target, risk = make_events(stress["drawdown"], threshold, config["event_merge_days"], config["event_forecast_days"])
        test_cluster_count = sum(1 for item in clusters if item["first_onset"] >= split)
        predictions_alt = fit_expanding(features, target, risk, dates, clusters, config)
        mask = test_mask & risk & np.isfinite(target)
        for values in predictions_alt.values():
            mask &= np.isfinite(values)
        if np.sum(mask) and len(np.unique(target[mask])) == 2:
            alt_delta = roc_auc_score(target[mask], predictions_alt["M1_F"][mask]) - roc_auc_score(target[mask], predictions_alt["M1"][mask])
            alternative_thresholds[str(threshold)] = {"delta_auc_M1_F": float(alt_delta), "anchor_count": int(np.sum(mask)), "event_clusters": test_cluster_count}
        else:
            alternative_thresholds[str(threshold)] = {"delta_auc_M1_F": None, "anchor_count": int(np.sum(mask)), "event_clusters": test_cluster_count}
    moderate_start = np.datetime64(config["calibration_start"])
    moderate_end = np.datetime64(config["calibration_end"])
    calibration = (dates >= moderate_start) & (dates <= moderate_end)
    vol_threshold = float(np.nanquantile(stress["vol"][calibration], 0.75))
    corr_threshold = float(np.nanquantile(stress["corr_mean"][calibration], 0.75))
    moderate = test_mask & (stress["vol"] < vol_threshold) & (stress["corr_mean"] < corr_threshold)
    moderate_cluster_count = sum(1 for item in test_clusters if moderate[item["first_onset"]])
    if moderate_cluster_count >= 3:
        moderate_mask = common_mask & moderate
        if len(np.unique(labels[moderate_mask])) == 2:
            moderate_result = {"status": "estimable", "event_clusters": moderate_cluster_count, "anchor_count": int(np.sum(moderate_mask)), "delta_auc_M1_F": float(roc_auc_score(labels[moderate_mask], predictions["M1_F"][moderate_mask]) - roc_auc_score(labels[moderate_mask], predictions["M1"][moderate_mask]))}
        else:
            moderate_result = {"status": "not estimable", "event_clusters": moderate_cluster_count, "anchor_count": int(np.sum(moderate_mask)), "reason": "only one target class is present"}
    else:
        moderate_result = {"status": "not estimable", "event_clusters": moderate_cluster_count, "anchor_count": int(np.sum(common_mask & moderate)), "vol_threshold": vol_threshold, "correlation_threshold": corr_threshold}
    study = {
        "tier": "frozen",
        "data_period": {"start": str(dates[0]), "test_start": config["test_start"], "end": str(dates[-1])},
        "selected_lambda": forgetting,
        "event_clusters": len(test_clusters),
        "event_onsets": [str(dates[item["first_onset"]]) for item in test_clusters],
        "anchor_count": int(np.sum(common_mask)),
        "H1": {
            "verdict": h1_verdict,
            "surrogate_observed_medians": observed_h1,
            "surrogate_p_raw": raw_surrogate_p,
            "surrogate_p_holm": adjusted_surrogate_p,
            "surrogate_minimum_p": 1.0 / (config["surrogate_draws"] + 1),
            "surrogate_medians": surrogate_values,
            "row_shuffle_medians": row_shuffle_values,
            "event_placebo_p_raw": placebo_p,
            "event_placebo_p_holm": placebo_adjusted,
            "event_placebos": placebo_results,
            "condition_a": condition_a,
            "condition_b": condition_b,
            "condition_c": condition_c,
            "condition_d": condition_d,
            "spearman": spearman,
            "max_abs_spearman": max_spearman,
            "pca_components_for_90_percent": pca_components,
            "pca_cumulative": pca_cumulative,
            "direction": {key: value for key, value in direction.items() if key not in ("angles", "haar_angles")},
            "direction_angles": direction["angles"],
            "haar_angles": direction["haar_angles"],
            "wedin": wedin
        },
        "H2": {"verdict": h2_verdict, "bootstrap": h2_bootstrap, "moderate_regime": moderate_result, "moderate_thresholds": {"vol_75": vol_threshold, "correlation_75": corr_threshold}},
        "robustness": robustness,
        "alternative_event_thresholds": alternative_thresholds,
        "time_series": {
            "dates": [str(value) for value in dates[test_mask]],
            "R": dynamic5["R"][test_mask],
            "H": dynamic5["H"][test_mask],
            "vol": stress["vol"][test_mask],
            "drawdown": stress["drawdown"][test_mask],
            "rho_tilde": stress["rho_tilde"][test_mask],
            "F5": dynamic5["F"][test_mask],
            "F20": dynamic20["F"][test_mask]
        },
        "event_aligned_offsets": placebo_results["R"]["aligned_offsets"]
    }
    write_json(ROOT / "results" / "study.json", study)
    write_json(ROOT / "results" / "h2_predictions.json", {"tier": "frozen", "dates": [str(value) for value in dates[common_mask]], "labels": labels[common_mask], "predictions": {key: value[common_mask] for key, value in predictions.items()}, "event_clusters": len(test_clusters), "anchor_count": int(np.sum(common_mask))})
    write_json(ROOT / "results" / "robustness.json", {"tier": "frozen", "variants": robustness, "event_thresholds": alternative_thresholds, "moderate_regime": moderate_result})
    write_json(ROOT / "results" / "surrogates.json", {"tier": "frozen", "block_medians": surrogate_values, "row_shuffle_medians": row_shuffle_values, "adjusted_p": adjusted_surrogate_p, "minimum_p": 1.0 / (config["surrogate_draws"] + 1), "anchor_count": int(np.sum(test_mask)), "event_clusters": len(test_clusters)})
    study["runtime_seconds"] = float(time.perf_counter() - started)
    write_json(ROOT / "results" / "study.json", study)
    return study
