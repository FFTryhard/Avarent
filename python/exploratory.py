"""Post-freeze exploratory analyses; frozen primary result files are read-only."""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import ROOT, hash_verified, load_config, sha256_file, write_json
from dataset import iter_daily_rows, load_derived, load_panel
from estimation import estimate_parameters
from gates import core_run
from metrics import dynamic_metrics, empirical_two_sided, holm_adjust, principal_angle, stress_metrics
from real_analysis import block_shuffle, circular_block_indices, event_placebos, make_events


RESULTS = ROOT / "results" / "exploratory"
PLAN = RESULTS / "analysis_plan.json"
PLAN_SHA = RESULTS / "analysis_plan.sha256"
FROZEN_STUDY_SHA = "732f6b8b15661ec9b176489c989b5ccba4c09d453d24597aee78c430cdc0d0ae"
FROZEN_PROTOCOL_SHA = "a860e120e56490bcfe14675241d33f5958929c0e78747a8da334c79c7fa311c0"
FROZEN_ANALYSIS_PLAN_SHA = "7ea3f2c36f7150e3a13478dbfe6b7205e31d0934132d04eb6087935fe5688388"
SURROGATE_DRAWS = 499
BOOTSTRAP_DRAWS = 399
BLOCK_LENGTH = 250


def verify_frozen_inputs():
    plan_digest = PLAN_SHA.read_text(encoding="ascii").strip()
    hash_verified(PLAN, plan_digest)
    checks = {
        ROOT / "results" / "study.json": FROZEN_STUDY_SHA,
        ROOT / "results" / "protocol.sha256": FROZEN_PROTOCOL_SHA,
        ROOT / "results" / "analysis_plan.sha256": FROZEN_ANALYSIS_PLAN_SHA,
    }
    for path, expected in checks.items():
        if path.suffix == ".sha256":
            actual = path.read_text(encoding="ascii").strip()
        else:
            actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"frozen input changed: {path.relative_to(ROOT)}")
    return plan_digest


def load_primary():
    dates, returns, equal_weight = load_derived()
    study = json.loads((ROOT / "results" / "study.json").read_text(encoding="utf-8"))
    validation = json.loads((ROOT / "results" / "validation.json").read_text(encoding="utf-8"))
    config = load_config()
    protocol = json.loads((ROOT / "results" / "protocol.json").read_text(encoding="utf-8"))
    params = {key: np.asarray(validation["G4"]["parameters"][key], dtype=np.float64) for key in ("mu", "phi", "q")}
    return dates, returns, equal_weight, study, validation, config, protocol, params


def zero_counts(member_path, dimension, periods):
    accum = {name: {"days": 0, "zero_cells": 0, "valid_cells": 0, "days_with_any_zero": 0} for name, _, _ in periods}
    for date, _, log_values, _ in iter_daily_rows(member_path, portfolio_count=dimension):
        for name, start, end in periods:
            if start <= date <= end:
                item = accum[name]
                zeros = int(np.count_nonzero(log_values == 0.0))
                item["days"] += 1
                item["zero_cells"] += zeros
                item["valid_cells"] += len(log_values)
                item["days_with_any_zero"] += int(zeros > 0)
                break
    for item in accum.values():
        item["zero_share"] = item["zero_cells"] / item["valid_cells"] if item["valid_cells"] else None
    return accum


def write_zero_counts():
    periods = [
        ("estimation_1926_1949", "1926-01-01", "1949-12-31"),
        ("calibration_1950_1969", "1950-01-01", "1969-12-31"),
        ("test_1970_2026_09", "1970-01-01", "2026-09-30"),
    ]
    primary = zero_counts(ROOT / "data" / "10_Industry_Portfolios_Daily.csv", 10, periods)
    secondary_path = ROOT / "data" / "exploratory" / "17_industry" / "17_Industry_Portfolios_Daily.csv"
    secondary = zero_counts(secondary_path, 17, periods)
    result = {"tier": "post_freeze_exploratory", "count_definition": "Exact zero log returns after the source simple-return conversion; cells are portfolio-date observations.", "periods": {name: {"start": start, "end": end} for name, start, end in periods}, "10_industry": primary, "17_industry": secondary}
    write_json(RESULTS / "zero_counts.json", result)
    return result


def h1_evidence(dates, returns, dynamic, clusters, parameters, config, forgetting, zero_as_missing=False, draws=SURROGATE_DRAWS, seed_offset=0, period_ranges=None):
    split = int(np.searchsorted(dates, np.datetime64(config["test_start"])))
    keys = ("R", "H", "F")
    if period_ranges is None:
        observed = [float(np.median(dynamic[key][split:])) for key in keys]
        generator = np.random.default_rng(config["seeds"]["surrogate"] + seed_offset)
        null = np.empty((draws, len(keys)))
        for draw in range(draws):
            shuffled = returns.copy()
            shuffled[split:] = block_shuffle(returns[split:], config["surrogate_block_days"], generator)
            core = core_run(shuffled, parameters, forgetting, config, zero_as_missing=zero_as_missing)
            start = split - config["rotation_lag_days"]
            local = dynamic_metrics(core["J"][start:], config["primary_horizon"], config["rotation_lag_days"], config["principal_subspace_rank"])
            null[draw] = [float(np.median(local[key][config["rotation_lag_days"]:])) for key in keys]
            if (draw + 1) % 50 == 0:
                print(f"H1 surrogate {draw + 1}/{draws}", flush=True)
        raw_p = [empirical_two_sided(observed[index], null[:, index]) for index in range(len(keys))]
        adjusted = holm_adjust(raw_p).tolist()
        test_clusters = [event for event in clusters if event["first_onset"] >= split]
        placebo = {}
        for index, key in enumerate(keys):
            placebo[key] = event_placebos(dynamic[key], test_clusters, config["placebo_draws"], config["event_merge_days"], config["seeds"]["placebo"] + index + seed_offset, config["event_forecast_days"], split)
        placebo_raw = [placebo[key]["p_value"] for key in keys]
        placebo_adjusted = holm_adjust(placebo_raw).tolist()
        return {
            "surrogate_observed_medians": dict(zip(keys, observed)),
            "surrogate_p_raw": raw_p,
            "surrogate_p_holm": adjusted,
            "surrogate_null_medians": {key: float(np.median(null[:, index])) for index, key in enumerate(keys)},
            "event_placebo_p_raw": placebo_raw,
            "event_placebo_p_holm": placebo_adjusted,
            "event_placebo_observed_means": {key: placebo[key]["observed_pre_onset_mean"] for key in keys},
            "event_placebo_null_medians": {key: placebo[key]["placebo_median"] for key in keys},
            "test_event_clusters": len(test_clusters),
            "surrogate_draws": draws,
        }
    results = {}
    generator = np.random.default_rng(config["seeds"]["surrogate"] + seed_offset)
    for period_index, (name, start_date, end_date) in enumerate(period_ranges):
        start = int(np.searchsorted(dates, np.datetime64(start_date)))
        end = int(np.searchsorted(dates, np.datetime64(end_date), side="right"))
        observed = float(np.median(dynamic["F"][start:end]))
        null = np.empty(draws)
        for draw in range(draws):
            shuffled = returns.copy()
            shuffled[start:end] = block_shuffle(returns[start:end], config["surrogate_block_days"], generator)
            core = core_run(shuffled, parameters, forgetting, config, zero_as_missing=zero_as_missing)
            local = dynamic_metrics(core["J"][start:end], config["primary_horizon"], config["rotation_lag_days"], config["principal_subspace_rank"])
            null[draw] = float(np.median(local["F"]))
            if (draw + 1) % 50 == 0:
                print(f"{name} F surrogate {draw + 1}/{draws}", flush=True)
        results[name] = {"start": start_date, "end": end_date, "observed_median_F": observed, "null_median": float(np.median(null)), "raw_p": empirical_two_sided(observed, null), "draws": draws}
    raw = [results[name]["raw_p"] for name, _, _ in period_ranges]
    adjusted = holm_adjust(raw).tolist()
    for item, p in zip(results.values(), adjusted):
        item["holm_adjusted_p_across_subperiods"] = p
    return results


def ewma_volatility(equal_weight, half_life=20):
    values = np.asarray(equal_weight, dtype=np.float64)
    beta = 0.5 ** (1.0 / half_life)
    variance = np.full(len(values), np.nan)
    if len(values) > 1:
        variance[0] = values[0] * values[0]
        for index in range(1, len(values)):
            variance[index] = beta * variance[index - 1] + (1.0 - beta) * values[index - 1] ** 2
    return np.sqrt(np.maximum(variance, 1e-20)) * np.sqrt(252.0)


MODEL_COLUMNS = {
    "M1": ("vol", "drawdown", "rho_tilde", "ar1", "absorption"),
    "M1_F": ("vol", "drawdown", "rho_tilde", "ar1", "absorption", "F"),
    "EWMA": ("ewma_vol",),
}


def fit_h2_models(features, labels, at_risk, dates, clusters, config):
    count = len(dates)
    predictions = {name: np.full(count, np.nan) for name in MODEL_COLUMNS}
    test_start = int(np.searchsorted(dates, np.datetime64(config["test_start"])))
    calibration_start = np.datetime64(config["calibration_start"])
    for refit_start in range(test_start, count, config["refit_days"]):
        cutoff = refit_start - config["purge_days"]
        matured = sum(1 for item in clusters if item["last_onset"] + config["event_forecast_days"] < cutoff)
        if matured < config["minimum_mature_clusters"]:
            continue
        base_train = (np.arange(count) < cutoff) & (dates >= calibration_start) & at_risk & np.isfinite(labels)
        current_end = min(count, refit_start + config["refit_days"])
        test_index = np.arange(refit_start, current_end)
        test_index = test_index[at_risk[test_index] & np.isfinite(labels[test_index])]
        for name, columns in MODEL_COLUMNS.items():
            matrix = np.column_stack([features[key] for key in columns])
            train_index = np.flatnonzero(base_train & np.all(np.isfinite(matrix), axis=1))
            valid_test = test_index[np.all(np.isfinite(matrix[test_index]), axis=1)]
            if len(np.unique(labels[train_index])) < 2 or valid_test.size == 0:
                continue
            mean = matrix[train_index].mean(axis=0)
            scale = np.maximum(matrix[train_index].std(axis=0, ddof=1), 1e-8)
            model = LogisticRegression(C=float(config["logistic_C"]), penalty="l2", solver="lbfgs", max_iter=1000)
            model.fit((matrix[train_index] - mean) / scale, labels[train_index].astype(int))
            predictions[name][valid_test] = model.predict_proba((matrix[valid_test] - mean) / scale)[:, 1]
    return predictions


def _auc_values(labels, predictions, mask):
    truth = labels[mask].astype(int)
    return {name: float(roc_auc_score(truth, scores[mask])) for name, scores in predictions.items() if np.all(np.isfinite(scores[mask]))}


def _alert_metrics(truth, scores, rate=0.05):
    n_alert = max(1, int(math.ceil(rate * len(truth))))
    selected = np.argsort(-scores, kind="stable")[:n_alert]
    alert = np.zeros(len(truth), dtype=bool)
    alert[selected] = True
    true_positive = int(np.sum(alert & (truth == 1)))
    return {"alert_rate": float(np.mean(alert)), "precision": float(true_positive / n_alert), "recall": float(true_positive / np.sum(truth == 1)) if np.sum(truth == 1) else None, "alerts": n_alert}


def h2_summary(dates, labels, at_risk, clusters, predictions, config, draws=BOOTSTRAP_DRAWS, seed=0, block_length=BLOCK_LENGTH):
    test_start = int(np.searchsorted(dates, np.datetime64(config["test_start"])))
    common = (np.arange(len(dates)) >= test_start) & at_risk & np.isfinite(labels)
    for scores in predictions.values():
        common &= np.isfinite(scores)
    test_clusters = [item for item in clusters if item["first_onset"] >= test_start]
    truth = labels[common].astype(int)
    result = {"event_clusters": len(test_clusters), "anchor_count": int(np.sum(common)), "event_day_count": int(np.sum(truth == 1)), "no_skill_auc": 0.5, "bootstrap_draws_requested": draws}
    if len(np.unique(truth)) < 2:
        result.update({"status": "not_estimable", "reason": "the common out-of-sample anchors contain fewer than two outcome classes", "auc": {}, "auc_intervals": {}})
        return result
    point = {name: float(roc_auc_score(truth, scores[common])) for name, scores in predictions.items()}
    result["status"] = "estimated"
    result["auc"] = point
    result["delta_auc_M1_F"] = point["M1_F"] - point["M1"]
    result["alert_rate_5pct"] = {name: _alert_metrics(truth, predictions[name][common], 0.05) for name in ("M1", "M1_F", "EWMA")}
    test_timeline = np.arange(test_start, len(dates))
    generator = np.random.default_rng(config["seeds"]["bootstrap"] + seed)
    samples = []
    alert_samples = {f"{metric}_{name}": [] for name in ("M1", "M1_F") for metric in ("precision", "recall")}
    redraws = 0
    for _ in range(draws):
        for attempt in range(25):
            sampled = test_timeline[circular_block_indices(len(test_timeline), block_length, generator)]
            valid = common[sampled]
            indices = sampled[valid]
            sampled_truth = labels[indices].astype(int)
            if len(indices) > 1 and np.unique(sampled_truth).size == 2:
                break
            redraws += 1
        else:
            continue
        aucs = {name: float(roc_auc_score(sampled_truth, predictions[name][indices])) for name in MODEL_COLUMNS}
        samples.append([aucs["M1"], aucs["M1_F"], aucs["EWMA"], aucs["M1_F"] - aucs["M1"]])
        for name in ("M1", "M1_F"):
            metrics = _alert_metrics(sampled_truth, predictions[name][indices], 0.05)
            alert_samples[f"precision_{name}"].append(metrics["precision"])
            alert_samples[f"recall_{name}"].append(metrics["recall"])
    if samples:
        array = np.asarray(samples)
        names = ("M1", "M1_F", "EWMA", "delta_auc_M1_F")
        result["auc_intervals_95"] = {name: [float(value) for value in np.quantile(array[:, index], [0.025, 0.975])] for index, name in enumerate(names)}
        result["bootstrap_replicates_used"] = int(array.shape[0])
        result["bootstrap_redraws"] = redraws
        result["alert_metric_intervals_95"] = {name: [float(value) for value in np.quantile(values, [0.025, 0.975])] for name, values in alert_samples.items() if values}
    return result


def h2_block_length_sensitivity(dates, labels, at_risk, clusters, predictions, config):
    extension = RESULTS / "block_length_extension_plan.json"
    extension_hash_path = RESULTS / "block_length_extension_plan.sha256"
    extension_digest = extension_hash_path.read_text(encoding="ascii").strip()
    hash_verified(extension, extension_digest)
    outputs = []
    for block_length in (60, 125, 250):
        summary = h2_summary(
            dates, labels, at_risk, clusters, predictions, config,
            draws=BOOTSTRAP_DRAWS,
            seed=20261119 + block_length,
            block_length=block_length,
        )
        outputs.append({"block_length_sessions": block_length, **summary})
    return {
        "tier": "post_freeze_exploratory",
        "extension_plan_sha256": extension_digest,
        "replicates_per_length": BOOTSTRAP_DRAWS,
        "method": "Paired circular blocks of score-label observations on the common out-of-sample anchor set; model fits remain the original expanding-window fits.",
        "scope_limit": "This bandwidth comparison resamples fixed scores and labels. The separate 250-session refit-aware bootstrap resamples training blocks and refits the logistic models.",
        "results": outputs,
    }


def make_features(dates, returns, equal_weight, dynamic, config):
    stress = stress_metrics(dates, returns, equal_weight, config)
    features = {"R": dynamic["R"], "H": dynamic["H"], "F": dynamic["F"], **stress, "ewma_vol": ewma_volatility(equal_weight, 20)}
    return features, stress


def run_h2_configuration(dates, equal_weight, features, config, threshold, horizon, penalty, seed):
    local = copy.deepcopy(config)
    local["event_drawdown"] = float(threshold)
    local["event_forecast_days"] = int(horizon)
    local["logistic_C"] = float(penalty)
    clusters, labels, at_risk = make_events(features["drawdown"], threshold, config["event_merge_days"], horizon)
    predictions = fit_h2_models(features, labels, at_risk, dates, clusters, local)
    summary = h2_summary(dates, labels, at_risk, clusters, predictions, local, draws=BOOTSTRAP_DRAWS, seed=seed)
    summary.update({"threshold": float(threshold), "horizon_sessions": int(horizon), "logistic_C": float(penalty)})
    return summary, labels, at_risk, clusters, predictions


def h2_sensitivity_grid(dates, equal_weight, features, config):
    rows = []
    counter = 0
    for threshold in (-0.10, -0.15, -0.20):
        for horizon in (20, 60, 120):
            for penalty in (0.1, 1.0, 10.0):
                counter += 1
                row, _, _, _, _ = run_h2_configuration(dates, equal_weight, features, config, threshold, horizon, penalty, counter * 101)
                rows.append(row)
                print(f"H2 grid {counter}/27: threshold={threshold}, horizon={horizon}, C={penalty}, status={row['status']}", flush=True)
    return {"tier": "post_freeze_exploratory", "interval_method": "unadjusted paired circular 250-session block bootstrap; sensitivity grid is exploratory", "grid": rows}


def _fit_logistic(matrix, labels, indices, penalty):
    x = matrix[indices]
    y = labels[indices].astype(int)
    if len(np.unique(y)) < 2:
        return None
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0, ddof=1), 1e-8)
    model = LogisticRegression(C=float(penalty), penalty="l2", solver="lbfgs", max_iter=1000)
    model.fit((x - mean) / scale, y)
    return model, mean, scale


def refit_aware_h2_bootstrap(dates, labels, at_risk, clusters, features, predictions, config, replicates=BOOTSTRAP_DRAWS, seed=20261119):
    count = len(dates)
    test_start = int(np.searchsorted(dates, np.datetime64(config["test_start"])))
    calibration_start = int(np.searchsorted(dates, np.datetime64(config["calibration_start"])))
    test_timeline = np.arange(test_start, count)
    feature_matrices = {name: np.column_stack([features[key] for key in columns]) for name, columns in MODEL_COLUMNS.items() if name in ("M1", "M1_F")}
    common = (np.arange(count) >= test_start) & at_risk & np.isfinite(labels)
    for scores in predictions.values():
        common &= np.isfinite(scores)
    folds = []
    for refit_start in range(test_start, count, config["refit_days"]):
        cutoff = refit_start - config["purge_days"]
        matured = sum(1 for item in clusters if item["last_onset"] + config["event_forecast_days"] < cutoff)
        if matured < config["minimum_mature_clusters"]:
            continue
        train_timeline = np.arange(calibration_start, cutoff)
        test_index = np.arange(refit_start, min(count, refit_start + config["refit_days"]))
        test_index = test_index[common[test_index]]
        if train_timeline.size and test_index.size:
            folds.append((train_timeline, test_index))
    generator = np.random.default_rng(seed)
    values = []
    failures = 0
    for draw in range(replicates):
        for attempt in range(25):
            sampled_test = test_timeline[circular_block_indices(len(test_timeline), BLOCK_LENGTH, generator)]
            sampled_truth = labels[sampled_test].astype(int)
            sampled_valid = common[sampled_test]
            sampled_truth = sampled_truth[sampled_valid]
            if sampled_truth.size > 1 and np.unique(sampled_truth).size == 2:
                break
        else:
            failures += 1
            continue
        test_weights = np.bincount(sampled_test[sampled_valid] - test_start, minlength=len(test_timeline))
        weights = np.zeros(count, dtype=np.float64)
        weights[test_timeline] = test_weights
        refit_predictions = {name: np.full(count, np.nan) for name in ("M1", "M1_F")}
        draw_failed = False
        for train_timeline, test_index in folds:
            best_sample = None
            for _ in range(8):
                sampled_train = train_timeline[circular_block_indices(len(train_timeline), BLOCK_LENGTH, generator)]
                eligible = at_risk[sampled_train] & np.isfinite(labels[sampled_train])
                eligible &= np.all(np.isfinite(feature_matrices["M1"][sampled_train]), axis=1)
                eligible &= np.all(np.isfinite(feature_matrices["M1_F"][sampled_train]), axis=1)
                sampled_train = sampled_train[eligible]
                if sampled_train.size and np.unique(labels[sampled_train]).size == 2:
                    best_sample = sampled_train
                    break
            if best_sample is None:
                draw_failed = True
                break
            for name, matrix in feature_matrices.items():
                fitted = _fit_logistic(matrix, labels, best_sample, config["logistic_C"])
                if fitted is None:
                    draw_failed = True
                    break
                model, mean, scale = fitted
                valid_test = test_index[np.all(np.isfinite(matrix[test_index]), axis=1)]
                refit_predictions[name][valid_test] = model.predict_proba((matrix[valid_test] - mean) / scale)[:, 1]
            if draw_failed:
                break
        if draw_failed:
            failures += 1
            continue
        mask = common & (weights > 0) & np.isfinite(refit_predictions["M1"]) & np.isfinite(refit_predictions["M1_F"])
        y = labels[mask].astype(int)
        if np.unique(y).size < 2:
            failures += 1
            continue
        weight = weights[mask]
        auc0 = roc_auc_score(y, refit_predictions["M1"][mask], sample_weight=weight)
        auc1 = roc_auc_score(y, refit_predictions["M1_F"][mask], sample_weight=weight)
        values.append([float(auc0), float(auc1), float(auc1 - auc0)])
        if (draw + 1) % 25 == 0:
            print(f"Refit-aware H2 bootstrap {draw + 1}/{replicates}", flush=True)
    if not values:
        return {"status": "not_estimable", "replicates_requested": replicates, "replicates_used": 0, "failed_replicates": failures}
    array = np.asarray(values)
    return {
        "status": "estimated",
        "method": "250-session circular blocks in the training history and test window; refit M1 and M1+F logistic models at every original expanding-window cutoff.",
        "scope_limit": "The state filter and RLS path are held fixed; this interval captures logistic fitting and test-block uncertainty, not upstream state-estimation uncertainty.",
        "replicates_requested": replicates,
        "replicates_used": len(values),
        "failed_replicates": failures,
        "auc_M1_interval_95": [float(x) for x in np.quantile(array[:, 0], [0.025, 0.975])],
        "auc_M1_F_interval_95": [float(x) for x in np.quantile(array[:, 1], [0.025, 0.975])],
        "delta_auc_M1_F_interval_95": [float(x) for x in np.quantile(array[:, 2], [0.025, 0.975])],
    }


def _integrated_time_effective_n(values):
    x = np.asarray(values, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 3 or float(x @ x) == 0.0:
        return {"integrated_autocorrelation_time": 1.0, "effective_n": float(n)}
    nfft = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, n=nfft)
    autocov = np.fft.irfft(spectrum * spectrum.conjugate(), n=nfft)[:n]
    autocov /= autocov[0]
    positive = []
    for lag in range(1, min(n // 2, 2000)):
        if autocov[lag] <= 0.0:
            break
        positive.append(float(autocov[lag]))
    tau = max(1.0, 1.0 + 2.0 * sum(positive))
    return {"integrated_autocorrelation_time": tau, "effective_n": float(max(1.0, min(n, n / tau)))}


def factor_aligned_null(covariances, right_vectors, dates, config, draws=1000, block_replicates=1000):
    positions = np.flatnonzero(np.asarray(dates) >= np.datetime64(config["test_start"]))
    rank = int(config["principal_subspace_rank"])
    dim = int(right_vectors.shape[1])
    factors = []
    observed_spaces = []
    observed_angles = []
    valid_dates = []
    for index in positions:
        covariance = covariances[index]
        if not np.all(np.isfinite(covariance)):
            continue
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        factor = eigenvectors[:, -1]
        observed = right_vectors[index, :, :rank]
        factors.append(factor)
        observed_spaces.append(observed)
        observed_angles.append(principal_angle(eigenvectors[:, -rank:], observed))
        valid_dates.append(index)
    factors = np.asarray(factors)
    observed_spaces = np.asarray(observed_spaces)
    observed_angles = np.asarray(observed_angles)
    percentiles = np.empty(len(factors))
    generator = np.random.default_rng(config["seeds"]["haar"] + 81)
    chunk_size = 64
    for start in range(0, len(factors), chunk_size):
        end = min(len(factors), start + chunk_size)
        factor = factors[start:end]
        gaussian = generator.normal(size=(end - start, draws, dim, rank - 1))
        component = np.sum(gaussian * factor[:, None, :, None], axis=2, keepdims=True)
        gaussian -= factor[:, None, :, None] * component
        basis = np.linalg.qr(gaussian, mode="reduced")[0]
        fixed = np.broadcast_to(factor[:, None, :, None], (end - start, draws, dim, 1))
        null_spaces = np.concatenate((fixed, basis), axis=-1)
        cross = np.matmul(observed_spaces[start:end, None, :, :].transpose(0, 1, 3, 2), null_spaces)
        singular = np.linalg.svd(cross, compute_uv=False)
        angles = np.arccos(np.clip(singular, 0.0, 1.0)).mean(axis=-1)
        percentiles[start:end] = np.mean(angles <= observed_angles[start:end, None], axis=1)
        if end % (chunk_size * 8) == 0 or end == len(factors):
            print(f"factor-aligned null: dates {end}/{len(factors)}", flush=True)
    generator = np.random.default_rng(config["seeds"]["bootstrap"] + 82)
    boot = np.empty(block_replicates)
    for draw in range(block_replicates):
        indices = circular_block_indices(len(percentiles), BLOCK_LENGTH, generator)
        boot[draw] = np.median(percentiles[indices])
    return {
        "tier": "post_freeze_exploratory",
        "construction": "The leading rolling covariance eigenvector is fixed in every null subspace; the remaining dimensions are random orthonormal directions in its orthogonal complement.",
        "draws_per_date": draws,
        "date_count": int(len(percentiles)),
        "median_percentile": float(np.median(percentiles)),
        "mean_percentile": float(np.mean(percentiles)),
        "median_percentile_block_bootstrap_95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "block_length_sessions": BLOCK_LENGTH,
        "bootstrap_replicates": block_replicates,
        "autocorrelation": _integrated_time_effective_n(percentiles),
        "haar_median_percentile_frozen": None,
        "date_indices_are_derived": True,
    }


def haar_percentile_dependence(study, config, block_replicates=1000):
    plan = RESULTS / "haar_dependence_extension_plan.json"
    plan_digest = (RESULTS / "haar_dependence_extension_plan.sha256").read_text(encoding="ascii").strip()
    hash_verified(plan, plan_digest)
    direction = study["H1"]
    observed_angles = np.asarray(direction["direction_angles"], dtype=np.float64)
    reference_angles = np.asarray(direction["haar_angles"], dtype=np.float64)
    percentiles = np.mean(reference_angles[None, :] <= observed_angles[:, None], axis=1)
    generator = np.random.default_rng(config["seeds"]["bootstrap"] + 83)
    bootstrap = np.empty(block_replicates, dtype=np.float64)
    for draw in range(block_replicates):
        indices = circular_block_indices(len(percentiles), BLOCK_LENGTH, generator)
        bootstrap[draw] = np.median(percentiles[indices])
    return {
        "tier": "post_freeze_exploratory",
        "extension_plan_sha256": plan_digest,
        "date_count": int(len(percentiles)),
        "reference_draws": int(len(reference_angles)),
        "original_frozen_median_percentile": float(direction["direction"]["median_haar_percentile"]),
        "recomputed_median_percentile": float(np.median(percentiles)),
        "mean_percentile": float(np.mean(percentiles)),
        "block_bootstrap_95_interval": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "block_length_sessions": BLOCK_LENGTH,
        "bootstrap_replicates": int(block_replicates),
        "autocorrelation": _integrated_time_effective_n(percentiles),
        "construction": "For each date, the percentile is the fraction of frozen Haar reference angles no larger than the observed mean principal angle; inference resamples ordered dates in circular blocks.",
    }


def run_h1_variant(name, dates, returns, equal_weight, config, floor, zero_as_missing, forgetting, parameters, seed_offset):
    local = copy.deepcopy(config)
    local["variance_floor"] = float(floor)
    output = core_run(returns, parameters, forgetting, local, zero_as_missing=zero_as_missing)
    dynamic = dynamic_metrics(output["J"], local["primary_horizon"], local["rotation_lag_days"], local["principal_subspace_rank"])
    features, stress = make_features(dates, returns, equal_weight, dynamic, local)
    clusters, labels, at_risk = make_events(stress["drawdown"], local["event_drawdown"], local["event_merge_days"], local["event_forecast_days"])
    evidence = h1_evidence(dates, returns, dynamic, clusters, parameters, local, forgetting, zero_as_missing, seed_offset=seed_offset)
    h2_predictions = fit_h2_models(features, labels, at_risk, dates, clusters, local)
    h2 = h2_summary(dates, labels, at_risk, clusters, h2_predictions, local, seed=seed_offset)
    return {
        "tier": "post_freeze_exploratory",
        "variant": name,
        "variance_floor": floor,
        "zero_as_missing": zero_as_missing,
        "boundary_assets_in_estimation": np.asarray(parameters.get("boundary_assets", []), dtype=int).tolist(),
        "h1": evidence,
        "h2": h2,
    }


def zero_variants(dates, returns, equal_weight, config, study, validation):
    periods = {"estimation": (dates >= np.datetime64(config["estimate_start"])) & (dates <= np.datetime64(config["estimate_end"]))}
    baseline_params = {key: np.asarray(validation["G4"]["parameters"][key], dtype=np.float64) for key in ("mu", "phi", "q")}
    outputs = []
    variants = [
        ("variance_floor_1e-6", 1e-6, False, 301),
        ("zero_returns_missing", float(config["variance_floor"]), True, 701),
    ]
    for name, floor, missing, seed_offset in variants:
        local = copy.deepcopy(config)
        local["variance_floor"] = floor
        parameters = estimate_parameters(returns[periods["estimation"]], local, zero_as_missing=missing)
        outputs.append(run_h1_variant(name, dates, returns, equal_weight, local, floor, missing, float(study["selected_lambda"]), parameters, seed_offset))
    original_event_raw = study["H1"]["event_placebo_p_raw"]
    original_event_adjusted = study["H1"]["event_placebo_p_holm"]
    return {
        "tier": "post_freeze_exploratory",
        "baseline_event_placebo": {"raw_p_R_H_F": original_event_raw, "holm_adjusted_p_R_H_F": original_event_adjusted, "explanation": "Holm-adjusted values equal one because all three raw placebo probabilities are large; they are not evidence that each observed statistic equals its placebo median."},
        "variants": outputs,
    }


def run_subperiod_f(dates, returns, config, study, parameters):
    primary = [("1970_1999", "1970-01-01", "1999-12-31"), ("2000_2026_09", "2000-01-01", "2026-09-30")]
    split = int(np.searchsorted(dates, np.datetime64(config["test_start"])))
    base = core_run(returns, parameters, float(study["selected_lambda"]), config)
    original = dynamic_metrics(base["J"], config["primary_horizon"], config["rotation_lag_days"], config["principal_subspace_rank"])["F"]
    spans = [(name, int(np.searchsorted(dates, np.datetime64(start))), int(np.searchsorted(dates, np.datetime64(end), side="right")), start, end) for name, start, end in primary]
    rng = np.random.default_rng(config["seeds"]["surrogate"] + 409)
    null = {name: np.empty(SURROGATE_DRAWS) for name, *_ in spans}
    for draw in range(SURROGATE_DRAWS):
        shuffled = returns.copy()
        for name, start, end, _, _ in spans:
            shuffled[start:end] = block_shuffle(returns[start:end], config["surrogate_block_days"], rng)
        surrogate_core = core_run(shuffled, parameters, float(study["selected_lambda"]), config)
        for name, start, end, _, _ in spans:
            local = dynamic_metrics(surrogate_core["J"][start:end], config["primary_horizon"], config["rotation_lag_days"], config["principal_subspace_rank"])
            null[name][draw] = float(np.median(local["F"]))
        if (draw + 1) % 50 == 0:
            print(f"subperiod F surrogate {draw + 1}/{SURROGATE_DRAWS}", flush=True)
    results = {}
    for name, start, end, start_date, end_date in spans:
        observed = float(np.median(original[start:end]))
        results[name] = {"start": start_date, "end": end_date, "observed_median_F": observed, "null_median": float(np.median(null[name])), "raw_p": empirical_two_sided(observed, null[name]), "draws": SURROGATE_DRAWS}
    adjusted = holm_adjust([results[name]["raw_p"] for name, *_ in spans]).tolist()
    for item, p in zip(results.values(), adjusted):
        item["holm_adjusted_p"] = p
    return {"tier": "post_freeze_exploratory", "adjustment": "Holm across the two prespecified subperiods", "results": results}


def run_lambda_grid(dates, returns, config, study, params):
    values = (0.990, 0.995, 0.998, 0.999)
    split = int(np.searchsorted(dates, np.datetime64(config["test_start"])))
    rng = np.random.default_rng(config["seeds"]["surrogate"] + 997)
    rows = []
    for position, forgetting in enumerate(values):
        output = core_run(returns, params, forgetting, config)
        dynamic = dynamic_metrics(output["J"], config["primary_horizon"], config["rotation_lag_days"], config["principal_subspace_rank"])
        observed = [float(np.median(dynamic[key][split:])) for key in ("R", "H", "F")]
        null = np.empty((SURROGATE_DRAWS, 3))
        for draw in range(SURROGATE_DRAWS):
            shuffled = returns.copy()
            shuffled[split:] = block_shuffle(returns[split:], config["surrogate_block_days"], rng)
            surrogate_core = core_run(shuffled, params, forgetting, config)
            start = split - config["rotation_lag_days"]
            local = dynamic_metrics(surrogate_core["J"][start:], config["primary_horizon"], config["rotation_lag_days"], config["principal_subspace_rank"])
            null[draw] = [float(np.median(local[key][config["rotation_lag_days"]:])) for key in ("R", "H", "F")]
            if (draw + 1) % 50 == 0:
                print(f"lambda {forgetting} surrogate {draw + 1}/{SURROGATE_DRAWS}", flush=True)
        raw = [empirical_two_sided(observed[index], null[:, index]) for index in range(3)]
        adjusted = holm_adjust(raw).tolist()
        rows.append({"lambda": forgetting, "median_R": observed[0], "median_H": observed[1], "median_F": observed[2], "surrogate_p_raw_R_H_F": raw, "surrogate_p_holm_R_H_F": adjusted, "null_median_R_H_F": [float(np.median(null[:, index])) for index in range(3)], "draws": SURROGATE_DRAWS})
        rows[-1]["distribution_summary"] = {
            key: {
                "p05": float(np.quantile(dynamic[key][split:], 0.05)),
                "p25": float(np.quantile(dynamic[key][split:], 0.25)),
                "median": float(np.median(dynamic[key][split:])),
                "p75": float(np.quantile(dynamic[key][split:], 0.75)),
                "p95": float(np.quantile(dynamic[key][split:], 0.95)),
            }
            for key in ("R", "H", "F")
        }
    return {"tier": "post_freeze_exploratory", "parameters_fixed_to_primary_estimation": True, "grid": rows}


def run_17_panel(config, primary_study):
    member = ROOT / "data" / "exploratory" / "17_industry" / "17_Industry_Portfolios_Daily.csv"
    dates, returns, equal_weight, labels = load_panel(member_path=member, portfolio_count=17)
    estimate = (dates >= np.datetime64(config["estimate_start"])) & (dates <= np.datetime64(config["estimate_end"]))
    params = estimate_parameters(returns[estimate], config)
    output = core_run(returns, params, 0.998, config)
    dynamic = dynamic_metrics(output["J"], config["primary_horizon"], config["rotation_lag_days"], config["principal_subspace_rank"])
    features, stress = make_features(dates, returns, equal_weight, dynamic, config)
    split = int(np.searchsorted(dates, np.datetime64(config["test_start"])))
    clusters, labels_target, at_risk = make_events(stress["drawdown"], config["event_drawdown"], config["event_merge_days"], config["event_forecast_days"])
    observed = float(np.median(dynamic["F"][split:]))
    rng = np.random.default_rng(config["seeds"]["surrogate"] + 1701)
    null = np.empty(SURROGATE_DRAWS)
    for draw in range(SURROGATE_DRAWS):
        shuffled = returns.copy()
        shuffled[split:] = block_shuffle(returns[split:], config["surrogate_block_days"], rng)
        surrogate_core = core_run(shuffled, params, 0.998, config)
        start = split - config["rotation_lag_days"]
        local = dynamic_metrics(surrogate_core["J"][start:], config["primary_horizon"], config["rotation_lag_days"], config["principal_subspace_rank"])
        null[draw] = float(np.median(local["F"][config["rotation_lag_days"]:]))
        if (draw + 1) % 25 == 0:
            print(f"17-industry F surrogate {draw + 1}/{SURROGATE_DRAWS}", flush=True)
    f_raw_p = empirical_two_sided(observed, null)
    predictions = fit_h2_models(features, labels_target, at_risk, dates, clusters, config)
    h2 = h2_summary(dates, labels_target, at_risk, clusters, predictions, config, seed=1701)
    primary_onsets = [np.datetime64(value) for value in primary_study["event_onsets"]]
    secondary_onsets = [np.datetime64(dates[item["first_onset"]]) for item in clusters if item["first_onset"] >= split]
    session_index = {date: index for index, date in enumerate(dates)}
    distances = np.asarray([[abs(session_index[a] - session_index[b]) for b in secondary_onsets] for a in primary_onsets], dtype=int)
    matched = 0
    if distances.size:
        from scipy.optimize import linear_sum_assignment
        costs = distances.astype(np.float64)
        costs[costs > config["event_merge_days"]] = 1e9
        rows, columns = linear_sum_assignment(costs)
        matched = int(np.sum(costs[rows, columns] <= config["event_merge_days"]))
    return {
        "tier": "post_freeze_exploratory",
        "source": "17 U.S. industry portfolios; aggregation check, not an independent market replication",
        "dates": {"start": str(dates[0]), "end": str(dates[-1]), "test_start": config["test_start"]},
        "portfolio_labels": labels,
        "selected_lambda": 0.998,
        "test_event_clusters": len(secondary_onsets),
        "primary_10_industry_event_clusters": len(primary_onsets),
        "event_onsets_matched_within_120_sessions": matched,
        "event_overlap_share_of_17_panel": float(matched / len(secondary_onsets)) if secondary_onsets else None,
        "F_surrogate": {"observed_median": observed, "null_median": float(np.median(null)), "raw_p": f_raw_p, "draws": SURROGATE_DRAWS},
        "H2": h2,
        "estimated_parameter_boundaries": np.asarray(params.get("boundary_assets", []), dtype=int).tolist(),
        "panel_dimensions": int(returns.shape[1]),
        "derived_time_series_hashes_only": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the post-freeze exploratory addendum without overwriting frozen results.")
    parser.add_argument("--stage", choices=("all", "zeros", "factor", "haar_dependence", "h2", "block_sensitivity", "subperiod", "lambda", "panel17"), default="all")
    args = parser.parse_args()
    plan_hash = verify_frozen_inputs()
    RESULTS.mkdir(parents=True, exist_ok=True)
    dates, returns, equal_weight, study, validation, config, protocol, params = load_primary()
    if args.stage in ("all", "zeros"):
        counts = write_zero_counts()
        print("zero counts written", flush=True)
        zero = zero_variants(dates, returns, equal_weight, config, study, validation)
        write_json(RESULTS / "zero_variants.json", zero)
        print("zero-return variants written", flush=True)
    if args.stage in ("all", "factor", "haar_dependence"):
        if args.stage in ("all", "factor"):
            baseline = core_run(returns, params, float(study["selected_lambda"]), config)
            dynamic = dynamic_metrics(baseline["J"], config["primary_horizon"], config["rotation_lag_days"], config["principal_subspace_rank"])
            stress = stress_metrics(dates, returns, equal_weight, config)
            aligned = factor_aligned_null(stress["cov250"], dynamic["right_vectors"], dates, config)
            aligned["haar_median_percentile_frozen"] = float(study["H1"]["direction"]["median_haar_percentile"])
            write_json(RESULTS / "factor_aligned_null.json", aligned)
            print("factor-aligned null written", flush=True)
        if args.stage in ("all", "factor", "haar_dependence"):
            haar = haar_percentile_dependence(study, config)
            write_json(RESULTS / "haar_dependence.json", haar)
            print("Haar percentile dependence summary written", flush=True)
    if args.stage in ("all", "h2", "block_sensitivity"):
        baseline = core_run(returns, params, float(study["selected_lambda"]), config)
        dynamic = dynamic_metrics(baseline["J"], config["primary_horizon"], config["rotation_lag_days"], config["principal_subspace_rank"])
        features, stress = make_features(dates, returns, equal_weight, dynamic, config)
        grid = h2_sensitivity_grid(dates, equal_weight, features, config)
        write_json(RESULTS / "h2_sensitivity.json", grid)
        clusters, labels, at_risk = make_events(stress["drawdown"], config["event_drawdown"], config["event_merge_days"], config["event_forecast_days"])
        primary_predictions = fit_h2_models(features, labels, at_risk, dates, clusters, config)
        primary_h2 = h2_summary(dates, labels, at_risk, clusters, primary_predictions, config, seed=443)
        if args.stage in ("all", "h2"):
            write_json(RESULTS / "h2_prediction_metrics.json", {"tier": "post_freeze_exploratory", "primary_comparison": primary_h2, "refit_aware": refit_aware_h2_bootstrap(dates, labels, at_risk, clusters, features, primary_predictions, config)})
            print("H2 metrics and refit-aware bootstrap written", flush=True)
        if args.stage in ("all", "block_sensitivity"):
            block_sensitivity = h2_block_length_sensitivity(dates, labels, at_risk, clusters, primary_predictions, config)
            write_json(RESULTS / "h2_block_length_sensitivity.json", block_sensitivity)
            print("H2 block-length sensitivity written", flush=True)
    if args.stage in ("all", "subperiod"):
        result = run_subperiod_f(dates, returns, config, study, params)
        write_json(RESULTS / "subperiod_F_surrogate.json", result)
        print("subperiod F surrogate written", flush=True)
    if args.stage in ("all", "lambda"):
        result = run_lambda_grid(dates, returns, config, study, params)
        write_json(RESULTS / "lambda_grid.json", result)
        print("lambda grid written", flush=True)
    if args.stage in ("all", "panel17"):
        result = run_17_panel(config, study)
        write_json(RESULTS / "panel_17_industry.json", result)
        print("17-industry panel written", flush=True)
    if verify_frozen_inputs() != plan_hash:
        raise RuntimeError("post-freeze plan digest changed during execution")
    run_record = {
        "tier": "post_freeze_exploratory",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_plan_sha256": plan_hash,
        "block_length_extension_plan_sha256": (RESULTS / "block_length_extension_plan.sha256").read_text(encoding="ascii").strip() if (RESULTS / "block_length_extension_plan.sha256").exists() else None,
        "haar_dependence_extension_plan_sha256": (RESULTS / "haar_dependence_extension_plan.sha256").read_text(encoding="ascii").strip() if (RESULTS / "haar_dependence_extension_plan.sha256").exists() else None,
        "frozen_study_sha256": sha256_file(ROOT / "results" / "study.json"),
        "frozen_protocol_sha256": FROZEN_PROTOCOL_SHA,
        "frozen_analysis_plan_sha256": FROZEN_ANALYSIS_PLAN_SHA,
        "exploratory_script_sha256": sha256_file(ROOT / "python" / "exploratory.py"),
        "core_source_sha256": sha256_file(ROOT / "core" / "market_core.cpp"),
        "core_binary_sha256": sha256_file(ROOT / "python" / "market_core.cp312-win_amd64.pyd"),
        "secondary_data_snapshot_sha256": sha256_file(RESULTS / "secondary_data_snapshot.json") if (RESULTS / "secondary_data_snapshot.json").exists() else None,
    }
    write_json(RESULTS / "reproducibility.json", run_record)
    print(json.dumps(run_record, indent=2))


if __name__ == "__main__":
    main()
