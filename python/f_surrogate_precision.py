"""Hashed 4,999-draw, F-only post-freeze surrogate precision rerun."""

import copy
import json
import time

import numpy as np

from common import ROOT, hash_verified, sha256_file, write_json
from dataset import load_derived
from estimation import estimate_parameters
from exploratory import load_primary
from gates import core_run
from metrics import empirical_two_sided, holm_adjust
from real_analysis import block_shuffle


RESULTS = ROOT / "results" / "exploratory"
PLAN = RESULTS / "f_surrogate_precision_plan.json"
PLAN_SHA = RESULTS / "f_surrogate_precision_plan.sha256"
DRAWS = 4999
FROZEN_STUDY_SHA = "732f6b8b15661ec9b176489c989b5ccba4c09d453d24597aee78c430cdc0d0ae"
FROZEN_PROTOCOL_SHA = "a860e120e56490bcfe14675241d33f5958929c0e78747a8da334c79c7fa311c0"
FROZEN_ANALYSIS_PLAN_SHA = "7ea3f2c36f7150e3a13478dbfe6b7205e31d0934132d04eb6087935fe5688388"


def verify_inputs():
    plan_hash = PLAN_SHA.read_text(encoding="ascii").strip()
    hash_verified(PLAN, plan_hash)
    checks = {
        ROOT / "results" / "study.json": FROZEN_STUDY_SHA,
        ROOT / "results" / "protocol.sha256": FROZEN_PROTOCOL_SHA,
        ROOT / "results" / "analysis_plan.sha256": FROZEN_ANALYSIS_PLAN_SHA,
    }
    for path, expected in checks.items():
        actual = path.read_text(encoding="ascii").strip() if path.suffix == ".sha256" else sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"frozen input changed: {path.relative_to(ROOT)}")
    return plan_hash


def median_f(jacobians, horizon, start, end):
    power = np.linalg.matrix_power(jacobians[start:end], horizon)
    largest = np.linalg.svd(power, compute_uv=False)[:, 0]
    return float(np.median(largest))


def rerun_f(dates, returns, parameters, forgetting, config, observed_jacobians, seed_offset=0, spans=None, generator=None):
    split = int(np.searchsorted(dates, np.datetime64(config["test_start"])))
    if spans is None:
        observed = median_f(observed_jacobians, config["primary_horizon"], split, len(dates))
        ranges = [("test", split, len(dates))]
    else:
        ranges = [(name, int(np.searchsorted(dates, np.datetime64(start))), int(np.searchsorted(dates, np.datetime64(end), side="right"))) for name, start, end in spans]
        observed = {name: median_f(observed_jacobians, config["primary_horizon"], start, end) for name, start, end in ranges}
    if generator is None:
        generator = np.random.default_rng(config["seeds"]["surrogate"] + seed_offset)
    null = {name: np.empty(DRAWS, dtype=np.float64) for name, *_ in ranges}
    for draw in range(DRAWS):
        shuffled = returns.copy()
        if spans is None:
            shuffled[split:] = block_shuffle(returns[split:], config["surrogate_block_days"], generator)
        else:
            for _, start, end in ranges:
                shuffled[start:end] = block_shuffle(returns[start:end], config["surrogate_block_days"], generator)
        core = core_run(shuffled, parameters, forgetting, config)
        for name, start, end in ranges:
            null[name][draw] = median_f(core["J"], config["primary_horizon"], start, end)
        if (draw + 1) % 500 == 0:
            print(f"F surrogate seed offset {seed_offset}: {draw + 1}/{DRAWS}", flush=True)
    result = {
        name: {
            "observed_median_F": observed[name] if isinstance(observed, dict) else observed,
            "null_median_F": float(np.median(null[name])),
            "raw_p": empirical_two_sided(observed[name] if isinstance(observed, dict) else observed, null[name]),
            "draws": DRAWS,
            "seed": int(config["seeds"]["surrogate"] + seed_offset),
        }
        for name, *_ in ranges
    }
    return result


def update_h1_f(h1, f_result, plan_hash):
    updated = copy.deepcopy(h1)
    raw = list(updated["surrogate_p_raw"])
    raw[2] = f_result["raw_p"]
    updated["surrogate_p_raw"] = raw
    updated["surrogate_p_holm"] = holm_adjust(raw).tolist()
    updated.setdefault("surrogate_null_medians", {})["F"] = f_result["null_median_F"]
    updated.setdefault("surrogate_observed_medians", {})["F"] = f_result["observed_median_F"]
    updated["surrogate_draws_by_signal"] = {"R": int(h1.get("surrogate_draws", 499)), "H": int(h1.get("surrogate_draws", 499)), "F": DRAWS}
    updated["precision_plan_sha256"] = plan_hash
    return updated


def previous_values(study, zero, subperiod, lambdas):
    return {
        "primary": {"raw_p": study["H1"]["surrogate_p_raw"][2], "holm_adjusted_p": study["H1"]["surrogate_p_holm"][2], "draws": int(json.loads((ROOT / "results" / "protocol.json").read_text(encoding="utf-8"))["config"]["surrogate_draws"])},
        "zero_variants": {
            row["variant"]: {"raw_p": row["h1"]["surrogate_p_raw"][2], "holm_adjusted_p": row["h1"]["surrogate_p_holm"][2], "draws": row["h1"].get("surrogate_draws", 499)}
            for row in zero["variants"]
        },
        "subperiods": {name: {"raw_p": row["raw_p"], "holm_adjusted_p": row["holm_adjusted_p"], "draws": row["draws"]} for name, row in subperiod["results"].items()},
        "lambda_grid": {str(row["lambda"]): {"raw_p": row["surrogate_p_raw_R_H_F"][2], "holm_adjusted_p": row["surrogate_p_holm_R_H_F"][2], "draws": row["draws"]} for row in lambdas["grid"]},
    }


def main():
    plan_hash = verify_inputs()
    dates, returns, equal_weight = load_derived()
    dates_check, returns_check, equal_weight_check, study, validation, config, _, parameters = load_primary()
    if not (np.array_equal(dates, dates_check) and np.array_equal(returns, returns_check) and np.array_equal(equal_weight, equal_weight_check)):
        raise RuntimeError("derived input panels differ between verified loaders")
    old_zero = json.loads((RESULTS / "zero_variants.json").read_text(encoding="utf-8"))
    old_subperiod = json.loads((RESULTS / "subperiod_F_surrogate.json").read_text(encoding="utf-8"))
    old_lambda = json.loads((RESULTS / "lambda_grid.json").read_text(encoding="utf-8"))
    previous = previous_values(study, old_zero, old_subperiod, old_lambda)
    split = int(np.searchsorted(dates, np.datetime64(config["test_start"])))
    outputs = {}

    print(f"starting primary F surrogate ({DRAWS} draws)", flush=True)
    primary_core = core_run(returns, parameters, float(study["selected_lambda"]), config)
    primary_f = rerun_f(dates, returns, parameters, float(study["selected_lambda"]), config, primary_core["J"], seed_offset=0)["test"]
    primary_raw = list(study["H1"]["surrogate_p_raw"])
    primary_raw[2] = primary_f["raw_p"]
    primary_holm = holm_adjust(primary_raw).tolist()
    outputs["primary"] = primary_f | {"holm_adjusted_p": primary_holm[2], "p_R_H_F_raw_used_for_Holm": primary_raw, "p_R_H_F_holm": primary_holm, "reused_R_H_draws": 499}

    zero = copy.deepcopy(old_zero)
    zero_specs = {"variance_floor_1e-6": (1e-6, False, 301), "zero_returns_missing": (float(config["variance_floor"]), True, 701)}
    estimation = (dates >= np.datetime64(config["estimate_start"])) & (dates <= np.datetime64(config["estimate_end"]))
    zero_rows = {row["variant"]: row for row in zero["variants"]}
    for name, (floor, missing, seed_offset) in zero_specs.items():
        print(f"starting {name} F surrogate ({DRAWS} draws)", flush=True)
        local = copy.deepcopy(config)
        local["variance_floor"] = floor
        fitted = estimate_parameters(returns[estimation], local, zero_as_missing=missing)
        baseline = core_run(returns, fitted, float(study["selected_lambda"]), local, zero_as_missing=missing)
        f_value = rerun_f(dates, returns, fitted, float(study["selected_lambda"]), local, baseline["J"], seed_offset=seed_offset)["test"]
        row = zero_rows[name]
        row["h1"] = update_h1_f(row["h1"], f_value, plan_hash)
        row["h1"]["surrogate_f_raw_p"] = f_value["raw_p"]
        row["h1"]["surrogate_f_holm_p"] = row["h1"]["surrogate_p_holm"][2]
        outputs[name] = f_value | {"holm_adjusted_p": row["h1"]["surrogate_p_holm"][2], "p_R_H_F_raw_used_for_Holm": row["h1"]["surrogate_p_raw"], "p_R_H_F_holm": row["h1"]["surrogate_p_holm"], "reused_R_H_draws": 499}
    zero["precision_plan_sha256"] = plan_hash
    zero["F_surrogate_draws"] = DRAWS
    zero["multiplicity_note"] = "R and H raw probabilities are retained from the prior 499-draw runs; Holm values are recomputed with the 4,999-draw F probability."

    print(f"starting subperiod F surrogates ({DRAWS} draws per period)", flush=True)
    subperiod_spans = [("1970_1999", "1970-01-01", "1999-12-31"), ("2000_2026_09", "2000-01-01", "2026-09-30")]
    baseline = core_run(returns, parameters, float(study["selected_lambda"]), config)
    subperiod_values = rerun_f(dates, returns, parameters, float(study["selected_lambda"]), config, baseline["J"], seed_offset=409, spans=subperiod_spans)
    subperiod = {"tier": "post_freeze_exploratory", "precision_plan_sha256": plan_hash, "draws_per_period": DRAWS, "adjustment": "Holm across the two prespecified subperiod F probabilities", "results": {}}
    sub_raw = [subperiod_values[name]["raw_p"] for name, _, _ in subperiod_spans]
    sub_holm = holm_adjust(sub_raw).tolist()
    for index, (name, start_date, end_date) in enumerate(subperiod_spans):
        subperiod["results"][name] = {"start": start_date, "end": end_date, **subperiod_values[name], "holm_adjusted_p": sub_holm[index]}
        outputs[name] = subperiod["results"][name]

    print(f"starting lambda-grid F surrogates ({DRAWS} draws per value)", flush=True)
    lambda_grid = copy.deepcopy(old_lambda)
    lambda_rng = np.random.default_rng(config["seeds"]["surrogate"] + 997)
    lambda_rows = {float(row["lambda"]): row for row in lambda_grid["grid"]}
    for forgetting in (0.990, 0.995, 0.998, 0.999):
        print(f"starting lambda={forgetting:.3f} F surrogate", flush=True)
        baseline = core_run(returns, parameters, forgetting, config)
        f_value = rerun_f(dates, returns, parameters, forgetting, config, baseline["J"], seed_offset=997, generator=lambda_rng)["test"]
        row = lambda_rows[forgetting]
        raw = list(row["surrogate_p_raw_R_H_F"])
        raw[2] = f_value["raw_p"]
        adjusted = holm_adjust(raw).tolist()
        row["surrogate_p_raw_R_H_F"] = raw
        row["surrogate_p_holm_R_H_F"] = adjusted
        row["null_median_R_H_F"][2] = f_value["null_median_F"]
        row["surrogate_draws_by_signal"] = {"R": 499, "H": 499, "F": DRAWS}
        outputs[str(forgetting)] = f_value | {"holm_adjusted_p": adjusted[2], "p_R_H_F_raw_used_for_Holm": raw, "p_R_H_F_holm": adjusted, "reused_R_H_draws": 499}
    lambda_grid["precision_plan_sha256"] = plan_hash
    lambda_grid["F_surrogate_draws"] = DRAWS
    lambda_grid["multiplicity_note"] = "R and H raw probabilities are retained from the prior 499-draw runs; Holm values are recomputed with the 4,999-draw F probability."

    primary_output = {
        "tier": "post_freeze_exploratory",
        "precision_plan_sha256": plan_hash,
        "draws": DRAWS,
        "minimum_attainable_empirical_p": 1.0 / (DRAWS + 1),
        "method": "F-only circular block shuffle; R and H raw probabilities remain from the valid 499-draw analysis and are included in the recomputed Holm family.",
        "previous_frozen_499_draw_result": previous["primary"],
        "rerun": outputs["primary"],
    }
    comparison = {
        "tier": "post_freeze_exploratory",
        "precision_plan_sha256": plan_hash,
        "draws_per_F_test": DRAWS,
        "minimum_attainable_empirical_p": 1.0 / (DRAWS + 1),
        "previous_499_draw_values": previous,
        "rerun_F_values": outputs,
    }

    write_json(RESULTS / "primary_F_surrogate_precision.json", primary_output)
    write_json(RESULTS / "zero_variants.json", zero)
    write_json(RESULTS / "subperiod_F_surrogate.json", subperiod)
    write_json(RESULTS / "lambda_grid.json", lambda_grid)
    write_json(RESULTS / "f_surrogate_precision_comparison.json", comparison)
    if verify_inputs() != plan_hash:
        raise RuntimeError("frozen input or precision plan hash changed during the rerun")

    names = ("primary_F_surrogate_precision.json", "zero_variants.json", "subperiod_F_surrogate.json", "lambda_grid.json", "f_surrogate_precision_comparison.json")
    record = json.loads((RESULTS / "reproducibility.json").read_text(encoding="utf-8"))
    record.update({
        "precision_run_completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "precision_plan_sha256": plan_hash,
        "precision_runner_sha256": sha256_file(ROOT / "python" / "f_surrogate_precision.py"),
        "precision_draws_per_F_test": DRAWS,
        "precision_seed_offsets": {"primary": 0, "floor_1e-6": 301, "zeros_missing": 701, "subperiods": 409, "lambda_grid": 997},
        "precision_output_sha256": {name: sha256_file(RESULTS / name) for name in names},
    })
    write_json(RESULTS / "reproducibility.json", record)
    print(json.dumps(outputs, indent=2), flush=True)
    print(json.dumps({"precision_plan_sha256": plan_hash, "precision_runner_sha256": record["precision_runner_sha256"], "frozen_study_sha256": sha256_file(ROOT / "results" / "study.json"), "frozen_protocol_sha256": FROZEN_PROTOCOL_SHA}, indent=2), flush=True)


if __name__ == "__main__":
    main()
