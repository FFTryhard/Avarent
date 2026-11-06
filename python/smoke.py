import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from common import ROOT, write_json
from documents import render
from plots import create_figures


def run_smoke():
    out = ROOT / "paper" / "smoke"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    dates = np.arange(300)
    bootstrap = np.random.default_rng(1).normal(0.01, 0.02, size=(2, 1000))
    event = {key: {"aligned_median": np.sin(np.linspace(-3.0, 3.0, 121)) * scale, "aligned_offsets": np.arange(-60, 61), "p_value": 0.2, "placebo_values": np.linspace(-1.0, 1.0, 1000)} for key, scale in (("R", 0.2), ("H", 1.0), ("F", 0.5))}
    study = {
        "tier": "diagnostic",
        "data_period": {"start": "1926-07-01", "test_start": "1970-01-01", "end": "2026-09-30"},
        "selected_lambda": 0.998,
        "event_clusters": 4,
        "anchor_count": 1000,
        "H1": {
            "verdict": "inconclusive",
            "surrogate_p_holm": [0.2, 0.3, 0.4],
            "event_placebo_p_holm": [0.2, 0.3, 0.4],
            "condition_a": False,
            "condition_b": False,
            "condition_c": True,
            "condition_d": True,
            "max_abs_spearman": 0.4,
            "pca_components_for_90_percent": 4,
            "direction": {"median_haar_percentile": 0.5},
            "direction_angles": np.linspace(0.1, 1.0, 300),
            "haar_angles": np.linspace(0.2, 1.2, 1000),
            "event_placebos": event,
            "wedin": {"median_gap": 0.2, "median_perturbation_norm": 0.1, "informative_share": 0.3}
        },
        "H2": {
            "verdict": "inconclusive",
            "bootstrap": {
                "point_estimates": [0.01, 0.015],
                "holm_adjusted_95_intervals": [[-0.02, 0.04], [-0.03, 0.05]],
                "holm_adjusted_p_at_zero": [0.7, 0.6],
                "bootstrap": bootstrap,
                "event_clusters": 4,
                "anchor_count": 1000
            },
            "moderate_regime": {"status": "not estimable", "event_clusters": 1, "anchor_count": 200}
        },
        "robustness": {
            "alternative_lambda": {"lambda": 0.995, "median_R": 0.3, "median_H": 4.0, "median_F5": 1.1},
            "ema_before_rls": {"half_life_days": 10, "median_R": 0.25, "median_H": 4.2, "median_F5": 1.0},
            "horizon_20": {"median_F20": 1.3, "median_rho_power_20": 0.6}
        },
        "time_series": {
            "dates": [str(np.datetime64("1970-01-01") + np.timedelta64(i, "D")) for i in dates],
            "R": np.abs(np.sin(dates / 20.0)),
            "H": 3.0 + np.cos(dates / 25.0),
            "vol": 0.15 + 0.03 * np.sin(dates / 10.0),
            "drawdown": -0.1 + 0.05 * np.cos(dates / 17.0),
            "rho_tilde": 0.1 * np.sin(dates / 13.0),
            "F5": 1.0 + 0.1 * np.sin(dates / 8.0),
            "F20": 1.2 + 0.2 * np.sin(dates / 8.0)
        }
    }
    validation = {
        "tier": "diagnostic",
        "U": {"ukf_kf_abs_error": 1e-14, "finite_difference_abs_error": 1e-10},
        "E1": {"assets_passing": 10},
        "G1": {"selected_lambda": 0.998, "trials": [{"lambda": 0.995, "relative_frobenius_error": 0.05, "off_diagonal_correlation": 0.8}, {"lambda": 0.998, "relative_frobenius_error": 0.04, "off_diagonal_correlation": 0.86}]},
        "G2": {"rotation_contrast_radians": 0.26, "amplification_log_contrast": 2.3},
        "G3": {"relative_improvement": 0.59},
        "G4": {"median_rho": 0.99, "median_F5": 1.07, "maximum_F5": 1.33, "median_direct_rls_relative_gap": 0.08}
    }
    config = {
        "variance_floor": 1e-8, "log_chi_square_mean": -1.2704, "observation_variance": 4.9348,
        "ukf_alpha": 0.5, "ukf_beta": 2.0, "ukf_kappa": 0.0, "ridge_prior": 10.0,
        "primary_horizon": 5, "secondary_horizon": 20, "rotation_lag_days": 20,
        "principal_subspace_rank": 3, "event_drawdown": -0.15, "event_merge_days": 120,
        "event_forecast_days": 60, "logistic_C": 1.0, "purge_days": 60, "refit_days": 252,
        "bootstrap_block_days": 250, "surrogate_block_days": 60, "surrogate_draws": 499,
        "ema_half_life_days": 10
        , "estimate_start": "1926-01-01", "estimate_end": "1949-12-31",
        "calibration_start": "1950-01-01", "calibration_end": "1969-12-31",
        "portfolio_count": 10, "volatility_window_days": 20, "drawdown_peak_window_days": 250, "correlation_window_days": 60,
        "covariance_window_days": 250, "annualization_days": 252,
        "standardization_floor": 0.05, "standardization_clip": 6.0,
        "minimum_surrogate_p": 0.01, "event_placebo_p_threshold": 0.05,
        "h1_max_abs_spearman": 0.7, "pca_variance_share": 0.9,
        "gates": {
            "ukf_kf_abs_error_max": 1e-10, "finite_difference_abs_error_max": 1e-9,
            "e1_assets_pass_min": 8, "g1_relative_frobenius_error_max": 0.35,
            "g1_off_diagonal_correlation_min": 0.5, "g2_rotation_contrast_min": 0.15,
            "g2_amplification_contrast_min": 0.15,
            "g3_rmse_improvement_min": 0.05, "g4_median_rho_max": 1.05,
            "g4_median_f5_max": 3.0, "g4_max_f5_max": 1000.0,
            "g4_direct_rls_gap_max": 0.5
        }
    }
    protocol = {
        "tier": "diagnostic",
        "config": config,
        "development_attempts": 4,
        "analysis_plan": {"rules": {"maximum_attempts": 5}},
        "analysis_plan_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
        "verdict_rules": {"H2": {"support_point_threshold": 0.02}},
        "references": {key: {"year": year} for key, year in (("Julier", 2004), ("HRS", 1994), ("DavisKahan", 1970), ("Wedin", 1972), ("Kunsch", 1989), ("Roy", 2007), ("Kritzman", 2011), ("Scheffer", 2009))}
    }
    snapshot = {"retrieval_utc": "2026-11-05T22:41:44+00:00", "archive_sha256": "c" * 64, "member_sha256": "d" * 64, "derived_returns_sha256": "e" * 64}
    reproducibility = {"executable_sha256": "f" * 64, "compiler": "MSVC", "cpp_standard": "C++17", "cmake": "4", "python": "3.12", "numpy": "2", "scipy": "1.16", "pybind11": "3", "Eigen": "not used"}
    for name, value in (("smoke_validation.json", validation), ("smoke_study.json", study), ("smoke_protocol.json", protocol), ("smoke_snapshot.json", snapshot), ("smoke_reproducibility.json", reproducibility)):
        write_json(ROOT / "results" / name, value)
    render(validation, study, protocol, snapshot, reproducibility, out)
    shutil.copyfile(ROOT / "paper" / "main.tex", out / "main.tex")
    shutil.copyfile(ROOT / "paper" / "supplement.tex", out / "supplement.tex")
    create_figures(ROOT / "results" / "smoke_study.json", ROOT / "results" / "smoke_protocol.json", out / "figures")
    tectonic = os.environ.get("CODEX_TECTONIC_PATH") or shutil.which("tectonic")
    if not tectonic:
        raise RuntimeError("Tectonic was not found")
    for source in ("main.tex", "supplement.tex"):
        subprocess.run([tectonic, "-o", str(out), source], cwd=out, check=True)
    for name in ("smoke_validation.json", "smoke_study.json", "smoke_protocol.json", "smoke_snapshot.json", "smoke_reproducibility.json"):
        (ROOT / "results" / name).unlink()
    (ROOT / "results" / "claim_map.json").unlink(missing_ok=True)
    shutil.rmtree(out)
    return True


if __name__ == "__main__":
    run_smoke()
