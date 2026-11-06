import json
from pathlib import Path

import numpy as np

from common import ROOT, hash_verified, sha256_file
from common import sha256_file


def fmt(value, digits=4):
    if value is None:
        return "--"
    if isinstance(value, (float, np.floating)):
        return format(float(value), f".{digits}g")
    return str(value)


def math_fmt(value, digits=4):
    text = fmt(value, digits).lower()
    if "e" in text:
        mantissa, exponent = text.split("e", 1)
        return rf"{mantissa}\times10^{{{int(exponent)}}}"
    return text


def escape(value):
    return str(value).replace("\\", "\\textbackslash{}").replace("%", "\\%").replace("_", "\\_").replace("&", "\\&")


def render(validation, study, protocol, snapshot, reproducibility, destination):
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    h1 = study["H1"]
    h2 = study["H2"]["bootstrap"]
    g1 = next(row for row in validation["G1"]["trials"] if float(row["lambda"]) == float(validation["G1"]["selected_lambda"]))
    values = h2["point_estimates"]
    intervals = h2["holm_adjusted_95_intervals"]
    adjusted_p = h2["holm_adjusted_p_at_zero"]
    if h1["verdict"] == "unsupported":
        reason = "all three adjusted block-shuffle probabilities exceed the frozen unsupported threshold"
    elif h1["verdict"] == "supported":
        reason = "all four frozen conditions pass"
    else:
        reason = "the frozen conditions do not jointly pass and do not meet the unsupported rule"
    percentile = h1["direction"]["median_haar_percentile"]
    if percentile <= 0.1:
        direction = "more closely aligned than most Haar reference subspaces"
    elif percentile >= 0.9:
        direction = "more separated than most Haar reference subspaces"
    else:
        direction = "about as aligned as Haar reference subspaces"
    macros = {
        "HoneVerdict": h1["verdict"],
        "HtwoVerdict": study["H2"]["verdict"],
        "SampleStart": study["data_period"]["start"],
        "SampleEnd": study["data_period"]["end"],
        "TestStart": study["data_period"]["test_start"],
        "EstimationStart": study["data_period"]["start"],
        "EstimationEnd": protocol["config"]["estimate_end"],
        "CalibrationStart": protocol["config"]["calibration_start"],
        "CalibrationEnd": protocol["config"]["calibration_end"],
        "EventClusters": study["event_clusters"],
        "AnchorCount": study["anchor_count"],
        "SelectedLambda": fmt(study["selected_lambda"]),
        "VarianceFloor": fmt(protocol["config"]["variance_floor"]),
        "LogChiMean": fmt(protocol["config"]["log_chi_square_mean"]),
        "ObservationVariance": fmt(protocol["config"]["observation_variance"]),
        "UKFAlpha": fmt(protocol["config"]["ukf_alpha"]),
        "UKFBeta": fmt(protocol["config"]["ukf_beta"]),
        "UKFKappa": fmt(protocol["config"]["ukf_kappa"]),
        "RidgePrior": fmt(protocol["config"]["ridge_prior"]),
        "PrimaryHorizon": protocol["config"]["primary_horizon"],
        "SecondaryHorizon": protocol["config"]["secondary_horizon"],
        "RotationLag": protocol["config"]["rotation_lag_days"],
        "SubspaceRank": protocol["config"]["principal_subspace_rank"],
        "VolWindow": protocol["config"]["volatility_window_days"],
        "CorrelationWindow": protocol["config"]["correlation_window_days"],
        "CovarianceWindow": protocol["config"]["covariance_window_days"],
        "PeakWindow": protocol["config"]["drawdown_peak_window_days"],
        "AnnualizationDays": protocol["config"]["annualization_days"],
        "StandardFloor": fmt(protocol["config"]["standardization_floor"]),
        "StandardClip": fmt(protocol["config"]["standardization_clip"]),
        "EventThreshold": fmt(protocol["config"]["event_drawdown"] * 100) + r"\%",
        "MergeDays": protocol["config"]["event_merge_days"],
        "ForecastDays": protocol["config"]["event_forecast_days"],
        "LogisticC": fmt(protocol["config"]["logistic_C"]),
        "PurgeDays": protocol["config"]["purge_days"],
        "RefitDays": protocol["config"]["refit_days"],
        "BootstrapBlock": protocol["config"]["bootstrap_block_days"],
        "SurrogateBlock": protocol["config"]["surrogate_block_days"],
        "MinimumSurrogateP": fmt(1.0 / (protocol["config"]["surrogate_draws"] + 1)),
        "SupportThreshold": fmt(protocol["verdict_rules"]["H2"]["support_point_threshold"]),
        "AttemptCount": protocol["development_attempts"],
        "UKFError": fmt(validation["U"]["ukf_kf_abs_error"], 3),
        "FiniteDifferenceError": fmt(validation["U"]["finite_difference_abs_error"], 3),
        "EOnePassCount": validation["E1"]["assets_passing"],
        "GOneRelativeError": fmt(g1["relative_frobenius_error"]),
        "GOneOffDiagonalCorrelation": fmt(g1["off_diagonal_correlation"]),
        "GTwoRotationContrast": fmt(validation["G2"]["rotation_contrast_radians"]),
        "GTwoAmplificationContrast": fmt(validation["G2"]["amplification_log_contrast"]),
        "GThreeImprovement": fmt(validation["G3"]["relative_improvement"] * 100) + r"\%",
        "GFourMedianRho": fmt(validation["G4"]["median_rho"]),
        "GFourMedianF": fmt(validation["G4"]["median_F5"]),
        "GFourMaximumF": fmt(validation["G4"]["maximum_F5"]),
        "GFourDirectGap": fmt(validation["G4"]["median_direct_rls_relative_gap"]),
        "DeltaMZero": fmt(values[0]),
        "DeltaMZeroLow": fmt(intervals[0][0]),
        "DeltaMZeroHigh": fmt(intervals[0][1]),
        "DeltaMOne": fmt(values[1]),
        "DeltaMOneLow": fmt(intervals[1][0]),
        "DeltaMOneHigh": fmt(intervals[1][1]),
        "HTwoPZero": fmt(adjusted_p[0]),
        "HTwoPOne": fmt(adjusted_p[1]),
        "HoneAdjpR": fmt(h1["surrogate_p_holm"][0]),
        "HoneAdjpH": fmt(h1["surrogate_p_holm"][1]),
        "HoneAdjpF": fmt(h1["surrogate_p_holm"][2]),
        "HoneEventAdjpR": fmt(h1["event_placebo_p_holm"][0]),
        "HoneEventAdjpH": fmt(h1["event_placebo_p_holm"][1]),
        "HoneEventAdjpF": fmt(h1["event_placebo_p_holm"][2]),
        "HoneEventRawpR": fmt(h1["event_placebo_p_raw"][0]),
        "HoneEventRawpH": fmt(h1["event_placebo_p_raw"][1]),
        "HoneEventRawpF": fmt(h1["event_placebo_p_raw"][2]),
        "MaxSpearman": fmt(h1["max_abs_spearman"]),
        "PCANinetyComponents": h1["pca_components_for_90_percent"],
        "DirectionPercentile": fmt(percentile * 100) + r"\%",
        "DirectionDateCount": h1["direction"]["date_count"],
        "WedinShare": fmt(h1["wedin"]["informative_share"] * 100, 2) + r"\%",
        "SurrogateThreshold": fmt(protocol["config"]["minimum_surrogate_p"]),
        "EventPlaceboThreshold": fmt(protocol["config"]["event_placebo_p_threshold"]),
        "HoneUnsupportedRule": protocol["verdict_rules"]["H1"]["unsupported"],
        "DirectionInterpretation": escape("The covariance and susceptibility spaces are " + direction),
        "DirectionVarianceText": escape(direction),
        "DirectionSusceptibilityText": escape(direction),
        "HoneReason": escape(reason),
        "MedianF": fmt(np.median(study["time_series"]["F5"])),
        "MedianH": fmt(np.median(study["time_series"]["H"])),
        "RetrievalDate": snapshot["retrieval_utc"][:10],
        "SecondaryRetrievalDate": json.loads((ROOT / "results" / "exploratory" / "secondary_data_snapshot.json").read_text(encoding="utf-8"))["retrieval_utc"][:10],
        "PlanHashShort": protocol["analysis_plan_sha256"][:12],
        "ProtocolHashShort": protocol["protocol_sha256"][:12],
        "DataArchiveHashShort": snapshot["archive_sha256"][:12],
        "ExecutableHashShort": reproducibility["executable_sha256"][:12],
        "CompilerName": escape(reproducibility["compiler"]),
        "CppStandard": escape(reproducibility["cpp_standard"]),
        "CmakeVersion": escape(reproducibility["cmake"]),
        "PythonVersion": escape(reproducibility["python"]),
        "NumpyVersion": escape(reproducibility["numpy"]),
        "ScipyVersion": escape(reproducibility["scipy"]),
        "PybindVersion": escape(reproducibility["pybind11"]),
        "EigenStatus": "not used; small fixed-dimension kernels are implemented in C++"
    }
    for key, item in protocol["references"].items():
        if isinstance(item, dict) and "year" in item:
            macros["Year" + key] = item["year"]
    rules = protocol["analysis_plan"]["rules"]
    gate_cfg = protocol["config"]["gates"]
    gates = [
        ("U", f"UKF error {fmt(validation['U']['ukf_kf_abs_error'], 2)}; finite difference {fmt(validation['U']['finite_difference_abs_error'], 2)}", f"$\\leq {math_fmt(gate_cfg['ukf_kf_abs_error_max'])}$; $\\leq {math_fmt(gate_cfg['finite_difference_abs_error_max'])}$"),
        ("E1", f"{validation['E1']['assets_passing']}/{protocol['config']['portfolio_count']} assets", f"$\\geq {fmt(gate_cfg['e1_assets_pass_min'])}/{protocol['config']['portfolio_count']}$ and no box boundary"),
        ("G1", f"error {fmt(g1['relative_frobenius_error'])}; correlation {fmt(g1['off_diagonal_correlation'])}", f"$\\leq {fmt(gate_cfg['g1_relative_frobenius_error_max'])}$; $\\geq {fmt(gate_cfg['g1_off_diagonal_correlation_min'])}$"),
        ("G2", f"rotation {fmt(validation['G2']['rotation_contrast_radians'])}; log amplification {fmt(validation['G2']['amplification_log_contrast'])}", f"$\\geq {fmt(gate_cfg['g2_rotation_contrast_min'])}$; $\\geq {fmt(gate_cfg['g2_amplification_contrast_min'])}$"),
        ("G3", f"RMSE improvement {fmt(validation['G3']['relative_improvement'] * 100)}\\%", f"$\\geq {fmt(gate_cfg['g3_rmse_improvement_min'] * 100)}\\%$"),
        ("G4", f"median rho {fmt(validation['G4']['median_rho'])}; median F5 {fmt(validation['G4']['median_F5'])}; max F5 {fmt(validation['G4']['maximum_F5'])}; direct gap {fmt(validation['G4']['median_direct_rls_relative_gap'])}", f"$\\leq {fmt(gate_cfg['g4_median_rho_max'])}$; $\\leq {fmt(gate_cfg['g4_median_f5_max'])}$; $\\leq {fmt(gate_cfg['g4_max_f5_max'])}$; $\\leq {fmt(gate_cfg['g4_direct_rls_gap_max'])}$"),
        ("Attempts", str(protocol["development_attempts"]), f"at most {rules['maximum_attempts']}")
    ]
    gate_rows = [f"{name} & {value} & {threshold} & pass " + r"\\" for name, value, threshold in gates]
    h2_rows = [f"{name} & {fmt(values[i])} & [{fmt(intervals[i][0])}, {fmt(intervals[i][1])}] & {fmt(adjusted_p[i])} & {study['event_clusters']} & {study['anchor_count']} " + r"\\" for i, name in enumerate(("M0+F vs M0", "M1+F vs M1"))]
    placebo_pairs = "; ".join(
        f"{signal}: {fmt(h1['event_placebo_p_raw'][i], 3)}/{fmt(h1['event_placebo_p_holm'][i], 3)}"
        for i, signal in enumerate(("R", "H", "F"))
    )
    conditions = [
        ("(a) Surrogates", f"$p_R={fmt(h1['surrogate_p_holm'][0])}$, $p_H={fmt(h1['surrogate_p_holm'][1])}$, $p_F={fmt(h1['surrogate_p_holm'][2])}$", f"$p\\leq {fmt(protocol['config']['minimum_surrogate_p'])}$ for all", h1["condition_a"]),
        ("(b) Pre-onset", f"raw/Holm: {placebo_pairs}", f"adjusted $p<{fmt(protocol['config']['event_placebo_p_threshold'])}$ for one or more", h1["condition_b"]),
        ("(c) Redundancy", f"{fmt(h1['max_abs_spearman'], 4)} < {fmt(protocol['config']['h1_max_abs_spearman'])}", f"maximum absolute Spearman correlation", h1["condition_c"]),
        ("(d) Dimension", f"{h1['pca_components_for_90_percent']} components", f"$\\geq {protocol['config']['principal_subspace_rank']}$ for {fmt(protocol['config']['pca_variance_share'] * 100)}\\%", h1["condition_d"])
    ]
    h1_rows = [f"{name} & {value} & {threshold} & {'pass' if passed else 'fail'} " + r"\\" for name, value, threshold, passed in conditions]
    robust = study["robustness"]
    robust_rows = [
        f"Primary, lambda {fmt(study['selected_lambda'])} & {fmt(np.median(study['time_series']['R']))} & {fmt(np.median(study['time_series']['H']))} & {fmt(np.median(study['time_series']['F5']))} " + r"\\",
        f"Alternate lambda {fmt(robust['alternative_lambda']['lambda'])} & {fmt(robust['alternative_lambda']['median_R'])} & {fmt(robust['alternative_lambda']['median_H'])} & {fmt(robust['alternative_lambda']['median_F5'])} " + r"\\",
        f"EMA before RLS & {fmt(robust['ema_before_rls']['median_R'])} & {fmt(robust['ema_before_rls']['median_H'])} & {fmt(robust['ema_before_rls']['median_F5'])} " + r"\\",
        f"Horizon tau {protocol['config']['secondary_horizon']} & -- & -- & {fmt(robust['horizon_20']['median_F20'])} " + r"\\"
    ]
    f_values = np.asarray(study["time_series"]["F5"])
    wedin = h1["wedin"]
    diagnostic_rows = [
        f"Wedin median singular gap & {fmt(wedin['median_gap'])} & Subspace separation " + r"\\",
        f"Wedin median perturbation norm & {fmt(wedin['median_perturbation_norm'])} & Map change over lag " + r"\\",
        f"Informative Wedin share & {fmt(wedin['informative_share'] * 100)}\\% & Bound below one " + r"\\",
        f"F5 median and interquartile range & {fmt(np.median(f_values))}; [{fmt(np.quantile(f_values, .25))}, {fmt(np.quantile(f_values, .75))}] & Test-period distribution " + r"\\"
    ]

    gate_definitions = [
        ("U", "Numerical equivalence and local-Jacobian checks", f"UKF/Kalman error $\\leq {math_fmt(gate_cfg['ukf_kf_abs_error_max'])}$; finite-difference error $\\leq {math_fmt(gate_cfg['finite_difference_abs_error_max'])}$"),
        ("E1", "Variance-parameter recovery across assets", f"At least {gate_cfg['e1_assets_pass_min']}/{protocol['config']['portfolio_count']} assets; $|\\Delta\\phi|\\leq {fmt(rules['E1_phi_abs_error_max'])}$; $q$ ratio $\\leq {fmt(rules['E1_q_ratio_max'])}$; no box-boundary estimate"),
        ("G1", "Recovery of a planted heterogeneous transition map", f"Relative Frobenius error $\\leq {fmt(gate_cfg['g1_relative_frobenius_error_max'])}$; off-diagonal correlation $\\geq {fmt(gate_cfg['g1_off_diagonal_correlation_min'])}$"),
        ("G2", "Separation of rotating and amplifying synthetic regimes", f"Rotation contrast $\\geq {fmt(rules['G2_rotation_contrast_min'])}$ radians; log-amplification contrast $\\geq {fmt(rules['G2_amplification_contrast_min'])}$; stationary rotation $\\leq {math_fmt(rules['G2_stationary_rotation_max'])}$"),
        ("G3", "Held-out filtering improvement over a constant-state comparator", f"Relative RMSE improvement $\\geq {fmt(rules['G3_rmse_improvement_min'] * 100)}\\%$"),
        ("G4", "Historical boundedness and direct-RLS comparison", f"Median $\\rho\\leq {fmt(rules['G4_median_rho_max'])}$; median $F_5\\leq {fmt(rules['G4_median_F5_max'])}$; maximum $F_5\\leq {fmt(rules['G4_max_F5_max'])}$; direct-RLS relative gap $\\leq {fmt(rules['G4_direct_RLS_relative_gap_max'])}$")
    ]
    gate_definition_rows = [f"{name} & {meaning} & {threshold} " + r"\\" for name, meaning, threshold in gate_definitions]
    hash_rows = []
    study_digest = sha256_file(ROOT / "results" / "study.json")
    exploratory_run = json.loads((ROOT / "results" / "exploratory" / "reproducibility.json").read_text(encoding="utf-8"))
    secondary_snapshot = json.loads((ROOT / "results" / "exploratory" / "secondary_data_snapshot.json").read_text(encoding="utf-8"))
    extension_hash_path = ROOT / "results" / "exploratory" / "block_length_extension_plan.sha256"
    extension_digest = extension_hash_path.read_text(encoding="ascii").strip()
    haar_extension_digest = (ROOT / "results" / "exploratory" / "haar_dependence_extension_plan.sha256").read_text(encoding="ascii").strip()
    precision_run = json.loads((ROOT / "results" / "exploratory" / "reproducibility.json").read_text(encoding="utf-8"))
    precision_folder = ROOT / "results" / "exploratory"
    for label, digest in (
        ("Source archive", snapshot["archive_sha256"]),
        ("Extracted member", snapshot["member_sha256"]),
        ("Derived returns", snapshot["derived_returns_sha256"]),
        ("Frozen study JSON", study_digest),
        ("Analysis plan", protocol["analysis_plan_sha256"]),
        ("Post-freeze plan", (ROOT / "results" / "exploratory" / "analysis_plan.sha256").read_text(encoding="ascii").strip()),
        ("Block-length plan", extension_digest),
        ("Haar dependence plan", haar_extension_digest),
        ("F precision plan", precision_run["precision_plan_sha256"]),
        ("F precision runner", precision_run["precision_runner_sha256"]),
        ("F primary precision output", sha256_file(precision_folder / "primary_F_surrogate_precision.json")),
        ("Zero-variant precision output", sha256_file(precision_folder / "zero_variants.json")),
        ("Subperiod precision output", sha256_file(precision_folder / "subperiod_F_surrogate.json")),
        ("Lambda-grid precision output", sha256_file(precision_folder / "lambda_grid.json")),
        ("F precision comparison", sha256_file(precision_folder / "f_surrogate_precision_comparison.json")),
        ("Protocol amendment log", sha256_file(ROOT / "results" / "protocol_amendments.json")),
        ("Protocol", protocol["protocol_sha256"]),
        ("17-industry source archive", secondary_snapshot["archive_sha256"]),
        ("17-industry extracted member", secondary_snapshot["member_sha256"]),
        ("17-industry derived returns", secondary_snapshot["derived_returns_sha256"]),
        ("Frozen core executable", reproducibility["executable_sha256"]),
        ("Exploratory core source", exploratory_run["core_source_sha256"]),
        ("Exploratory core executable", exploratory_run["core_binary_sha256"]),
        ("Exploratory analysis script", exploratory_run["exploratory_script_sha256"])
    ):
        hash_lines = r"\shortstack[l]{\texttt{" + digest[:32] + r"}\\\texttt{" + digest[32:] + r"}}"
        hash_rows.append(f"{label} & {hash_lines} " + r"\\")
    table_values = {
        "GateRows": "\n".join(gate_rows),
        "HtwoRows": "\n".join(h2_rows),
        "HoneRows": "\n".join(h1_rows),
        "RobustnessRows": "\n".join(robust_rows),
        "DiagnosticRows": "\n".join(diagnostic_rows),
        "GateDefinitionRows": "\n".join(gate_definition_rows),
        "HashRows": "\n".join(hash_rows)
    }
    lines = [f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macros.items()]
    tables = [f"\\newcommand{{\\{key}}}{{\n{value}\n}}" for key, value in table_values.items()]
    (output / "results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (output / "tables.tex").write_text("\n".join(tables) + "\n", encoding="utf-8", newline="\n")
    return macros, table_values


def render_exploratory(output):
    folder = ROOT / "results" / "exploratory"

    def read(name):
        return json.loads((folder / name).read_text(encoding="utf-8"))

    def interval(values):
        return f"[{fmt(values[0], 3)}, {fmt(values[1], 3)}]"

    def interval_percent(values):
        return f"[{fmt(100 * values[0], 3)}\\%, {fmt(100 * values[1], 3)}\\%]"

    counts = read("zero_counts.json")["10_industry"]
    period_labels = {
        "estimation_1926_1949": "Estimation (1926--1949)",
        "calibration_1950_1969": "Calibration (1950--1969)",
        "test_1970_2026_09": "Test (1970--2026-09)",
    }
    zero_count_rows = []
    for key, label in period_labels.items():
        row = counts[key]
        zero_count_rows.append(
            f"{label} & {row['days']} & {row['zero_cells']} / {row['valid_cells']} & "
            f"{fmt(100 * row['zero_share'], 3)}\\% & {row['days_with_any_zero']} " + r"\\"
        )

    zero = read("zero_variants.json")
    zero_rows = []
    for item in zero["variants"]:
        h1 = item["h1"]
        h2 = item["h2"]
        delta_interval = h2.get("auc_intervals_95", {}).get("delta_auc_M1_F", [None, None])
        zero_rows.append(
            f"{escape(item['variant'].replace('_', ' '))} & "
            f"{fmt(h1['surrogate_p_holm'][0])}, {fmt(h1['surrogate_p_holm'][1])}, {fmt(h1['surrogate_p_holm'][2])} & "
            f"{fmt(h2.get('delta_auc_M1_F'))} & {interval(delta_interval)} & "
            f"{h2.get('event_clusters', '--')} & {h2.get('anchor_count', '--')} " + r"\\"
        )

    factor = read("factor_aligned_null.json")
    factor_summary = (
        f"{fmt(100 * factor['median_percentile'], 3)}\\% "
        f"(95\\% block interval {interval_percent(factor['median_percentile_block_bootstrap_95'])}); "
        f"effective sample size {fmt(factor['autocorrelation']['effective_n'], 4)} "
        f"of {factor['date_count']} dates"
    )
    haar = read("haar_dependence.json")
    haar_summary = (
        f"median {fmt(100 * haar['recomputed_median_percentile'], 3)}\\% "
        f"(95\\% block interval {interval_percent(haar['block_bootstrap_95_interval'])}); "
        f"effective sample size {fmt(haar['autocorrelation']['effective_n'], 4)} "
        f"of {haar['date_count']} dates"
    )

    h2 = read("h2_prediction_metrics.json")
    prediction = h2["primary_comparison"]
    auc = prediction["auc"]
    auc_intervals = prediction["auc_intervals_95"]
    alerts = prediction["alert_rate_5pct"]
    refit = h2["refit_aware"]
    h2_grid = read("h2_sensitivity.json")["grid"]
    sensitivity_rows = []
    for row in h2_grid:
        if row.get("status") != "estimated":
            sensitivity_rows.append(
                f"{fmt(100 * row['threshold'], 2)}\\% & {row['horizon_sessions']} & {fmt(row['logistic_C'], 2)} & "
                f"{row.get('event_clusters', '--')} & {row.get('anchor_count', '--')} & not estimable & -- & -- & -- & -- " + r"\\"
            )
            continue
        values = row["auc"]
        limits = row.get("auc_intervals_95", {}).get("delta_auc_M1_F", [None, None])
        sensitivity_rows.append(
            f"{fmt(100 * row['threshold'], 2)}\\% & {row['horizon_sessions']} & {fmt(row['logistic_C'], 2)} & "
            f"{row['event_clusters']} & {row['anchor_count']} & {fmt(values['M1'], 3)} & "
            f"{fmt(values['M1_F'], 3)} & {fmt(row['delta_auc_M1_F'], 3)} & {interval(limits)} & "
            f"{fmt(values['EWMA'], 3)} " + r"\\"
        )

    block = read("h2_block_length_sensitivity.json")
    block_rows = []
    for row in block["results"]:
        block_rows.append(
            f"{row['block_length_sessions']} & {fmt(row['auc']['M1'], 3)} & {fmt(row['auc']['M1_F'], 3)} & "
            f"{fmt(row['delta_auc_M1_F'], 3)} & {interval(row['auc_intervals_95']['delta_auc_M1_F'])} " + r"\\"
        )

    subperiod = read("subperiod_F_surrogate.json")["results"]
    subperiod_rows = []
    for value in subperiod.values():
        subperiod_rows.append(
            f"{value['start'][:4]}--{value['end'][:4]} & {value['observed_median_F']:.2f} & "
            f"{value.get('null_median_F', value.get('null_median')):.2f} & {fmt(value['raw_p'], 4)} & "
            f"{fmt(value['holm_adjusted_p'], 3)} " + r"\\"
        )

    lambda_grid = read("lambda_grid.json")["grid"]
    lambda_rows = []
    for row in lambda_grid:
        distributions = row["distribution_summary"]
        lambda_rows.append(
            f"{row['lambda']:.3f} & {fmt(distributions['R']['median'], 3)} [{fmt(distributions['R']['p25'], 3)}, {fmt(distributions['R']['p75'], 3)}] & "
            f"{fmt(distributions['H']['median'], 3)} [{fmt(distributions['H']['p25'], 3)}, {fmt(distributions['H']['p75'], 3)}] & "
            f"{fmt(distributions['F']['median'], 3)} [{fmt(distributions['F']['p25'], 3)}, {fmt(distributions['F']['p75'], 3)}] & "
            f"{', '.join(fmt(value, 3) for value in row['surrogate_p_holm_R_H_F'])} " + r"\\"
        )

    panel = read("panel_17_industry.json")
    panel_h2 = panel["H2"]
    panel_summary = (
        f"I find {panel['test_event_clusters']} test-period event clusters in the 17-industry aggregation and "
        f"match {panel['event_onsets_matched_within_120_sessions']} to the {panel['primary_10_industry_event_clusters']} "
        f"ten-industry events within 120 sessions ({fmt(100 * panel['event_overlap_share_of_17_panel'], 3)}\\% of the 17-industry events). "
        f"Its $F$ surrogate comparison gives raw $p={fmt(panel['F_surrogate']['raw_p'], 3)}$ from "
        f"{panel['F_surrogate']['draws']} draws. I estimate $\\Delta$AUC for $M_1+F$ versus $M_1$ as "
        f"{fmt(panel_h2.get('delta_auc_M1_F'), 3)} (95\\% interval {interval(panel_h2['auc_intervals_95']['delta_auc_M1_F'])}) with "
        f"{panel_h2.get('anchor_count', '--')} anchors."
    )

    precision = read("f_surrogate_precision_comparison.json")["rerun_F_values"]
    primary_precision = precision["primary"]
    floor_precision = precision["variance_floor_1e-6"]
    missing_precision = precision["zero_returns_missing"]
    early_precision = precision["1970_1999"]
    late_precision = precision["2000_2026_09"]
    lambda_precision = [precision[key] for key in ("0.99", "0.995", "0.998", "0.999")]
    scale_grid = read("lambda_grid.json")["grid"]
    f_lambda_medians = {round(row["lambda"], 3): row["distribution_summary"]["F"]["median"] for row in scale_grid}
    precision_macros = {
        "FPrimaryObserved": fmt(primary_precision["observed_median_F"], 3),
        "FPrimaryNull": fmt(primary_precision["null_median_F"], 3),
        "FPrimaryRawP": fmt(primary_precision["raw_p"], 4),
        "FPrimaryHolmP": fmt(primary_precision["holm_adjusted_p"], 4),
        "FFloorObserved": fmt(floor_precision["observed_median_F"], 3),
        "FFloorNull": fmt(floor_precision["null_median_F"], 3),
        "FFloorRawP": fmt(floor_precision["raw_p"], 4),
        "FFloorHolmP": fmt(floor_precision["holm_adjusted_p"], 4),
        "FMissingObserved": fmt(missing_precision["observed_median_F"], 3),
        "FMissingNull": fmt(missing_precision["null_median_F"], 3),
        "FMissingRawP": fmt(missing_precision["raw_p"], 4),
        "FMissingHolmP": fmt(missing_precision["holm_adjusted_p"], 4),
        "FSubperiodSummary": (
            f"1970--1999: {early_precision['observed_median_F']:.2f} vs {early_precision['null_median_F']:.2f} "
            f"(raw/Holm $p={fmt(early_precision['raw_p'], 4)}/{fmt(early_precision['holm_adjusted_p'], 4)}$); "
            f"2000--2026-09: {late_precision['observed_median_F']:.2f} vs {late_precision['null_median_F']:.2f} "
            f"(raw/Holm $p={fmt(late_precision['raw_p'], 4)}/{fmt(late_precision['holm_adjusted_p'], 4)}$)"
        ),
        "FLambdaHolmSummary": ", ".join(fmt(value["holm_adjusted_p"], 4) for value in lambda_precision),
        "FScaleLambdaRange": f"{fmt(f_lambda_medians[0.999], 3)} at 0.999 to {fmt(f_lambda_medians[0.99], 3)} at 0.990",
        "FPrecisionDraws": "4,999"
    }

    macros = {
        "ZeroCountRows": "\n".join(zero_count_rows),
        "ZeroVariantRows": "\n".join(zero_rows),
        "FactorAlignedSummary": factor_summary,
        "HaarDependenceSummary": haar_summary,
        "FactorAlignedMedianPercentile": fmt(100 * factor["median_percentile"], 3) + r"\%",
        "HaarMedianPercentile": fmt(100 * haar["recomputed_median_percentile"], 3) + r"\%",
        "HTwoSensitivityRows": "\n".join(sensitivity_rows),
        "BlockLengthRows": "\n".join(block_rows),
        "SubperiodRows": "\n".join(subperiod_rows),
        "LambdaRows": "\n".join(lambda_rows),
        "PanelSeventeenSummary": panel_summary,
        "HTwoNoSkillAUC": fmt(prediction["no_skill_auc"], 3),
        "HTwoEWMAAUC": fmt(auc["EWMA"], 3),
        "HTwoEWMAInterval": interval(auc_intervals["EWMA"]),
        "HTwoMOneAUC": fmt(auc["M1"], 3),
        "HTwoMOneInterval": interval(auc_intervals["M1"]),
        "HTwoMOneFAUC": fmt(auc["M1_F"], 3),
        "HTwoMOneFInterval": interval(auc_intervals["M1_F"]),
        "HTwoDeltaAUC": fmt(prediction["delta_auc_M1_F"], 3),
        "HTwoDeltaInterval": interval(auc_intervals["delta_auc_M1_F"]),
        "HTwoAnchorCount": prediction["anchor_count"],
        "HTwoPrecisionMOne": fmt(alerts["M1"]["precision"], 3),
        "HTwoRecallMOne": fmt(alerts["M1"]["recall"], 3),
        "HTwoPrecisionMOneF": fmt(alerts["M1_F"]["precision"], 3),
        "HTwoRecallMOneF": fmt(alerts["M1_F"]["recall"], 3),
        "HTwoPrecisionEWMA": fmt(alerts["EWMA"]["precision"], 3),
        "HTwoRecallEWMA": fmt(alerts["EWMA"]["recall"], 3),
        "HTwoRefitReplicates": refit.get("replicates_used", 0),
        "HTwoRefitMOneInterval": interval(refit.get("auc_M1_interval_95", [None, None])),
        "HTwoRefitMOneFInterval": interval(refit.get("auc_M1_F_interval_95", [None, None])),
        "HTwoRefitDeltaInterval": interval(refit.get("delta_auc_M1_F_interval_95", [None, None])),
        "ExtensionPlanHash": block["extension_plan_sha256"],
        **precision_macros,
    }
    text = "\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macros.items()) + "\n"
    (Path(output) / "exploratory.tex").write_text(text, encoding="utf-8", newline="\n")


def defense_text(study, validation, protocol):
    low, high = study["H2"]["bootstrap"]["holm_adjusted_95_intervals"][1]
    return f"""# Project notes

I find H1 **{study['H1']['verdict']}** and H2 **{study['H2']['verdict']}** under the frozen rules. Adding F to the richer baseline changes AUC by {fmt(study['H2']['bootstrap']['point_estimates'][1])}, with its adjusted interval [{fmt(low)}, {fmt(high)}]. I report {study['event_clusters']} test-period event clusters and {study['anchor_count']} common forecast anchors.

I use a Kalman filter for the linear state and observation equations. The original unscented transform produces the same moments in this linear case. I define F as the largest singular-value amplification of the fitted local response over the declared horizon.

I do not claim causality or trading performance. My principal limitations are the small number of drawdown events, the single U.S. return source, and the absence of funding, leverage, and liquidity measures. I disclose the analysis chronology in the paper and amendment record.

I record funding, competing interests, AI assistance, code and data availability, licensing, and the disclaimer in the paper's declarations. I give the clean-clone instructions in README.md.
"""


def generate_documents():
    def read(name):
        return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
    validation = read("validation.json")
    study = read("study.json")
    if sha256_file(ROOT / "results" / "study.json") != "732f6b8b15661ec9b176489c989b5ccba4c09d453d24597aee78c430cdc0d0ae":
        raise RuntimeError("the frozen study JSON hash differs from the recorded primary result")
    protocol = read("protocol.json")
    protocol_hash = (ROOT / "results" / "protocol.sha256").read_text(encoding="ascii").strip()
    hash_verified(ROOT / "results" / "protocol.json", protocol_hash)
    if read("protocol_hash.json")["protocol_sha256"] != protocol_hash:
        raise RuntimeError("protocol hash record does not match the frozen protocol")
    protocol["protocol_sha256"] = protocol_hash
    snapshot = read("data_snapshot.json")
    reproducibility = read("reproducibility.json")
    reproducibility["compiler"] = read("reproducibility_correction.json")["compiler"]
    render(validation, study, protocol, snapshot, reproducibility, ROOT / "paper")
    render_exploratory(ROOT / "paper")
    (ROOT / "DEFENSE.md").write_text(defense_text(study, validation, protocol), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    generate_documents()
