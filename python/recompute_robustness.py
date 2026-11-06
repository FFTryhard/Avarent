import json

import numpy as np

from common import ROOT, load_config, sha256_file, write_json
from dataset import load_derived
from gates import core_run
from metrics import dynamic_metrics
from real_analysis import verify_protocol


def recompute_alternative_lambda():
    protocol = verify_protocol()
    snapshot = json.loads((ROOT / "results" / "data_snapshot.json").read_text(encoding="utf-8"))
    derived_path = ROOT / "data" / "returns.csv"
    if sha256_file(derived_path) != snapshot["derived_returns_sha256"]:
        raise RuntimeError("derived panel does not match its frozen snapshot")
    config = load_config()
    dates, returns, equal_weight = load_derived()
    validation = json.loads((ROOT / "results" / "validation.json").read_text(encoding="utf-8"))
    study = json.loads((ROOT / "results" / "study.json").read_text(encoding="utf-8"))
    parameters = {key: np.asarray(validation["G4"]["parameters"][key], dtype=np.float64) for key in ("mu", "phi", "q")}
    selected = float(protocol["selected_lambda"])
    alternatives = [value for value in config["lambda_candidates"] if not np.isclose(value, selected, rtol=0.0, atol=1e-12)]
    alternative = float(alternatives[0]) if alternatives else selected
    core = core_run(returns, parameters, alternative, config)
    metrics = dynamic_metrics(core["J"], config["primary_horizon"], config["rotation_lag_days"], config["principal_subspace_rank"])
    test = dates >= np.datetime64(config["test_start"])
    summary = {"lambda": alternative, "median_F5": float(np.median(metrics["F"][test])), "median_R": float(np.median(metrics["R"][test])), "median_H": float(np.median(metrics["H"][test]))}
    study["robustness"]["alternative_lambda"] = summary
    write_json(ROOT / "results" / "study.json", study)
    robustness_path = ROOT / "results" / "robustness.json"
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    robustness["variants"]["alternative_lambda"] = summary
    write_json(robustness_path, robustness)
    return summary


if __name__ == "__main__":
    print(json.dumps(recompute_alternative_lambda(), indent=2))
