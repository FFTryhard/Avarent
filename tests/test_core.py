import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import market_core
from common import hash_verified, load_config
from gates import finite_difference_gate, numerical_safety_gate
from metrics import dynamic_metrics


class CoreTests(unittest.TestCase):
    def test_exp_helper_clamps(self):
        self.assertAlmostEqual(market_core.bounded_exp(-100.0), np.exp(-30.0))
        self.assertAlmostEqual(market_core.bounded_exp(100.0), np.exp(30.0))

    def test_unscented_linear_filter_matches_kalman(self):
        config = load_config()
        observations = np.random.default_rng(config["seed"]).normal(-11.0, 2.0, size=500)
        error = market_core.ukf_kf_error(observations.tolist(), -10.0, 0.95, 0.2, config["observation_variance"], config["ukf_alpha"], config["ukf_beta"], config["ukf_kappa"])
        self.assertLessEqual(error, 1e-10)

    def test_finite_difference_recovers_transition(self):
        self.assertLessEqual(finite_difference_gate(), 1e-9)

    def test_numerical_safety_panels(self):
        result = numerical_safety_gate(load_config())
        self.assertEqual(len(result["numerical_safety_cases"]), 4)

    def test_stationary_spectral_metrics(self):
        matrices = np.repeat((0.8 * np.eye(10))[None, :, :], 80, axis=0)
        result = dynamic_metrics(matrices, 5, 20)
        self.assertEqual(float(np.max(result["R"])), 0.0)
        self.assertAlmostEqual(float(np.median(result["F"])), 0.8 ** 5, places=12)
        self.assertAlmostEqual(float(np.median(result["H"])), 10.0, places=12)

    def test_core_accepts_secondary_panel_dimension(self):
        config = load_config()
        generator = np.random.default_rng(29)
        returns = generator.normal(0.0, 0.01, size=(300, 17))
        parameters = {"mu": np.full(17, -10.0), "phi": np.full(17, 0.95), "q": np.full(17, 0.2)}
        from gates import core_run
        output = core_run(returns, parameters, 0.998, config)
        self.assertEqual(output["J"].shape, (300, 17, 17))

    def test_zero_missing_skips_only_the_observation_update(self):
        config = load_config()
        generator = np.random.default_rng(31)
        returns = generator.normal(0.0, 0.01, size=(300, 10))
        returns[123, 3] = 0.0
        parameters = {"mu": np.full(10, -10.0), "phi": np.full(10, 0.95), "q": np.full(10, 0.2)}
        from gates import core_run
        output = core_run(returns, parameters, 0.998, config, zero_as_missing=True)
        expected = parameters["mu"][3] + parameters["phi"][3] * (output["h"][122, 3] - parameters["mu"][3])
        self.assertAlmostEqual(float(output["h"][123, 3]), float(expected), places=12)
        self.assertTrue(np.isfinite(output["h"]).all())

    def test_plan_digest(self):
        digest = (ROOT / "results" / "analysis_plan.sha256").read_text(encoding="ascii").strip()
        self.assertEqual(hash_verified(ROOT / "results" / "analysis_plan.json", digest), digest)

    def test_config_contains_referenced_keys(self):
        config = load_config()
        keys = set()
        for path in (ROOT / "python").glob("*.py"):
            keys.update(re.findall(r'config\["([^"]+)"\]', path.read_text(encoding="utf-8")))
        self.assertFalse(keys.difference(config))

    def test_result_json_has_tier(self):
        for path in (ROOT / "results").glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("tier", value, path.name)


if __name__ == "__main__":
    unittest.main()
