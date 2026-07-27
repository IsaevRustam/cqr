import importlib.util
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "cqr" / "calibration.py"
SPEC = importlib.util.spec_from_file_location("calibration_under_test", MODULE_PATH)
CALIBRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CALIBRATION)
LocalConformalOptimizer = CALIBRATION.LocalConformalOptimizer
global_calibration = CALIBRATION.global_calibration


class GlobalCalibrationTests(unittest.TestCase):
    def test_uses_observed_order_statistic(self):
        scores = np.arange(10, dtype=float)

        # k = ceil(11 * 0.8) = 9, hence the ninth order statistic is 8.
        self.assertEqual(global_calibration(scores, alpha=0.2), 8.0)

    def test_clips_rank_at_largest_observation(self):
        scores = np.array([1.0, 4.0, 2.0])

        self.assertEqual(global_calibration(scores, alpha=0.01), 4.0)


class LocalCalibrationTests(unittest.TestCase):
    def test_returns_kish_ess_from_same_kernel_weights(self):
        calibrator = LocalConformalOptimizer(
            X_cal=np.array([[-0.5], [0.5]]),
            scores=np.array([0.0, 1.0]),
            h=1.0,
        )

        _, ess = calibrator.predict_corrections(
            np.array([[0.0]]), alpha=0.5,
            return_effective_sample_size=True,
        )

        self.assertAlmostEqual(float(ess[0]), 2.0)

    def test_no_neighbor_uses_global_order_statistic_and_zero_ess(self):
        calibrator = LocalConformalOptimizer(
            X_cal=np.array([[0.0], [1.0]]),
            scores=np.array([0.0, 2.0]),
            h=0.1,
        )

        correction, ess = calibrator.predict_corrections(
            np.array([[10.0]]), alpha=0.5,
            return_effective_sample_size=True,
        )

        self.assertEqual(float(correction[0]), 2.0)
        self.assertEqual(float(ess[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
