import importlib.util
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "cqr" / "metrics.py"
SPEC = importlib.util.spec_from_file_location("metrics_under_test", MODULE_PATH)
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)
conditional_coverage = METRICS.conditional_coverage


class ConditionalCoverageTests(unittest.TestCase):
    def test_rank_binning_has_equal_counts_even_with_ties(self):
        y = np.ones(10)
        lo = np.concatenate([np.zeros(5), np.full(5, 2.0)])
        hi = np.concatenate([np.full(5, 2.0), np.full(5, 3.0)])
        grouping = np.zeros(10)

        result = conditional_coverage(
            y, lo, hi, np.zeros((10, 1)),
            n_bins=2,
            min_bin_size=0,
            grouping_values=grouping,
            binning="rank",
        )

        self.assertEqual(result["bin_counts"], [5, 5])
        self.assertEqual(result["best_bin_coverage"], 1.0)
        self.assertEqual(result["worst_bin_coverage"], 0.0)

    def test_grouping_values_must_match_test_length(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            conditional_coverage(
                np.ones(3), np.zeros(3), np.ones(3), np.zeros((3, 1)),
                grouping_values=np.ones(2),
            )


if __name__ == "__main__":
    unittest.main()
