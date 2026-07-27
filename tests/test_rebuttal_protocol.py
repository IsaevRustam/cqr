import unittest

from rebuttal.protocol import (
    CONFIRMATORY_BANDWIDTH,
    CONFIRMATORY_METHOD_KEYS,
    CONFIRMATORY_SEEDS,
    INNER_SPLIT_FRACS,
    PRIMARY_WGC_BINNING,
    PRIMARY_WGC_GROUPING,
    SELECTED_METHOD_KEYS,
    SELECTED_PROTOCOL,
    SELECTED_SEEDS,
)


class ConfirmatoryProtocolTests(unittest.TestCase):
    def test_frozen_primary_choices(self):
        self.assertEqual(CONFIRMATORY_BANDWIDTH, 1.4)
        self.assertEqual(CONFIRMATORY_SEEDS, list(range(142, 162)))
        self.assertEqual(
            CONFIRMATORY_METHOD_KEYS,
            ("global", "local_fixed_1.4"),
        )
        self.assertEqual(PRIMARY_WGC_GROUPING, "base_interval_width")
        self.assertEqual(PRIMARY_WGC_BINNING, "rank")

    def test_frozen_train_selected_choices(self):
        self.assertEqual(SELECTED_PROTOCOL, "train_selected")
        self.assertEqual(INNER_SPLIT_FRACS, (0.70, 0.15, 0.15))
        self.assertEqual(SELECTED_METHOD_KEYS, ("global", "local_selected"))
        self.assertEqual(SELECTED_SEEDS, list(range(142, 162)))


if __name__ == "__main__":
    unittest.main()
