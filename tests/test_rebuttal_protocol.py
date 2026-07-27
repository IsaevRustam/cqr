import unittest

from rebuttal.protocol import (
    CONFIRMATORY_BANDWIDTH,
    CONFIRMATORY_METHOD_KEYS,
    CONFIRMATORY_SEEDS,
    PRIMARY_WGC_BINNING,
    PRIMARY_WGC_GROUPING,
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


if __name__ == "__main__":
    unittest.main()
