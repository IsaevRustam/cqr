"""
Leak-freedom tests for train-only bandwidth selection (protocol
``train_selected``): the source greps and the synthetic runtime check from
rebuttal.verify_train_only_selection, run as unit tests.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rebuttal.verify_train_only_selection import (  # noqa: E402
    runtime_check_failures,
    static_check_failures,
)


class TrainOnlySelectionTests(unittest.TestCase):
    def test_static_no_leakage(self):
        self.assertEqual(static_check_failures(), [])

    def test_runtime_selection_on_synthetic_data(self):
        self.assertEqual(runtime_check_failures(), [])


if __name__ == "__main__":
    unittest.main()
