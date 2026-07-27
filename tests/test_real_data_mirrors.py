import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "cqr" / "real_data.py"
SPEC = importlib.util.spec_from_file_location("real_data_under_test", MODULE_PATH)
REAL_DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REAL_DATA)
_OPENML_DATAGIT_MIRRORS = REAL_DATA._OPENML_DATAGIT_MIRRORS
_load_verified_openml_mirror = REAL_DATA._load_verified_openml_mirror


class VerifiedOpenMLMirrorTests(unittest.TestCase):
    def test_cached_mirror_is_checked_and_target_selected(self):
        frame = pd.DataFrame({
            "x1": [1.0, 2.0],
            "x2": [3.0, 4.0],
            "y1": [5.0, 6.0],
            "y2": [7.0, 8.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            mirror_dir = cache_dir / "openml_datagit"
            mirror_dir.mkdir()
            commit = "a" * 40
            path = mirror_dir / f"toy_{commit[:12]}.csv"
            frame.to_csv(path, index=False)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            spec = {
                "openml_id": 1,
                "commit": commit,
                "sha256": digest,
                "n_samples": 2,
                "n_features": 2,
                "n_targets": 2,
            }
            with patch.dict(_OPENML_DATAGIT_MIRRORS, {"toy": spec}):
                X, y, info = _load_verified_openml_mirror(
                    "toy", "test", target_col=1, cache_dir=cache_dir,
                )

        np.testing.assert_array_equal(X, np.array([[1, 3], [2, 4]], dtype=np.float32))
        np.testing.assert_array_equal(y, np.array([7, 8], dtype=np.float32))
        self.assertEqual(info["n_samples"], 2)
        self.assertEqual(info["n_features"], 2)

    def test_cached_mirror_checksum_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            mirror_dir = cache_dir / "openml_datagit"
            mirror_dir.mkdir()
            commit = "b" * 40
            path = mirror_dir / f"toy_{commit[:12]}.csv"
            path.write_text("x,y\n1,2\n")
            spec = {
                "openml_id": 1,
                "commit": commit,
                "sha256": "0" * 64,
                "n_samples": 1,
                "n_features": 1,
                "n_targets": 1,
            }
            with patch.dict(_OPENML_DATAGIT_MIRRORS, {"toy": spec}):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    _load_verified_openml_mirror(
                        "toy", "test", cache_dir=cache_dir,
                    )


if __name__ == "__main__":
    unittest.main()
