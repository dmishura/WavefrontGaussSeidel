#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
SOURCE = TOOLS / "analyze_psi_temporal_locality.py"
SPEC = importlib.util.spec_from_file_location("psi_temporal_locality", SOURCE)
assert SPEC and SPEC.loader
locality = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = locality
SPEC.loader.exec_module(locality)


class PsiTemporalLocalityTests(unittest.TestCase):
    def test_reuse_gaps_are_between_execution_rows(self) -> None:
        touch_rows = np.array([0, 1, 4, 2, 3], dtype=np.int32)
        touch_lines = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        identity = np.arange(8, dtype=np.int32)
        result = locality.analyze_permutation(
            8, touch_rows, touch_lines, identity
        )
        self.assertEqual(result["psiLinesTouched"], 2)
        self.assertEqual(result["totalLineTouches"], 5)
        self.assertEqual(result["totalReuseEvents"], 3)
        self.assertAlmostEqual(result["reuseGapMean"], 5/3)
        self.assertEqual(result["reuseGapP50"], 1)
        self.assertEqual(result["reuseGapMax"], 3)
        self.assertAlmostEqual(result["reuseWithin32"], 1.0)
        self.assertEqual(result["lifetimeP50"], 1)
        self.assertEqual(result["lifetimeP90"], 1)

    def test_sliding_window_counts_exact_unique_lines(self) -> None:
        touch_rows = np.array([0, 1, 4, 2, 3], dtype=np.int32)
        touch_lines = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        keys, _shift, mask = locality.sorted_line_position_keys(
            8, touch_rows, touch_lines, np.arange(8, dtype=np.int32)
        )
        same = ((keys[1:] ^ keys[:-1]) > np.uint64(mask)) == 0
        gaps = (keys[1:] & np.uint64(mask)) - (keys[:-1] & np.uint64(mask))
        result = locality.working_set_distribution(
            keys, same, gaps, mask, 8, 2
        )
        self.assertAlmostEqual(result["mean"], 1.0)
        self.assertEqual(result["p50"], 1)
        self.assertEqual(result["p90"], 2)
        self.assertEqual(result["p99"], 2)

    def test_permutation_changes_temporal_gaps_without_relabeling_lines(self) -> None:
        touch_rows = np.array([0, 1, 4], dtype=np.int32)
        touch_lines = np.zeros(3, dtype=np.int32)
        # Execution order old cells: [0, 4, 1, 2, 3].
        old_to_new = np.array([0, 2, 3, 4, 1], dtype=np.int32)
        result = locality.analyze_permutation(
            5, touch_rows, touch_lines, old_to_new
        )
        self.assertEqual(result["psiLinesTouched"], 1)
        self.assertEqual(result["reuseGapMax"], 1)
        self.assertEqual(result["totalReuseEvents"], 2)


if __name__ == "__main__":
    unittest.main()
