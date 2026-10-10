#!/usr/bin/env python3

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import analyze_psi_boundary_continuity as continuity


class PsiBoundaryContinuityTests(unittest.TestCase):
    def test_level_starts_are_reconstructed_from_saved_chunk_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"trace.csv"
            path.write_text(
                "chunkIndex,chosenChunkSize\n0,3\n1,2\n2,1\n",
                encoding="utf-8",
            )
            starts = continuity.level_starts_from_trace(path, 6)
        np.testing.assert_array_equal(starts, [0, 3, 5, 6])

    def test_consecutive_overlap_and_hot_sets_use_execution_rows(self) -> None:
        # Rows read: {0,1}, {1,2}, {2,3}; line ids are already unique per row.
        touch_rows = np.repeat(np.arange(3, dtype=np.int32), 2)
        touch_lines = np.array([0, 1, 1, 2, 2, 3], dtype=np.int32)
        identity = np.arange(3, dtype=np.int32)
        counts, new_lines, overlap, jaccard, hot = continuity.continuity_arrays(
            3, touch_rows, touch_lines, identity, identity
        )
        np.testing.assert_array_equal(counts, [2, 2, 2])
        np.testing.assert_array_equal(new_lines, [2, 1, 1])
        np.testing.assert_allclose(overlap, [0.0, 0.5, 0.5])
        np.testing.assert_allclose(jaccard, [0.0, 1/3, 1/3])
        np.testing.assert_allclose(hot[8], [0.0, 0.5, 0.5])

    def test_recovery_length_uses_fixed_tolerance_and_eight_row_window(self) -> None:
        hot32 = np.full(600, 0.8, dtype=np.float32)
        hot32[300:304] = 0.0
        starts = np.array([0, 300, 600], dtype=np.int32)
        stats, fraction, baseline, censored = continuity.recovery_statistics(
            hot32, starts
        )
        self.assertAlmostEqual(baseline, 0.8)
        self.assertEqual(stats["p50"], 4)
        self.assertEqual(stats["p90"], 4)
        self.assertAlmostEqual(fraction, 4/600)
        self.assertEqual(censored, 0.0)

    def test_recovery_profile_stops_at_next_chunk_end(self) -> None:
        boundaries = np.array([2], dtype=np.int64)
        ends = np.array([4], dtype=np.int64)
        new_lines = np.array([0, 0, 7, 9, 11], dtype=np.int32)
        overlap = np.arange(5, dtype=np.float32)/10
        hot = {window: np.arange(5, dtype=np.float32)/10
               for window in continuity.HOT_WINDOWS}
        rows = continuity.recovery_profile(
            boundaries, ends, new_lines, overlap, hot
        )
        self.assertEqual(rows[0]["samples"], 1)
        self.assertEqual(rows[0]["newLines_mean"], 7)
        self.assertEqual(rows[1]["newLines_mean"], 9)
        self.assertEqual(rows[2]["samples"], 0)


if __name__ == "__main__":
    unittest.main()
