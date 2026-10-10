#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest

import numpy as np


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import analyze_psi_chunk_offset as offset


class PsiChunkOffsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.starts = np.array([0, 4, 8], dtype=np.int64)
        self.new = np.arange(8, dtype=np.float64)
        self.hot32 = self.new/10
        self.hot128 = self.new/20
        self.prefixes = {
            "newLines": offset.prefix_sum(self.new),
            "hot32": offset.prefix_sum(self.hot32),
            "hot128": offset.prefix_sum(self.hot128),
        }

    def test_absolute_offset_combines_same_position_from_each_chunk(self) -> None:
        rows = offset.absolute_offset_profile(
            self.starts, self.new, self.hot32, self.hot128
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["samples"], 2)
        self.assertEqual(rows[0]["newLines_mean"], 2.0)
        self.assertEqual(rows[3]["newLines_mean"], 5.0)

    def test_normalized_profile_partitions_remaining_rows_after_exclusion(self) -> None:
        rows = offset.normalized_profile(
            self.starts, self.prefixes, exclusion=2, bins=2
        )
        self.assertEqual(rows[0]["samples"], 2)
        self.assertEqual(rows[0]["newLines_mean"], 4.0)  # rows 2 and 6
        self.assertEqual(rows[1]["newLines_mean"], 5.0)  # rows 3 and 7

    def test_half_summary_resplits_post_exclusion_interval(self) -> None:
        result = offset.half_summary(self.starts, self.prefixes, exclusion=2)
        self.assertEqual(result["firstHalf_newLines"], 4.0)
        self.assertEqual(result["secondHalf_newLines"], 5.0)
        self.assertEqual(result["newLines_secondToFirst"], 1.25)

    def test_quartiles_cover_chunks_without_overlap(self) -> None:
        result = offset.quartile_summary(self.starts, self.prefixes)
        self.assertEqual(result["Q1_newLines"], 2.0)
        self.assertEqual(result["Q4_newLines"], 5.0)
        self.assertEqual(result["newLines_Q4ToQ1"], 2.5)


if __name__ == "__main__":
    unittest.main()
