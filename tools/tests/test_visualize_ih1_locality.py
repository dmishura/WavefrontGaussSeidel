#!/usr/bin/env python3
"""Unit tests for IH1 locality field and projection calculations."""

from __future__ import annotations

from array import array
from pathlib import Path
import sys
import unittest

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from ih_permutation import IHPermutation  # noqa: E402
from openfoam_mesh_reader import PolyMeshTopology  # noqa: E402
from visualize_ih1_locality import (  # noqa: E402
    aligned_boundary_profile,
    chunk_size_statistics,
    chunk_boundary_fields,
    execution_row_jumps,
    execution_to_original,
    log_brightness,
    normalization_threshold,
    projection_coordinates,
    psi_jump_fields,
)


class LocalityFieldTest(unittest.TestCase):
    def test_sequential_and_one_large_jump(self) -> None:
        self.assertEqual(
            execution_row_jumps(np.array([0, 1, 2, 3], dtype=np.int32)).tolist(),
            [0.0, 1.0, 1.0, 1.0],
        )
        self.assertEqual(
            execution_row_jumps(np.array([0, 1, 9, 2], dtype=np.int32)).tolist(),
            [0.0, 1.0, 8.0, 7.0],
        )

    def test_execution_to_original_and_chunk_boundaries(self) -> None:
        permutation = np.array([2, 0, 3, 1], dtype=np.int32)
        mapped = execution_to_original(
            np.array([10.0, 20.0, 30.0, 40.0]), permutation
        )
        self.assertEqual(mapped.tolist(), [20.0, 40.0, 10.0, 30.0])
        row, boundary, boundary_jump = chunk_boundary_fields(
            permutation, np.array([0, 2, 4], dtype=np.int32)
        )
        self.assertEqual(boundary.tolist(), [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(row.tolist(), [2.0, 2.0, 0.0, 3.0])
        self.assertEqual(boundary_jump.tolist(), [0.0, 0.0, 0.0, 3.0])

    def test_known_psi_column_sequence(self) -> None:
        mesh = PolyMeshTopology(
            poly_mesh_dir=Path("."),
            owner=array("i", [0, 0, 0]),
            neighbour=array("i", [1, 4, 2]),
            n_cells=5,
            n_faces=3,
            n_points=0,
        )
        identity = np.arange(5, dtype="<i4")
        permutation = IHPermutation(
            path=Path("identity"),
            variant="identity",
            level=0,
            level_starts=np.array([0, 5], dtype="<i4"),
            new_to_old=identity,
            old_to_new=identity.copy(),
        )
        mean, p90, maximum, reads = psi_jump_fields(mesh, permutation)
        # Cell 0 reads [1, 4, 2], whose jumps are [3, 2].
        self.assertEqual(reads, 6)
        self.assertEqual(mean[0], 2.5)
        self.assertEqual(p90[0], 2.0)
        self.assertEqual(maximum[0], 3.0)

    def test_shared_threshold_and_log_saturation(self) -> None:
        shared = np.concatenate(
            [np.arange(100, dtype=float), np.arange(100, 200, dtype=float)]
        )
        threshold = normalization_threshold(shared)
        self.assertEqual(threshold, 197.0)
        brightness = log_brightness(
            np.array([0.0, threshold, 10*threshold]), threshold
        )
        self.assertEqual(brightness[0], 0)
        self.assertEqual(brightness[1], 255)
        self.assertEqual(brightness[2], 255)

    def test_projection_uses_one_equal_axis_scale(self) -> None:
        centres = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
        x, y, scale = projection_coordinates(centres, (0, 1), 1001, 10)
        x_distance = abs(int(x[1]) - int(x[0]))
        y_distance = abs(int(y[1]) - int(y[0]))
        self.assertGreater(scale, 0)
        self.assertLessEqual(abs(x_distance - 2*y_distance), 1)

    def test_aligned_boundary_profile(self) -> None:
        permutation = np.array([0, 1, 9, 2, 3], dtype=np.int32)
        offsets, means, counts = aligned_boundary_profile(
            permutation, np.array([2], dtype=np.int32), 1
        )
        self.assertEqual(offsets.tolist(), [-1, 0, 1])
        self.assertEqual(means.tolist(), [1.0, 8.0, 7.0])
        self.assertEqual(counts.tolist(), [1, 1, 1])

    def test_chunk_size_statistics(self) -> None:
        stats = chunk_size_statistics(np.array([0, 3, 8, 10]))
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["min"], 2)
        self.assertEqual(stats["max"], 5)
        self.assertEqual(stats["p50"], 3.0)


if __name__ == "__main__":
    unittest.main()
