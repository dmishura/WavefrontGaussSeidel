#!/usr/bin/env python3
"""Tests for IH1 permutation files and remapped topology analysis."""

from __future__ import annotations

from array import array
from dataclasses import asdict
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from analyze_mesh_ordering import analyse_mesh  # noqa: E402
from ih_permutation import (  # noqa: E402
    MAGIC,
    PermutationReadError,
    read_ih_permutation,
)


def write_permutation(
    path: Path,
    variant: str,
    new_to_old: list[int],
    old_to_new: list[int],
    level_starts: list[int] | None = None,
) -> None:
    encoded = variant.encode("utf-8")
    version = 2 if level_starts is not None else 1
    with path.open("wb") as output:
        output.write(
            struct.pack(
                "<8sIIQII", MAGIC, version, 4, len(new_to_old), 0, len(encoded)
            )
        )
        if level_starts is not None:
            output.write(struct.pack("<Q", len(level_starts)))
        output.write(encoded)
        if level_starts is not None:
            output.write(np.asarray(level_starts, dtype="<i4").tobytes())
        output.write(np.asarray(new_to_old, dtype="<i4").tobytes())
        output.write(np.asarray(old_to_new, dtype="<i4").tobytes())


def write_ascii_list(
    path: Path, object_name: str, values: list[int], n_cells: int
) -> None:
    payload = "\n".join(str(value) for value in values)
    path.write_text(
        "FoamFile\n{\n    format ascii;\n    class labelList;\n"
        f'    note "nCells:{n_cells}  nFaces:{len(values)}";\n'
        f"    object {object_name};\n}}\n{len(values)}\n(\n{payload}\n)\n",
        encoding="ascii",
    )


class PermutationTest(unittest.TestCase):
    def test_read_and_inverse_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.ihperm"
            write_permutation(path, "IH1Test", [2, 0, 1], [1, 2, 0])
            result = read_ih_permutation(path)
            self.assertEqual(result.variant, "IH1Test")
            self.assertEqual(result.level, 0)
            self.assertEqual(result.new_to_old.tolist(), [2, 0, 1])
            self.assertEqual(result.old_to_new.tolist(), [1, 2, 0])
            self.assertIsNone(result.level_starts)

    def test_version_two_level_starts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test-v2.ihperm"
            write_permutation(
                path, "IH1W2", [2, 0, 1], [1, 2, 0], [0, 2, 3]
            )
            result = read_ih_permutation(path)
            self.assertEqual(result.level_starts.tolist(), [0, 2, 3])

    def test_invalid_inverse_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.ihperm"
            write_permutation(path, "bad", [2, 0, 1], [0, 1, 2])
            with self.assertRaises(PermutationReadError):
                read_ih_permutation(path)

    def test_analyser_uses_new_index_space(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            # Three-cell path: internal faces (0,1), (1,2).
            write_ascii_list(directory / "owner", "owner", [0, 1], 3)
            write_ascii_list(directory / "neighbour", "neighbour", [1, 2], 3)
            permutation_path = directory / "test.ihperm"
            write_permutation(permutation_path, "IH1Test", [2, 0, 1], [1, 2, 0])
            permutation = read_ih_permutation(permutation_path)

            original = analyse_mesh("path", directory)
            remapped = analyse_mesh("path", directory, permutation)
            self.assertEqual(remapped.numbering, "IH1Test")
            self.assertEqual(remapped.cells, original.cells)
            self.assertEqual(remapped.internal_faces, original.internal_faces)
            self.assertNotEqual(
                remapped.normalized_mean_edge_delta,
                original.normalized_mean_edge_delta,
            )
            payload = asdict(remapped)
            self.assertNotIn("backward_fraction", payload)
            self.assertNotIn("direction_change_rate", payload)


if __name__ == "__main__":
    unittest.main()
