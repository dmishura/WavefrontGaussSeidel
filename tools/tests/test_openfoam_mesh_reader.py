#!/usr/bin/env python3
"""Parity tests for tools/openfoam_mesh_reader.py and the C++ fixture."""

from __future__ import annotations

from array import array
import os
from pathlib import Path
import tempfile
import unittest

import sys

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from openfoam_mesh_reader import MeshReadError, read_poly_mesh  # noqa: E402


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_MOTORBIKE = REPOSITORY / "build-cmake/mesh-data/MTB_example_polyMesh"


def write_ascii_list(
    path: Path, object_name: str, values: list[int], n_cells: int
) -> None:
    payload = "\n".join(str(value) for value in values)
    path.write_text(
        "FoamFile\n"
        "{\n"
        "    format ascii;\n"
        "    class labelList;\n"
        f'    note "nCells:{n_cells}  nFaces:{len(values)}";\n'
        f"    object {object_name};\n"
        "}\n"
        f"{len(values)}\n(\n{payload}\n)\n",
        encoding="ascii",
    )


class MotorBikeReaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mesh_path = Path(
            os.environ.get("OPENFOAM_TEST_POLYMESH", DEFAULT_MOTORBIKE)
        )
        if not (cls.mesh_path / "owner").is_file():
            raise unittest.SkipTest(f"motorBike fixture is absent: {cls.mesh_path}")
        cls.mesh = read_poly_mesh(cls.mesh_path)

    def test_cpp_reader_counts_and_ranges(self) -> None:
        mesh = self.mesh
        self.assertEqual(mesh.n_cells, 352253)
        self.assertEqual(mesh.n_points, 404931)
        self.assertEqual(mesh.n_faces, 1104083)
        self.assertEqual(mesh.n_internal_faces, 1053964)
        self.assertEqual(len(mesh.owner), mesh.n_faces)
        self.assertEqual(len(mesh.neighbour), mesh.n_internal_faces)
        self.assertEqual((min(mesh.owner), max(mesh.owner)), (0, 352252))
        self.assertEqual((min(mesh.neighbour), max(mesh.neighbour)), (1, 352252))
        self.assertTrue(
            all(
                mesh.owner[face] < mesh.neighbour[face]
                for face in range(mesh.n_internal_faces)
            )
        )

    def test_cpp_reader_incidence_checks(self) -> None:
        mesh = self.mesh
        incident = array("I", [0]) * mesh.n_cells
        for cell in mesh.owner:
            incident[cell] += 1
        for cell in mesh.neighbour:
            incident[cell] += 1
        self.assertEqual(min(incident), 4)
        self.assertEqual(max(incident), 21)
        self.assertEqual(sum(incident), 2158047)

    def test_deterministic_topology_checksum(self) -> None:
        self.assertEqual(
            self.mesh.topology_checksum(),
            "81fecfde472dc8659254959da8a624be289a4fd375ba93a4705f18788f7afbb0",
        )


class AsciiReaderTest(unittest.TestCase):
    def test_ascii_label_lists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_ascii_list(directory / "owner", "owner", [0], 2)
            write_ascii_list(directory / "neighbour", "neighbour", [1], 2)
            mesh = read_poly_mesh(directory)
            self.assertEqual(mesh.n_cells, 2)
            self.assertEqual(mesh.n_faces, 1)
            self.assertEqual(mesh.n_internal_faces, 1)
            self.assertEqual(mesh.owner.tolist(), [0])
            self.assertEqual(mesh.neighbour.tolist(), [1])

    def test_out_of_range_neighbour_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_ascii_list(directory / "owner", "owner", [0], 2)
            write_ascii_list(directory / "neighbour", "neighbour", [3], 2)
            with self.assertRaises(MeshReadError):
                read_poly_mesh(directory)


if __name__ == "__main__":
    unittest.main()
