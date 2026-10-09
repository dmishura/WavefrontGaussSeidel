#!/usr/bin/env python3
"""Small reusable reader for OpenFOAM polyMesh LDU topology.

The supported labelList formats intentionally match the standalone C++
PolyMeshReader: ASCII labels and little-endian 32-bit binary labels.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import struct
import sys
from typing import BinaryIO


class MeshReadError(RuntimeError):
    """Raised when an OpenFOAM labelList or topology is invalid."""


_HEADER_NUMBER = {
    name: re.compile(rb"\b" + name.encode("ascii") + rb"\s*:\s*([0-9]+)")
    for name in ("nCells", "nFaces", "nPoints", "nInternalFaces")
}


@dataclass(frozen=True)
class LabelList:
    values: array
    n_cells: int = 0
    n_faces: int = 0
    n_points: int = 0
    n_internal_faces: int = 0
    binary: bool = False


@dataclass(frozen=True)
class PolyMeshTopology:
    poly_mesh_dir: Path
    owner: array
    neighbour: array
    n_cells: int
    n_faces: int
    n_points: int

    @property
    def n_internal_faces(self) -> int:
        return len(self.neighbour)

    @property
    def internal_owner(self) -> memoryview:
        return memoryview(self.owner)[: self.n_internal_faces]

    def topology_checksum(self) -> str:
        """SHA-256 over lengths and little-endian signed 32-bit labels."""
        digest = hashlib.sha256()
        digest.update(struct.pack("<QQ", len(self.owner), len(self.neighbour)))
        for values in (self.owner, self.neighbour):
            if sys.byteorder == "little":
                digest.update(memoryview(values).cast("B"))
            else:
                copy = array("i", values)
                copy.byteswap()
                digest.update(memoryview(copy).cast("B"))
        return digest.hexdigest()

    def validate(self) -> None:
        if self.n_cells <= 0:
            raise MeshReadError("polyMesh has no cells")
        if len(self.owner) != self.n_faces:
            raise MeshReadError("owner size differs from nFaces")
        if len(self.neighbour) > len(self.owner):
            raise MeshReadError("too many neighbour entries")

        incident = array("I", [0]) * self.n_cells
        previous_owner = -1
        for face, owner in enumerate(self.owner):
            if owner < 0 or owner >= self.n_cells:
                raise MeshReadError(f"owner[{face}] is outside cell range")
            incident[owner] += 1
            if face < self.n_internal_faces:
                neighbour = self.neighbour[face]
                if neighbour < 0 or neighbour >= self.n_cells:
                    raise MeshReadError(
                        f"neighbour[{face}] is outside cell range"
                    )
                if owner >= neighbour:
                    raise MeshReadError(
                        f"internal face {face} is not upper triangular"
                    )
                if owner < previous_owner:
                    raise MeshReadError(
                        "internal-face owners are not grouped"
                    )
                previous_owner = owner
                incident[neighbour] += 1
        if 0 in incident:
            raise MeshReadError("polyMesh contains a cell without faces")


def resolve_poly_mesh(path: str | Path) -> Path:
    """Resolve either a polyMesh directory or an OpenFOAM case directory."""
    candidate = Path(path).expanduser().resolve()
    if (candidate / "owner").is_file() and (candidate / "neighbour").is_file():
        return candidate
    nested = candidate / "constant" / "polyMesh"
    if (nested / "owner").is_file() and (nested / "neighbour").is_file():
        return nested
    raise MeshReadError(
        f"{candidate} is neither polyMesh nor a case containing constant/polyMesh"
    )


def _read_header(stream: BinaryIO, path: Path) -> bytes:
    data = stream.read(65536)
    begin = data.find(b"FoamFile")
    end = data.find(b"}", begin)
    if begin < 0 or end < 0:
        raise MeshReadError(f"invalid FoamFile header in {path}")
    stream.seek(end + 1)
    return data[begin : end + 1]


def _skip_space_and_comments(stream: BinaryIO, path: Path) -> None:
    while True:
        byte = stream.read(1)
        if not byte:
            return
        if byte.isspace():
            continue
        if byte != b"/":
            stream.seek(-1, 1)
            return
        second = stream.read(1)
        if second == b"/":
            while (byte := stream.read(1)) and byte != b"\n":
                pass
            continue
        if second == b"*":
            previous = b""
            while byte := stream.read(1):
                if previous == b"*" and byte == b"/":
                    break
                previous = byte
            else:
                raise MeshReadError(f"unterminated comment in {path}")
            continue
        stream.seek(-len(second) - 1, 1)
        return


def _read_unsigned(stream: BinaryIO, path: Path, what: str) -> int:
    digits = bytearray()
    while True:
        byte = stream.read(1)
        if byte and byte.isdigit():
            digits.extend(byte)
        else:
            if byte:
                stream.seek(-1, 1)
            break
    if not digits:
        raise MeshReadError(f"missing {what} in {path}")
    return int(digits)


def _header_number(header: bytes, name: str) -> int:
    match = _HEADER_NUMBER[name].search(header)
    return int(match.group(1)) if match else 0


def read_label_list(path: str | Path) -> LabelList:
    """Read an OpenFOAM labelList in ASCII or LE label=32 binary form."""
    source = Path(path)
    try:
        stream = source.open("rb")
    except OSError as error:
        raise MeshReadError(f"cannot open {source}: {error}") from error
    with stream:
        header = _read_header(stream, source)
        if not re.search(rb"\bclass\s+labelList\s*;", header):
            raise MeshReadError(f"expected labelList in {source}")
        binary = bool(re.search(rb"\bformat\s+binary\s*;", header))
        if binary:
            arch = re.search(rb"\barch\s+\"([^\"]+)\"\s*;", header)
            if arch and (b"MSB" in arch.group(1) or b"label=64" in arch.group(1)):
                raise MeshReadError(
                    f"binary {source} is not little-endian label=32"
                )

        _skip_space_and_comments(stream, source)
        count = _read_unsigned(stream, source, "list size")
        _skip_space_and_comments(stream, source)
        if stream.read(1) != b"(":
            raise MeshReadError(f"missing list opening in {source}")

        values = array("i")
        if values.itemsize != 4:
            raise MeshReadError("Python array('i') is not 32-bit on this platform")
        if binary:
            payload = stream.read(count * 4)
            if len(payload) != count * 4:
                raise MeshReadError(f"truncated binary payload in {source}")
            values.frombytes(payload)
            if sys.byteorder != "little":
                values.byteswap()
        else:
            remainder = stream.read()
            remainder = re.sub(rb"//[^\n]*", b" ", remainder)
            remainder = re.sub(rb"/\*.*?\*/", b" ", remainder, flags=re.S)
            closing = remainder.find(b")")
            if closing < 0:
                raise MeshReadError(f"missing list closing in {source}")
            tokens = re.findall(rb"[+-]?[0-9]+", remainder[:closing])
            if len(tokens) != count:
                raise MeshReadError(
                    f"ASCII list size mismatch in {source}: "
                    f"declared {count}, found {len(tokens)}"
                )
            values.fromlist([int(token) for token in tokens])

    return LabelList(
        values=values,
        n_cells=_header_number(header, "nCells"),
        n_faces=_header_number(header, "nFaces"),
        n_points=_header_number(header, "nPoints"),
        n_internal_faces=_header_number(header, "nInternalFaces"),
        binary=binary,
    )


def read_poly_mesh(path: str | Path) -> PolyMeshTopology:
    """Read and validate owner/neighbour topology from a serial polyMesh."""
    directory = resolve_poly_mesh(path)
    owner_file = read_label_list(directory / "owner")
    neighbour_file = read_label_list(directory / "neighbour")

    n_cells = owner_file.n_cells
    if not n_cells and owner_file.values:
        n_cells = max(owner_file.values) + 1
    n_faces = len(owner_file.values)
    n_internal = len(neighbour_file.values)

    if owner_file.n_faces and owner_file.n_faces != n_faces:
        raise MeshReadError("owner header nFaces mismatch")
    if owner_file.n_internal_faces and owner_file.n_internal_faces != n_internal:
        raise MeshReadError("owner header nInternalFaces mismatch")
    if neighbour_file.n_cells and neighbour_file.n_cells != n_cells:
        raise MeshReadError("owner/neighbour nCells mismatch")
    if neighbour_file.n_faces and neighbour_file.n_faces != n_faces:
        raise MeshReadError("owner/neighbour nFaces mismatch")
    if (
        neighbour_file.n_internal_faces
        and neighbour_file.n_internal_faces != n_internal
    ):
        raise MeshReadError("neighbour header nInternalFaces mismatch")
    if (
        owner_file.n_points
        and neighbour_file.n_points
        and owner_file.n_points != neighbour_file.n_points
    ):
        raise MeshReadError("owner/neighbour nPoints mismatch")

    mesh = PolyMeshTopology(
        poly_mesh_dir=directory,
        owner=owner_file.values,
        neighbour=neighbour_file.values,
        n_cells=n_cells,
        n_faces=n_faces,
        n_points=owner_file.n_points,
    )
    mesh.validate()
    return mesh


__all__ = [
    "LabelList",
    "MeshReadError",
    "PolyMeshTopology",
    "read_label_list",
    "read_poly_mesh",
    "resolve_poly_mesh",
]
