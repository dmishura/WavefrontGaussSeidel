#!/usr/bin/env python3
"""Reader for the compact IH1 permutation dump written by gamg_level_bench.

The format is entirely little-endian.  Version 1 stores ``IH1PERM\0``, version
(u32), label width (u32), cell count (u64), GAMG level (u32), UTF-8 name length
(u32), name bytes, then ``newToOld[N]`` and ``oldToNew[N]`` as i32.  Version 2
adds a level-start count (u64) before the name and ``levelStarts[]`` before the
two permutations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


MAGIC = b"IH1PERM\0"
VERSION = 2
_HEADER = struct.Struct("<8sIIQII")


class PermutationReadError(RuntimeError):
    """Raised when an IH1 permutation file is malformed."""


@dataclass(frozen=True)
class IHPermutation:
    path: Path
    variant: str
    level: int
    level_starts: np.ndarray | None
    new_to_old: np.ndarray
    old_to_new: np.ndarray

    @property
    def n_cells(self) -> int:
        return int(self.new_to_old.size)

    def validate(self) -> None:
        n_cells = self.n_cells
        if self.old_to_new.size != n_cells:
            raise PermutationReadError("newToOld/oldToNew size mismatch")
        for name, values in (
            ("newToOld", self.new_to_old),
            ("oldToNew", self.old_to_new),
        ):
            if values.dtype != np.dtype("<i4"):
                raise PermutationReadError(f"{name} is not little-endian int32")
            if values.size and (
                int(values.min()) < 0 or int(values.max()) >= n_cells
            ):
                raise PermutationReadError(f"{name} contains an out-of-range cell")

        chunk = 4_000_000
        for begin in range(0, n_cells, chunk):
            end = min(begin + chunk, n_cells)
            rows = np.arange(begin, end, dtype=np.int32)
            old_cells = self.new_to_old[begin:end]
            if not np.array_equal(self.old_to_new[old_cells], rows):
                raise PermutationReadError(
                    "newToOld and oldToNew are not inverse permutations"
                )


def read_ih_permutation(path: str | Path) -> IHPermutation:
    """Read and validate a version-1 binary ``.ihperm`` file."""
    source = Path(path)
    try:
        with source.open("rb") as stream:
            header = stream.read(_HEADER.size)
            if len(header) != _HEADER.size:
                raise PermutationReadError(f"truncated header in {source}")
            magic, version, label_bytes, n_cells, level, name_bytes = (
                _HEADER.unpack(header)
            )
            if magic != MAGIC:
                raise PermutationReadError(f"invalid magic in {source}")
            if version not in (1, VERSION):
                raise PermutationReadError(
                    f"unsupported permutation version {version} in {source}"
                )
            if label_bytes != 4:
                raise PermutationReadError(
                    f"unsupported label width {label_bytes} in {source}"
                )
            level_start_count = 0
            if version >= 2:
                encoded_count = stream.read(8)
                if len(encoded_count) != 8:
                    raise PermutationReadError(
                        f"truncated level-start count in {source}"
                    )
                level_start_count = struct.unpack("<Q", encoded_count)[0]
            encoded_name = stream.read(name_bytes)
            if len(encoded_name) != name_bytes:
                raise PermutationReadError(f"truncated variant name in {source}")
            try:
                variant = encoded_name.decode("utf-8")
            except UnicodeDecodeError as error:
                raise PermutationReadError(
                    f"invalid UTF-8 variant name in {source}"
                ) from error
            payload_offset = stream.tell()
    except OSError as error:
        raise PermutationReadError(f"cannot read {source}: {error}") from error

    expected_size = payload_offset + (
        level_start_count + 2*n_cells
    ) * label_bytes
    try:
        actual_size = source.stat().st_size
    except OSError as error:
        raise PermutationReadError(f"cannot stat {source}: {error}") from error
    if actual_size != expected_size:
        raise PermutationReadError(
            f"payload size mismatch in {source}: expected {expected_size}, "
            f"found {actual_size}"
        )

    level_starts = None
    permutation_offset = payload_offset
    if level_start_count:
        level_starts = np.fromfile(
            source, dtype="<i4", count=level_start_count, offset=payload_offset
        )
        permutation_offset += level_start_count*label_bytes
        if (
            level_starts.size < 2
            or int(level_starts[0]) != 0
            or int(level_starts[-1]) != n_cells
            or np.any(level_starts[1:] <= level_starts[:-1])
        ):
            raise PermutationReadError(f"invalid levelStarts in {source}")
    new_to_old = np.fromfile(
        source, dtype="<i4", count=n_cells, offset=permutation_offset
    )
    old_to_new = np.fromfile(
        source,
        dtype="<i4",
        count=n_cells,
        offset=permutation_offset + n_cells * label_bytes,
    )
    result = IHPermutation(
        path=source.resolve(),
        variant=variant,
        level=level,
        level_starts=level_starts,
        new_to_old=new_to_old,
        old_to_new=old_to_new,
    )
    result.validate()
    return result
