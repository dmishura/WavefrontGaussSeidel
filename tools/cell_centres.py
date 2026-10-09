#!/usr/bin/env python3
"""Read exact geometric cell centres exported by mesh_centres_export."""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np


MAGIC = b"IHCENTR\0"
_HEADER = struct.Struct("<8sIIQ")


def read_cell_centres(path: str | Path) -> np.ndarray:
    source = Path(path)
    with source.open("rb") as stream:
        header = stream.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise RuntimeError(f"truncated cell-centre header in {source}")
        magic, version, scalar_bytes, n_cells = _HEADER.unpack(header)
        if magic != MAGIC:
            raise RuntimeError(f"invalid cell-centre magic in {source}")
        if version != 1 or scalar_bytes != 8:
            raise RuntimeError(
                f"unsupported cell-centre format {version}/{scalar_bytes}"
            )
        offset = stream.tell()
    expected = offset + 3*n_cells*scalar_bytes
    if source.stat().st_size != expected:
        raise RuntimeError(f"cell-centre payload size mismatch in {source}")
    values = np.fromfile(source, dtype="<f8", count=3*n_cells, offset=offset)
    return values.reshape((n_cells, 3))
