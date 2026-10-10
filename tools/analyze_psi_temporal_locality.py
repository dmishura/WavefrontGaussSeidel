#!/usr/bin/env python3
"""Offline temporal-locality analysis for actual IH1 ``psi[cols]`` reads.

Rows execute in ``newToOld`` order, while neighbour indices remain in the
original OpenFOAM cell-numbering space.  A cache-line touch is counted once
per executed row even when that row reads several doubles from the same 64-B
line.
"""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path
import statistics
from typing import Iterable

import numpy as np

from analyze_mesh_ordering import _build_ldu_adjacency_stream
from ih_permutation import read_ih_permutation
from openfoam_mesh_reader import read_poly_mesh


REUSE_THRESHOLDS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
WINDOW_SIZES = (64, 128, 256, 512, 1024, 2048, 4096, 8192)


def lower_percentiles(
    values: np.ndarray, fractions: Iterable[float]
) -> list[float]:
    fractions = tuple(fractions)
    if values.size == 0:
        return [0.0 for _ in fractions]
    indices = [int(fraction*(values.size - 1)) for fraction in fractions]
    work = values.copy()
    work.partition(tuple(sorted(set(indices))))
    return [float(work[index]) for index in indices]


def build_unique_row_line_touches(poly_mesh: Path) -> tuple[int, np.ndarray, np.ndarray]:
    """Return unique ``(original row, original psi cache line)`` touches."""
    mesh = read_poly_mesh(poly_mesh)
    n_cells = mesh.n_cells
    n_faces = mesh.n_internal_faces
    owner = np.frombuffer(mesh.owner, dtype=np.int32, count=n_faces)
    neighbour = np.frombuffer(mesh.neighbour, dtype=np.int32)
    stream, degree = _build_ldu_adjacency_stream(n_cells, owner, neighbour)
    row_starts = np.empty(n_cells + 1, dtype=np.int64)
    row_starts[0] = 0
    np.cumsum(degree, out=row_starts[1:])
    n_lines = (n_cells + 7)//8
    keys = np.empty(stream.size, dtype=np.uint64)
    rows_per_batch = 500_000
    for row_begin in range(0, n_cells, rows_per_batch):
        row_end = min(row_begin + rows_per_batch, n_cells)
        entry_begin = int(row_starts[row_begin])
        entry_end = int(row_starts[row_end])
        rows = np.repeat(
            np.arange(row_begin, row_end, dtype=np.uint64),
            degree[row_begin:row_end],
        )
        keys[entry_begin:entry_end] = (
            rows*np.uint64(n_lines)
            + (stream[entry_begin:entry_end].astype(np.uint64, copy=False) >> 3)
        )
    del stream, degree, row_starts, owner, neighbour, mesh
    keys = np.unique(keys)
    touch_rows = (keys//np.uint64(n_lines)).astype(np.int32)
    touch_lines = (keys%np.uint64(n_lines)).astype(np.int32)
    return n_cells, touch_rows, touch_lines


def sorted_line_position_keys(
    n_cells: int,
    touch_rows: np.ndarray,
    touch_lines: np.ndarray,
    old_to_new: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    shift = max(1, (n_cells - 1).bit_length())
    mask = (1 << shift) - 1
    keys = (
        touch_lines.astype(np.uint64) << np.uint64(shift)
    ) | old_to_new[touch_rows].astype(np.uint64)
    keys.sort()
    return keys, shift, mask


def working_set_distribution(
    keys: np.ndarray,
    same_line: np.ndarray,
    adjacent_gaps: np.ndarray,
    mask: int,
    n_cells: int,
    window: int,
) -> dict[str, float]:
    window_count = n_cells - window + 1
    if window_count <= 0 or keys.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    group_start = np.empty(keys.size, dtype=bool)
    group_start[0] = True
    group_start[1:] = (~same_line) | (adjacent_gaps > window)
    starts_index = np.flatnonzero(group_start)
    ends_index = np.empty_like(starts_index)
    ends_index[:-1] = starts_index[1:] - 1
    ends_index[-1] = keys.size - 1
    first_positions = (keys[starts_index] & np.uint64(mask)).astype(np.int64)
    last_positions = (keys[ends_index] & np.uint64(mask)).astype(np.int64)
    interval_starts = np.maximum(0, first_positions - window + 1)
    interval_ends = np.minimum(window_count - 1, last_positions)
    valid = interval_starts <= interval_ends
    interval_starts = interval_starts[valid]
    interval_ends = interval_ends[valid]
    difference = np.bincount(
        interval_starts, minlength=window_count + 1
    ).astype(np.int64, copy=False)
    difference -= np.bincount(
        interval_ends + 1, minlength=window_count + 1
    )
    counts = np.cumsum(difference[:-1])
    mean = float(counts.mean())
    p50, p90, p99 = lower_percentiles(counts, (0.50, 0.90, 0.99))
    return {"mean": mean, "p50": p50, "p90": p90, "p99": p99}


def analyze_permutation(
    n_cells: int,
    touch_rows: np.ndarray,
    touch_lines: np.ndarray,
    old_to_new: np.ndarray,
) -> dict[str, float | int]:
    keys, _shift, mask = sorted_line_position_keys(
        n_cells, touch_rows, touch_lines, old_to_new
    )
    same_line = np.empty(max(0, keys.size - 1), dtype=bool)
    if keys.size > 1:
        same_line[:] = ((keys[1:] ^ keys[:-1]) > np.uint64(mask)) == 0
        adjacent_gaps = (
            (keys[1:] & np.uint64(mask))
            - (keys[:-1] & np.uint64(mask))
        )
        reuse_gaps = adjacent_gaps[same_line]
    else:
        adjacent_gaps = np.empty(0, dtype=np.uint64)
        reuse_gaps = np.empty(0, dtype=np.uint64)

    result: dict[str, float | int] = {
        "psiLinesTouched": int(np.count_nonzero(
            np.r_[True, ~same_line] if keys.size else np.empty(0, dtype=bool)
        )),
        "totalLineTouches": int(keys.size),
        "totalReuseEvents": int(reuse_gaps.size),
        "reuseGapMean": float(reuse_gaps.mean()) if reuse_gaps.size else 0.0,
        "reuseGapMax": int(reuse_gaps.max(initial=0)),
    }
    p50, p90, p99 = lower_percentiles(reuse_gaps, (0.50, 0.90, 0.99))
    result.update(reuseGapP50=p50, reuseGapP90=p90, reuseGapP99=p99)
    for threshold in REUSE_THRESHOLDS:
        result[f"reuseWithin{threshold}"] = (
            float(np.count_nonzero(reuse_gaps <= threshold))/reuse_gaps.size
            if reuse_gaps.size else 0.0
        )

    line_start = np.empty(keys.size, dtype=bool)
    if keys.size:
        line_start[0] = True
        line_start[1:] = ~same_line
        starts = np.flatnonzero(line_start)
        ends = np.empty_like(starts)
        ends[:-1] = starts[1:] - 1
        ends[-1] = keys.size - 1
        first = keys[starts] & np.uint64(mask)
        last = keys[ends] & np.uint64(mask)
        lifetime = last - first
        touch_count = ends - starts + 1
    else:
        lifetime = np.empty(0, dtype=np.uint64)
        touch_count = np.empty(0, dtype=np.int64)
    for name, values in (("lifetime", lifetime), ("touchCount", touch_count)):
        values_p50, values_p90, values_p99 = lower_percentiles(
            values, (0.50, 0.90, 0.99)
        )
        result[f"{name}P50"] = values_p50
        result[f"{name}P90"] = values_p90
        result[f"{name}P99"] = values_p99

    for window in WINDOW_SIZES:
        stats = working_set_distribution(
            keys, same_line, adjacent_gaps, mask, n_cells, window
        )
        for statistic, value in stats.items():
            result[f"window{window}UniqueLines{statistic.capitalize()}"] = value
    return result


def runtime_medians(paths: list[Path]) -> dict[tuple[str, int], float]:
    samples: dict[tuple[str, int], list[float]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                solver = row.get("solver", "")
                if not solver.startswith("W") or not solver[1:].isdigit():
                    continue
                key = (row["mesh"], int(solver[1:]))
                samples.setdefault(key, []).append(
                    1000.0*float(row["real_seconds"])
                )
    return {key: statistics.median(values) for key, values in samples.items()}


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-name", required=True)
    parser.add_argument("--poly-mesh", required=True, type=Path)
    parser.add_argument("--permutation-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--widths", nargs="+", type=int,
        default=[256, 512, 1024, 2048, 4096, 8192],
    )
    parser.add_argument(
        "--runtime-results", action="append", default=[], type=Path
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    runtimes = runtime_medians(arguments.runtime_results)
    n_cells, touch_rows, touch_lines = build_unique_row_line_touches(
        arguments.poly_mesh
    )
    rows: list[dict[str, object]] = []
    for width in arguments.widths:
        permutation_path = arguments.permutation_dir/f"IH1W{width}.ihperm"
        permutation = read_ih_permutation(permutation_path)
        if permutation.n_cells != n_cells or permutation.level != 0:
            raise RuntimeError(f"incompatible permutation: {permutation_path}")
        metrics = analyze_permutation(
            n_cells, touch_rows, touch_lines, permutation.old_to_new
        )
        row: dict[str, object] = {
            "mesh": arguments.mesh_name,
            "width": width,
            "cells": n_cells,
            "runtimeMedianMs": runtimes.get((arguments.mesh_name, width), ""),
            **metrics,
        }
        rows.append(row)
        write_rows(
            arguments.output_dir/
            f"{arguments.mesh_name}_W{width}_summary.csv", [row]
        )
        print(
            f"psi-locality mesh={arguments.mesh_name} width={width} "
            f"touches={metrics['totalLineTouches']} "
            f"reuseEvents={metrics['totalReuseEvents']}"
        )
        gc.collect()
    write_rows(arguments.output_dir/f"{arguments.mesh_name}_summary.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
