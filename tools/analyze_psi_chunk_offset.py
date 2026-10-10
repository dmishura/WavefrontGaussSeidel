#!/usr/bin/env python3
"""Offline ``psi`` locality profiles over fixed-width IH1W chunks.

The graph remains in original cell numbering.  Only row execution order is
permuted.  Excluded profiles remove the leading rows and normalize position
over the remaining interval ``[exclude, chunkSize)``.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np

from analyze_psi_boundary_continuity import (
    continuity_arrays,
    level_starts_from_trace,
)
from analyze_psi_temporal_locality import (
    build_unique_row_line_touches,
    lower_percentiles,
    runtime_medians,
    write_rows,
)
from ih_permutation import read_ih_permutation


WIDTHS = (256, 512, 1024, 2048, 4096, 8192)
EXCLUSIONS = (0, 32, 128)
NORMALIZED_BINS = 20


def _ratio(numerator: float, denominator: float) -> float:
    return numerator/denominator if denominator else 0.0


def prefix_sum(values: np.ndarray) -> np.ndarray:
    result = np.empty(values.size + 1, dtype=np.float64)
    result[0] = 0.0
    np.cumsum(values, dtype=np.float64, out=result[1:])
    return result


def range_aggregate(
    prefix: np.ndarray, begins: np.ndarray, ends: np.ndarray
) -> tuple[int, float]:
    valid = ends > begins
    count = int(np.sum(ends[valid] - begins[valid], dtype=np.int64))
    if not count:
        return 0, 0.0
    total = float(np.sum(prefix[ends[valid]] - prefix[begins[valid]]))
    return count, total/count


def absolute_offset_profile(
    level_starts: np.ndarray,
    new_lines: np.ndarray,
    hot32: np.ndarray,
    hot128: np.ndarray,
) -> list[dict[str, object]]:
    starts = level_starts[:-1].astype(np.int64, copy=False)
    sizes = np.diff(level_starts).astype(np.int64, copy=False)
    rows: list[dict[str, object]] = []
    for offset in range(int(sizes.max(initial=0))):
        positions = starts[sizes > offset] + offset
        new = new_lines[positions]
        h32 = hot32[positions]
        h128 = hot128[positions]
        rows.append({
            "offset": offset,
            "samples": int(positions.size),
            "newLines_mean": float(new.mean()),
            "newLines_median": lower_percentiles(new, (0.5,))[0],
            "hot32_mean": float(h32.mean()),
            "hot32_median": lower_percentiles(h32, (0.5,))[0],
            "hot128_mean": float(h128.mean()),
            "hot128_median": lower_percentiles(h128, (0.5,))[0],
        })
    return rows


def normalized_profile(
    level_starts: np.ndarray,
    prefixes: dict[str, np.ndarray],
    exclusion: int,
    bins: int = NORMALIZED_BINS,
) -> list[dict[str, object]]:
    starts = level_starts[:-1].astype(np.int64, copy=False)
    sizes = np.diff(level_starts).astype(np.int64, copy=False)
    valid = sizes > exclusion
    starts = starts[valid]
    remaining = sizes[valid] - exclusion
    rows: list[dict[str, object]] = []
    for bin_index in range(bins):
        # Integer boundaries partition every remaining chunk exactly once.
        local_begin = exclusion + (remaining*bin_index)//bins
        local_end = exclusion + (remaining*(bin_index + 1))//bins
        begins = starts + local_begin
        ends = starts + local_end
        row: dict[str, object] = {
            "exclusion": exclusion,
            "normalizedBinStart": bin_index/bins,
            "normalizedBinEnd": (bin_index + 1)/bins,
        }
        samples = 0
        for name, prefix in prefixes.items():
            count, mean = range_aggregate(prefix, begins, ends)
            samples = count
            row[f"{name}_mean"] = mean
        row["samples"] = samples
        rows.append(row)
    return rows


def half_summary(
    level_starts: np.ndarray,
    prefixes: dict[str, np.ndarray],
    exclusion: int,
) -> dict[str, float | int]:
    starts = level_starts[:-1].astype(np.int64, copy=False)
    sizes = np.diff(level_starts).astype(np.int64, copy=False)
    valid = sizes > exclusion
    starts = starts[valid]
    remaining = sizes[valid] - exclusion
    middle = exclusion + remaining//2
    intervals = {
        "firstHalf": (starts + exclusion, starts + middle),
        "secondHalf": (starts + middle, starts + exclusion + remaining),
    }
    result: dict[str, float | int] = {
        "exclusion": exclusion,
        "chunksUsed": int(starts.size),
    }
    for half, (begins, ends) in intervals.items():
        for name, prefix in prefixes.items():
            count, mean = range_aggregate(prefix, begins, ends)
            result[f"{half}_{name}"] = mean
            result[f"{half}_samples"] = count
    for name in prefixes:
        result[f"{name}_secondToFirst"] = _ratio(
            float(result[f"secondHalf_{name}"]),
            float(result[f"firstHalf_{name}"]),
        )
    return result


def quartile_summary(
    level_starts: np.ndarray, prefixes: dict[str, np.ndarray]
) -> dict[str, float]:
    starts = level_starts[:-1].astype(np.int64, copy=False)
    sizes = np.diff(level_starts).astype(np.int64, copy=False)
    result: dict[str, float] = {}
    for quartile in range(4):
        begins = starts + (sizes*quartile)//4
        ends = starts + (sizes*(quartile + 1))//4
        for name, prefix in prefixes.items():
            _count, mean = range_aggregate(prefix, begins, ends)
            result[f"Q{quartile + 1}_{name}"] = mean
    for name in prefixes:
        result[f"{name}_Q4ToQ1"] = _ratio(
            result[f"Q4_{name}"], result[f"Q1_{name}"]
        )
    return result


def per_chunk_rows(
    level_starts: np.ndarray, prefixes: dict[str, np.ndarray]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for chunk, (start, end) in enumerate(
        zip(level_starts[:-1], level_starts[1:])
    ):
        start = int(start)
        end = int(end)
        middle = start + (end - start)//2
        row: dict[str, object] = {"chunkId": chunk, "chunkSize": end - start}
        for name, prefix in prefixes.items():
            row[f"mean_{name}"] = float(prefix[end] - prefix[start])/(end - start)
            first_count = middle - start
            second_count = end - middle
            row[f"firstHalf_{name}"] = (
                float(prefix[middle] - prefix[start])/first_count
                if first_count else 0.0
            )
            row[f"secondHalf_{name}"] = (
                float(prefix[end] - prefix[middle])/second_count
                if second_count else 0.0
            )
        rows.append(row)
    return rows


def analyze_width(
    mesh_name: str,
    width: int,
    runtime_ms: float | str,
    permutation_path: Path,
    trace_dir: Path,
    output_dir: Path,
    n_cells: int,
    touch_rows: np.ndarray,
    touch_lines: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    permutation = read_ih_permutation(permutation_path)
    if permutation.n_cells != n_cells or permutation.level != 0:
        raise RuntimeError(f"incompatible permutation: {permutation_path}")
    level_starts = permutation.level_starts
    if level_starts is None:
        level_starts = level_starts_from_trace(
            trace_dir/f"{mesh_name}_W{width}_release_trace.csv", n_cells
        )

    _counts, new_lines, _overlap, _jaccard, hot = continuity_arrays(
        n_cells, touch_rows, touch_lines,
        permutation.new_to_old, permutation.old_to_new,
    )
    prefixes = {
        "newLines": prefix_sum(new_lines),
        "hot32": prefix_sum(hot[32]),
        "hot128": prefix_sum(hot[128]),
    }

    absolute = absolute_offset_profile(
        level_starts, new_lines, hot[32], hot[128]
    )
    for row in absolute:
        row.update(mesh=mesh_name, width=width)
    # Put identifiers first without relying on dict insertion order above.
    absolute = [{
        "mesh": row.pop("mesh"), "width": row.pop("width"), **row
    } for row in absolute]
    write_rows(output_dir/f"{mesh_name}_W{width}_offset.csv", absolute)

    normalized: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    quartiles = quartile_summary(level_starts, prefixes)
    for exclusion in EXCLUSIONS:
        profile = normalized_profile(level_starts, prefixes, exclusion)
        for row in profile:
            normalized.append({"mesh": mesh_name, "width": width, **row})
        summary: dict[str, object] = {
            "mesh": mesh_name,
            "width": width,
            "runtimeMedianMs": runtime_ms,
            "chunks": int(level_starts.size - 1),
            **half_summary(level_starts, prefixes, exclusion),
        }
        if exclusion == 0:
            summary.update(quartiles)
        summaries.append(summary)

    write_rows(
        output_dir/f"{mesh_name}_W{width}_chunks.csv",
        per_chunk_rows(level_starts, prefixes),
    )
    return normalized, summaries


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-name", required=True)
    parser.add_argument("--poly-mesh", required=True, type=Path)
    parser.add_argument("--permutation-dir", required=True, type=Path)
    parser.add_argument("--chunk-trace-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--widths", nargs="+", type=int, default=list(WIDTHS))
    parser.add_argument("--runtime-results", action="append", default=[], type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    runtimes = runtime_medians(arguments.runtime_results)
    n_cells, touch_rows, touch_lines = build_unique_row_line_touches(
        arguments.poly_mesh
    )
    normalized_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for width in arguments.widths:
        normalized, summaries = analyze_width(
            arguments.mesh_name,
            width,
            runtimes.get((arguments.mesh_name, width), ""),
            arguments.permutation_dir/f"IH1W{width}.ihperm",
            arguments.chunk_trace_dir,
            arguments.output_dir,
            n_cells,
            touch_rows,
            touch_lines,
        )
        normalized_rows.extend(normalized)
        summary_rows.extend(summaries)
        print(f"psi-offset mesh={arguments.mesh_name} width={width}")
        gc.collect()
    write_rows(
        arguments.output_dir/f"{arguments.mesh_name}_normalized.csv",
        normalized_rows,
    )
    write_rows(
        arguments.output_dir/f"{arguments.mesh_name}_summary.csv", summary_rows
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
