#!/usr/bin/env python3
"""Offline IH1 chunk-boundary continuity analysis for ``psi[cols]`` reads."""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import numpy as np

from analyze_psi_temporal_locality import (
    build_unique_row_line_touches,
    lower_percentiles,
    runtime_medians,
    sorted_line_position_keys,
    write_rows,
)
from ih_permutation import read_ih_permutation


HOT_WINDOWS = (8, 32, 128, 512)
PROFILE_OFFSETS = 256
RECOVERY_SEARCH = 512
RECOVERY_WINDOW = 8
RECOVERY_TOLERANCE = 0.02


def level_starts_from_trace(path: Path, n_cells: int) -> np.ndarray:
    """Reconstruct exact fixed-width chunk boundaries from a saved trace."""
    sizes: list[int] = []
    with path.open(newline="", encoding="utf-8") as source:
        for expected_index, row in enumerate(csv.DictReader(source)):
            if int(row["chunkIndex"]) != expected_index:
                raise RuntimeError(f"non-consecutive chunk index in {path}")
            size = int(row["chosenChunkSize"])
            if size <= 0:
                raise RuntimeError(f"non-positive chunk size in {path}")
            sizes.append(size)
    starts = np.empty(len(sizes) + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(np.asarray(sizes, dtype=np.int64), out=starts[1:])
    if not sizes or int(starts[-1]) != n_cells:
        raise RuntimeError(
            f"chunk trace covers {int(starts[-1]) if sizes else 0} of "
            f"{n_cells} cells: {path}"
        )
    return starts


def distribution(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    p50, p90, p99 = lower_percentiles(values, (0.50, 0.90, 0.99))
    return {
        "mean": float(values.mean()),
        "p50": p50,
        "p90": p90,
        "p99": p99,
    }


def prefixed(result: dict[str, object], prefix: str, values: np.ndarray) -> None:
    for name, value in distribution(values).items():
        result[f"{prefix}_{name}"] = value


def continuity_arrays(
    n_cells: int,
    touch_rows: np.ndarray,
    touch_lines: np.ndarray,
    new_to_old: np.ndarray,
    old_to_new: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    touch_count = np.bincount(touch_rows, minlength=n_cells).astype(np.int32)
    current_count = touch_count[new_to_old]
    keys, _shift, position_mask = sorted_line_position_keys(
        n_cells, touch_rows, touch_lines, old_to_new
    )
    same_line = ((keys[1:] ^ keys[:-1]) > np.uint64(position_mask)) == 0
    gaps = (
        (keys[1:] & np.uint64(position_mask))
        - (keys[:-1] & np.uint64(position_mask))
    ).astype(np.uint32)
    current_positions = (keys[1:] & np.uint64(position_mask)).astype(np.int32)

    overlap = np.bincount(
        current_positions[same_line & (gaps == 1)], minlength=n_cells
    ).astype(np.int32)
    previous_count = np.empty_like(current_count)
    previous_count[0] = 0
    previous_count[1:] = current_count[:-1]
    union_count = previous_count + current_count - overlap
    overlap_fraction = np.divide(
        overlap, current_count,
        out=np.zeros(n_cells, dtype=np.float32), where=current_count != 0,
    )
    jaccard = np.divide(
        overlap, union_count,
        out=np.zeros(n_cells, dtype=np.float32), where=union_count != 0,
    )

    hot_fractions: dict[int, np.ndarray] = {}
    for window in HOT_WINDOWS:
        hot_count = np.bincount(
            current_positions[same_line & (gaps <= window)], minlength=n_cells
        ).astype(np.int32)
        hot_fractions[window] = np.divide(
            hot_count, current_count,
            out=np.zeros(n_cells, dtype=np.float32), where=current_count != 0,
        )
    del keys, same_line, gaps, current_positions, touch_count
    return current_count, current_count - overlap, overlap_fraction, jaccard, hot_fractions


def recovery_statistics(
    hot32: np.ndarray, level_starts: np.ndarray
) -> tuple[dict[str, float], float, float, float]:
    n_cells = hot32.size
    starts = level_starts[:-1].astype(np.int64, copy=False)
    ends = level_starts[1:].astype(np.int64, copy=False)
    boundaries = starts[1:]
    boundary_ends = ends[1:]

    steady = np.zeros(n_cells, dtype=bool)
    for start, end in zip(starts, ends):
        steady_begin = int(start) + 256
        if steady_begin < end:
            steady[steady_begin:int(end)] = True
    if np.any(steady):
        baseline = float(np.median(hot32[steady]))
    else:
        inside = np.ones(n_cells, dtype=bool)
        inside[0] = False
        inside[boundaries] = False
        baseline = float(np.median(hot32[inside])) if np.any(inside) else 0.0
    threshold = baseline - RECOVERY_TOLERANCE

    prefix = np.empty(n_cells + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(hot32, dtype=np.float64, out=prefix[1:])
    recovery = np.full(boundaries.size, -1, dtype=np.int32)
    for offset in range(RECOVERY_SEARCH):
        unresolved = recovery < 0
        positions = boundaries + offset
        valid = unresolved & (positions + RECOVERY_WINDOW <= boundary_ends)
        if not np.any(valid):
            continue
        selected = positions[valid]
        rolling_mean = (
            prefix[selected + RECOVERY_WINDOW] - prefix[selected]
        )/RECOVERY_WINDOW
        recovered_indices = np.flatnonzero(valid)[rolling_mean >= threshold]
        recovery[recovered_indices] = offset
    unresolved = recovery < 0
    censored_fraction = float(np.count_nonzero(unresolved))/recovery.size if recovery.size else 0.0
    if np.any(unresolved):
        available = np.minimum(
            RECOVERY_SEARCH,
            np.maximum(0, boundary_ends - boundaries - RECOVERY_WINDOW + 1),
        )
        recovery[unresolved] = available[unresolved].astype(np.int32)
    stats = distribution(recovery)
    transition_fraction = float(recovery.sum(dtype=np.int64))/n_cells
    return stats, transition_fraction, baseline, censored_fraction


def recovery_profile(
    boundaries: np.ndarray,
    boundary_ends: np.ndarray,
    new_lines: np.ndarray,
    overlap_fraction: np.ndarray,
    hot: dict[int, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset in range(PROFILE_OFFSETS):
        positions = boundaries + offset
        valid = positions < boundary_ends
        selected = positions[valid]
        if selected.size:
            new_values = new_lines[selected]
            new_median = lower_percentiles(new_values, (0.50,))[0]
            row = {
                "offset": offset,
                "samples": int(selected.size),
                "newLines_mean": float(new_values.mean()),
                "newLines_median": new_median,
                "overlapFrac_mean": float(overlap_fraction[selected].mean()),
                "hotFrac8_mean": float(hot[8][selected].mean()),
                "hotFrac32_mean": float(hot[32][selected].mean()),
                "hotFrac128_mean": float(hot[128][selected].mean()),
            }
        else:
            row = {
                "offset": offset, "samples": 0,
                "newLines_mean": 0.0, "newLines_median": 0.0,
                "overlapFrac_mean": 0.0, "hotFrac8_mean": 0.0,
                "hotFrac32_mean": 0.0, "hotFrac128_mean": 0.0,
            }
        rows.append(row)
    return rows


def analyze_width(
    mesh_name: str,
    width: int,
    runtime_ms: float | str,
    permutation_path: Path,
    n_cells: int,
    touch_rows: np.ndarray,
    touch_lines: np.ndarray,
    output_dir: Path,
    chunk_trace_dir: Path | None,
) -> dict[str, object]:
    permutation = read_ih_permutation(permutation_path)
    if permutation.n_cells != n_cells or permutation.level != 0:
        raise RuntimeError(f"incompatible permutation: {permutation_path}")
    level_starts = permutation.level_starts
    if level_starts is None:
        if chunk_trace_dir is None:
            raise RuntimeError(
                f"permutation has no chunk boundaries and no trace was supplied: "
                f"{permutation_path}"
            )
        level_starts = level_starts_from_trace(
            chunk_trace_dir/f"{mesh_name}_W{width}_release_trace.csv", n_cells
        )
    current_count, new_lines, overlap_fraction, jaccard, hot = continuity_arrays(
        n_cells, touch_rows, touch_lines,
        permutation.new_to_old, permutation.old_to_new,
    )
    boundaries = level_starts[1:-1].astype(np.int64, copy=False)
    boundary_ends = level_starts[2:].astype(np.int64, copy=False)
    transition_positions = np.arange(1, n_cells, dtype=np.int64)
    boundary_mask = np.zeros(n_cells, dtype=bool)
    boundary_mask[boundaries] = True
    boundary_positions = transition_positions[boundary_mask[1:]]
    inside_positions = transition_positions[~boundary_mask[1:]]

    result: dict[str, object] = {
        "mesh": mesh_name,
        "width": width,
        "runtimeMedianMs": runtime_ms,
        "chunks": int(level_starts.size - 1),
        "boundaryCount": int(boundaries.size),
    }
    for category, positions in (
        ("inside", inside_positions), ("boundary", boundary_positions)
    ):
        result[f"{category}_count"] = int(positions.size)
        prefixed(result, f"{category}_newLines", new_lines[positions])
        prefixed(
            result, f"{category}_overlapFracCur", overlap_fraction[positions]
        )
        prefixed(result, f"{category}_jaccard", jaccard[positions])
        for window in (8, 32, 128):
            result[f"{category}_hotFrac{window}_mean"] = (
                float(hot[window][positions].mean()) if positions.size else 0.0
            )

    inside_new = float(result["inside_newLines_mean"])
    boundary_new = float(result["boundary_newLines_mean"])
    result["boundary_newLinesPenalty"] = boundary_new - inside_new
    result["boundaryNewLinesToInsideRatio"] = (
        boundary_new/inside_new if inside_new else 0.0
    )
    inside_hot = float(result["inside_hotFrac32_mean"])
    boundary_hot = float(result["boundary_hotFrac32_mean"])
    result["boundaryHotFrac32AbsoluteDrop"] = inside_hot - boundary_hot
    result["boundaryHotFrac32RelativeDrop"] = (
        (inside_hot - boundary_hot)/inside_hot if inside_hot else 0.0
    )

    for window in HOT_WINDOWS:
        prefixed(result, f"hotFrac{window}", hot[window])
        cold = np.rint((1.0 - hot[window])*current_count).astype(np.int32)
        prefixed(result, f"coldLines{window}", cold)

    recovery, transition_fraction, baseline, censored_fraction = recovery_statistics(
        hot[32], level_starts
    )
    result["steadyHotFrac32Baseline"] = baseline
    result["recoveryTolerance"] = RECOVERY_TOLERANCE
    result["recoveryWindow"] = RECOVERY_WINDOW
    result["recoveryCensoredFraction"] = censored_fraction
    for name, value in recovery.items():
        result[f"recoveryLength_{name}"] = value
    result["transitionRowsFraction"] = transition_fraction

    profile = recovery_profile(
        boundaries, boundary_ends, new_lines, overlap_fraction, hot
    )
    write_rows(
        output_dir/f"{mesh_name}_W{width}_recovery.csv", profile
    )
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-name", required=True)
    parser.add_argument("--poly-mesh", required=True, type=Path)
    parser.add_argument("--permutation-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-trace-dir", type=Path)
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
        row = analyze_width(
            arguments.mesh_name,
            width,
            runtimes.get((arguments.mesh_name, width), ""),
            arguments.permutation_dir/f"IH1W{width}.ihperm",
            n_cells,
            touch_rows,
            touch_lines,
            arguments.output_dir,
            arguments.chunk_trace_dir,
        )
        rows.append(row)
        print(
            f"psi-boundary mesh={arguments.mesh_name} width={width} "
            f"chunks={row['chunks']} recoveryP90={row['recoveryLength_p90']}"
        )
        gc.collect()
    write_rows(arguments.output_dir/f"{arguments.mesh_name}_summary.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
