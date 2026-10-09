#!/usr/bin/env python3
"""Visualize IH1 execution and psi-read locality in physical mesh space."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

from analyze_mesh_ordering import _build_ldu_adjacency_stream
from cell_centres import read_cell_centres
from ih_permutation import IHPermutation, read_ih_permutation
from openfoam_mesh_reader import PolyMeshTopology, read_poly_mesh


@dataclass
class LocalityFields:
    row_prev_jump: np.ndarray
    chunk_boundary: np.ndarray
    chunk_boundary_jump: np.ndarray
    psi_jump_mean: np.ndarray
    psi_jump_p90: np.ndarray
    psi_jump_max: np.ndarray


@dataclass
class CaseData:
    name: str
    mesh: PolyMeshTopology
    permutation: IHPermutation
    centres: np.ndarray
    fields: LocalityFields
    statistics: dict[str, object]


def lower_percentile(values: np.ndarray, fraction: float) -> float:
    if values.size == 0:
        return 0.0
    index = int(fraction*(values.size - 1))
    return float(np.partition(values, index)[index])


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean(dtype=np.float64)) if values.size else 0.0,
        "p50": lower_percentile(values, 0.50),
        "p90": lower_percentile(values, 0.90),
        "p99": lower_percentile(values, 0.99),
        "max": float(values.max(initial=0)),
    }


def execution_row_jumps(new_to_old: np.ndarray) -> np.ndarray:
    jumps = np.zeros(new_to_old.size, dtype=np.float64)
    if new_to_old.size > 1:
        jumps[1:] = np.abs(
            new_to_old[1:].astype(np.int64)
            - new_to_old[:-1].astype(np.int64)
        )
    return jumps


def execution_to_original(
    execution_values: np.ndarray, new_to_old: np.ndarray
) -> np.ndarray:
    result = np.empty_like(execution_values)
    result[new_to_old] = execution_values
    return result


def chunk_boundary_fields(
    new_to_old: np.ndarray, level_starts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if level_starts is None:
        raise ValueError("permutation file has no levelStarts; regenerate as v2")
    row_jumps = execution_row_jumps(new_to_old)
    boundary_execution = np.zeros(new_to_old.size, dtype=np.float64)
    boundary_execution[level_starts[:-1]] = 1.0
    boundary_jump_execution = boundary_execution*row_jumps
    return (
        execution_to_original(row_jumps, new_to_old),
        execution_to_original(boundary_execution, new_to_old),
        execution_to_original(boundary_jump_execution, new_to_old),
    )


def psi_jump_fields(
    mesh: PolyMeshTopology, permutation: IHPermutation
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Calculate per-physical-cell jumps from actual incoming|outgoing rows.

    Rows are visited in IH execution order, but columns remain original cell
    indices.  Neighbours are never sorted: incoming and outgoing groups retain
    internal-face ID order, matching the packed IH1 construction.
    """
    n_faces = mesh.n_internal_faces
    owner = np.frombuffer(mesh.owner, dtype=np.int32, count=n_faces)
    neighbour = np.frombuffer(mesh.neighbour, dtype=np.int32)
    stream, degree = _build_ldu_adjacency_stream(
        mesh.n_cells, owner, neighbour
    )
    row_starts = np.empty(mesh.n_cells + 1, dtype=np.int64)
    row_starts[0] = 0
    np.cumsum(degree, out=row_starts[1:])

    means = np.zeros(mesh.n_cells, dtype=np.float64)
    p90s = np.zeros(mesh.n_cells, dtype=np.float64)
    maxima = np.zeros(mesh.n_cells, dtype=np.float64)
    execution_degree = degree[permutation.new_to_old]
    max_degree = int(execution_degree.max(initial=0))
    batch = 250_000
    for row_degree in range(2, max_degree + 1):
        rows = np.flatnonzero(execution_degree == row_degree)
        for begin in range(0, rows.size, batch):
            selected_rows = rows[begin : begin + batch]
            old_cells = permutation.new_to_old[selected_rows]
            positions = (
                row_starts[old_cells, None]
                + np.arange(row_degree, dtype=np.int64)[None, :]
            )
            columns = stream[positions].astype(np.int64, copy=False)
            jumps = np.abs(np.diff(columns, axis=1))
            means[old_cells] = jumps.mean(axis=1)
            maxima[old_cells] = jumps.max(axis=1)
            p90_index = int(0.9*(jumps.shape[1] - 1))
            p90s[old_cells] = np.partition(
                jumps, p90_index, axis=1
            )[:, p90_index]
    return means, p90s, maxima, int(stream.size)


def log_brightness(values: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.log1p(np.maximum(values, 0))/math.log1p(threshold)
    return np.rint(255*np.minimum(scaled, 1.0)).astype(np.uint8)


def normalization_threshold(values: np.ndarray) -> float:
    threshold = lower_percentile(values, 0.99)
    if threshold <= 0:
        threshold = float(values.max(initial=0))
    return threshold if threshold > 0 else 1.0


def projection_coordinates(
    centres: np.ndarray,
    axes: tuple[int, int],
    panel_size: int,
    margin: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    first = centres[:, axes[0]]
    second = centres[:, axes[1]]
    first_min, first_max = float(first.min()), float(first.max())
    second_min, second_max = float(second.min()), float(second.max())
    first_range = max(first_max - first_min, np.finfo(float).eps)
    second_range = max(second_max - second_min, np.finfo(float).eps)
    usable = panel_size - 2*margin
    scale = min(usable/first_range, usable/second_range)
    first_offset = margin + 0.5*(usable - scale*first_range)
    second_offset = margin + 0.5*(usable - scale*second_range)
    x = np.rint(first_offset + scale*(first - first_min)).astype(np.int32)
    y = np.rint(
        panel_size - 1 - (second_offset + scale*(second - second_min))
    ).astype(np.int32)
    np.clip(x, 0, panel_size - 1, out=x)
    np.clip(y, 0, panel_size - 1, out=y)
    return x, y, scale


def save_execution_raster(
    path: Path,
    permutation: IHPermutation,
    width: int,
    threshold: float,
    show_markers: bool,
) -> None:
    jumps = execution_row_jumps(permutation.new_to_old)
    pixels = log_brightness(jumps, threshold)
    height = (pixels.size + width - 1)//width
    raster = np.zeros(height*width, dtype=np.uint8)
    raster[: pixels.size] = pixels
    raster = raster.reshape((height, width))
    if show_markers:
        markers = np.zeros_like(raster)
        positions = permutation.level_starts[:-1]
        markers[positions//width, positions % width] = 255
        canvas = np.zeros((2*height + 1, width), dtype=np.uint8)
        canvas[:height] = raster
        canvas[height + 1 :] = markers
    else:
        canvas = raster
    Image.fromarray(canvas, mode="L").save(path)


def save_projection_triptych(
    path: Path,
    centres: np.ndarray,
    values: np.ndarray,
    threshold: float,
    panel_size: int = 900,
    margin: int = 18,
) -> None:
    brightness = log_brightness(values, threshold)
    projections = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    panels: list[np.ndarray] = []
    for first, second, label in projections:
        x, y, _ = projection_coordinates(
            centres, (first, second), panel_size, margin
        )
        panel = np.zeros((panel_size, panel_size), dtype=np.uint8)
        np.maximum.at(panel, (y, x), brightness)
        image = Image.fromarray(panel, mode="L")
        ImageDraw.Draw(image).text((4, 3), label, fill=160)
        panels.append(np.asarray(image))
    Image.fromarray(np.hstack(panels), mode="L").save(path)


def transition_statistics(
    permutation: IHPermutation,
) -> dict[str, object]:
    jumps = execution_row_jumps(permutation.new_to_old)
    boundaries = np.zeros(jumps.size, dtype=bool)
    boundaries[permutation.level_starts[:-1]] = True
    transition = np.arange(jumps.size) > 0
    boundary_values = jumps[transition & boundaries]
    intra_values = jumps[transition & ~boundaries]
    total = float(jumps.sum(dtype=np.float64))
    return {
        "all": {
            **percentile_summary(jumps),
            "totalMagnitude": total,
        },
        "intraChunk": {
            "count": int(intra_values.size),
            **percentile_summary(intra_values),
        },
        "chunkBoundary": {
            "count": int(boundary_values.size),
            **percentile_summary(boundary_values),
        },
        "chunkBoundaryMagnitudeFraction": (
            float(boundary_values.sum(dtype=np.float64))/total
            if total else 0.0
        ),
    }


def chunk_size_statistics(level_starts: np.ndarray) -> dict[str, float | int]:
    sizes = np.diff(level_starts.astype(np.int64, copy=False))
    if sizes.size == 0:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0,
                "p50": 0.0, "p90": 0.0}
    return {
        "count": int(sizes.size),
        "min": int(sizes.min()),
        "max": int(sizes.max()),
        "mean": float(sizes.mean()),
        "p50": float(np.percentile(sizes, 50, method="lower")),
        "p90": float(np.percentile(sizes, 90, method="lower")),
    }


def aligned_boundary_profile(
    permutation: np.ndarray,
    reference_boundaries: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average row jump at offsets around fixed reference boundaries."""
    if radius < 0:
        raise ValueError("boundary profile radius must be non-negative")
    jumps = execution_row_jumps(permutation)
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    sums = np.zeros(offsets.size, dtype=np.float64)
    counts = np.zeros(offsets.size, dtype=np.int64)
    for boundary in np.asarray(reference_boundaries, dtype=np.int64):
        positions = boundary + offsets
        valid = (positions >= 0) & (positions < jumps.size)
        sums[valid] += jumps[positions[valid]]
        counts[valid] += 1
    means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums),
        where=counts != 0,
    )
    return offsets, means, counts


def build_case(
    name: str, mesh_path: Path, permutation_path: Path, centres_path: Path
) -> CaseData:
    mesh = read_poly_mesh(mesh_path)
    permutation = read_ih_permutation(permutation_path)
    if permutation.level_starts is None:
        raise ValueError(f"{permutation_path} is v1 and lacks levelStarts")
    if permutation.n_cells != mesh.n_cells:
        raise ValueError("mesh/permutation cell count mismatch")
    centres = read_cell_centres(centres_path)
    if centres.shape != (mesh.n_cells, 3):
        raise ValueError("mesh/centroid cell count mismatch")

    row_jump, boundary, boundary_jump = chunk_boundary_fields(
        permutation.new_to_old, permutation.level_starts
    )
    psi_mean, psi_p90, psi_max, reads = psi_jump_fields(mesh, permutation)
    fields = LocalityFields(
        row_jump, boundary, boundary_jump, psi_mean, psi_p90, psi_max
    )
    transitions = transition_statistics(permutation)
    log_row = np.log1p(row_jump)
    log_psi = np.log1p(psi_p90)
    locality_correlation = float(np.corrcoef(log_row, log_psi)[0, 1])
    row_top = row_jump >= normalization_threshold(row_jump)
    psi_top = psi_p90 >= normalization_threshold(psi_p90)
    top_intersection = int(np.count_nonzero(row_top & psi_top))
    top_psi_count = int(np.count_nonzero(psi_top))
    statistics: dict[str, object] = {
        "mesh": name,
        "cells": mesh.n_cells,
        "internalFaces": mesh.n_internal_faces,
        "chunks": int(permutation.level_starts.size - 1),
        "chunkSize": chunk_size_statistics(permutation.level_starts),
        "rowPrevJump": transitions["all"],
        "intraChunk": transitions["intraChunk"],
        "chunkBoundary": transitions["chunkBoundary"],
        "chunkBoundaryMagnitudeFraction": transitions[
            "chunkBoundaryMagnitudeFraction"
        ],
        "psiReads": reads,
        "psiJumpMean": percentile_summary(psi_mean),
        "psiJumpP90": percentile_summary(psi_p90),
        "psiJumpMax": percentile_summary(psi_max),
        "logRowPsiP90Correlation": locality_correlation,
        "psiTop1AlsoRowTop1Fraction": (
            top_intersection/top_psi_count if top_psi_count else 0.0
        ),
    }
    return CaseData(name, mesh, permutation, centres, fields, statistics)


def parse_named(values: list[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{option} requires NAME=PATH")
        name, path = value.split("=", 1)
        if name in result:
            raise SystemExit(f"duplicate {option} name: {name}")
        result[name] = Path(path)
    return result


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument(
        "--permutation", action="append", default=[], metavar="NAME=FILE"
    )
    parser.add_argument(
        "--centroids", action="append", default=[], metavar="NAME=FILE"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raster-width", type=int, default=2048)
    parser.add_argument(
        "--normalization", choices=("local", "shared"), default="local"
    )
    parser.add_argument("--level-markers", action="store_true")
    parser.add_argument(
        "--execution-only",
        action="store_true",
        help="write execution rasters but skip physical projection PNGs",
    )
    parser.add_argument(
        "--boundary-reference",
        help="case NAME whose chunk boundaries align all execution profiles",
    )
    parser.add_argument("--boundary-profile-radius", type=int, default=256)
    arguments = parser.parse_args()

    meshes = parse_named(arguments.mesh, "--mesh")
    permutations = parse_named(arguments.permutation, "--permutation")
    centroids = parse_named(arguments.centroids, "--centroids")
    if not meshes or meshes.keys() != permutations.keys() or meshes.keys() != centroids.keys():
        raise SystemExit("--mesh, --permutation and --centroids names must match")
    if arguments.raster_width < 1:
        raise SystemExit("--raster-width must be positive")
    if arguments.boundary_profile_radius < 0:
        raise SystemExit("--boundary-profile-radius must be non-negative")

    cases = [
        build_case(name, meshes[name], permutations[name], centroids[name])
        for name in meshes
    ]
    reference = None
    if arguments.boundary_reference:
        reference = next(
            (case for case in cases if case.name == arguments.boundary_reference),
            None,
        )
        if reference is None:
            raise SystemExit(
                f"unknown --boundary-reference: {arguments.boundary_reference}"
            )
        large_jump_threshold = float(
            reference.statistics["rowPrevJump"]["p99"]
        )
        for case in cases:
            jumps = execution_row_jumps(case.permutation.new_to_old)
            case.statistics["largeJumpThreshold"] = large_jump_threshold
            case.statistics["largeJumpCount"] = int(
                np.count_nonzero(jumps >= large_jump_threshold)
            )
    field_names = {
        "rowJump": "row_prev_jump",
        "psiP90": "psi_jump_p90",
        "chunkBoundary": "chunk_boundary",
        "chunkBoundaryJump": "chunk_boundary_jump",
        "psiMax": "psi_jump_max",
    }
    thresholds: dict[tuple[str, str], float] = {}
    for output_name, attribute in field_names.items():
        if arguments.normalization == "shared":
            combined = np.concatenate(
                [getattr(case.fields, attribute) for case in cases]
            )
            threshold = normalization_threshold(combined)
            del combined
            for case in cases:
                thresholds[(case.name, output_name)] = threshold
        else:
            for case in cases:
                thresholds[(case.name, output_name)] = normalization_threshold(
                    getattr(case.fields, attribute)
                )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        prefix = (
            f"{safe_name(case.name)}_L{case.permutation.level}_"
            f"{safe_name(case.permutation.variant)}"
        )
        execution_path = arguments.output_dir/f"{prefix}_execution.png"
        save_execution_raster(
            execution_path,
            case.permutation,
            arguments.raster_width,
            thresholds[(case.name, "rowJump")],
            arguments.level_markers,
        )
        print(f"PNG {execution_path}")
        if not arguments.execution_only:
            for output_name, attribute in field_names.items():
                path = arguments.output_dir/f"{prefix}_mesh_{output_name}.png"
                save_projection_triptych(
                    path,
                    case.centres,
                    getattr(case.fields, attribute),
                    thresholds[(case.name, output_name)],
                )
                print(f"PNG {path}")
        print("LOCALITY_STATS " + json.dumps(case.statistics, sort_keys=True))

    if arguments.boundary_reference:
        assert reference is not None
        reference_boundaries = reference.permutation.level_starts[1:-1]
        profiles = {
            case.name: aligned_boundary_profile(
                case.permutation.new_to_old,
                reference_boundaries,
                arguments.boundary_profile_radius,
            )
            for case in cases
        }
        profile_path = arguments.output_dir / (
            f"{safe_name(reference.name)}_aligned_boundary_profile.csv"
        )
        offsets = profiles[reference.name][0]
        with profile_path.open("w", encoding="utf-8") as stream:
            stream.write("offset," + ",".join(case.name for case in cases) + "\n")
            for index, offset in enumerate(offsets):
                values = ",".join(
                    f"{profiles[case.name][1][index]:.17g}" for case in cases
                )
                stream.write(f"{int(offset)},{values}\n")
        print(f"BOUNDARY_PROFILE {profile_path}")

    print()
    print(
        "| Mesh | Cells | Faces | Chunks | row P50 | row P90 | row P99 | "
        "boundary | boundary total Δ | psi reads | psiMean P90 | "
        "psiP90 P90 | psiMax P90 |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for case in cases:
        item = case.statistics
        row = item["rowPrevJump"]
        print(
            f"| {case.name} | {item['cells']} | {item['internalFaces']} | "
            f"{item['chunks']} | {row['p50']:.0f} | {row['p90']:.0f} | "
            f"{row['p99']:.0f} | {item['chunkBoundary']['count']} | "
            f"{100*item['chunkBoundaryMagnitudeFraction']:.3f}% | "
            f"{item['psiReads']} | {item['psiJumpMean']['p90']:.0f} | "
            f"{item['psiJumpP90']['p90']:.0f} | "
            f"{item['psiJumpMax']['p90']:.0f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
