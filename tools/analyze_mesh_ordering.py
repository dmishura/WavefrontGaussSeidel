#!/usr/bin/env python3
"""Analyse locality and smoothness of existing OpenFOAM cell numbering.

The deterministic adjacency stream uses original cell order.  Within each
cell row it lists incoming neighbours in internal-face ID order, followed by
outgoing neighbours in internal-face ID order.  Stream roughness is a
heuristic normalized against two independent indices spanning [0, N); it is
not a probabilistic model of an adjacency graph.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - environment diagnostic
    raise SystemExit("analyze_mesh_ordering.py requires NumPy") from error

from ih_permutation import IHPermutation, read_ih_permutation
from openfoam_mesh_reader import PolyMeshTopology, read_poly_mesh


@dataclass
class OrderingMetrics:
    mesh: str
    numbering: str
    path: str
    permutation_path: str | None
    cells: int
    internal_faces: int
    mean_edge_delta: float
    median_edge_delta: int
    p90_edge_delta: int
    p99_edge_delta: int
    max_edge_delta: int
    normalized_mean_edge_delta: float
    normalized_median_edge_delta: float
    normalized_p90_edge_delta: float
    normalized_p99_edge_delta: float
    same64b: float
    same4kib: float
    same2mib: float
    mean_abs_jump: float
    median_abs_jump: int
    p90_abs_jump: int
    p99_abs_jump: int
    stream_roughness: float
    mean_backward_distance: float
    normalized_backward_distance: float
    normalized_neighbour_center_offset: float
    checksum: str
    permutation_checksum: str | None


BENCHMARK_PATHS = (
    (
        "MotorBike",
        "motorBike / MTB_example",
        ("mesh-data/MTB_example_polyMesh", "all-mesh-data/MotorBike/polyMesh"),
    ),
    (
        "AHBody_original", "AHBody original",
        ("all-mesh-data/AHBody_original/polyMesh",),
    ),
    (
        "AHBody_renumbered", "AHBody renumbered",
        ("all-mesh-data/AHBody_renumbered/polyMesh",),
    ),
    (
        "MTBHPC_small_original", "MTBHPC_small original",
        ("all-mesh-data/MTBHPC_original/polyMesh",),
    ),
    (
        "MTBHPC_small_renumbered",
        "MTBHPC_small renumbered",
        (
            "all-mesh-data/MTBHPC_renumbered/polyMesh",
            "mesh-data/MTBHPC_small_renumbered/polyMesh",
        ),
    ),
    (
        "WD_DamBreak_original", "WD_DamBreak original",
        ("all-mesh-data/WD_DamBreak_original/polyMesh",),
    ),
    (
        "WD_DamBreak_renumbered",
        "WD_DamBreak renumbered",
        ("all-mesh-data/WD_DamBreak_renumbered/polyMesh",),
    ),
)


def discover_benchmark_meshes(root: Path) -> list[tuple[str, str, Path]]:
    result: list[tuple[str, str, Path]] = []
    for mesh_id, name, relative_candidates in BENCHMARK_PATHS:
        for relative in relative_candidates:
            candidate = root / relative
            if (candidate / "owner").is_file() and (
                candidate / "neighbour"
            ).is_file():
                result.append((mesh_id, name, candidate))
                break
        else:
            tried = ", ".join(str(root / item) for item in relative_candidates)
            raise FileNotFoundError(f"cannot locate {name}; tried: {tried}")
    return result


def _lower_percentiles(values: np.ndarray, fractions: Iterable[float]) -> list[int]:
    """Exact lower empirical percentiles, calculated by in-place partition."""
    if values.size == 0:
        return [0 for _ in fractions]
    indices = [int(fraction * (values.size - 1)) for fraction in fractions]
    values.partition(tuple(sorted(set(indices))))
    return [int(values[index]) for index in indices]


def _build_ldu_adjacency_stream(
    n_cells: int, owner: np.ndarray, neighbour: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pack rows as incoming then outgoing, preserving face-ID order in each."""
    n_faces = owner.size
    incoming_counts = np.bincount(neighbour, minlength=n_cells).astype(
        np.int64, copy=False
    )
    outgoing_counts = np.bincount(owner, minlength=n_cells).astype(
        np.int64, copy=False
    )
    degree = incoming_counts + outgoing_counts
    row_starts = np.empty(n_cells + 1, dtype=np.int64)
    row_starts[0] = 0
    np.cumsum(degree, out=row_starts[1:])
    incoming_starts = np.empty(n_cells + 1, dtype=np.int64)
    incoming_starts[0] = 0
    np.cumsum(incoming_counts, out=incoming_starts[1:])
    outgoing_starts = np.empty(n_cells + 1, dtype=np.int64)
    outgoing_starts[0] = 0
    np.cumsum(outgoing_counts, out=outgoing_starts[1:])

    stream = np.empty(2 * n_faces, dtype=np.int32)
    incoming_order = np.argsort(neighbour, kind="stable")
    incoming_shift = row_starts[:-1] - incoming_starts[:-1]
    chunk = 1_000_000
    for begin in range(0, n_faces, chunk):
        end = min(begin + chunk, n_faces)
        faces = incoming_order[begin:end]
        cells = neighbour[faces]
        positions = np.arange(begin, end, dtype=np.int64) + incoming_shift[cells]
        stream[positions] = owner[faces]
    del incoming_order, incoming_shift, incoming_starts

    outgoing_order = np.argsort(owner, kind="stable")
    outgoing_shift = row_starts[:-1] + incoming_counts - outgoing_starts[:-1]
    for begin in range(0, n_faces, chunk):
        end = min(begin + chunk, n_faces)
        faces = outgoing_order[begin:end]
        cells = owner[faces]
        positions = np.arange(begin, end, dtype=np.int64) + outgoing_shift[cells]
        stream[positions] = neighbour[faces]
    return stream, degree


def _remap_adjacency_stream(
    natural_stream: np.ndarray,
    natural_degree: np.ndarray,
    permutation: IHPermutation,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder complete natural rows and remap their column indices."""
    n_cells = natural_degree.size
    old_row_starts = np.empty(n_cells + 1, dtype=np.int64)
    old_row_starts[0] = 0
    np.cumsum(natural_degree, out=old_row_starts[1:])
    degree = natural_degree[permutation.new_to_old]
    row_starts = np.empty(n_cells + 1, dtype=np.int64)
    row_starts[0] = 0
    np.cumsum(degree, out=row_starts[1:])
    stream = np.empty_like(natural_stream)

    chunk = 4_000_000
    for begin in range(0, stream.size, chunk):
        end = min(begin + chunk, stream.size)
        positions = np.arange(begin, end, dtype=np.int64)
        rows = np.searchsorted(row_starts, positions, side="right") - 1
        old_rows = permutation.new_to_old[rows]
        source_positions = (
            old_row_starts[old_rows] + positions - row_starts[rows]
        )
        stream[begin:end] = permutation.old_to_new[
            natural_stream[source_positions]
        ]
    return stream, degree


def _stream_statistics(stream: np.ndarray, n_cells: int) -> dict[str, float | int]:
    count = stream.size - 1
    if count <= 0:
        return {
            "mean_abs_jump": 0.0,
            "median_abs_jump": 0,
            "p90_abs_jump": 0,
            "p99_abs_jump": 0,
            "stream_roughness": 0.0,
            "mean_backward_distance": 0.0,
            "normalized_backward_distance": 0.0,
        }

    abs_jumps = np.empty(count, dtype=np.uint32)
    absolute_sum = 0
    backward_sum = 0
    chunk = 4_000_000
    for begin in range(0, count, chunk):
        end = min(begin + chunk, count)
        delta = (
            stream[begin + 1 : end + 1].astype(np.int64)
            - stream[begin:end].astype(np.int64)
        )
        absolute = np.abs(delta)
        abs_jumps[begin:end] = absolute
        absolute_sum += int(absolute.sum(dtype=np.uint64))
        backward = delta < 0
        backward_sum += int((-delta[backward]).sum(dtype=np.uint64))

    median, p90, p99 = _lower_percentiles(abs_jumps, (0.5, 0.9, 0.99))
    mean_abs = absolute_sum / count
    mean_backward = backward_sum / count
    return {
        "mean_abs_jump": mean_abs,
        "median_abs_jump": median,
        "p90_abs_jump": p90,
        "p99_abs_jump": p99,
        "stream_roughness": 3.0 * mean_abs / (n_cells + 1),
        "mean_backward_distance": mean_backward,
        "normalized_backward_distance": mean_backward / n_cells,
    }


def analyse_mesh(
    name: str,
    path: Path,
    permutation: IHPermutation | None = None,
    mesh: PolyMeshTopology | None = None,
    natural_stream: np.ndarray | None = None,
    natural_degree: np.ndarray | None = None,
) -> OrderingMetrics:
    if mesh is None:
        mesh = read_poly_mesh(path)
    cells = mesh.n_cells
    faces = mesh.n_internal_faces
    original_owner = np.frombuffer(mesh.owner, dtype=np.int32, count=faces)
    original_neighbour = np.frombuffer(mesh.neighbour, dtype=np.int32)
    if permutation is None:
        numbering = "Original"
        permutation_path = None
        permutation_checksum = None
        owner = original_owner
        neighbour = original_neighbour
    else:
        if permutation.level != 0:
            raise ValueError(
                f"{permutation.path} describes GAMG level {permutation.level}; "
                "polyMesh topology analysis requires level 0"
            )
        if permutation.n_cells != cells:
            raise ValueError(
                f"{permutation.path} has {permutation.n_cells} cells, "
                f"but {path} has {cells}"
            )
        numbering = permutation.variant
        permutation_path = str(permutation.path)
        digest = hashlib.sha256()
        digest.update(permutation.new_to_old.tobytes())
        digest.update(permutation.old_to_new.tobytes())
        permutation_checksum = digest.hexdigest()
        mapped_owner = permutation.old_to_new[original_owner]
        mapped_neighbour = permutation.old_to_new[original_neighbour]
        owner = np.minimum(mapped_owner, mapped_neighbour)
        neighbour = np.maximum(mapped_owner, mapped_neighbour)

    edge_delta = np.maximum(owner, neighbour) - np.minimum(owner, neighbour)
    mean_edge_delta = float(edge_delta.mean(dtype=np.float64))
    max_edge_delta = int(edge_delta.max(initial=0))
    same64b = float(np.count_nonzero(owner // 8 == neighbour // 8) / faces)
    same4kib = float(np.count_nonzero(owner // 512 == neighbour // 512) / faces)
    same2mib = float(
        np.count_nonzero(owner // 262144 == neighbour // 262144) / faces
    )
    edge_median, edge_p90, edge_p99 = _lower_percentiles(
        edge_delta, (0.5, 0.9, 0.99)
    )
    del edge_delta

    if natural_stream is None or natural_degree is None:
        natural_stream, natural_degree = _build_ldu_adjacency_stream(
            cells, original_owner, original_neighbour
        )
    if permutation is None:
        stream = natural_stream
        degree = natural_degree
    else:
        stream, degree = _remap_adjacency_stream(
            natural_stream, natural_degree, permutation
        )
    stream_stats = _stream_statistics(stream, cells)
    if permutation is not None:
        del stream

    neighbour_sum = np.bincount(
        owner, weights=neighbour, minlength=cells
    ) + np.bincount(neighbour, weights=owner, minlength=cells)
    nonisolated = degree != 0
    indices = np.arange(cells, dtype=np.float64)
    center_offset = float(
        np.mean(
            np.abs(
                indices[nonisolated]
                - neighbour_sum[nonisolated] / degree[nonisolated]
            )
        )
        / cells
    )

    return OrderingMetrics(
        mesh=name,
        numbering=numbering,
        path=str(mesh.poly_mesh_dir),
        permutation_path=permutation_path,
        cells=cells,
        internal_faces=faces,
        mean_edge_delta=mean_edge_delta,
        median_edge_delta=edge_median,
        p90_edge_delta=edge_p90,
        p99_edge_delta=edge_p99,
        max_edge_delta=max_edge_delta,
        normalized_mean_edge_delta=mean_edge_delta / cells,
        normalized_median_edge_delta=edge_median / cells,
        normalized_p90_edge_delta=edge_p90 / cells,
        normalized_p99_edge_delta=edge_p99 / cells,
        same64b=same64b,
        same4kib=same4kib,
        same2mib=same2mib,
        normalized_neighbour_center_offset=center_offset,
        checksum=mesh.topology_checksum(),
        permutation_checksum=permutation_checksum,
        **stream_stats,
    )


def print_markdown(metrics: list[OrderingMetrics]) -> None:
    print()
    print(
        "| Mesh | Numbering | Cells | Faces | meanEdgeDelta/N | edge P50/N | "
        "edge P90/N | edge P99/N | same64B | same4KiB | same2MiB | "
        "roughness R | backward distance/N | neighbour-center offset/N |"
    )
    print(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|"
    )
    for item in metrics:
        print(
            f"| {item.mesh} | {item.numbering} | {item.cells} | "
            f"{item.internal_faces} | "
            f"{item.normalized_mean_edge_delta:.6g} | "
            f"{item.normalized_median_edge_delta:.6g} | "
            f"{item.normalized_p90_edge_delta:.6g} | "
            f"{item.normalized_p99_edge_delta:.6g} | "
            f"{100*item.same64b:.4f}% | {100*item.same4kib:.4f}% | "
            f"{100*item.same2mib:.4f}% | {item.stream_roughness:.6g} | "
            f"{item.normalized_backward_distance:.6g} | "
            f"{item.normalized_neighbour_center_offset:.6g} |"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        help="build directory containing mesh-data and all-mesh-data",
    )
    parser.add_argument(
        "--permutation-root",
        type=Path,
        help=(
            "root containing MESH_ID/*.ihperm dumps; requires "
            "--benchmark-root"
        ),
    )
    parser.add_argument(
        "--permutation",
        action="append",
        default=[],
        type=Path,
        help="permutation file applied to every explicitly supplied --mesh",
    )
    parser.add_argument(
        "--mesh",
        action="append",
        default=[],
        metavar="NAME=POLYMESH",
        help="analyse an explicitly named mesh; may be repeated",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.permutation_root and not arguments.benchmark_root:
        raise SystemExit("--permutation-root requires --benchmark-root")
    if arguments.permutation and not arguments.mesh:
        raise SystemExit("--permutation requires at least one explicit --mesh")
    meshes: list[tuple[str, str, Path]] = []
    if arguments.benchmark_root:
        meshes.extend(discover_benchmark_meshes(arguments.benchmark_root))
    for specification in arguments.mesh:
        if "=" not in specification:
            raise SystemExit("--mesh requires NAME=POLYMESH")
        name, path = specification.split("=", 1)
        meshes.append((name, name, Path(path)))
    if not meshes:
        raise SystemExit("provide --benchmark-root or at least one --mesh")

    results: list[OrderingMetrics] = []
    for mesh_id, name, path in meshes:
        print(f"Analysing {name}: {path}", file=sys.stderr, flush=True)
        mesh_topology = read_poly_mesh(path)
        base_owner = np.frombuffer(
            mesh_topology.owner,
            dtype=np.int32,
            count=mesh_topology.n_internal_faces,
        )
        base_neighbour = np.frombuffer(
            mesh_topology.neighbour, dtype=np.int32
        )
        natural_stream, natural_degree = _build_ldu_adjacency_stream(
            mesh_topology.n_cells, base_owner, base_neighbour
        )
        permutation_paths: list[Path | None] = [None]
        if arguments.permutation_root:
            mesh_permutations = sorted(
                (arguments.permutation_root / mesh_id).glob("*.ihperm")
            )
            if not mesh_permutations:
                raise FileNotFoundError(
                    f"no .ihperm files for {mesh_id} below "
                    f"{arguments.permutation_root}"
                )
            permutation_paths.extend(mesh_permutations)
        if arguments.mesh and not arguments.benchmark_root:
            permutation_paths.extend(arguments.permutation)

        for permutation_path in permutation_paths:
            permutation = (
                read_ih_permutation(permutation_path)
                if permutation_path is not None
                else None
            )
            result = analyse_mesh(
                name,
                path,
                permutation,
                mesh_topology,
                natural_stream,
                natural_degree,
            )
            results.append(result)
            print("MESH_METRICS " + json.dumps(asdict(result), sort_keys=True))
            del result
            del permutation
            gc.collect()
        del natural_stream, natural_degree, mesh_topology
        gc.collect()
    print_markdown(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
