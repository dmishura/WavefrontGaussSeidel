# Wavefront Gauss-Seidel Lab

This public repository explores Gauss-Seidel smoothers for OpenFOAM-style
LDU matrices. It contains a standalone compatibility layer, realistic mesh
workloads, correctness tests, performance benchmarks, and experimental
wavefront/Index-Kahn implementations for serial, OpenMP, and SIMD execution.

The main goal is to understand how dependency scheduling, data layout, cache
locality, and parallel execution affect smoothing quality and performance,
while keeping the original OpenFOAM Gauss-Seidel implementation as the
reference baseline.

The OpenMP implementation is included as a straightforward parallelization
example. No OpenMP-specific tuning or optimization has been applied: it runs
essentially "as is", and its current performance results are unlikely to be
optimal.

The repository also contains experimental OpenFOAM v2606 integration patches.
OpenFOAM source trees themselves are treated as external, immutable references.
The author is currently working on integration with OpenFOAM/GAMG, but
this integration is still under development and is not ready for production
use.

## Sequential performance: packing vs. Index-Kahn ordering

- `ReferenceGS` — original OpenFOAM-style Gauss–Seidel implementation
- `PackedReferenceGS` — the same lexicographic Gauss–Seidel sweep on a packed CSR-like matrix representation
- `IndexKahn1024` — packed Index-Kahn traversal with width 1024 (IH1)

Benchmark conditions:

- GCC `-O2`, 1 thread, pinned to CPU 2
- 5 samples × 81 sweeps, 3 sweeps per benchmark call
- randomized benchmark order

| Mesh | Original, ms | Packed, ms | IK-1024, ms | Packed / Original | IK / Original | IK / Packed |
|---|---:|---:|---:|---:|---:|---:|
| MTB example | 3.562 | 2.434 | 2.448 | 1.463× | 1.455× | 0.994× |
| AHBody original | 23.406 | 19.934 | 20.344 | 1.174× | 1.151× | 0.980× |
| AHBody renumbered | 20.394 | 17.356 | 16.478 | 1.175× | 1.238× | 1.053× |
| MTBHPC original | 109.009 | 82.856 | 81.589 | 1.316× | 1.336× | 1.016× |
| MTBHPC renumbered | 94.579 | 64.010 | 61.412 | 1.478× | 1.540× | 1.042× |
| WD DamBreak original | 111.299 | 100.805 | 79.516 | 1.104× | 1.400× | 1.268× |
| WD DamBreak renumbered | 54.949 | 45.649 | 44.848 | 1.204× | 1.225× | 1.018× |

### Interpretation

For most meshes, matrix packing accounts for the majority of the single-thread performance improvement. Once the matrix is packed, `IndexKahn1024` is usually within a few percent of packed lexicographic Gauss–Seidel.

This means that Index-Kahn should not be interpreted primarily as a single-thread optimization. Its main purpose is to expose parallelism while retaining nearly the same sequential throughput as a well-packed lexicographic Gauss–Seidel implementation.

There is, however, one important exception: `WD DamBreak original`. On this mesh, Index-Kahn is 1.268× faster than packed lexicographic Gauss–Seidel. After `renumberMesh`, the difference drops to only 1.018×.

This strongly suggests that the large gain on the original WD DamBreak mesh comes from improved memory locality caused by the Index-Kahn traversal order. In other words, Index-Kahn can sometimes act as an implicit renumbering/locality optimization in addition to exposing parallelism.

Overall, the sequential results suggest three partially independent performance effects:

1. **Matrix packing** reduces the overhead of the original sparse matrix representation and makes row data contiguous.
2. Standard OpenFOAM **mesh renumbering** improves memory locality of the mesh data structures in general, benefiting Gauss–Seidel as well as other mesh-based operations.
3. **Index-Kahn ordering** exposes parallelism and can additionally improve locality on poorly numbered meshes.

Across most tested meshes, packing dominates the single-thread speedup, while Index-Kahn preserves that performance and provides the structure required for parallel execution.

## Test meshes

The benchmark meshes used by this project are published in the dedicated
[`dmishura/OpenFOAM-Benchmark-Meshes`](https://github.com/dmishura/OpenFOAM-Benchmark-Meshes)
repository as GitHub Release assets. The current dataset is `dataset-v1`.
Archives are downloaded on demand and cached under `meshes/`; they are not
stored in this repository.

```bash
python3 meshes/fetch_mesh.py MTB_example
python3 meshes/fetch_mesh.py MTBHPC_small renumbered
python3 meshes/fetch_mesh.py --all
```

The default Make and CMake MotorBike preparation paths fetch the one required
archive automatically. See [`meshes/README.md`](meshes/README.md) for cache,
verification, offline-use, and custom-workload details.

## Usage and attribution

The original work in this repository may be used, modified, and redistributed
freely, provided that credit is given to **Dmitry Mishura** and this repository
is identified as the source.

Files copied or derived from OpenFOAM remain subject to their original GNU
General Public License and copyright notices. Third-party components retain
their respective licenses.
