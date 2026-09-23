# Test harness refactor map

This document records the behavior that must remain stable while the legacy
`Test-GaussSeidel.C` driver is migrated to registries.

## Existing workload groups

| Legacy entry point | Workload / purpose | Timing model |
|---|---|---|
| `runTest` | motorBike/default mesh, all baseline diagnostics | 7 samples, 250 sweeps; production mode is 100 calls x 3 sweeps |
| `--width-benchmark-only` | Geometry-X widths 256..16384 | 7 samples, 100 calls x 3 sweeps |
| `--hpc-series` | MTBHPC Geometry-X width series | 5 samples, 27 calls x 3 sweeps |
| `--hpc-prefetch-series` | packed and Geometry-8192 prefetch | 5 samples, 27 calls x 3 sweeps |
| `--hpc-best-series` | Reference, packed, Geometry-8192 | 5 samples, 27 calls x 3 sweeps |
| `--hpc-hybrid-series` | hybrid edge-value representation | 5 samples, 27 calls x 3 sweeps |
| `--hpc-geometry-kernels` | scalar, generic-4 and AVX-512 rows | 5 samples, 27 calls x 3 sweeps |
| `--hpc-row-degree` | row-degree scalar/unrolled/AVX-512 divide | randomized production batching |
| `--hpc-blocked-row-degree` | persistent OMP block sizes 32..256 | randomized production batching |
| `--hpc-index-series` | Index/Index-window Kahn widths | 5 samples, 27 calls x 3 sweeps |
| `--hpc-direct-coeff-series` | packed coefficients versus live coefficients | randomized production batching |
| `--profile-variant` | short VTune workloads | warm-up plus about 20 sweeps |

The synthetic asymmetric-LDU test is a correctness workload and is never
timed. Mesh reading has its own standalone correctness executable.

## Correctness policies

The unchanged OpenFOAM forward Gauss-Seidel implementation is the reference.
Scalar packed, reordered same-level schedules, compressed addressing,
row-degree, and OpenMP wavefront variants require exact equality. AVX2/AVX-512
intra-row reductions use a `1e-12` tolerance. Symmetric Gauss-Seidel is not
compared with a forward sweep; it must reduce residual and converge to the
known solution. Every schedule must cover every cell and satisfy
`level[owner] < level[neighbour]`.

Index schedules additionally retain the non-zero initial-field, three-sweep,
two-call and asymmetric-matrix checks from the legacy driver.

## Shared registry model

`WorkloadRegistry` owns mesh, matrix, fields and all preprocessing. Its
objects outlive solver callables. `SmootherRegistry` stores generic
`VariantMetadata`, correctness policy, batching metadata and one outer-call
`std::function`. Tier/tag parsing and matching live in `VariantSelection` and
have no smoother-specific dependencies. No callable is invoked per cell or
per contribution.

GoogleTest consumes correctness-enabled entries and applies their declared
policy. Google Benchmark consumes benchmark-enabled entries and performs field
reset outside manual timing. Schedule construction, mesh I/O and matrix setup
occur before registration and outside all timed regions.

The legacy executable remains available during migration. It is the parity
oracle for experimental CLI workloads until each group has moved to a named
workload factory. It must only be removed after every row in the table above
is represented in `WorkloadRegistry`.
