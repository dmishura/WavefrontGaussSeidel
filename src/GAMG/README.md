# Standalone GAMG baseline

This directory contains a serial, symmetric-matrix adaptation of the stock
OpenFOAM v2606 GAMG algorithm. It is deliberately independent of the previous
Index-Kahn OpenFOAM patches in this repository.

The source mapping is:

- `GAMGAgglomeration.*`: `pairGAMGAgglomeration` and the serial/internal-face
  portion of `GAMGAgglomerateLduAddressing.C`;
- `GAMGMatrix.*`: the symmetric branch of
  `GAMGSolver::agglomerateMatrix` and standalone LDU storage;
- `GAMGKernels.*`: restriction, injection prolongation, `A*x`, residual,
  correction scaling, and OpenFOAM-style lexicographic Gauss-Seidel;
- `GAMGSolver.*`: the default non-interpolating V-cycle and convergence loop;
- `GAMGControls.H`: standalone equivalents of the relevant dictionary
  controls.

The hierarchy uses the stock `algebraicPair` weighting, `abs(upper)`. The test
compatibility matrix has no `fvMesh` face-area field, so the finite-volume
`faceAreaPair` front-end is outside this standalone component. The underlying
pair selection and coarse-addressing algorithms are the v2606 algorithms.

Current intentional limits are: serial execution, scalar symmetric matrices,
no interfaces, no MPI/process agglomeration, and no `interpolateCorrection`.
The coarsest level is solved by a small dense pivoted solve because the current
standalone tree has no retained PCG/DIC implementation. This corresponds to
the OpenFOAM `directSolveCoarsest` path, with a compatibility-layer dense
solver replacing `LUscalarMatrix`.

Files adapted from OpenFOAM remain under the GNU General Public License,
version 3 or later, with their original project attribution retained in the
source headers.

## Baseline timing

`GAMGTimingStats` is an optional accumulator passed to `GAMGSolver`. A null
pointer leaves instrumentation disabled. Enabled instrumentation times whole
operations with `std::chrono::steady_clock`; timer calls are never placed in
cell or face loops.

The timing conventions are:

- topology hierarchy construction and per-transition coarse matrix
  construction are separate setup categories;
- smoothing, `Amul`, residual update, restriction, prolongation and the
  coarsest solve are exclusive kernel timings;
- complete `scaleCorrection` is inclusive of its separately recorded `Amul`;
- V-cycle and solve durations are inclusive wall times, so overlapping
  categories must not be summed.

The dedicated Google Benchmark executable provides setup, V-cycle and full
solve entries, plus an accumulated per-level timing report after warm-up:

```bash
OMP_NUM_THREADS=1 taskset -c 2 ./build-cmake/gamg_bench \
    --mesh=build-cmake/mesh-data/MTB_example_polyMesh \
    --gamg_warmup=1 \
    --gamg_samples=7 \
    --benchmark_repetitions=7 \
    --benchmark_report_aggregates_only=true
```

Add `--gamg_csv` to print the per-level counters as CSV.
