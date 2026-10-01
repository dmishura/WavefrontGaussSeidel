# fwGAMG experimental standalone solver

`fwGAMG` is a serial experimental multigrid path for the standalone test
harness.  The existing `src/GAMG` solver remains the stock-style reference.

The fine OpenFOAM-compatible `lduMatrix` is crossed once during construction.
Every retained level is then stored as an `fwgamg::Matrix`: diagonal values and
symmetric row contributions in CSR form.  `HierarchyOptions::rowOrdering`
selects either natural row order or Index-Kahn order.  The hot smoother, SpMV,
residual, restriction, prolongation and scale-correction paths operate directly
in the selected numbering and do not use `waveCells` or convert fields back to
original numbering.

The optional experimental
`CoarseRenumbering::MinimumFineIndex` pass runs after each fine-to-coarse
operator construction.  It assigns each coarse cell the minimum current-level
fine-cell index in its aggregate, sorts coarse cells by this key, permutes the
coarse matrix consistently and builds a direct restriction/prolongation map for
the new numbering.  The pair-agglomeration algorithm and its mappings are not
changed.  Locality renumbering is timed separately as
`localityRenumberSeconds` for every coarse level.

The current implementation reuses the validated standalone OpenFOAM-v2606
pair-agglomeration metadata.  That metadata builder still creates temporary
`GAMGMatrix` levels internally to propagate algebraic-pair weights.  This is the
only intermediate-level LDU bridge and is reported as `intermediate coarse LDU
construction`.  The actual fw coarse operators are independently accumulated
directly from the preceding fw matrix; there is no coarse LDU-to-fw repacking.
The coarsest matrix is solved directly from fw rows with the existing small
dense pivoted algorithm, so coarsest fw-to-LDU conversion is currently zero.

Fine input vectors are permuted once at the start of `solve()` and the solution
is inverse-permuted once at the end.  Each transition owns a precomputed direct
fine-fw-row to coarse-fw-row map.

Current limitations match the standalone GAMG baseline: symmetric matrices,
serial scalar execution, no interfaces/MPI/processor agglomeration and no
`interpolateCorrection`.

Build and validate:

```bash
cmake --build build-cmake -j4
ctest --test-dir build-cmake -R motorBike_gamg_correctness --output-on-failure
```

Pinned 1-thread comparison:

```bash
OMP_NUM_THREADS=1 taskset -c 2 ./build-cmake/gamg_bench \
  --mesh=build-cmake/mesh-data/MTB_example_polyMesh \
  --gamg_warmup=1 --gamg_samples=7 \
  '--benchmark_filter=.*GAMG/MotorBike/(Setup|Solve).*' \
  --benchmark_repetitions=7 \
  --benchmark_report_aggregates_only=true
```

Per-level LDU, natural-CSR, Index-Kahn-CSR and locality-renumbered diagnostics:

```bash
OMP_NUM_THREADS=1 taskset -c 2 ./build-cmake/gamg_level_bench \
  --mesh=build-cmake/mesh-data/MTB_example_polyMesh \
  '--benchmark_filter=^GAMGLevel/.*' \
  --benchmark_repetitions=5 \
  --benchmark_enable_random_interleaving=true \
  --benchmark_report_aggregates_only=true
```

This diagnostic reports GS, SpMV, residual and scale-correction independently,
as well as locality and dependency-level statistics.  The relevant variants
are `B_CSRNatural`, `D_CSRLocalNatural` and `E_CSRLocalIndexKahn`.

The complete setup/solve comparison is registered as:

```text
fwGAMG/MotorBike/Natural/{Setup,Solve}
fwGAMG/MotorBike/LocalNatural/{Setup,Solve}
fwGAMG/MotorBike/LocalIndexKahn/{Setup,Solve}
```

The production fwGAMG SpMV uses a direct symmetric edge view containing one
`owner`, `column` and `coefficient` entry per internal face.  Residual and
`scaleCorrection` use the same direct traversal.  Packed CSR remains the
unchanged IH1/GS representation; its row-wise SpMV remains available as a
benchmark/reference control.

An experimental indirect symmetric flat edge-wise SpMV reuses the existing CSR arrays.
It stores `owner` and `upperEntry` for each unique edge, then processes all
faces in one long loop and updates both result cells.  Metadata construction is
reported separately as `flatEdgeMetadata`.
This sub-timer is contained within the inclusive `packing` setup timer.
The production direct view removes the `upperEntry -> CSR entry` lookup.  Its
construction time is reported as `directFlatEdgeMetadata`; it is also contained
within the inclusive `packing` timer.  Neither diagnostic view changes the CSR
rows used by IH1/GS.
The complete comparison is registered as:

```text
fwGAMG/MotorBike/{Setup,Solve}
fwGAMG/MotorBike/FlatEdgeSpMV/{Setup,Solve}
GAMG/Production/ReferenceGS/{Setup,Solve}
fwGAMG/Production/RowCSR/{Setup,Solve}
fwGAMG/Production/DirectFlat/{Setup,Solve}
```

Per-level SpMV names are `A_LDU`, `B_CSRNatural`,
`B2_CSRFlatEdgeNatural` and `B3_CSRDirectFlatEdgeNatural`.  Their counters
include `ns/cell`, `ns/nnz` and `ns/internal-face`.  The level benchmark also
validates all three packed traversals against LDU before timing and prints the
metadata construction time and storage of both flat views.
