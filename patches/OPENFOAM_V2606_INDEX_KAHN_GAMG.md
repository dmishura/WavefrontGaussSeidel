# Size-gated Index-Kahn smoother inside GAMG (OpenFOAM v2606)

The production patch keeps the existing `indexKahn` runtime smoother and adds
a configurable `minCells` cutoff. A matrix with fewer local cells executes
OpenFOAM's standard `GaussSeidelSmoother::smooth()` and never accesses the
Index-Kahn cache or coefficient-packing path.

No `IndexKahnGAMG` class is introduced. The temporary `GAMGDBG` lifetime
instrumentation is not included.

## Patch choice and application

After applying the GAMG/minCells integration, apply the nested-controls
forwarding fix:

```sh
git apply --check \
    /home/dmishura/uGS/SmootherTest/patches/openfoam-v2606-index-kahn-gamg-controls.patch
git apply \
    /home/dmishura/uGS/SmootherTest/patches/openfoam-v2606-index-kahn-gamg-controls.patch
```

This patch makes `GAMGSolver::initVcycle()` extract the smoother name and the
exact nested smoother dictionary once, then pass both explicitly to the
factory for the finest and every coarse level. The original factory overload
is retained, so `smoothSolver` and primitive `smoother GaussSeidel;` syntax
remain unchanged.

The primary patch for a tree that already has the previous GAMG integration
is:

```sh
git apply --check \
    /home/dmishura/uGS/SmootherTest/patches/openfoam-v2606-index-kahn-gamg-min-cells.patch
git apply \
    /home/dmishura/uGS/SmootherTest/patches/openfoam-v2606-index-kahn-gamg-min-cells.patch
```

It changes only four files in `src/indexKahnSmoother`.

For a fresh tree where only `openfoam-v2606-index-kahn.patch` has been applied,
use the combined GAMG patch instead:

```sh
git apply --check \
    /home/dmishura/uGS/SmootherTest/patches/openfoam-v2606-index-kahn-gamg.patch
git apply \
    /home/dmishura/uGS/SmootherTest/patches/openfoam-v2606-index-kahn-gamg.patch
```

The combined patch also adds the topology-lifetime cache slot to
`lduPrimitiveMesh`. Do not apply both patches.

## Rebuild

For the incremental `minCells` patch, rebuild only the user library:

```sh
wmake libso src/indexKahnSmoother
```

The nested-controls forwarding patch changes `lduMatrix.H`,
`lduMatrixSmoother.C`, and `GAMGSolverSolve.C`, so rebuild `libOpenFOAM` as
well as the user library after applying it. Follow the normal v2606 build
procedure used for core library changes; do not mix the changed header with
an older `libOpenFOAM` binary.

For the combined patch, `lduPrimitiveMesh.H` changes layout. A full rebuild is
the safe choice, followed by rebuilding the user library:

```sh
./Allwmake
wmake libso src/indexKahnSmoother
```

The affected user-library target is
`$(FOAM_USER_LIBBIN)/libIndexKahnSmoother.so`. The combined patch additionally
affects core `libOpenFOAM` ABI through `lduPrimitiveMesh`; do not mix binaries
built against the old header. Neither patch was compiled or run while being
prepared.

## fvSolution syntax

OpenFOAM v2606 requires the nested smoother dictionary for smoother-specific
controls:

```foam
p
{
    solver          GAMG;

    smoother
    {
        smoother    indexKahn;
        width       1024;
        minCells    10000;
    }

    tolerance       1e-7;
    relTol          0.01;
}
```

`width` defaults to 1024 and `minCells` defaults to 10000. `minCells 0` enables
Index-Kahn on every non-empty level. The cutoff is based on the actual local
matrix size on each MPI rank:

```cpp
matrix.diag().size() >= minCells
```

The value 10000 is only an initial experimental cutoff. It is not encoded as
an assumed optimum and can be tested with values such as 0, 2000, 5000,
10000, 20000, and 50000.

Keep the user library in `system/controlDict`:

```foam
libs ("libIndexKahnSmoother.so");

DebugSwitches
{
    indexKahn                 1;
    IndexKahnScheduleCache    1;
}
```

No separate `IndexKahnGAMGDebug` switch is required. The existing `indexKahn`
switch reports the level decision and timing; `IndexKahnScheduleCache` reports
cache build/reuse.

## Selection and fallback path

The constructor validates `width` and `minCells`, then checks the cell count
before any Index-Kahn-specific operation. For a small level it returns before:

* `IndexKahnScheduleCache::New()`;
* `getOrCreate()`;
* schedule allocation or construction;
* `coeffs_` allocation;
* coefficient packing.

The smoother object retains `schedule_ == nullptr` and an empty `coeffs_`.
Its `smoothInternal()` immediately calls the unchanged static
`GaussSeidelSmoother::smooth()` implementation with the current matrix,
source, interfaces, component, and sweep count. Thus coupled-interface
handling and local arithmetic are the standard OpenFOAM Gauss-Seidel path;
the fallback does not construct and reinterpret an Index-Kahn schedule.

For an eligible level the existing behavior is unchanged: obtain the
topology-owned schedule, pack current off-diagonal coefficients in the
smoother constructor, read the current matrix diagonal directly, and execute
the Index-Kahn kernel. Coefficient-packing and kernel timing remain guarded by
the `indexKahn` debug switch. The Gauss-Seidel fallback is not timed by the
plugin.

## Hierarchy ownership and cache lifetime

The earlier v2606 lifetime diagnostics established that `GAMGAgglomeration`
persists across pressure solves, while `GAMGSolver`, level smoothers, and
coarse `lduMatrix` objects are recreated. Each coarse level's
`lduPrimitiveMesh` and `lduAddressing` remain stable and are owned by the
persistent agglomeration hierarchy.

Accordingly:

* the finest `fvMesh` uses the existing topology-invalidated
  `TopologicalMeshObject` cache;
* an eligible coarse `lduPrimitiveMesh` owns one opaque cache slot whose
  lifetime matches its addressing;
* schedules are keyed by `width`, not by `lduMatrix*`;
* multiple widths can coexist in a topology cache;
* `minCells` is not a schedule key because it only decides whether the cache
  is consulted;
* ineligible levels never create or populate an Index-Kahn cache;
* actual upper/lower/diagonal/source/psi values are never stored in the
  topology cache.

When the GAMG hierarchy is destroyed or rebuilt, its coarse
`lduPrimitiveMesh` objects destroy their cache slots and schedules. No
process-global pointer map is used. `lduPrimitiveMesh` is non-copyable in
v2606, so the opaque slot is not accidentally duplicated by a mesh copy.

The finest GAMG matrix is backed by the original `fvMesh`, so it naturally
reuses the same mesh-owned schedule as ordinary `smoothSolver` use with the
same `width`; no second finest-level cache is introduced.

## Expected diagnostics

The forwarding patch temporarily prints the values parsed by every
`IndexKahnSmoother` constructor. For the example configuration it must show:

```text
IndexKahnSmoother controls: width=2048 minCells=50000
```

This line is intentionally unconditional for the verification run and should
be removed after the hand-off has been confirmed.

The generic v2606 smoother factory does not pass the GAMG ordinal to a
runtime-selected smoother. To avoid modifying `GAMGSolver` merely for logging,
the production diagnostics identify each level by its local cell count. The
lines occur in hierarchy order during smoother construction.

First pressure solve:

```text
IndexKahn GAMG: cells=176840 minCells=10000 action=IndexKahn width=1024 schedule=build
IndexKahn GAMG: cells=82311 minCells=10000 action=IndexKahn width=1024 schedule=build
IndexKahn GAMG: cells=37120 minCells=10000 action=IndexKahn width=1024 schedule=build
IndexKahn GAMG: cells=14210 minCells=10000 action=IndexKahn width=1024 schedule=build
IndexKahn GAMG: cells=6180 minCells=10000 action=GaussSeidel
IndexKahn GAMG: cells=2470 minCells=10000 action=GaussSeidel
```

Subsequent solve:

```text
IndexKahn GAMG: cells=176840 minCells=10000 action=IndexKahn width=1024 schedule=reuse
IndexKahn GAMG: cells=82311 minCells=10000 action=IndexKahn width=1024 schedule=reuse
...
IndexKahn GAMG: cells=6180 minCells=10000 action=GaussSeidel
```

With `IndexKahnScheduleCache 1`, matching `building schedule` or `reusing
schedule` lines are also printed. Small levels must produce no schedule-cache
message and no coefficient-packing timing line.

## Unchanged limitations

The compact `uint8_t` row-degree limit remains. An eligible level containing a
row with more than 255 contributions fails loudly instead of truncating it.
Agglomeration, operators, restriction/prolongation, pre/post-sweep counts,
convergence controls, and interface communication are unchanged. Correctness,
MPI behavior, and performance require manual validation after rebuilding.
