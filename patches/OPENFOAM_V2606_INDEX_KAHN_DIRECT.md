# Direct-coefficient Index-Kahn smoother for OpenFOAM v2606

This experimental patch adds `indexKahnDirect` alongside the existing packed
`indexKahn` smoother. It reuses the same persistent topology schedule but
reads `lower[faceIds[p]]` and `upper[faceIds[p]]` from the current matrix in
every sweep. The existing smoother and schedule/cache implementation are not
changed.

## Prerequisites and application

Apply this after the existing Index-Kahn GAMG/minCells integration and the
nested smoother-controls forwarding fix:

```sh
cd /path/to/OpenFOAM-v2606
git apply --check \
    /home/dmishura/uGS/SmootherTest/patches/openfoam-v2606-index-kahn-direct.patch
git apply \
    /home/dmishura/uGS/SmootherTest/patches/openfoam-v2606-index-kahn-direct.patch
```

The patch changes only the existing user library:

```text
src/indexKahnSmoother/IndexKahnDirectSmoother.H  (new)
src/indexKahnSmoother/IndexKahnDirectSmoother.C  (new)
src/indexKahnSmoother/Make/files                 (modified)
```

Rebuild the user library with the same command used for the packed smoother:

```sh
wmake libso src/indexKahnSmoother
```

No additional OpenFOAM core change is introduced by this patch. The earlier
GAMG nested-controls patch still requires its corresponding `libOpenFOAM`
rebuild.

## Runtime registration

`IndexKahnDirectSmoother` is registered in both the symmetric and asymmetric
`lduMatrix::smoother` constructor tables with:

```cpp
TypeName("indexKahnDirect");
```

It is built into the existing `libIndexKahnSmoother.so`; retain the normal
library loading entry:

```foam
libs ("libIndexKahnSmoother.so");
```

## smoothSolver example

```foam
U
{
    solver          smoothSolver;

    smoother
    {
        smoother    indexKahnDirect;
        width       1024;
        minCells    10000;
    }

    tolerance       1e-8;
    relTol          0.1;
    nSweeps         1;
}
```

`width` defaults to 1024 and `minCells` defaults to 10000, matching the packed
implementation.

## GAMG example

```foam
p
{
    solver          GAMG;

    smoother
    {
        smoother    indexKahnDirect;
        width       1024;
        minCells    50000;
    }

    tolerance       1e-7;
    relTol          0.01;
}
```

The `nCells >= minCells` decision occurs before cache access. Smaller levels
call the standard `GaussSeidelSmoother::smooth()` path and allocate no
Index-Kahn-specific data.

## Data and lifetime

Both `indexKahn` and `indexKahnDirect` call:

```cpp
IndexKahnScheduleCache::New(matrix_.mesh()).getOrCreate(width)
```

Consequently a schedule already built for a topology/width by either smoother
is reused by the other. The direct smoother adds no schedule format and no
second cache key.

The direct smoother has no `coeffs_` member. Its constructor performs no
coefficient allocation, copy, or coefficient-packing timing. The persistent
cache continues to contain only topology-derived arrays: `waveCells`,
`levelStarts`, `cols`, `faceIds`, `degrees`, and `incomingDegrees`. Matrix
`lower`, `upper`, and `diag` values are accessed from the current `lduMatrix`
inside `smoothInternal()` and are never placed in the cache.

## Ordering and interfaces

For each row, the direct kernel uses the cached running contribution index and
executes incoming entries first with `lower[faceIds[p]]`, followed by outgoing
entries with `upper[faceIds[p]]`. Both loops preserve cached entry order.

The source copy and the `initMatrixInterfaces()` / `updateMatrixInterfaces()`
sequence are copied from the working packed smoother. No GAMG transfer,
convergence, or sweep-count logic is changed.

## Diagnostics

The direct smoother reports diagnostics when either the existing shared
`indexKahn` switch or its own `indexKahnDirect` switch is enabled. For an
isolated direct-kernel experiment use:

```foam
DebugSwitches
{
    indexKahnDirect          1;
    IndexKahnScheduleCache   1;
}
```

Existing cases that already set `indexKahn 1;` also receive the direct
smoother diagnostics without changing `controlDict`.

Eligible levels report selection and cache status:

```text
IndexKahnDirect GAMG: cells=176840 minCells=50000 action=IndexKahnDirect width=1024 schedule=reuse
IndexKahnDirect: sweep kernel field=p sweeps=2 total=... ms average=... ms/sweep
```

Small levels report `action=GaussSeidel`. There is intentionally no
coefficient-packing timing for `indexKahnDirect`.

## Static-inspection note

The v2606 API/lifetime model is unchanged: a smoother retains a reference to
its current `lduMatrix`, while the persistent schedule is owned by the
topology-lifetime cache associated with the fine `fvMesh` or coarse
`lduPrimitiveMesh`. Coarse matrices may be recreated between GAMG solves;
the direct kernel therefore resolves coefficient pointers from `matrix_` for
the current smoother invocation and never caches them. No additional
lifetime issue was found statically.

The patch was prepared and checked statically. OpenFOAM was not compiled or
run.
