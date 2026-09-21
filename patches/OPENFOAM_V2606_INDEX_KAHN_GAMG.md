# Index-Kahn smoother inside GAMG (OpenFOAM v2606)

This is an **incremental** patch for a v2606 tree that already contains the
working `src/indexKahnSmoother` user library. It does not include or require
the temporary `GAMGDBG` lifetime instrumentation. No `IndexKahnGAMG` class is
introduced; GAMG selects the existing `indexKahn` smoother.

## Apply and rebuild

From the OpenFOAM v2606 source root:

```sh
git apply --check /home/dmishura/uGS/SmootherTest/patches/openfoam-v2606-index-kahn-gamg.patch
git apply /home/dmishura/uGS/SmootherTest/patches/openfoam-v2606-index-kahn-gamg.patch
```

The patch adds one opaque cache slot to `lduPrimitiveMesh`, which changes
the C++ layout of that core class. A rebuild of **all OpenFOAM libraries
and applications that use the core headers** is safer than rebuilding only
`libOpenFOAM`; then rebuild the existing user library:

```sh
./Allwmake
wmake libso src/indexKahnSmoother
```

Use the usual OpenFOAM environment and build procedure for your installation.
Do not mix old binaries/libraries compiled against the earlier
`lduPrimitiveMesh.H` with the rebuilt ones. This patch was not compiled or run
while being prepared.

If the temporary GAMG lifetime diagnostic patch is currently applied, this
new patch does not remove it. Reverse the diagnostic patch separately when
you no longer want its logging; the two patches touch different files.

## First pressure test

Keep `libIndexKahnSmoother.so` in `system/controlDict`'s `libs` list. To check
cache reuse, enable the existing schedule-cache debug switch:

```foam
DebugSwitches
{
    IndexKahnScheduleCache 1;
}
```

In `system/fvSolution`, change only the smoother selection for `p`:

```foam
p
{
    solver          GAMG;
    smoother
    {
        smoother    indexKahn;
        width       1024;
    }
    tolerance       1e-7;
    relTol          0.01;
}
```

The nested dictionary is required when setting `width`: the v2606 smoother
factory passes that dictionary to the smoother constructor. `width` still
defaults to 1024 if omitted. Use the previous `GaussSeidel` configuration as
the numerical comparison baseline.

## Ownership and invalidation

For the finest `fvMesh` level and ordinary `smoothSolver` fields, the cache
remains mesh-owned and topology-invalidated as before (`TopologicalMeshObject`).
For a GAMG coarse level, the same `IndexKahnScheduleCache` data is held in
one opaque slot of that level's `lduPrimitiveMesh`. Its `lduAddressing` is
the topology identity used to build the schedule. The coarse mesh and slot
are destroyed together when `GAMGAgglomeration` discards that level; no
process-global pointer map or generic `lduAddressing` cache facility is
introduced. Each cache retains schedules for multiple `width` values.

No schedule is keyed by or stored in a `lduMatrix` or smoother instance.
Coefficients are still packed from the current matrix in each smoother
constructor; the diagonal is read directly from the current matrix during
the sweep. This patch does **not** remove coefficient-repacking cost.

The original compact `uint8_t` degree path and its explicit degree limit
remain unchanged. If a GAMG coarse row has more than 255 contributions,
schedule construction will fail loudly; handling such rows is a separate
follow-up change, not silently truncated. Row order, contribution order,
arithmetic, and coupled-interface calls are otherwise unchanged.

With the debug switch, expect one `building schedule` message for each
distinct active addressing/width on a rank, followed by `reusing schedule`
messages on subsequent GAMG solves. A rebuild is expected if the
agglomeration/coarse mesh is recreated or the finest mesh topology changes.
Correctness, MPI behavior, and performance still require your manual tests.
