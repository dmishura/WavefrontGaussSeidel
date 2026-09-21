# GAMG hierarchy/lifetime diagnostics for OpenFOAM v2606

This is a temporary, diagnostic-only source patch. It does not change the
configured smoother, GAMG calculations, matrix coefficients, sweep count, or
Index-Kahn code. The pressure smoother remains `GaussSeidel`.

## Apply and build

From the OpenFOAM v2606 source root:

```sh
git apply --check /home/dmishura/uGS/SmootherTest/patches/openfoam-v2606-gamg-lifetime-diagnostics.patch
git apply /home/dmishura/uGS/SmootherTest/patches/openfoam-v2606-gamg-lifetime-diagnostics.patch
wmake libso src/OpenFOAM
```

Only `libOpenFOAM` is changed by this patch. It touches `GAMGSolver.C`,
`GAMGSolverSolve.C`, and `GAMGAgglomeration.C`. It does not touch the existing
Index-Kahn user library. If the model uses that library for U/k/omega, it can
remain configured as before.

Enable the single switch in `system/controlDict`:

```foam
DebugSwitches
{
    IndexKahnGAMGDebug 1;
}
```

Keep `p` in `system/fvSolution` on the standard solver/smoother:

```foam
p
{
    solver          GAMG;
    smoother        GaussSeidel;
    tolerance       1e-7;
    relTol          0.01;
}
```

Run the usual model for at least two pressure solves and save the log. The
lines of interest start with `GAMGDBG`. Output uses `Pout`, so a parallel run
also has the usual OpenFOAM rank prefix. The `solve` counter is per process
and counts all GAMG solves, not just `p`; filter with `field=p`.

No diagnostics are printed when the switch is absent or zero. The only
always-on work is a branch checking the switch.

## Reading the records

`solve-begin` prints the field, GAMGSolver instance, registered
GAMGAgglomeration instance, and total number of levels (finest plus coarse).
Each `level` record prints local cell/face counts and the identities of its
`lduMesh`, `lduAddressing`, and `lduMatrix`. A level with `matrix=none` is
absent on the local rank, which can happen with processor agglomeration.

`smoother-create` appears only if the initial residual did not already meet
the convergence test. The level smoothers are created by `initVcycle()` into
a solve-local list. `smoothers-destroy` is emitted immediately before that
list goes out of scope; `smoothers-skipped` marks an initially converged
solve. The per-level pointers and the create/scope-end records together show
whether smoothers are recreated on subsequent solves.

`solver-construct`/`solver-destroy` bracket a GAMGSolver instance and its
coarse matrix list. `agglomeration-construct`/`agglomeration-destroy` bracket
the mesh-registry-owned hierarchy. `agglomeration-reuse` means a registry
cache hit; `agglomeration-mark-rebuild` and `agglomeration-rebuild` identify
the update/invalidation path. The `cacheAgglomeration` and
`deleteAgglomeration` values show whether the solver destructor retains or
deletes the hierarchy.

Address identities are printed as decimal integers. They are **observations
only**, not future cache keys. Allocators can reuse the same address after
an object is destroyed. Read pointer equality together with the explicit
construct/destroy/rebuild events. In particular:

```text
same agglomeration/mesh/addr with a new solver/matrix construction
    => persistent addressing, temporary matrix

new agglomeration plus new coarse mesh/addr on each solve
    => temporary addressing and temporary matrix
```

The v2606 source suggests the former with `cacheAgglomeration yes`:
`GAMGAgglomeration::New()` looks up a registered object in the mesh's
`objectRegistry`, whereas `GAMGSolver` owns `matrixLevels_` and `solve()` owns
the smoother list. Mesh motion may mark the agglomeration for rebuild. The
runtime log is needed to confirm actual lifetime in the chosen case.

## Limits

This patch does not profile timings and does not test whether Index-Kahn is
valid on GAMG coarse levels. It is source-only: OpenFOAM was not compiled or
run while preparing it. If a compiler diagnostic appears, send it back with
the affected source line; no runtime or numerical claim is made here.
