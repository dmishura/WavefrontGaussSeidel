# OpenFOAM v2606 Gauss–Seidel call counters

`openfoam-v2606-gs-counters.patch` is a diagnostic patch for the v2606 source
tree. It is **not applied** to the `OpenFOAM-v2606` symlink by this project.

From the OpenFOAM source root, after deciding to modify that checkout:

```sh
git apply --check /home/dmishura/uGS/SmootherTest/patches/openfoam-v2606-gs-counters.patch
git apply /home/dmishura/uGS/SmootherTest/patches/openfoam-v2606-gs-counters.patch
```

Rebuild `libOpenFOAM` and `libfiniteVolume` using the usual v2606 environment
and build process. The new header must be present in the OpenFOAM `lnInclude`
links before `libfiniteVolume` is rebuilt.

At **normal process exit**, the patch writes `GS_COUNTERS_*` lines to stderr.
Each MPI rank writes its own counters, identified by `rank`; there is no MPI
reduction at exit. Redirect stderr to a file if the model has many solves.

The `GS_COUNTERS_BEGIN` line reports process-wide totals:

- `fvSolves`: calls to the usual `fvMatrix::solveSegregatedOrCoupled` path,
  plus calls through the scalar `fvSolver::solve` path. A vector equation is
  one `fvSolves` call but may cause several component-wise LDU solves.
- `smoothSolverSolves`: calls to `smoothSolver::solve`, including calls that
  converge without constructing a smoother.
- `smootherNew`: requests to `lduMatrix::smoother::New` (all smoother types).
- `gsConstructors`: actual `GaussSeidelSmoother` constructor calls.
- `gsSmoothCalls`: calls to the original static GS sweep entry point.
- `gsSweeps`: sum of the `nSweeps` arguments to that entry point.

Each `GS_COUNTERS_FV_SOLVE` line is **one model-level** `fvMatrix` solve,
identified by `id` and `field`. It includes smoother construction and GS
activity inside any component-wise LDU solves, including GAMG. Its
`smoothSolverCalls` field counts nested `smoothSolver::solve` invocations.
Each `GS_COUNTERS_SMOOTH_SOLVE` line is one such nested invocation, with its
own constructor/call/sweep counts. Global GS totals can exceed the sum of FV
records if GS is invoked directly outside an `fvMatrix` solve. A GS call with
`nSweeps=3` counts as one `gsSmoothCalls` and three `gsSweeps`.

No counter is incremented inside the GS cell/face loops. This is temporary
diagnostic instrumentation; it keeps a record for every FV and
`smoothSolver::solve` until process exit, so memory use and final log size grow
with the number of solves. An abnormal termination may not print the report.
