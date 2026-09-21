# Index-Kahn smoother patch for OpenFOAM v2606

The source patch is `openfoam-v2606-index-kahn.patch`. It is based on the
unmodified v2606 tree and has **not** been applied to the `OpenFOAM-v2606`
symlink. The source files used to generate it are also kept under
`SmootherTest/openfoam-v2606/src/indexKahnSmoother/`.

## Apply and build manually

From the OpenFOAM v2606 source root:

```sh
git apply --check /home/dmishura/uGS/SmootherTest/patches/openfoam-v2606-index-kahn.patch
git apply /home/dmishura/uGS/SmootherTest/patches/openfoam-v2606-index-kahn.patch
```

After loading the normal v2606 environment, build only the new user library:

```sh
wmake libso src/indexKahnSmoother
```

The plugin target is `$(FOAM_USER_LIBBIN)/libIndexKahnSmoother.so` and links
against the existing `libOpenFOAM` and `libfiniteVolume`. This patch does not
modify either core library, so neither needs rebuilding for `indexKahn`.

## First case configuration

Add the library to `system/controlDict`:

```foam
libs ("libIndexKahnSmoother.so");

DebugSwitches
{
    IndexKahnScheduleCache 1;
    IndexKahnSmoother 1;
}
```

Use the new smoother only for ordinary `smoothSolver` fields in
`system/fvSolution`:

```foam
solvers
{
    p
    {
        solver GAMG;
        smoother GaussSeidel;
        tolerance 1e-7;
        relTol 0.01;
    }
    U
    {
        solver smoothSolver;
        smoother
        {
            smoother indexKahn;
            width 1024;
        }
        tolerance 1e-8;
        relTol 0.1;
        nSweeps 1;
    }
    k
    {
        solver smoothSolver;
        smoother
        {
            smoother indexKahn;
            width 1024;
        }
        tolerance 1e-8;
        relTol 0.1;
        nSweeps 1;
    }
    omega
    {
        solver smoothSolver;
        smoother
        {
            smoother indexKahn;
            width 1024;
        }
        tolerance 1e-8;
        relTol 0.1;
        nSweeps 1;
    }
}
```

`width` defaults to 1024 if omitted from the nested smoother dictionary.
The v2606 factory already passes that nested dictionary to the smoother
constructor. A sibling `width` beside `smoother` is **not** passed to the
constructor; use the nested form whenever setting a non-default width.

With the debug switch enabled, a serial run should show one line like:

```text
IndexKahn: building schedule width=1024 cells=...
```

and subsequent smoother constructions on the same mesh should show:

```text
IndexKahn: reusing schedule width=1024 cells=...
```

If `k` instead uses nested `width 512;`, its first solve should additionally show
one `building schedule width=512` line; later k solves should reuse 512 while
U and omega continue to reuse 1024. In parallel, each MPI rank owns its own
mesh cache; these messages use the OpenFOAM master stream and thus normally
appear only for the master rank.

With `IndexKahnSmoother 1`, each new smoother constructor also reports
`coefficient packing ... time=... ms`; each `smooth()` call reports
`sweep kernel ... total=... ms average=... ms/sweep`. Packing time includes
coefficient allocation/copy but excludes cache lookup and schedule build.
The sweep-kernel time covers only the cell traversal, not source copy or
coupled-interface updates. Timing messages are disabled by default.

## What to check after building

1. Confirm the plugin loads and `indexKahn` is selected for U/k/omega while
   p remains GAMG/GaussSeidel.
2. Check build/reuse messages over multiple solver calls and across fields.
3. Compare residuals and solution fields with a GaussSeidel baseline, first
   for one sweep and then for the complete model.
4. Check coupled-boundary/MPI cases separately. The patch preserves the
   reference `initMatrixInterfaces` and `updateMatrixInterfaces` sequence,
   but these paths have not been run here.
5. Measure separately schedule construction, coefficient packing per new
   smoother, sweep-kernel time, and complete `solve()` time. The patch
   instruments packing and sweep-kernel time; whole-solve timing requires
   external measurement.

The first implementation is **serial**. It uses the tested Index-Kahn
topological order, `uint8_t` row degree, and running contribution index;
OpenMP execution is deliberately deferred. The mesh cache stores no matrix
values. Every new smoother instance packs off-diagonal coefficients from its
current matrix once; the packed values are reused by all of that instance's
sweeps. The diagonal is not packed and is read as `matrix_.diag()[cell]`.
GAMG coarse levels are outside this patch's supported scope. No OpenFOAM
compilation or runtime validation was performed while preparing the patch.
