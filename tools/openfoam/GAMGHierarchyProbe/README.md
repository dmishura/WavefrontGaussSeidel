# GAMGHierarchyProbe

`GAMGHierarchyProbe` is a serial OpenFOAM v2606 utility used as a reference
oracle for the standalone GAMG implementation under `src/GAMG`.

It reads an undecomposed `constant/polyMesh`, constructs exactly the synthetic
symmetric matrix used by `tests/harness/ScheduleBuilder.C::makeMatrix()`, calls
the stock OpenFOAM `GAMGAgglomeration::New(matrix, controls)` path with
`algebraicPair`, prints cells/faces for every level, and exits. It does not
construct a linear solver or execute a V-cycle.

Build in an OpenFOAM v2606 environment:

```bash
export WM_PROJECT_DIR=/path/to/OpenFOAM-v2606
source "$WM_PROJECT_DIR/etc/bashrc"
cd tools/openfoam/GAMGHierarchyProbe
wmake
```

Run from any undecomposed OpenFOAM case:

```bash
GAMGHierarchyProbe -case /path/to/motorBike
```

The utility calls `argList::noParallel()`. A `processor0` directory is not a
replacement for the complete serial `constant/polyMesh`.

The case must be a normal OpenFOAM case with `system/controlDict`,
`system/fvSchemes`, `system/fvSolution`, and the undecomposed
`constant/polyMesh`. None of the field files are read and no solve is run.

The matrix coefficients reproduce `tests/harness/ScheduleBuilder.C`:

```text
c = 1 + 0.001*((17*owner + 13*neighbour) % 101)
upper[face] = lower[face] = -c
diag[owner] += c
diag[neighbour] += c
diag[cell] += 0.25*mean(diag)
```

OpenFOAM symmetric storage uses a single `upper` array; the stock
`algebraicPair` implementation consequently uses `mag(matrix.upper())` as its
face weights. The effective hierarchy controls are deliberately fixed and
printed by the utility:

```text
agglomerator=algebraicPair
nCellsInCoarsestLevel=10
mergeLevels=1
renumber=false
maxLevels=50
parallel=false
```
