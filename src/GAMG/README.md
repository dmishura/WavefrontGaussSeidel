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
