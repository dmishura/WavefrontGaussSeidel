/*---------------------------------------------------------------------------*\
  Copyright (C) 2011-2017 OpenFOAM Foundation
  Copyright (C) 2016-2023 OpenCFD Ltd.

  Standalone serial symmetric GAMG solver adapted from OpenFOAM v2606.

  OpenFOAM is distributed under the GNU General Public License, version 3
  or later. This adapted file is distributed under the same terms.
\*---------------------------------------------------------------------------*/

#ifndef SmootherTest_GAMGSolver_H
#define SmootherTest_GAMGSolver_H

#include "GAMGAgglomeration.H"
#include "GAMGControls.H"
#include "GAMGMatrix.H"

#include <vector>

namespace smootherTest::gamg
{

struct GAMGLevelDiagnostics
{
    Foam::label level = 0;
    Foam::label cells = 0;
    Foam::label internalFaces = 0;
    Foam::scalar agglomerationRatio = 1;
    Foam::label preSweeps = 0;
    Foam::label postSweeps = 0;
    Foam::scalar diagonalSum = 0;
    Foam::scalar coefficientL1 = 0;
};

struct GAMGSolverPerformance
{
    bool converged = false;
    Foam::label iterations = 0;
    Foam::scalar initialResidual = 0;
    Foam::scalar finalResidual = 0;
    std::vector<Foam::scalar> residualHistory;
};

class GAMGSolver
{
    GAMGControls controls_;
    GAMGAgglomeration agglomeration_;
    GAMGMatrixHierarchy matrices_;

    void vCycle
    (
        Foam::scalarField& psi,
        const Foam::scalarField& source,
        const Foam::scalarField& finestResidual
    ) const;

    Foam::scalar normFactor
    (
        const Foam::scalarField& psi,
        const Foam::scalarField& source,
        const Foam::scalarField& matrixPsi
    ) const;

public:
    GAMGSolver(const Foam::lduMatrix& matrix, GAMGControls controls = {});

    GAMGSolverPerformance solve
    (
        Foam::scalarField& psi,
        const Foam::scalarField& source
    ) const;

    const GAMGAgglomeration& agglomeration() const noexcept
    {
        return agglomeration_;
    }
    const GAMGMatrixHierarchy& matrices() const noexcept { return matrices_; }
    std::vector<GAMGLevelDiagnostics> diagnostics() const;
};

} // namespace smootherTest::gamg

#endif
