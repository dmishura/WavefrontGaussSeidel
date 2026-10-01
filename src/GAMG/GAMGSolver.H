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
#include "GAMGKernels.H"
#include "GAMGTiming.H"

#include <chrono>
#include <cstdint>
#include <vector>

namespace smootherTest::gamg
{

enum class GAMGSpmvTraversal
{
    NativeLdu,
    DirectFlat
};

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
    std::chrono::steady_clock::time_point setupBegin_;
    GAMGControls controls_;
    GAMGAgglomeration agglomeration_;
    GAMGMatrixHierarchy matrices_;
    GAMGSpmvTraversal spmvTraversal_ = GAMGSpmvTraversal::NativeLdu;
    std::vector<DirectFlatMatrix> directFlatMatrices_;
    bool evictL1BeforeDirectFlat_ = false;
    std::vector<std::uint8_t> l1EvictionBuffer_;
    mutable volatile std::uint64_t l1EvictionChecksum_ = 0;
    GAMGTimingStats* timing_ = nullptr;

    void evictL1BeforeDirectFlat() const;

    void multiplyLevel
    (
        std::size_t level,
        Foam::scalarField& result,
        const Foam::scalarField& field
    ) const;

    void residualLevel
    (
        std::size_t level,
        Foam::scalarField& result,
        const Foam::scalarField& field,
        const Foam::scalarField& source
    ) const;

    void scaleCorrectionLevel
    (
        std::size_t level,
        Foam::scalarField& field,
        const Foam::scalarField& source,
        Foam::scalarField& work
    ) const;

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
    GAMGSolver
    (
        const Foam::lduMatrix& matrix,
        GAMGControls controls = {},
        GAMGTimingStats* timing = nullptr,
        GAMGSpmvTraversal spmvTraversal = GAMGSpmvTraversal::NativeLdu,
        bool evictL1BeforeDirectFlat = false
    );

    GAMGSolverPerformance solve
    (
        Foam::scalarField& psi,
        const Foam::scalarField& source
    ) const;

    void applyVCycle
    (
        Foam::scalarField& psi,
        const Foam::scalarField& source,
        const Foam::scalarField& finestResidual
    ) const
    {
        vCycle(psi, source, finestResidual);
    }

    const GAMGAgglomeration& agglomeration() const noexcept
    {
        return agglomeration_;
    }
    const GAMGMatrixHierarchy& matrices() const noexcept { return matrices_; }
    std::vector<GAMGLevelDiagnostics> diagnostics() const;
};

} // namespace smootherTest::gamg

#endif
