/*---------------------------------------------------------------------------*\
  Copyright (C) 2011-2017 OpenFOAM Foundation
  Copyright (C) 2016-2023 OpenCFD Ltd.

  Serial symmetric V-cycle adapted from the stock OpenFOAM v2606
  GAMGSolverSolve.C implementation.

  OpenFOAM is distributed under the GNU General Public License, version 3
  or later. This adapted file is distributed under the same terms.
\*---------------------------------------------------------------------------*/

#include "GAMGSolver.H"

#include "GAMGKernels.H"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace smootherTest::gamg
{

GAMGSolver::GAMGSolver
(
    const Foam::lduMatrix& matrix,
    GAMGControls controls
)
:
    controls_(std::move(controls)),
    agglomeration_(GAMGMatrix::fromLduMatrix(matrix), controls_),
    matrices_(GAMGMatrix::fromLduMatrix(matrix), agglomeration_.levels())
{
    if (controls_.maxIterations < 1 || controls_.minIterations < 0)
        throw std::runtime_error("invalid GAMG iteration controls");
    if (controls_.diagnostics)
    {
        std::cout << "GAMG hierarchy:\n";
        for (const GAMGLevelDiagnostics& level : diagnostics())
        {
            std::cout
                << "  level=" << level.level
                << " cells=" << level.cells
                << " faces=" << level.internalFaces
                << " ratio=" << level.agglomerationRatio
                << " preSweeps=" << level.preSweeps
                << " postSweeps=" << level.postSweeps
                << " diagSum=" << level.diagonalSum
                << " coeffL1=" << level.coefficientL1 << '\n';
        }
    }
}

Foam::scalar GAMGSolver::normFactor
(
    const Foam::scalarField& psi,
    const Foam::scalarField& source,
    const Foam::scalarField& matrixPsi
) const
{
    const GAMGMatrix& matrix = matrices_[0];
    const Foam::scalar average = std::accumulate(psi.begin(), psi.end(), 0.0)
        /static_cast<Foam::scalar>(psi.size());
    Foam::scalarField rowSum(matrix.diag());
    for (Foam::label face=0; face<matrix.nFaces(); ++face)
    {
        rowSum[matrix.lowerAddr()[face]] += matrix.upper()[face];
        rowSum[matrix.upperAddr()[face]] += matrix.upper()[face];
    }
    Foam::scalar factor = 0;
    for (Foam::label cell=0; cell<matrix.nCells(); ++cell)
    {
        const Foam::scalar reference = average*rowSum[cell];
        factor += std::abs(matrixPsi[cell] - reference)
            + std::abs(source[cell] - reference);
    }
    return factor + 1e-300;
}

void GAMGSolver::vCycle
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const Foam::scalarField& finestResidual
) const
{
    const std::size_t nLevels = matrices_.size();
    std::vector<Foam::scalarField> corrections(nLevels);
    std::vector<Foam::scalarField> sources(nLevels);
    sources[0] = finestResidual;
    for (std::size_t level=1; level<nLevels; ++level)
    {
        corrections[level].assign(matrices_[level].nCells(), 0.0);
        restrictField(sources[level], sources[level - 1], agglomeration_[level - 1]);
        if (level + 1 == nLevels) break;

        if (controls_.nPreSweeps > 0)
        {
            const Foam::label sweeps = std::min
            (
                controls_.nPreSweeps
              + controls_.preSweepsLevelMultiplier*static_cast<Foam::label>(level - 1),
                controls_.maxPreSweeps
            );
            gaussSeidelSmooth
            (
                corrections[level], matrices_[level], sources[level], sweeps
            );
            if (controls_.scaleCorrection && level + 2 < nLevels)
            {
                Foam::scalarField work;
                scaleCorrection
                (
                    corrections[level], matrices_[level], sources[level], work
                );
            }
            Foam::scalarField levelResidual;
            residual
            (
                levelResidual,
                matrices_[level],
                corrections[level],
                sources[level]
            );
            sources[level] = std::move(levelResidual);
        }
    }

    solveCoarsest
    (
        corrections.back(), matrices_[nLevels - 1], sources.back()
    );

    if (nLevels > 2)
    {
        for (std::size_t level=nLevels - 1; level-- > 1;)
        {
            Foam::scalarField preSmoothed;
            if (controls_.nPreSweeps > 0) preSmoothed = corrections[level];
            prolongField
            (
                corrections[level], corrections[level + 1], agglomeration_[level]
            );
            if (controls_.scaleCorrection && level + 2 < nLevels)
            {
                Foam::scalarField work;
                scaleCorrection
                (
                    corrections[level], matrices_[level], sources[level], work
                );
            }
            if (controls_.nPreSweeps > 0)
                for (Foam::label cell=0; cell<matrices_[level].nCells(); ++cell)
                    corrections[level][cell] += preSmoothed[cell];

            const Foam::label sweeps = std::min
            (
                controls_.nPostSweeps
              + controls_.postSweepsLevelMultiplier*static_cast<Foam::label>(level - 1),
                controls_.maxPostSweeps
            );
            gaussSeidelSmooth
            (
                corrections[level], matrices_[level], sources[level], sweeps
            );
        }
    }

    Foam::scalarField finestCorrection;
    prolongField(finestCorrection, corrections[1], agglomeration_[0]);
    if (controls_.scaleCorrection)
    {
        Foam::scalarField work;
        scaleCorrection
        (
            finestCorrection, matrices_[0], finestResidual, work
        );
    }
    for (Foam::label cell=0; cell<matrices_[0].nCells(); ++cell)
        psi[cell] += finestCorrection[cell];
    gaussSeidelSmooth
    (
        psi, matrices_[0], source, controls_.nFinestSweeps
    );
}

GAMGSolverPerformance GAMGSolver::solve
(
    Foam::scalarField& psi,
    const Foam::scalarField& source
) const
{
    if
    (
        psi.size() != static_cast<std::size_t>(matrices_[0].nCells())
     || source.size() != psi.size()
    )
        throw std::runtime_error("GAMG solve field size mismatch");

    Foam::scalarField matrixPsi;
    multiply(matrixPsi, matrices_[0], psi);
    const Foam::scalar normalization = normFactor(psi, source, matrixPsi);
    Foam::scalarField finestResidual(source.size());
    for (Foam::label cell=0; cell<matrices_[0].nCells(); ++cell)
        finestResidual[cell] = source[cell] - matrixPsi[cell];

    GAMGSolverPerformance performance;
    performance.initialResidual = l1Norm(finestResidual)/normalization;
    performance.finalResidual = performance.initialResidual;
    performance.residualHistory.push_back(performance.initialResidual);
    const auto converged = [&]()
    {
        return performance.finalResidual <= controls_.tolerance
            ||
            (
                controls_.relativeTolerance > 0
             && performance.finalResidual
                <= controls_.relativeTolerance*performance.initialResidual
            );
    };

    performance.converged = converged() && controls_.minIterations == 0;
    while
    (
        performance.iterations < controls_.maxIterations
     && (performance.iterations < controls_.minIterations || !performance.converged)
    )
    {
        vCycle(psi, source, finestResidual);
        multiply(matrixPsi, matrices_[0], psi);
        for (Foam::label cell=0; cell<matrices_[0].nCells(); ++cell)
            finestResidual[cell] = source[cell] - matrixPsi[cell];
        ++performance.iterations;
        performance.finalResidual = l1Norm(finestResidual)/normalization;
        performance.residualHistory.push_back(performance.finalResidual);
        performance.converged = converged()
            && performance.iterations >= controls_.minIterations;
    }
    return performance;
}

std::vector<GAMGLevelDiagnostics> GAMGSolver::diagnostics() const
{
    std::vector<GAMGLevelDiagnostics> result;
    result.reserve(matrices_.size());
    for (std::size_t level=0; level<matrices_.size(); ++level)
    {
        const GAMGMatrix& matrix = matrices_[level];
        GAMGLevelDiagnostics entry;
        entry.level = static_cast<Foam::label>(level);
        entry.cells = matrix.nCells();
        entry.internalFaces = matrix.nFaces();
        entry.agglomerationRatio = level == 0
            ? 1.0
            : Foam::scalar(matrices_[level - 1].nCells())/matrix.nCells();
        if (level == 0)
        {
            entry.preSweeps = 0;
            entry.postSweeps = controls_.nFinestSweeps;
        }
        else if (level + 1 < matrices_.size())
        {
            if (controls_.nPreSweeps > 0)
                entry.preSweeps = std::min
                (
                    controls_.nPreSweeps
                  + controls_.preSweepsLevelMultiplier*static_cast<Foam::label>(level - 1),
                    controls_.maxPreSweeps
                );
            entry.postSweeps = std::min
            (
                controls_.nPostSweeps
              + controls_.postSweepsLevelMultiplier*static_cast<Foam::label>(level - 1),
                controls_.maxPostSweeps
            );
        }
        entry.diagonalSum = std::accumulate
        (
            matrix.diag().begin(), matrix.diag().end(), 0.0
        );
        for (const Foam::scalar coefficient : matrix.upper())
            entry.coefficientL1 += std::abs(coefficient);
        result.push_back(entry);
    }
    return result;
}

} // namespace smootherTest::gamg
