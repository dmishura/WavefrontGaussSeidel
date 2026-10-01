#include "fwGAMG.H"
#include "GAMGKernels.H"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace smootherTest::fwgamg
{
namespace
{

double elapsed(const std::chrono::steady_clock::time_point begin)
{
    return std::chrono::duration<double>
    (
        std::chrono::steady_clock::now() - begin
    ).count();
}

} // namespace

Solver::Solver
(
    const Foam::lduMatrix& matrix,
    Controls controls,
    TimingStats* const timing
)
:
    setupBegin_(std::chrono::steady_clock::now()),
    controls_(std::move(controls)),
    hierarchy_
    (
        matrix,
        controls_.gamg,
        HierarchyOptions
        {
            controls_.width,
            controls_.rowOrdering,
            controls_.coarseRenumbering,
            controls_.spmvTraversal == SpmvTraversal::FlatEdgeWise,
            controls_.spmvTraversal == SpmvTraversal::DirectFlatEdgeWise
         || controls_.finestSpmvFusion == FinestSpmvFusion::DeltaCorrection
         || controls_.finestSpmvFusion
                == FinestSpmvFusion::DeltaCorrectionCurrent
         || controls_.finestSpmvFusion
                == FinestSpmvFusion::DeltaCorrectionEdgeRows,
            controls_.spmvTraversal
                == SpmvTraversal::NativeFaceOrderDirectFlatEdgeWise
        },
        timing
    ),
    timing_(timing)
{
    if (controls_.width < 1)
        throw std::runtime_error("fwGAMG width must be positive");
    if
    (
        controls_.gamg.maxIterations < 1
     || controls_.gamg.minIterations < 0
    ) throw std::runtime_error("invalid fwGAMG iteration controls");
    if (controls_.evictL1BeforeDirectFlat)
        l1EvictionBuffer_.assign
        (
            gamg::l1EvictionBufferBytes, std::uint8_t{1}
        );
    if (timing_ && timing_->enabled)
        timing_->setupWallSeconds += elapsed(setupBegin_);
}

void Solver::evictL1BeforeDirectFlat() const
{
    if (!controls_.evictL1BeforeDirectFlat) return;
    l1EvictionChecksum_ = gamg::evictL1
    (
        l1EvictionBuffer_.data(), l1EvictionBuffer_.size()
    );
}

Foam::scalarField Solver::toFw
(
    const Foam::scalarField& original,
    const std::size_t level
) const
{
    const Matrix& matrix = hierarchy_[level];
    if (original.size() != static_cast<std::size_t>(matrix.nCells()))
        throw std::runtime_error("fw input permutation size mismatch");
    Foam::scalarField result(original.size());
    for (Foam::label row=0; row<matrix.nCells(); ++row)
        result[row] = original[matrix.newToOld()[row]];
    return result;
}

Foam::scalarField Solver::toOriginal
(
    const Foam::scalarField& reordered,
    const std::size_t level
) const
{
    const Matrix& matrix = hierarchy_[level];
    if (reordered.size() != static_cast<std::size_t>(matrix.nCells()))
        throw std::runtime_error("fw output permutation size mismatch");
    Foam::scalarField result(reordered.size());
    for (Foam::label oldCell=0; oldCell<matrix.nCells(); ++oldCell)
        result[oldCell] = reordered[matrix.oldToNew()[oldCell]];
    return result;
}

Foam::scalar Solver::finestL1Norm(const Foam::scalarField& field) const
{
    const Matrix& matrix = hierarchy_[0];
    Foam::scalar result = 0;
    // Sum in original numbering to retain the baseline convergence criterion.
    for (Foam::label oldCell=0; oldCell<matrix.nCells(); ++oldCell)
        result += std::abs(field[matrix.oldToNew()[oldCell]]);
    return result;
}

void Solver::multiply
(
    const std::size_t level,
    Foam::scalarField& result,
    const Foam::scalarField& field
) const
{
    if (controls_.spmvTraversal == SpmvTraversal::DirectFlatEdgeWise)
    {
        evictL1BeforeDirectFlat();
        hierarchy_[level].multiplyDirectFlatEdgeWise
        (
            result, field, timing_, level
        );
    }
    else if
    (
        controls_.spmvTraversal
            == SpmvTraversal::NativeFaceOrderDirectFlatEdgeWise
    )
        hierarchy_[level].multiplyLduOrderDirectFlatEdgeWise
        (
            result, field, timing_, level
        );
    else if (controls_.spmvTraversal == SpmvTraversal::FlatEdgeWise)
        hierarchy_[level].multiplyFlatEdgeWise(result, field, timing_, level);
    else
        hierarchy_[level].multiply(result, field, timing_, level);
}

void Solver::residual
(
    const std::size_t level,
    Foam::scalarField& result,
    const Foam::scalarField& field,
    const Foam::scalarField& source
) const
{
    if (controls_.spmvTraversal == SpmvTraversal::DirectFlatEdgeWise)
    {
        evictL1BeforeDirectFlat();
        hierarchy_[level].residualDirectFlatEdgeWise
        (
            result, field, source, timing_, level
        );
    }
    else if
    (
        controls_.spmvTraversal
            == SpmvTraversal::NativeFaceOrderDirectFlatEdgeWise
    )
        hierarchy_[level].residualLduOrderDirectFlatEdgeWise
        (
            result, field, source, timing_, level
        );
    else if (controls_.spmvTraversal == SpmvTraversal::FlatEdgeWise)
        hierarchy_[level].residualFlatEdgeWise
        (
            result, field, source, timing_, level
        );
    else
        hierarchy_[level].residual(result, field, source, timing_, level);
}

void Solver::scaleCorrection
(
    const std::size_t level,
    Foam::scalarField& field,
    const Foam::scalarField& source,
    Foam::scalarField& work
) const
{
    if (controls_.spmvTraversal == SpmvTraversal::DirectFlatEdgeWise)
    {
        evictL1BeforeDirectFlat();
        hierarchy_[level].scaleCorrectionDirectFlatEdgeWise
        (
            field, source, work, timing_, level
        );
    }
    else if
    (
        controls_.spmvTraversal
            == SpmvTraversal::NativeFaceOrderDirectFlatEdgeWise
    )
        hierarchy_[level].scaleCorrectionLduOrderDirectFlatEdgeWise
        (
            field, source, work, timing_, level
        );
    else if (controls_.spmvTraversal == SpmvTraversal::FlatEdgeWise)
        hierarchy_[level].scaleCorrectionFlatEdgeWise
        (
            field, source, work, timing_, level
        );
    else
        hierarchy_[level].scaleCorrection
        (
            field, source, work, timing_, level
        );
}

Foam::scalar Solver::normFactor
(
    const Foam::scalarField& psi,
    const Foam::scalarField& source,
    const Foam::scalarField& matrixPsi
) const
{
    const Matrix& matrix = hierarchy_[0];
    Foam::scalar sum = 0;
    for (Foam::label oldCell=0; oldCell<matrix.nCells(); ++oldCell)
        sum += psi[matrix.oldToNew()[oldCell]];
    const Foam::scalar average = sum/static_cast<Foam::scalar>(matrix.nCells());

    Foam::scalar factor = 0;
    for (Foam::label oldCell=0; oldCell<matrix.nCells(); ++oldCell)
    {
        const Foam::label row = matrix.oldToNew()[oldCell];
        Foam::scalar rowSum = matrix.diag()[row];
        for
        (
            Foam::label p=matrix.rowStarts()[row];
            p<matrix.rowStarts()[row + 1];
            ++p
        ) rowSum += matrix.coeffs()[p];
        const Foam::scalar reference = average*rowSum;
        factor += std::abs(matrixPsi[row] - reference)
            + std::abs(source[row] - reference);
    }
    return factor + 1e-300;
}

void Solver::vCycle
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const Foam::scalarField& finestResidual,
    Foam::scalarField* const finalMatrixPsi
) const
{
    const auto cycleBegin = std::chrono::steady_clock::now();
    const std::size_t nLevels = hierarchy_.size();
    std::vector<Foam::scalarField> corrections(nLevels);
    std::vector<Foam::scalarField> sources(nLevels);
    sources[0] = finestResidual;

    for (std::size_t level=1; level<nLevels; ++level)
    {
        corrections[level].assign(hierarchy_[level].nCells(), 0.0);
        hierarchy_.restrictField
        (
            sources[level], sources[level - 1], level - 1, timing_
        );
        if (level + 1 == nLevels) break;

        if (controls_.gamg.nPreSweeps > 0)
        {
            const Foam::label sweeps = std::min
            (
                controls_.gamg.nPreSweeps
              + controls_.gamg.preSweepsLevelMultiplier
               *static_cast<Foam::label>(level - 1),
                controls_.gamg.maxPreSweeps
            );
            hierarchy_[level].smooth
            (
                corrections[level], sources[level], sweeps, true,
                timing_, level
            );
            if (controls_.gamg.scaleCorrection && level + 2 < nLevels)
            {
                Foam::scalarField work;
                scaleCorrection(level, corrections[level], sources[level], work);
            }
            Foam::scalarField levelResidual;
            residual
            (
                level, levelResidual, corrections[level], sources[level]
            );
            sources[level] = std::move(levelResidual);
        }
    }

    hierarchy_[nLevels - 1].solveDense
    (
        corrections.back(), sources.back(), timing_, nLevels - 1
    );

    if (nLevels > 2)
        for (std::size_t level=nLevels - 1; level-- > 1;)
        {
            Foam::scalarField preSmoothed;
            if (controls_.gamg.nPreSweeps > 0)
                preSmoothed = corrections[level];
            hierarchy_.prolongField
            (
                corrections[level], corrections[level + 1], level, timing_
            );
            if (controls_.gamg.scaleCorrection && level + 2 < nLevels)
            {
                Foam::scalarField work;
                scaleCorrection(level, corrections[level], sources[level], work);
            }
            if (controls_.gamg.nPreSweeps > 0)
                for (Foam::label row=0; row<hierarchy_[level].nCells(); ++row)
                    corrections[level][row] += preSmoothed[row];

            const Foam::label sweeps = std::min
            (
                controls_.gamg.nPostSweeps
              + controls_.gamg.postSweepsLevelMultiplier
               *static_cast<Foam::label>(level - 1),
                controls_.gamg.maxPostSweeps
            );
            hierarchy_[level].smooth
            (
                corrections[level], sources[level], sweeps, false,
                timing_, level
            );
        }

    Foam::scalarField finestCorrection;
    hierarchy_.prolongField
    (
        finestCorrection, corrections[1], 0, timing_
    );
    if (controls_.gamg.scaleCorrection)
    {
        Foam::scalarField work;
        scaleCorrection(0, finestCorrection, finestResidual, work);
    }
    for (Foam::label row=0; row<hierarchy_[0].nCells(); ++row)
        psi[row] += finestCorrection[row];
    if
    (
        controls_.finestSpmvFusion != FinestSpmvFusion::None
     && finalMatrixPsi
     && controls_.gamg.nFinestSweeps > 0
    )
    {
        if (controls_.finestSpmvFusion == FinestSpmvFusion::DeltaCorrection)
            hierarchy_[0].smoothAndMultiplyDelta
            (
                psi, source, *finalMatrixPsi,
                controls_.gamg.nFinestSweeps, false, timing_, 0
            );
        else if
        (
            controls_.finestSpmvFusion
                == FinestSpmvFusion::DeltaCorrectionCurrent
        )
            hierarchy_[0].smoothAndMultiplyDeltaCurrent
            (
                psi, source, *finalMatrixPsi,
                controls_.gamg.nFinestSweeps, false, timing_, 0
            );
        else if
        (
            controls_.finestSpmvFusion
                == FinestSpmvFusion::DeltaCorrectionEdgeRows
        )
            hierarchy_[0].smoothAndMultiplyDeltaEdgeRows
            (
                psi, source, *finalMatrixPsi,
                controls_.gamg.nFinestSweeps, false, timing_, 0
            );
        else if
        (
            controls_.finestSpmvFusion
                == FinestSpmvFusion::DeltaCorrectionPackedBaseline
        )
            hierarchy_[0].smoothAndMultiplyDeltaPackedBaseline
            (
                psi, source, *finalMatrixPsi,
                controls_.gamg.nFinestSweeps, false, timing_, 0
            );
        else
            hierarchy_[0].smoothAndMultiply
            (
                psi, source, *finalMatrixPsi,
                controls_.gamg.nFinestSweeps, false, timing_, 0
            );
    }
    else
        hierarchy_[0].smooth
        (
            psi, source, controls_.gamg.nFinestSweeps, false, timing_, 0
        );

    if (timing_ && timing_->enabled)
        timing_->cycleDurations.push_back(elapsed(cycleBegin));
}

Performance Solver::solve
(
    Foam::scalarField& originalPsi,
    const Foam::scalarField& originalSource
) const
{
    if
    (
        originalPsi.size() != static_cast<std::size_t>(hierarchy_[0].nCells())
     || originalSource.size() != originalPsi.size()
    ) throw std::runtime_error("fwGAMG solve field size mismatch");

    const auto solveBegin = std::chrono::steady_clock::now();
    auto begin = std::chrono::steady_clock::now();
    Foam::scalarField psi = toFw(originalPsi);
    Foam::scalarField source = toFw(originalSource);
    if (timing_ && timing_->enabled)
        add(timing_->inputPermutation, elapsed(begin));

    Foam::scalarField matrixPsi;
    multiply(0, matrixPsi, psi);
    const Foam::scalar normalization = normFactor(psi, source, matrixPsi);
    Foam::scalarField finestResidual(source.size());
    const auto residualBegin = std::chrono::steady_clock::now();
    for (Foam::label row=0; row<hierarchy_[0].nCells(); ++row)
        finestResidual[row] = source[row] - matrixPsi[row];
    if (timing_ && timing_->enabled)
        add(timing_->levels[0].residual, elapsed(residualBegin));

    Performance performance;
    performance.initialResidual = finestL1Norm(finestResidual)/normalization;
    performance.finalResidual = performance.initialResidual;
    performance.residualHistory.push_back(performance.initialResidual);
    const auto converged = [&]()
    {
        return performance.finalResidual <= controls_.gamg.tolerance
            ||
            (
                controls_.gamg.relativeTolerance > 0
             && performance.finalResidual
                <= controls_.gamg.relativeTolerance*performance.initialResidual
            );
    };

    performance.converged = converged() && controls_.gamg.minIterations == 0;
    while
    (
        performance.iterations < controls_.gamg.maxIterations
     &&
        (
            performance.iterations < controls_.gamg.minIterations
         || !performance.converged
        )
    )
    {
        const bool fuse = controls_.finestSpmvFusion != FinestSpmvFusion::None
            && controls_.gamg.nFinestSweeps > 0;
        vCycle
        (
            psi, source, finestResidual,
            fuse ? &matrixPsi : nullptr
        );
        if (!fuse) multiply(0, matrixPsi, psi);
        const auto iterationResidualBegin = std::chrono::steady_clock::now();
        for (Foam::label row=0; row<hierarchy_[0].nCells(); ++row)
            finestResidual[row] = source[row] - matrixPsi[row];
        if (timing_ && timing_->enabled)
            add(timing_->levels[0].residual, elapsed(iterationResidualBegin));
        ++performance.iterations;
        performance.finalResidual = finestL1Norm(finestResidual)/normalization;
        performance.residualHistory.push_back(performance.finalResidual);
        performance.converged = converged()
            && performance.iterations >= controls_.gamg.minIterations;
    }

    begin = std::chrono::steady_clock::now();
    originalPsi = toOriginal(psi);
    if (timing_ && timing_->enabled)
    {
        add(timing_->outputPermutation, elapsed(begin));
        timing_->solveDurations.push_back(elapsed(solveBegin));
    }
    return performance;
}

void Solver::applyVCycle
(
    Foam::scalarField& originalPsi,
    const Foam::scalarField& originalSource,
    const Foam::scalarField& originalResidual
) const
{
    Foam::scalarField psi = toFw(originalPsi);
    const Foam::scalarField source = toFw(originalSource);
    const Foam::scalarField residual = toFw(originalResidual);
    vCycle(psi, source, residual);
    originalPsi = toOriginal(psi);
}

} // namespace smootherTest::fwgamg
