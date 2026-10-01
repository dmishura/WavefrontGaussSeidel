#ifndef SmootherTest_fwGAMG_H
#define SmootherTest_fwGAMG_H

#include "GAMGControls.H"
#include "fwHierarchy.H"

#include <chrono>
#include <cstdint>
#include <vector>

namespace smootherTest::fwgamg
{

enum class SpmvTraversal
{
    RowWise,
    FlatEdgeWise,
    DirectFlatEdgeWise,
    NativeFaceOrderDirectFlatEdgeWise
};

enum class FinestSpmvFusion
{
    None,
    IncomingHeld,
    DeltaCorrection,
    DeltaCorrectionCurrent,
    DeltaCorrectionEdgeRows,
    DeltaCorrectionPackedBaseline
};

struct Controls
{
    gamg::GAMGControls gamg;
    Foam::label width = 1024;
    RowOrdering rowOrdering = RowOrdering::IndexKahn;
    CoarseRenumbering coarseRenumbering = CoarseRenumbering::None;
    SpmvTraversal spmvTraversal =
        SpmvTraversal::NativeFaceOrderDirectFlatEdgeWise;
    // Diagnostic only: read a separate 256 KiB buffer immediately before
    // every Direct-flat traversal.  Disabled for every production/default
    // configuration.
    bool evictL1BeforeDirectFlat = false;
    // Experimental 1T path: fuse the final finest-level IH1 sweep with the
    // A*psi used by the convergence residual immediately after the V-cycle.
    FinestSpmvFusion finestSpmvFusion = FinestSpmvFusion::None;
};

struct Performance
{
    bool converged = false;
    Foam::label iterations = 0;
    Foam::scalar initialResidual = 0;
    Foam::scalar finalResidual = 0;
    std::vector<Foam::scalar> residualHistory;
};

class Solver
{
    std::chrono::steady_clock::time_point setupBegin_;
    Controls controls_;
    Hierarchy hierarchy_;
    std::vector<std::uint8_t> l1EvictionBuffer_;
    mutable volatile std::uint64_t l1EvictionChecksum_ = 0;
    TimingStats* timing_ = nullptr;

    void evictL1BeforeDirectFlat() const;

    void vCycle
    (
        Foam::scalarField& psi,
        const Foam::scalarField& source,
        const Foam::scalarField& finestResidual,
        Foam::scalarField* finalMatrixPsi = nullptr
    ) const;

    Foam::scalar normFactor
    (
        const Foam::scalarField& psi,
        const Foam::scalarField& source,
        const Foam::scalarField& matrixPsi
    ) const;

    Foam::scalar finestL1Norm(const Foam::scalarField& field) const;

    void multiply
    (
        std::size_t level,
        Foam::scalarField& result,
        const Foam::scalarField& field
    ) const;

    void residual
    (
        std::size_t level,
        Foam::scalarField& result,
        const Foam::scalarField& field,
        const Foam::scalarField& source
    ) const;

    void scaleCorrection
    (
        std::size_t level,
        Foam::scalarField& field,
        const Foam::scalarField& source,
        Foam::scalarField& work
    ) const;

public:
    Solver
    (
        const Foam::lduMatrix& matrix,
        Controls controls = {},
        TimingStats* timing = nullptr
    );

    Performance solve
    (
        Foam::scalarField& psi,
        const Foam::scalarField& source
    ) const;

    void applyVCycle
    (
        Foam::scalarField& psi,
        const Foam::scalarField& source,
        const Foam::scalarField& residual
    ) const;

    const Hierarchy& hierarchy() const noexcept { return hierarchy_; }

    Foam::scalarField toFw
    (
        const Foam::scalarField& original,
        std::size_t level = 0
    ) const;

    Foam::scalarField toOriginal
    (
        const Foam::scalarField& reordered,
        std::size_t level = 0
    ) const;
};

} // namespace smootherTest::fwgamg

#endif
