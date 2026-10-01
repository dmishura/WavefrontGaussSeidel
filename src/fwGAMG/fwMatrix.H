#ifndef SmootherTest_fwMatrix_H
#define SmootherTest_fwMatrix_H

#include "WavefrontGaussSeidel.H"
#include "fwTiming.H"

namespace smootherTest::fwgamg
{

struct NativeMatrixData
{
    Foam::label nCells = 0;
    Foam::labelField lowerAddr;
    Foam::labelField upperAddr;
    Foam::scalarField diag;
    Foam::scalarField upper;
};

NativeMatrixData nativeMatrixData(const Foam::lduMatrix& matrix);

class Matrix
{
    Foam::label nFaces_ = 0;
    std::vector<Foam::label> levelStarts_;
    std::vector<Foam::label> newToOld_;
    std::vector<Foam::label> oldToNew_;
    std::vector<Foam::label> rowStarts_;
    std::vector<Foam::label> outgoingStarts_;
    std::vector<Foam::label> edgeOwners_;
    std::vector<Foam::label> upperEntries_;
    std::vector<Foam::label> directEdgeOwners_;
    std::vector<Foam::label> directEdgeRowStarts_;
    std::vector<Foam::label> directEdgeColumns_;
    std::vector<Foam::scalar> directEdgeCoeffs_;
    // Diagnostic direct-edge view in the exact native LDU face order.  For a
    // reordered Matrix the endpoints are remapped, but face f remains entry f.
    std::vector<Foam::label> lduOrderDirectEdgeOwners_;
    std::vector<Foam::label> lduOrderDirectEdgeColumns_;
    std::vector<Foam::scalar> lduOrderDirectEdgeCoeffs_;
    std::vector<Foam::label> cols_;
    std::vector<Foam::label> faceIds_;
    std::vector<Foam::scalar> coeffs_;
    std::vector<Foam::scalar> diag_;

    static Matrix buildWithOrdering
    (
        const NativeMatrixData& native,
        IndexKahnOrdering ordering,
        double scheduleSeconds,
        LevelSetupTiming* timing,
        bool buildFlatEdges,
        bool buildDirectFlatEdges,
        bool buildNativeFaceOrderDirectEdges
    );

public:
    Matrix() = default;

    static Matrix build
    (
        const NativeMatrixData& native,
        Foam::label width,
        LevelSetupTiming* timing = nullptr,
        bool buildFlatEdges = false,
        bool buildDirectFlatEdges = false,
        bool buildNativeFaceOrderDirectEdges = false
    );

    static Matrix buildNatural
    (
        const NativeMatrixData& native,
        LevelSetupTiming* timing = nullptr,
        bool buildFlatEdges = false,
        bool buildDirectFlatEdges = false,
        bool buildNativeFaceOrderDirectEdges = false
    );

    Foam::label nCells() const noexcept
    {
        return static_cast<Foam::label>(diag_.size());
    }
    Foam::label nFaces() const noexcept { return nFaces_; }

    const std::vector<Foam::label>& levelStarts() const noexcept
    {
        return levelStarts_;
    }
    const std::vector<Foam::label>& newToOld() const noexcept
    {
        return newToOld_;
    }
    const std::vector<Foam::label>& oldToNew() const noexcept
    {
        return oldToNew_;
    }
    const std::vector<Foam::label>& rowStarts() const noexcept
    {
        return rowStarts_;
    }
    const std::vector<Foam::label>& outgoingStarts() const noexcept
    {
        return outgoingStarts_;
    }
    const std::vector<Foam::label>& edgeOwners() const noexcept
    {
        return edgeOwners_;
    }
    const std::vector<Foam::label>& upperEntries() const noexcept
    {
        return upperEntries_;
    }
    const std::vector<Foam::label>& directEdgeOwners() const noexcept
    {
        return directEdgeOwners_;
    }
    const std::vector<Foam::label>& directEdgeRowStarts() const noexcept
    {
        return directEdgeRowStarts_;
    }
    const std::vector<Foam::label>& directEdgeColumns() const noexcept
    {
        return directEdgeColumns_;
    }
    const std::vector<Foam::scalar>& directEdgeCoeffs() const noexcept
    {
        return directEdgeCoeffs_;
    }
    const std::vector<Foam::label>& lduOrderDirectEdgeOwners() const noexcept
    {
        return lduOrderDirectEdgeOwners_;
    }
    const std::vector<Foam::label>& lduOrderDirectEdgeColumns() const noexcept
    {
        return lduOrderDirectEdgeColumns_;
    }
    const std::vector<Foam::scalar>& lduOrderDirectEdgeCoeffs() const noexcept
    {
        return lduOrderDirectEdgeCoeffs_;
    }
    const std::vector<Foam::label>& cols() const noexcept { return cols_; }
    const std::vector<Foam::label>& faceIds() const noexcept
    {
        return faceIds_;
    }
    const std::vector<Foam::scalar>& coeffs() const noexcept
    {
        return coeffs_;
    }
    const std::vector<Foam::scalar>& diag() const noexcept { return diag_; }

    void multiply
    (
        Foam::scalarField& result,
        const Foam::scalarField& field,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    void multiplyFlatEdgeWise
    (
        Foam::scalarField& result,
        const Foam::scalarField& field,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    void multiplyDirectFlatEdgeWise
    (
        Foam::scalarField& result,
        const Foam::scalarField& field,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    void multiplyDirectFlatOwnerAccumulated
    (
        Foam::scalarField& result,
        const Foam::scalarField& field,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    // Experimental single edge-loop traversal.  The owner-side result stays
    // in a scalar register until the owner changes, without an outer row loop.
    void multiplyDirectFlatOwnerRunAccumulated
    (
        Foam::scalarField& result,
        const Foam::scalarField& field,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    void multiplyLduOrderDirectFlatEdgeWise
    (
        Foam::scalarField& result,
        const Foam::scalarField& field,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    // Diagnostic entry point for cache-colour experiments.  Storage is
    // supplied by the caller; the arithmetic and native face order match
    // multiplyLduOrderDirectFlatEdgeWise().
    void multiplyLduOrderDirectFlatEdgeWiseRaw
    (
        Foam::scalar* result,
        const Foam::scalar* field
    ) const;

    void residual
    (
        Foam::scalarField& result,
        const Foam::scalarField& field,
        const Foam::scalarField& source,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    void residualFlatEdgeWise
    (
        Foam::scalarField& result,
        const Foam::scalarField& field,
        const Foam::scalarField& source,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    void residualDirectFlatEdgeWise
    (
        Foam::scalarField& result,
        const Foam::scalarField& field,
        const Foam::scalarField& source,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    void residualLduOrderDirectFlatEdgeWise
    (
        Foam::scalarField& result,
        const Foam::scalarField& field,
        const Foam::scalarField& source,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    void smooth
    (
        Foam::scalarField& psi,
        const Foam::scalarField& source,
        Foam::label sweeps,
        bool preSmoothing,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    // Execute all requested forward IH1/Index-Kahn sweeps and produce the
    // matrix product of the final field during the last sweep.  The result is
    // mathematically A*psi after the completed smoother.  No O(nFaces)
    // temporary or duplicate matrix representation is used.
    void smoothAndMultiply
    (
        Foam::scalarField& psi,
        const Foam::scalarField& source,
        Foam::scalarField& matrixPsi,
        Foam::label sweeps,
        bool preSmoothing,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    // Reduced-pass formulation.  The final sweep stores psi_old per row;
    // afterwards each upper edge contributes a*(psi_new-psi_old) to the row
    // which had already been satisfied by its Gauss-Seidel update.
    void smoothAndMultiplyDelta
    (
        Foam::scalarField& psi,
        const Foam::scalarField& source,
        Foam::scalarField& matrixPsi,
        Foam::label sweeps,
        bool preSmoothing,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    // Retained implementation baseline: discover each owner range by reading
    // directEdgeOwners_ in the correction loop.
    void smoothAndMultiplyDeltaCurrent
    (
        Foam::scalarField& psi,
        const Foam::scalarField& source,
        Foam::scalarField& matrixPsi,
        Foam::label sweeps,
        bool preSmoothing,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    // Use owner-row offsets into the direct edge arrays.
    void smoothAndMultiplyDeltaEdgeRows
    (
        Foam::scalarField& psi,
        const Foam::scalarField& source,
        Foam::scalarField& matrixPsi,
        Foam::label sweeps,
        bool preSmoothing,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    // Packed outgoing-range representation before the final scalar cleanup.
    void smoothAndMultiplyDeltaPackedBaseline
    (
        Foam::scalarField& psi,
        const Foam::scalarField& source,
        Foam::scalarField& matrixPsi,
        Foam::label sweeps,
        bool preSmoothing,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    // Correction-only entry points used by the focused microbenchmarks.
    // matrixPsi contains delta on entry and A*psi on return.
    void deltaCorrectionCurrent
    (
        const Foam::scalarField& source,
        Foam::scalarField& matrixPsi
    ) const;
    void deltaCorrectionEdgeRows
    (
        const Foam::scalarField& source,
        Foam::scalarField& matrixPsi
    ) const;
    void deltaCorrectionPacked
    (
        const Foam::scalarField& source,
        Foam::scalarField& matrixPsi
    ) const;

    void scaleCorrection
    (
        Foam::scalarField& field,
        const Foam::scalarField& source,
        Foam::scalarField& matrixField,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    void scaleCorrectionFlatEdgeWise
    (
        Foam::scalarField& field,
        const Foam::scalarField& source,
        Foam::scalarField& matrixField,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    void scaleCorrectionDirectFlatEdgeWise
    (
        Foam::scalarField& field,
        const Foam::scalarField& source,
        Foam::scalarField& matrixField,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    void scaleCorrectionLduOrderDirectFlatEdgeWise
    (
        Foam::scalarField& field,
        const Foam::scalarField& source,
        Foam::scalarField& matrixField,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;

    void solveDense
    (
        Foam::scalarField& correction,
        const Foam::scalarField& source,
        TimingStats* timing = nullptr,
        std::size_t level = 0
    ) const;
};

} // namespace smootherTest::fwgamg

#endif
