#include "IndexKahnSmoother.H"
#include "IndexKahnScheduleCache.H"

#include "GaussSeidelSmoother.H"
#include "PrecisionAdaptor.H"
#include "UPstream.H"
#include "IOstreams.H"
#include "error.H"
#include "typeInfo.H"

#include <chrono>

namespace Foam
{
    defineTypeNameAndDebug(IndexKahnSmoother, 0);

    lduMatrix::smoother::addsymMatrixConstructorToTable<IndexKahnSmoother>
        addIndexKahnSmootherSymMatrixConstructorToTable_;

    lduMatrix::smoother::addasymMatrixConstructorToTable<IndexKahnSmoother>
        addIndexKahnSmootherAsymMatrixConstructorToTable_;
}


Foam::IndexKahnSmoother::IndexKahnSmoother
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::smoother
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces
    ),
    width_(solverControls.getOrDefault<label>("width", 1024)),
    minCells_(solverControls.getOrDefault<label>("minCells", 10000)),
    useIndexKahn_(matrix.diag().size() >= minCells_),
    schedule_(nullptr)
{
    Pout<< "IndexKahnSmoother controls: width=" << width_
        << " minCells=" << minCells_ << endl;

    if (width_ < 1)
    {
        FatalIOErrorInFunction(solverControls)
            << "indexKahn width must be positive, got " << width_
            << exit(FatalIOError);
    }
    if (minCells_ < 0)
    {
        FatalIOErrorInFunction(solverControls)
            << "indexKahn minCells must be non-negative, got " << minCells_
            << exit(FatalIOError);
    }

    const label nCells = matrix_.diag().size();
    if (!useIndexKahn_)
    {
        if (debug)
        {
            Pout<< "IndexKahn GAMG: cells=" << nCells
                << " minCells=" << minCells_
                << " action=GaussSeidel" << endl;
        }
        return;
    }

    const IndexKahnScheduleCache& cache =
        IndexKahnScheduleCache::New(matrix_.mesh());
    bool scheduleBuilt = false;
    schedule_ = &cache.getOrCreate(width_, &scheduleBuilt);

    if (debug)
    {
        Pout<< "IndexKahn GAMG: cells=" << nCells
            << " minCells=" << minCells_
            << " action=IndexKahn width=" << width_
            << " schedule=" << (scheduleBuilt ? "build" : "reuse") << endl;
    }

    if (schedule_->waveCells.size() != std::size_t(matrix_.diag().size()))
    {
        FatalErrorInFunction
            << "Index-Kahn schedule and matrix sizes differ"
            << exit(FatalError);
    }

    // Only current-matrix off-diagonal coefficients are packed here. The
    // mesh cache contains face IDs and row order, never coefficient values.
    // Time this separately from schedule construction and smoothing.
    const auto packStart = std::chrono::steady_clock::now();
    coeffs_.setSize(static_cast<label>(schedule_->faceIds.size()));
    const scalar* const lower = matrix_.lower().begin();
    const scalar* const upper = matrix_.upper().begin();
    const label* const faceIds = schedule_->faceIds.data();
    const std::uint8_t* const degrees = schedule_->degrees.data();
    const std::uint8_t* const incomingDegrees = schedule_->incomingDegrees.data();
    scalar* const packedCoeffs = coeffs_.begin();
    const label nRows = static_cast<label>(schedule_->waveCells.size());

    label p = 0;
    for (label row=0; row<nRows; ++row)
    {
        const label split = p + incomingDegrees[row];
        const label end = p + degrees[row];
        for (; p<split; ++p) packedCoeffs[p] = lower[faceIds[p]];
        for (; p<end; ++p) packedCoeffs[p] = upper[faceIds[p]];
    }
    if (debug)
    {
        const auto packEnd = std::chrono::steady_clock::now();
        const double packMs =
            std::chrono::duration<double, std::milli>(packEnd - packStart).count();
        Info<< "IndexKahn: coefficient packing field=" << fieldName_
            << " entries=" << coeffs_.size()
            << " time=" << packMs << " ms" << endl;
    }
}


void Foam::IndexKahnSmoother::smoothInternal
(
    solveScalarField& psi,
    const solveScalarField& source,
    const direction cmpt,
    const label nSweeps
) const
{
    if (!useIndexKahn_)
    {
        GaussSeidelSmoother::smooth
        (
            fieldName_,
            psi,
            matrix_,
            source,
            interfaceBouCoeffs_,
            interfaces_,
            cmpt,
            nSweeps
        );
        return;
    }

    const label nCells = psi.size();
    solveScalarField& bPrime = matrix_.work(nCells);
    const label* const waveCells = schedule_->waveCells.data();
    const label* const cols = schedule_->cols.data();
    const std::uint8_t* const degrees = schedule_->degrees.data();
    const scalar* const coeffs = coeffs_.begin();
    const scalar* const diag = matrix_.diag().begin();
    solveScalar* const psiPtr = psi.begin();
    solveScalar* const bPrimePtr = bPrime.begin();
    std::chrono::duration<double> kernelTime = std::chrono::duration<double>::zero();

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        bPrime = source;

        // Preserve the reference GaussSeidel coupled-boundary sequence.
        const label startRequest = UPstream::nRequests();
        matrix_.initMatrixInterfaces
        (
            false,
            interfaceBouCoeffs_,
            interfaces_,
            psi,
            bPrime,
            cmpt
        );
        matrix_.updateMatrixInterfaces
        (
            false,
            interfaceBouCoeffs_,
            interfaces_,
            psi,
            bPrime,
            cmpt,
            startRequest
        );

        // waveCells is already a valid topological order. Levels are needed
        // during preprocessing, but not for this serial sweep.
        std::chrono::steady_clock::time_point kernelStart;
        if (debug) kernelStart = std::chrono::steady_clock::now();
        label p = 0;
        for (label row=0; row<nCells; ++row)
        {
            const label cell = waveCells[row];
            solveScalar psii = bPrimePtr[cell];
            const label end = p + degrees[row];
            for (; p<end; ++p) psii -= coeffs[p]*psiPtr[cols[p]];
            psiPtr[cell] = psii/diag[cell];
        }
        if (debug) kernelTime += std::chrono::steady_clock::now() - kernelStart;
    }
    if (debug && nSweeps > 0)
    {
        const double kernelMs =
            std::chrono::duration<double, std::milli>(kernelTime).count();
        Info<< "IndexKahn: sweep kernel field=" << fieldName_
            << " sweeps=" << nSweeps
            << " total=" << kernelMs << " ms"
            << " average=" << kernelMs/nSweeps << " ms/sweep" << endl;
    }
}


void Foam::IndexKahnSmoother::smooth
(
    solveScalarField& psi,
    const scalarField& source,
    const direction cmpt,
    const label nSweeps
) const
{
    ConstPrecisionAdaptor<solveScalar, scalar> tsource(source);
    smoothInternal(psi, tsource(), cmpt, nSweeps);
}


void Foam::IndexKahnSmoother::scalarSmooth
(
    solveScalarField& psi,
    const solveScalarField& source,
    const direction cmpt,
    const label nSweeps
) const
{
    smoothInternal(psi, source, cmpt, nSweeps);
}
