#include "fwMatrix.H"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
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

template<std::size_t IncomingDegree>
inline void fusedSmoothRow
(
    const Foam::label row,
    const Foam::label beginRow,
    const Foam::label endRow,
    const Foam::label* const cols,
    const Foam::scalar* const coeffs,
    const Foam::scalar* const diag,
    const Foam::scalar* const source,
    Foam::scalar* const psi,
    Foam::scalar* const matrixPsi
)
{
    std::array<Foam::label, IncomingDegree> incomingCols;
    std::array<Foam::scalar, IncomingDegree> incomingCoeffs;
    Foam::scalar value = source[row];
    Foam::scalar finalIncoming = 0.0;

    for (std::size_t i=0; i<IncomingDegree; ++i)
    {
        const Foam::label p = beginRow + static_cast<Foam::label>(i);
        incomingCols[i] = cols[p];
        incomingCoeffs[i] = coeffs[p];
        const Foam::scalar contribution = incomingCoeffs[i]*psi[incomingCols[i]];
        value -= contribution;
        finalIncoming += contribution;
    }
    for (Foam::label p=beginRow + static_cast<Foam::label>(IncomingDegree); p<endRow; ++p)
        value -= coeffs[p]*psi[cols[p]];

    const Foam::scalar finalValue = value/diag[row];
    psi[row] = finalValue;
    matrixPsi[row] = diag[row]*finalValue + finalIncoming;
    for (std::size_t i=0; i<IncomingDegree; ++i)
        matrixPsi[incomingCols[i]] += incomingCoeffs[i]*finalValue;
}

inline void fusedSmoothWideRow
(
    const Foam::label row,
    const Foam::label beginRow,
    const Foam::label split,
    const Foam::label endRow,
    const Foam::label* const cols,
    const Foam::scalar* const coeffs,
    const Foam::scalar* const diag,
    const Foam::scalar* const source,
    Foam::scalar* const psi,
    Foam::scalar* const matrixPsi
)
{
    Foam::scalar value = source[row];
    Foam::scalar finalIncoming = 0.0;
    for (Foam::label p=beginRow; p<split; ++p)
    {
        const Foam::scalar contribution = coeffs[p]*psi[cols[p]];
        value -= contribution;
        finalIncoming += contribution;
    }
    for (Foam::label p=split; p<endRow; ++p)
        value -= coeffs[p]*psi[cols[p]];

    const Foam::scalar finalValue = value/diag[row];
    psi[row] = finalValue;
    matrixPsi[row] = diag[row]*finalValue + finalIncoming;
    for (Foam::label p=beginRow; p<split; ++p)
        matrixPsi[cols[p]] += coeffs[p]*finalValue;
}

} // namespace

NativeMatrixData nativeMatrixData(const Foam::lduMatrix& matrix)
{
    NativeMatrixData result;
    result.nCells = static_cast<Foam::label>(matrix.diag().size());
    result.upperAddr = matrix.lduAddr().upperAddr();
    result.lowerAddr.resize(result.upperAddr.size());
    const Foam::labelField& starts = matrix.lduAddr().ownerStartAddr();
    for (Foam::label cell=0; cell<result.nCells; ++cell)
        for (Foam::label face=starts[cell]; face<starts[cell + 1]; ++face)
            result.lowerAddr[face] = cell;
    if (matrix.lower().size() != matrix.upper().size())
        throw std::runtime_error("fwGAMG requires a symmetric matrix");
    for (Foam::label face=0; face<static_cast<Foam::label>(matrix.upper().size()); ++face)
        if (matrix.lower()[face] != matrix.upper()[face])
            throw std::runtime_error("fwGAMG requires identical lower/upper coefficients");
    result.diag = matrix.diag();
    result.upper = matrix.upper();
    return result;
}

Matrix Matrix::build
(
    const NativeMatrixData& native,
    const Foam::label width,
    LevelSetupTiming* const timing,
    const bool buildFlatEdges,
    const bool buildDirectFlatEdges,
    const bool buildNativeFaceOrderDirectEdges
)
{
    if
    (
        native.nCells < 1
     || native.diag.size() != static_cast<std::size_t>(native.nCells)
     || native.lowerAddr.size() != native.upperAddr.size()
     || native.upper.size() != native.upperAddr.size()
    ) throw std::runtime_error("invalid native matrix for fw packing");

    Foam::labelField ownerStarts(native.nCells + 1, 0);
    for (const Foam::label owner : native.lowerAddr) ++ownerStarts[owner + 1];
    for (Foam::label cell=0; cell<native.nCells; ++cell)
        ownerStarts[cell + 1] += ownerStarts[cell];

    const auto scheduleBegin = std::chrono::steady_clock::now();
    IndexKahnOrdering ordering = makeIndexKahnOrdering
    (
        native.nCells, ownerStarts, native.upperAddr, width
    );
    const double scheduleSeconds = elapsed(scheduleBegin);

    return buildWithOrdering
    (
        native, std::move(ordering), scheduleSeconds, timing,
        buildFlatEdges, buildDirectFlatEdges,
        buildNativeFaceOrderDirectEdges
    );
}

Matrix Matrix::buildNatural
(
    const NativeMatrixData& native,
    LevelSetupTiming* const timing,
    const bool buildFlatEdges,
    const bool buildDirectFlatEdges,
    const bool buildNativeFaceOrderDirectEdges
)
{
    if
    (
        native.nCells < 1
     || native.diag.size() != static_cast<std::size_t>(native.nCells)
     || native.lowerAddr.size() != native.upperAddr.size()
     || native.upper.size() != native.upperAddr.size()
    ) throw std::runtime_error("invalid native matrix for natural fw packing");
    IndexKahnOrdering ordering;
    ordering.levelStarts = {0, native.nCells};
    ordering.newToOld.resize(native.nCells);
    ordering.oldToNew.resize(native.nCells);
    for (Foam::label cell=0; cell<native.nCells; ++cell)
        ordering.newToOld[cell] = ordering.oldToNew[cell] = cell;
    return buildWithOrdering
    (
        native, std::move(ordering), 0.0, timing,
        buildFlatEdges, buildDirectFlatEdges,
        buildNativeFaceOrderDirectEdges
    );
}

Matrix Matrix::buildWithOrdering
(
    const NativeMatrixData& native,
    IndexKahnOrdering ordering,
    const double scheduleSeconds,
    LevelSetupTiming* const timing,
    const bool buildFlatEdges,
    const bool buildDirectFlatEdges,
    const bool buildNativeFaceOrderDirectEdges
)
{
    Foam::labelField ownerStarts(native.nCells + 1, 0);
    for (const Foam::label owner : native.lowerAddr) ++ownerStarts[owner + 1];
    for (Foam::label cell=0; cell<native.nCells; ++cell)
        ownerStarts[cell + 1] += ownerStarts[cell];

    Matrix result;
    result.nFaces_ = static_cast<Foam::label>(native.upper.size());
    const auto permutationBegin = std::chrono::steady_clock::now();
    result.levelStarts_ = ordering.levelStarts;
    result.newToOld_ = ordering.newToOld;
    const double permutationSeconds = elapsed(permutationBegin);
    const auto inverseBegin = std::chrono::steady_clock::now();
    result.oldToNew_ = ordering.oldToNew;
    const double inverseSeconds = elapsed(inverseBegin);

    const auto packingBegin = std::chrono::steady_clock::now();
    std::vector<Foam::label> incomingStarts(native.nCells + 1, 0);
    for (const Foam::label neighbour : native.upperAddr)
        ++incomingStarts[neighbour + 1];
    for (Foam::label cell=0; cell<native.nCells; ++cell)
        incomingStarts[cell + 1] += incomingStarts[cell];
    std::vector<Foam::label> incomingFaces(native.upper.size());
    std::vector<Foam::label> cursor = incomingStarts;
    for (Foam::label face=0; face<result.nFaces_; ++face)
        incomingFaces[cursor[native.upperAddr[face]]++] = face;

    result.rowStarts_.reserve(native.nCells + 1);
    result.outgoingStarts_.resize(native.nCells);
    result.cols_.reserve(2*native.upper.size());
    result.faceIds_.reserve(2*native.upper.size());
    result.coeffs_.reserve(2*native.upper.size());
    result.diag_.reserve(native.nCells);
    result.rowStarts_.push_back(0);
    for (Foam::label row=0; row<native.nCells; ++row)
    {
        const Foam::label oldCell = result.newToOld_[row];
        result.diag_.push_back(native.diag[oldCell]);
        for
        (
            Foam::label in=incomingStarts[oldCell];
            in<incomingStarts[oldCell + 1];
            ++in
        )
        {
            const Foam::label face = incomingFaces[in];
            result.cols_.push_back(result.oldToNew_[native.lowerAddr[face]]);
            result.faceIds_.push_back(face);
            result.coeffs_.push_back(native.upper[face]);
        }
        // Rows are intentionally packed as incoming | outgoing.  Record the
        // boundary without changing either group's contribution order.
        result.outgoingStarts_[row] = result.cols_.size();
        for
        (
            Foam::label face=ownerStarts[oldCell];
            face<ownerStarts[oldCell + 1];
            ++face
        )
        {
            result.cols_.push_back(result.oldToNew_[native.upperAddr[face]]);
            result.faceIds_.push_back(face);
            result.coeffs_.push_back(native.upper[face]);
        }
        result.rowStarts_.push_back(result.cols_.size());
    }
    if (result.cols_.size() != 2*native.upper.size())
        throw std::runtime_error("fw packed rows do not cover all faces");
    for (Foam::label row=0; row<native.nCells; ++row)
    {
        for
        (
            Foam::label p=result.rowStarts_[row];
            p<result.outgoingStarts_[row];
            ++p
        )
            if (result.cols_[p] >= row)
                throw std::runtime_error("fw incoming row range is not contiguous");
        for
        (
            Foam::label p=result.outgoingStarts_[row];
            p<result.rowStarts_[row + 1];
            ++p
        )
            if (result.cols_[p] <= row)
                throw std::runtime_error("fw outgoing row range is not contiguous");
    }

    double flatMetadataSeconds = 0;
    double directFlatMetadataSeconds = 0;
    if (buildFlatEdges)
    {
        const auto flatMetadataBegin = std::chrono::steady_clock::now();
        result.edgeOwners_.reserve(result.nFaces_);
        result.upperEntries_.reserve(result.nFaces_);
        for (Foam::label row=0; row<native.nCells; ++row)
            for
            (
                Foam::label p=result.rowStarts_[row];
                p<result.rowStarts_[row + 1];
                ++p
            )
                if (result.cols_[p] > row)
                {
                    result.edgeOwners_.push_back(row);
                    result.upperEntries_.push_back(p);
                }
        if
        (
            result.edgeOwners_.size() != static_cast<std::size_t>(result.nFaces_)
         || result.upperEntries_.size() != static_cast<std::size_t>(result.nFaces_)
        ) throw std::runtime_error
        (
            "fw flat edge metadata does not cover every symmetric face once"
        );
        flatMetadataSeconds = elapsed(flatMetadataBegin);
    }
    if (buildNativeFaceOrderDirectEdges)
    {
        const auto directFlatMetadataBegin = std::chrono::steady_clock::now();
        result.lduOrderDirectEdgeOwners_.resize(result.nFaces_);
        result.lduOrderDirectEdgeColumns_.resize(result.nFaces_);
        result.lduOrderDirectEdgeCoeffs_.resize(result.nFaces_);
        // Preserve the native OpenFOAM LDU face sequence exactly.  The
        // natural-order diagnostic therefore has, literally:
        //   owner[f] = lowerAddr[f], column[f] = upperAddr[f], coeff[f] = upper[f].
        // Endpoint remapping is only needed when the Matrix itself is reordered.
        for (Foam::label face=0; face<result.nFaces_; ++face)
        {
            result.lduOrderDirectEdgeOwners_[face] =
                result.oldToNew_[native.lowerAddr[face]];
            result.lduOrderDirectEdgeColumns_[face] =
                result.oldToNew_[native.upperAddr[face]];
            result.lduOrderDirectEdgeCoeffs_[face] = native.upper[face];
        }
        if
        (
            result.lduOrderDirectEdgeOwners_.size()
                != static_cast<std::size_t>(result.nFaces_)
         || result.lduOrderDirectEdgeColumns_.size()
                != static_cast<std::size_t>(result.nFaces_)
         || result.lduOrderDirectEdgeCoeffs_.size()
                != static_cast<std::size_t>(result.nFaces_)
        ) throw std::runtime_error
        (
            "fw native-face-order direct edge view is incomplete"
        );
        directFlatMetadataSeconds += elapsed(directFlatMetadataBegin);
    }
    if (buildDirectFlatEdges)
    {
        const auto directFlatMetadataBegin = std::chrono::steady_clock::now();
        result.directEdgeOwners_.reserve(result.nFaces_);
        result.directEdgeRowStarts_.reserve(native.nCells + 1);
        result.directEdgeColumns_.reserve(result.nFaces_);
        result.directEdgeCoeffs_.reserve(result.nFaces_);
        for (Foam::label row=0; row<native.nCells; ++row)
        {
            result.directEdgeRowStarts_.push_back
            (
                static_cast<Foam::label>(result.directEdgeColumns_.size())
            );
            for
            (
                Foam::label p=result.rowStarts_[row];
                p<result.rowStarts_[row + 1];
                ++p
            )
                if (result.cols_[p] > row)
                {
                    result.directEdgeOwners_.push_back(row);
                    result.directEdgeColumns_.push_back(result.cols_[p]);
                    result.directEdgeCoeffs_.push_back(result.coeffs_[p]);
                }
        }
        result.directEdgeRowStarts_.push_back(result.nFaces_);
        if
        (
            result.directEdgeOwners_.size()
                != static_cast<std::size_t>(result.nFaces_)
         || result.directEdgeColumns_.size()
                != static_cast<std::size_t>(result.nFaces_)
         || result.directEdgeCoeffs_.size()
                != static_cast<std::size_t>(result.nFaces_)
         || result.directEdgeRowStarts_.size()
                != static_cast<std::size_t>(native.nCells + 1)
         || result.directEdgeRowStarts_.back() != result.nFaces_
        ) throw std::runtime_error
        (
            "fw direct flat edge view does not cover every symmetric face once"
        );
        for (Foam::label row=0; row<native.nCells; ++row)
            if
            (
                result.directEdgeRowStarts_[row + 1]
              - result.directEdgeRowStarts_[row]
             != result.rowStarts_[row + 1] - result.outgoingStarts_[row]
            ) throw std::runtime_error
            (
                "fw direct edge owner ranges disagree with packed rows"
            );
        directFlatMetadataSeconds += elapsed(directFlatMetadataBegin);
    }

    if (timing)
    {
        timing->cells = native.nCells;
        timing->faces = result.nFaces_;
        timing->dependencyLevels = result.levelStarts_.size() - 1;
        timing->scheduleSeconds += scheduleSeconds;
        timing->permutationSeconds += permutationSeconds;
        timing->inversePermutationSeconds += inverseSeconds;
        timing->matrixPackingSeconds += elapsed(packingBegin);
        timing->flatEdgeMetadataSeconds += flatMetadataSeconds;
        timing->directFlatEdgeMetadataSeconds += directFlatMetadataSeconds;
    }
    return result;
}

void Matrix::multiplyDirectFlatEdgeWise
(
    Foam::scalarField& result,
    const Foam::scalarField& field,
    TimingStats* const timing,
    const std::size_t level
) const
{
    if (field.size() != diag_.size())
        throw std::runtime_error("fw direct flat edge SpMV field size mismatch");
    if
    (
        directEdgeOwners_.size() != static_cast<std::size_t>(nFaces_)
     || directEdgeColumns_.size() != static_cast<std::size_t>(nFaces_)
     || directEdgeCoeffs_.size() != static_cast<std::size_t>(nFaces_)
    ) throw std::runtime_error("fw direct flat edge view was not constructed");
    const auto begin = std::chrono::steady_clock::now();
    result.resize(field.size());
    for (Foam::label row=0; row<nCells(); ++row)
        result[row] = diag_[row]*field[row];
    for (Foam::label edge=0; edge<nFaces_; ++edge)
    {
        const Foam::label owner = directEdgeOwners_[edge];
        const Foam::label column = directEdgeColumns_[edge];
        const Foam::scalar coefficient = directEdgeCoeffs_[edge];
        result[owner] += coefficient*field[column];
        result[column] += coefficient*field[owner];
    }
    if (timing && timing->enabled)
        add(timing->levels.at(level).spmv, elapsed(begin));
}

void Matrix::multiplyDirectFlatOwnerAccumulated
(
    Foam::scalarField& result,
    const Foam::scalarField& field,
    TimingStats* const timing,
    const std::size_t level
) const
{
    if (field.size() != diag_.size())
        throw std::runtime_error
        (
            "fw owner-accumulated direct flat edge SpMV field size mismatch"
        );
    if
    (
        directEdgeRowStarts_.size() != diag_.size() + 1
     || directEdgeColumns_.size() != static_cast<std::size_t>(nFaces_)
     || directEdgeCoeffs_.size() != static_cast<std::size_t>(nFaces_)
    ) throw std::runtime_error
    (
        "fw owner-accumulated direct flat edge view was not constructed"
    );
    const auto begin = std::chrono::steady_clock::now();
    result.resize(field.size());
    for (Foam::label row=0; row<nCells(); ++row)
        result[row] = diag_[row]*field[row];

    const Foam::label* const edgeStarts = directEdgeRowStarts_.data();
    const Foam::label* const columns = directEdgeColumns_.data();
    const Foam::scalar* const coefficients = directEdgeCoeffs_.data();
    const Foam::scalar* const x = field.data();
    Foam::scalar* const y = result.data();
    for (Foam::label owner=0; owner<nCells(); ++owner)
    {
        Foam::scalar ownerResult = y[owner];
        const Foam::scalar xOwner = x[owner];
        for
        (
            Foam::label edge=edgeStarts[owner];
            edge<edgeStarts[owner + 1];
            ++edge
        )
        {
            const Foam::label column = columns[edge];
            const Foam::scalar coefficient = coefficients[edge];
            ownerResult += coefficient*x[column];
            y[column] += coefficient*xOwner;
        }
        y[owner] = ownerResult;
    }
    if (timing && timing->enabled)
        add(timing->levels.at(level).spmv, elapsed(begin));
}

void Matrix::multiplyDirectFlatOwnerRunAccumulated
(
    Foam::scalarField& result,
    const Foam::scalarField& field,
    TimingStats* const timing,
    const std::size_t level
) const
{
    if (field.size() != diag_.size())
        throw std::runtime_error
        (
            "fw owner-run direct flat edge SpMV field size mismatch"
        );
    if
    (
        directEdgeOwners_.size() != static_cast<std::size_t>(nFaces_)
     || directEdgeColumns_.size() != static_cast<std::size_t>(nFaces_)
     || directEdgeCoeffs_.size() != static_cast<std::size_t>(nFaces_)
    ) throw std::runtime_error
    (
        "fw owner-run direct flat edge view was not constructed"
    );
    const auto begin = std::chrono::steady_clock::now();
    result.resize(field.size());
    for (Foam::label row=0; row<nCells(); ++row)
        result[row] = diag_[row]*field[row];

    if (nFaces_ > 0)
    {
        const Foam::label* const owners = directEdgeOwners_.data();
        const Foam::label* const columns = directEdgeColumns_.data();
        const Foam::scalar* const coefficients = directEdgeCoeffs_.data();
        const Foam::scalar* const x = field.data();
        Foam::scalar* const y = result.data();

        Foam::label previousOwner = owners[0];
        Foam::scalar ownerResult = y[previousOwner];
        Foam::scalar xOwner = x[previousOwner];
        for (Foam::label edge=0; edge<nFaces_; ++edge)
        {
            const Foam::label owner = owners[edge];
            if (owner != previousOwner)
            {
                y[previousOwner] = ownerResult;
                previousOwner = owner;
                ownerResult = y[owner];
                xOwner = x[owner];
            }

            const Foam::label column = columns[edge];
            const Foam::scalar coefficient = coefficients[edge];
            ownerResult += coefficient*x[column];
            y[column] += coefficient*xOwner;
        }
        y[previousOwner] = ownerResult;
    }
    if (timing && timing->enabled)
        add(timing->levels.at(level).spmv, elapsed(begin));
}

void Matrix::multiplyLduOrderDirectFlatEdgeWise
(
    Foam::scalarField& result,
    const Foam::scalarField& field,
    TimingStats* const timing,
    const std::size_t level
) const
{
    if (field.size() != diag_.size())
        throw std::runtime_error
        (
            "fw LDU-order direct flat edge SpMV field size mismatch"
        );
    if
    (
        lduOrderDirectEdgeOwners_.size() != static_cast<std::size_t>(nFaces_)
     || lduOrderDirectEdgeColumns_.size() != static_cast<std::size_t>(nFaces_)
     || lduOrderDirectEdgeCoeffs_.size() != static_cast<std::size_t>(nFaces_)
    ) throw std::runtime_error
    (
        "fw LDU-order direct flat edge view was not constructed"
    );
    const auto begin = std::chrono::steady_clock::now();
    result.resize(field.size());
    for (Foam::label row=0; row<nCells(); ++row)
        result[row] = diag_[row]*field[row];
    for (Foam::label face=0; face<nFaces_; ++face)
    {
        const Foam::label owner = lduOrderDirectEdgeOwners_[face];
        const Foam::label column = lduOrderDirectEdgeColumns_[face];
        const Foam::scalar coefficient = lduOrderDirectEdgeCoeffs_[face];
        result[owner] += coefficient*field[column];
        result[column] += coefficient*field[owner];
    }
    if (timing && timing->enabled)
        add(timing->levels.at(level).spmv, elapsed(begin));
}

void Matrix::multiplyLduOrderDirectFlatEdgeWiseRaw
(
    Foam::scalar* const result,
    const Foam::scalar* const field
) const
{
    if (!result || !field)
        throw std::runtime_error("fw raw LDU-order SpMV received null storage");
    if
    (
        lduOrderDirectEdgeOwners_.size() != static_cast<std::size_t>(nFaces_)
     || lduOrderDirectEdgeColumns_.size() != static_cast<std::size_t>(nFaces_)
     || lduOrderDirectEdgeCoeffs_.size() != static_cast<std::size_t>(nFaces_)
    ) throw std::runtime_error
    (
        "fw raw LDU-order direct flat edge view was not constructed"
    );
    for (Foam::label row=0; row<nCells(); ++row)
        result[row] = diag_[row]*field[row];
    for (Foam::label face=0; face<nFaces_; ++face)
    {
        const Foam::label owner = lduOrderDirectEdgeOwners_[face];
        const Foam::label column = lduOrderDirectEdgeColumns_[face];
        const Foam::scalar coefficient = lduOrderDirectEdgeCoeffs_[face];
        result[owner] += coefficient*field[column];
        result[column] += coefficient*field[owner];
    }
}

void Matrix::multiplyFlatEdgeWise
(
    Foam::scalarField& result,
    const Foam::scalarField& field,
    TimingStats* const timing,
    const std::size_t level
) const
{
    if (field.size() != diag_.size())
        throw std::runtime_error("fw flat edge-wise SpMV field size mismatch");
    if
    (
        edgeOwners_.size() != static_cast<std::size_t>(nFaces_)
     || upperEntries_.size() != static_cast<std::size_t>(nFaces_)
    ) throw std::runtime_error("fw flat edge metadata was not constructed");
    const auto begin = std::chrono::steady_clock::now();
    result.resize(field.size());
    for (Foam::label row=0; row<nCells(); ++row)
        result[row] = diag_[row]*field[row];
    for (Foam::label edge=0; edge<nFaces_; ++edge)
    {
        const Foam::label row = edgeOwners_[edge];
        const Foam::label p = upperEntries_[edge];
        const Foam::scalar xRow = field[row];
        const Foam::label column = cols_[p];
        const Foam::scalar coefficient = coeffs_[p];
        result[row] += coefficient*field[column];
        result[column] += coefficient*xRow;
    }
    if (timing && timing->enabled)
        add(timing->levels.at(level).spmv, elapsed(begin));
}

void Matrix::multiply
(
    Foam::scalarField& result,
    const Foam::scalarField& field,
    TimingStats* const timing,
    const std::size_t level
) const
{
    if (field.size() != diag_.size())
        throw std::runtime_error("fw SpMV field size mismatch");
    const auto begin = std::chrono::steady_clock::now();
    result.resize(field.size());
    for (Foam::label row=0; row<nCells(); ++row)
    {
        Foam::scalar value = diag_[row]*field[row];
        for (Foam::label p=rowStarts_[row]; p<rowStarts_[row + 1]; ++p)
            value += coeffs_[p]*field[cols_[p]];
        result[row] = value;
    }
    if (timing && timing->enabled)
        add(timing->levels.at(level).spmv, elapsed(begin));
}

void Matrix::residual
(
    Foam::scalarField& result,
    const Foam::scalarField& field,
    const Foam::scalarField& source,
    TimingStats* const timing,
    const std::size_t level
) const
{
    multiply(result, field, timing, level);
    const auto begin = std::chrono::steady_clock::now();
    for (Foam::label row=0; row<nCells(); ++row)
        result[row] = source[row] - result[row];
    if (timing && timing->enabled)
        add(timing->levels.at(level).residual, elapsed(begin));
}

void Matrix::residualFlatEdgeWise
(
    Foam::scalarField& result,
    const Foam::scalarField& field,
    const Foam::scalarField& source,
    TimingStats* const timing,
    const std::size_t level
) const
{
    multiplyFlatEdgeWise(result, field, timing, level);
    const auto begin = std::chrono::steady_clock::now();
    for (Foam::label row=0; row<nCells(); ++row)
        result[row] = source[row] - result[row];
    if (timing && timing->enabled)
        add(timing->levels.at(level).residual, elapsed(begin));
}

void Matrix::residualDirectFlatEdgeWise
(
    Foam::scalarField& result,
    const Foam::scalarField& field,
    const Foam::scalarField& source,
    TimingStats* const timing,
    const std::size_t level
) const
{
    multiplyDirectFlatEdgeWise(result, field, timing, level);
    const auto begin = std::chrono::steady_clock::now();
    for (Foam::label row=0; row<nCells(); ++row)
        result[row] = source[row] - result[row];
    if (timing && timing->enabled)
        add(timing->levels.at(level).residual, elapsed(begin));
}

void Matrix::residualLduOrderDirectFlatEdgeWise
(
    Foam::scalarField& result,
    const Foam::scalarField& field,
    const Foam::scalarField& source,
    TimingStats* const timing,
    const std::size_t level
) const
{
    multiplyLduOrderDirectFlatEdgeWise(result, field, timing, level);
    const auto begin = std::chrono::steady_clock::now();
    for (Foam::label row=0; row<nCells(); ++row)
        result[row] = source[row] - result[row];
    if (timing && timing->enabled)
        add(timing->levels.at(level).residual, elapsed(begin));
}

void Matrix::smooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const Foam::label sweeps,
    const bool preSmoothing,
    TimingStats* const timing,
    const std::size_t level
) const
{
    if (psi.size() != diag_.size() || source.size() != diag_.size())
        throw std::runtime_error("fw smoother field size mismatch");
    const auto begin = std::chrono::steady_clock::now();
    for (Foam::label sweep=0; sweep<sweeps; ++sweep)
        for (Foam::label row=0; row<nCells(); ++row)
        {
            Foam::scalar value = source[row];
            for (Foam::label p=rowStarts_[row]; p<rowStarts_[row + 1]; ++p)
                value -= coeffs_[p]*psi[cols_[p]];
            psi[row] = value/diag_[row];
        }
    if (timing && timing->enabled)
    {
        LevelSolveTiming& value = timing->levels.at(level);
        if (preSmoothing)
        {
            add(value.preSmooth, elapsed(begin));
            value.preSweeps += sweeps;
        }
        else
        {
            add(value.postSmooth, elapsed(begin));
            value.postSweeps += sweeps;
        }
    }
}

void Matrix::smoothAndMultiply
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    Foam::scalarField& matrixPsi,
    const Foam::label sweeps,
    const bool preSmoothing,
    TimingStats* const timing,
    const std::size_t level
) const
{
    if (psi.size() != diag_.size() || source.size() != diag_.size())
        throw std::runtime_error("fw fused smoother field size mismatch");
    if (sweeps < 1)
        throw std::runtime_error("fw fused smoother requires at least one sweep");

    const auto begin = std::chrono::steady_clock::now();

    // Earlier sweeps are unchanged.  Only the final sweep is fused with the
    // A*psi which immediately follows finest smoothing in Solver::solve().
    for (Foam::label sweep=1; sweep<sweeps; ++sweep)
        for (Foam::label row=0; row<nCells(); ++row)
        {
            Foam::scalar value = source[row];
            for (Foam::label p=rowStarts_[row]; p<rowStarts_[row + 1]; ++p)
                value -= coeffs_[p]*psi[cols_[p]];
            psi[row] = value/diag_[row];
        }

    matrixPsi.assign(psi.size(), 0.0);
    const Foam::label* const rowStarts = rowStarts_.data();
    const Foam::label* const cols = cols_.data();
    const Foam::scalar* const coeffs = coeffs_.data();
    const Foam::scalar* const diag = diag_.data();
    const Foam::scalar* const sourcePtr = source.data();
    Foam::scalar* const psiPtr = psi.data();
    Foam::scalar* const matrixPsiPtr = matrixPsi.data();
    for (Foam::label row=0; row<nCells(); ++row)
    {
        const Foam::label beginRow = rowStarts[row];
        const Foam::label endRow = rowStarts[row + 1];
        Foam::label split = beginRow;
        while (split < endRow && cols[split] < row) ++split;

        // delta/finalValue is known only after the complete row reduction.
        // For the common small incoming degrees, retain the already-loaded
        // edge metadata locally so the correction scatter does not reread CSR.
        switch (split - beginRow)
        {
            case 0: fusedSmoothRow<0>(row, beginRow, endRow, cols, coeffs,
                diag, sourcePtr, psiPtr, matrixPsiPtr); break;
            case 1: fusedSmoothRow<1>(row, beginRow, endRow, cols, coeffs,
                diag, sourcePtr, psiPtr, matrixPsiPtr); break;
            case 2: fusedSmoothRow<2>(row, beginRow, endRow, cols, coeffs,
                diag, sourcePtr, psiPtr, matrixPsiPtr); break;
            case 3: fusedSmoothRow<3>(row, beginRow, endRow, cols, coeffs,
                diag, sourcePtr, psiPtr, matrixPsiPtr); break;
            case 4: fusedSmoothRow<4>(row, beginRow, endRow, cols, coeffs,
                diag, sourcePtr, psiPtr, matrixPsiPtr); break;
            case 5: fusedSmoothRow<5>(row, beginRow, endRow, cols, coeffs,
                diag, sourcePtr, psiPtr, matrixPsiPtr); break;
            case 6: fusedSmoothRow<6>(row, beginRow, endRow, cols, coeffs,
                diag, sourcePtr, psiPtr, matrixPsiPtr); break;
            case 7: fusedSmoothRow<7>(row, beginRow, endRow, cols, coeffs,
                diag, sourcePtr, psiPtr, matrixPsiPtr); break;
            case 8: fusedSmoothRow<8>(row, beginRow, endRow, cols, coeffs,
                diag, sourcePtr, psiPtr, matrixPsiPtr); break;
            default: fusedSmoothWideRow
            (
                row, beginRow, split, endRow, cols, coeffs, diag,
                sourcePtr, psiPtr, matrixPsiPtr
            ); break;
        }
    }

    if (timing && timing->enabled)
    {
        LevelSolveTiming& value = timing->levels.at(level);
        add(value.fusedSmoothSpmv, elapsed(begin));
        if (preSmoothing) value.preSweeps += sweeps;
        else value.postSweeps += sweeps;
    }
}

void Matrix::smoothAndMultiplyDeltaCurrent
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    Foam::scalarField& matrixPsi,
    const Foam::label sweeps,
    const bool preSmoothing,
    TimingStats* const timing,
    const std::size_t level
) const
{
    if (psi.size() != diag_.size() || source.size() != diag_.size())
        throw std::runtime_error("fw delta-fused smoother field size mismatch");
    if (sweeps < 1)
        throw std::runtime_error("fw delta-fused smoother requires at least one sweep");
    if
    (
        directEdgeOwners_.size() != static_cast<std::size_t>(nFaces_)
     || directEdgeColumns_.size() != static_cast<std::size_t>(nFaces_)
     || directEdgeCoeffs_.size() != static_cast<std::size_t>(nFaces_)
    ) throw std::runtime_error("fw delta fusion requires direct edge metadata");

    const auto begin = std::chrono::steady_clock::now();
    for (Foam::label sweep=1; sweep<sweeps; ++sweep)
        for (Foam::label row=0; row<nCells(); ++row)
        {
            Foam::scalar value = source[row];
            for (Foam::label p=rowStarts_[row]; p<rowStarts_[row + 1]; ++p)
                value -= coeffs_[p]*psi[cols_[p]];
            psi[row] = value/diag_[row];
        }

    // Reuse the output vector as delta storage for the final sweep.  This is
    // the original implementation retained as the experimental baseline.
    // When owner rows are consumed below, every column is larger than its
    // owner, so its delta has not yet been overwritten by the A*psi result.
    matrixPsi.resize(psi.size());
    for (Foam::label row=0; row<nCells(); ++row)
    {
        const Foam::scalar oldValue = psi[row];
        Foam::scalar value = source[row];
        for (Foam::label p=rowStarts_[row]; p<rowStarts_[row + 1]; ++p)
            value -= coeffs_[p]*psi[cols_[p]];
        psi[row] = value/diag_[row];
        matrixPsi[row] = psi[row] - oldValue;
    }

    Foam::label edge = 0;
    for (Foam::label row=0; row<nCells(); ++row)
    {
        Foam::scalar value = source[row];
        while
        (
            edge < nFaces_
         && directEdgeOwners_[edge] == row
        )
        {
            const Foam::label column = directEdgeColumns_[edge];
            value += directEdgeCoeffs_[edge]*matrixPsi[column];
            ++edge;
        }
        matrixPsi[row] = value;
    }
    if (edge != nFaces_)
        throw std::runtime_error("fw delta fusion edge view is not owner ordered");

    if (timing && timing->enabled)
    {
        LevelSolveTiming& value = timing->levels.at(level);
        add(value.fusedSmoothSpmv, elapsed(begin));
        if (preSmoothing) value.preSweeps += sweeps;
        else value.postSweeps += sweeps;
    }
}

void Matrix::deltaCorrectionCurrent
(
    const Foam::scalarField& source,
    Foam::scalarField& matrixPsi
) const
{
    if (source.size() != diag_.size() || matrixPsi.size() != diag_.size())
        throw std::runtime_error("fw current delta correction field size mismatch");
    Foam::label edge = 0;
    for (Foam::label row=0; row<nCells(); ++row)
    {
        Foam::scalar value = source[row];
        while (edge < nFaces_ && directEdgeOwners_[edge] == row)
        {
            const Foam::label column = directEdgeColumns_[edge];
            value += directEdgeCoeffs_[edge]*matrixPsi[column];
            ++edge;
        }
        matrixPsi[row] = value;
    }
}

void Matrix::deltaCorrectionEdgeRows
(
    const Foam::scalarField& source,
    Foam::scalarField& matrixPsi
) const
{
    if (source.size() != diag_.size() || matrixPsi.size() != diag_.size())
        throw std::runtime_error("fw edge-row delta correction field size mismatch");
    if (directEdgeRowStarts_.size() != diag_.size() + 1)
        throw std::runtime_error("fw direct edge row starts were not constructed");

    const Foam::label* const edgeStarts = directEdgeRowStarts_.data();
    const Foam::label* const columns = directEdgeColumns_.data();
    const Foam::scalar* const coefficients = directEdgeCoeffs_.data();
    const Foam::scalar* const sourcePtr = source.data();
    Foam::scalar* const result = matrixPsi.data();
    for (Foam::label row=0; row<nCells(); ++row)
    {
        Foam::scalar value = sourcePtr[row];
        for (Foam::label edge=edgeStarts[row]; edge<edgeStarts[row + 1]; ++edge)
            value += coefficients[edge]*result[columns[edge]];
        result[row] = value;
    }
}

void Matrix::deltaCorrectionPacked
(
    const Foam::scalarField& source,
    Foam::scalarField& matrixPsi
) const
{
    if (source.size() != diag_.size() || matrixPsi.size() != diag_.size())
        throw std::runtime_error("fw packed delta correction field size mismatch");
    if (outgoingStarts_.size() != diag_.size())
        throw std::runtime_error("fw outgoing row starts were not constructed");

    const Foam::label* const rowStarts = rowStarts_.data();
    const Foam::label* const outgoingStarts = outgoingStarts_.data();
    const Foam::label* const columns = cols_.data();
    const Foam::scalar* const coefficients = coeffs_.data();
    const Foam::scalar* const sourcePtr = source.data();
    Foam::scalar* const result = matrixPsi.data();
    for (Foam::label row=0; row<nCells(); ++row)
    {
        Foam::scalar value = sourcePtr[row];
        for
        (
            Foam::label p=outgoingStarts[row];
            p<rowStarts[row + 1];
            ++p
        )
            value += coefficients[p]*result[columns[p]];
        result[row] = value;
    }
}

void Matrix::smoothAndMultiplyDeltaEdgeRows
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    Foam::scalarField& matrixPsi,
    const Foam::label sweeps,
    const bool preSmoothing,
    TimingStats* const timing,
    const std::size_t level
) const
{
    if
    (
        psi.size() != diag_.size()
     || source.size() != diag_.size()
    ) throw std::runtime_error("fw edge-row fused smoother field size mismatch");
    if (sweeps < 1)
        throw std::runtime_error("fw edge-row fused smoother requires at least one sweep");
    if (directEdgeRowStarts_.size() != diag_.size() + 1)
        throw std::runtime_error("fw direct edge row starts were not constructed");

    const auto begin = std::chrono::steady_clock::now();
    for (Foam::label sweep=1; sweep<sweeps; ++sweep)
        for (Foam::label row=0; row<nCells(); ++row)
        {
            Foam::scalar value = source[row];
            for (Foam::label p=rowStarts_[row]; p<rowStarts_[row + 1]; ++p)
                value -= coeffs_[p]*psi[cols_[p]];
            psi[row] = value/diag_[row];
        }

    matrixPsi.resize(psi.size());
    for (Foam::label row=0; row<nCells(); ++row)
    {
        const Foam::scalar oldValue = psi[row];
        Foam::scalar value = source[row];
        for (Foam::label p=rowStarts_[row]; p<rowStarts_[row + 1]; ++p)
            value -= coeffs_[p]*psi[cols_[p]];
        psi[row] = value/diag_[row];
        matrixPsi[row] = psi[row] - oldValue;
    }
    deltaCorrectionEdgeRows(source, matrixPsi);

    if (timing && timing->enabled)
    {
        LevelSolveTiming& value = timing->levels.at(level);
        add(value.fusedSmoothSpmv, elapsed(begin));
        if (preSmoothing) value.preSweeps += sweeps;
        else value.postSweeps += sweeps;
    }
}

void Matrix::smoothAndMultiplyDeltaPackedBaseline
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    Foam::scalarField& matrixPsi,
    const Foam::label sweeps,
    const bool preSmoothing,
    TimingStats* const timing,
    const std::size_t level
) const
{
    if (psi.size() != diag_.size() || source.size() != diag_.size())
        throw std::runtime_error("fw packed baseline fused smoother field size mismatch");
    if (sweeps < 1)
        throw std::runtime_error("fw packed baseline requires at least one sweep");

    const auto begin = std::chrono::steady_clock::now();
    for (Foam::label sweep=1; sweep<sweeps; ++sweep)
        for (Foam::label row=0; row<nCells(); ++row)
        {
            Foam::scalar value = source[row];
            for (Foam::label p=rowStarts_[row]; p<rowStarts_[row + 1]; ++p)
                value -= coeffs_[p]*psi[cols_[p]];
            psi[row] = value/diag_[row];
        }

    matrixPsi.resize(psi.size());
    for (Foam::label row=0; row<nCells(); ++row)
    {
        const Foam::scalar oldValue = psi[row];
        Foam::scalar value = source[row];
        for (Foam::label p=rowStarts_[row]; p<rowStarts_[row + 1]; ++p)
            value -= coeffs_[p]*psi[cols_[p]];
        psi[row] = value/diag_[row];
        matrixPsi[row] = psi[row] - oldValue;
    }
    deltaCorrectionPacked(source, matrixPsi);

    if (timing && timing->enabled)
    {
        LevelSolveTiming& value = timing->levels.at(level);
        add(value.fusedSmoothSpmv, elapsed(begin));
        if (preSmoothing) value.preSweeps += sweeps;
        else value.postSweeps += sweeps;
    }
}

void Matrix::smoothAndMultiplyDelta
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    Foam::scalarField& matrixPsi,
    const Foam::label sweeps,
    const bool preSmoothing,
    TimingStats* const timing,
    const std::size_t level
) const
{
    if
    (
        psi.size() != diag_.size()
     || source.size() != diag_.size()
     || matrixPsi.size() != psi.size()
    ) throw std::runtime_error("fw cleaned delta fused smoother field size mismatch");
    if (sweeps < 1)
        throw std::runtime_error("fw cleaned delta fused smoother requires at least one sweep");
    if (directEdgeRowStarts_.size() != diag_.size() + 1)
        throw std::runtime_error("fw direct edge row starts were not constructed");

    const auto begin = std::chrono::steady_clock::now();
    for (Foam::label sweep=1; sweep<sweeps; ++sweep)
        for (Foam::label row=0; row<nCells(); ++row)
        {
            Foam::scalar value = source[row];
            for (Foam::label p=rowStarts_[row]; p<rowStarts_[row + 1]; ++p)
                value -= coeffs_[p]*psi[cols_[p]];
            psi[row] = value/diag_[row];
        }

    for (Foam::label row=0; row<nCells(); ++row)
    {
        const Foam::scalar oldPsi = psi[row];
        Foam::scalar value = source[row];
        for (Foam::label p=rowStarts_[row]; p<rowStarts_[row + 1]; ++p)
            value -= coeffs_[p]*psi[cols_[p]];
        const Foam::scalar newPsi = value/diag_[row];
        psi[row] = newPsi;
        matrixPsi[row] = newPsi - oldPsi;
    }
    deltaCorrectionEdgeRows(source, matrixPsi);

    if (timing && timing->enabled)
    {
        LevelSolveTiming& value = timing->levels.at(level);
        add(value.fusedSmoothSpmv, elapsed(begin));
        if (preSmoothing) value.preSweeps += sweeps;
        else value.postSweeps += sweeps;
    }
}

void Matrix::scaleCorrection
(
    Foam::scalarField& field,
    const Foam::scalarField& source,
    Foam::scalarField& matrixField,
    TimingStats* const timing,
    const std::size_t level
) const
{
    const auto begin = std::chrono::steady_clock::now();
    multiply(matrixField, field, timing, level);
    Foam::scalar numerator = 0;
    Foam::scalar denominator = 0;
    for (Foam::label row=0; row<nCells(); ++row)
    {
        numerator += field[row]*source[row];
        denominator += field[row]*matrixField[row];
    }
    constexpr Foam::scalar small = 1e-300;
    const Foam::scalar stableDenominator = std::abs(denominator) < small
        ? std::copysign(small, denominator == 0 ? 1.0 : denominator)
        : denominator;
    const Foam::scalar factor = numerator/stableDenominator;
    for (Foam::label row=0; row<nCells(); ++row)
        field[row] = factor*field[row]
            + (source[row] - factor*matrixField[row])/diag_[row];
    if (timing && timing->enabled)
        add(timing->levels.at(level).scaleCorrection, elapsed(begin));
}

void Matrix::scaleCorrectionFlatEdgeWise
(
    Foam::scalarField& field,
    const Foam::scalarField& source,
    Foam::scalarField& matrixField,
    TimingStats* const timing,
    const std::size_t level
) const
{
    const auto begin = std::chrono::steady_clock::now();
    multiplyFlatEdgeWise(matrixField, field, timing, level);
    Foam::scalar numerator = 0;
    Foam::scalar denominator = 0;
    for (Foam::label row=0; row<nCells(); ++row)
    {
        numerator += field[row]*source[row];
        denominator += field[row]*matrixField[row];
    }
    constexpr Foam::scalar small = 1e-300;
    const Foam::scalar stableDenominator = std::abs(denominator) < small
        ? std::copysign(small, denominator == 0 ? 1.0 : denominator)
        : denominator;
    const Foam::scalar factor = numerator/stableDenominator;
    for (Foam::label row=0; row<nCells(); ++row)
        field[row] = factor*field[row]
            + (source[row] - factor*matrixField[row])/diag_[row];
    if (timing && timing->enabled)
        add(timing->levels.at(level).scaleCorrection, elapsed(begin));
}

void Matrix::scaleCorrectionDirectFlatEdgeWise
(
    Foam::scalarField& field,
    const Foam::scalarField& source,
    Foam::scalarField& matrixField,
    TimingStats* const timing,
    const std::size_t level
) const
{
    const auto begin = std::chrono::steady_clock::now();
    multiplyDirectFlatEdgeWise(matrixField, field, timing, level);
    Foam::scalar numerator = 0;
    Foam::scalar denominator = 0;
    for (Foam::label row=0; row<nCells(); ++row)
    {
        numerator += field[row]*source[row];
        denominator += field[row]*matrixField[row];
    }
    constexpr Foam::scalar small = 1e-300;
    const Foam::scalar stableDenominator = std::abs(denominator) < small
        ? std::copysign(small, denominator == 0 ? 1.0 : denominator)
        : denominator;
    const Foam::scalar factor = numerator/stableDenominator;
    for (Foam::label row=0; row<nCells(); ++row)
        field[row] = factor*field[row]
            + (source[row] - factor*matrixField[row])/diag_[row];
    if (timing && timing->enabled)
        add(timing->levels.at(level).scaleCorrection, elapsed(begin));
}

void Matrix::scaleCorrectionLduOrderDirectFlatEdgeWise
(
    Foam::scalarField& field,
    const Foam::scalarField& source,
    Foam::scalarField& matrixField,
    TimingStats* const timing,
    const std::size_t level
) const
{
    const auto begin = std::chrono::steady_clock::now();
    multiplyLduOrderDirectFlatEdgeWise(matrixField, field, timing, level);
    Foam::scalar numerator = 0;
    Foam::scalar denominator = 0;
    for (Foam::label row=0; row<nCells(); ++row)
    {
        numerator += field[row]*source[row];
        denominator += field[row]*matrixField[row];
    }
    constexpr Foam::scalar small = 1e-300;
    const Foam::scalar stableDenominator = std::abs(denominator) < small
        ? std::copysign(small, denominator == 0 ? 1.0 : denominator)
        : denominator;
    const Foam::scalar factor = numerator/stableDenominator;
    for (Foam::label row=0; row<nCells(); ++row)
        field[row] = factor*field[row]
            + (source[row] - factor*matrixField[row])/diag_[row];
    if (timing && timing->enabled)
        add(timing->levels.at(level).scaleCorrection, elapsed(begin));
}

void Matrix::solveDense
(
    Foam::scalarField& correction,
    const Foam::scalarField& source,
    TimingStats* const timing,
    const std::size_t level
) const
{
    const auto begin = std::chrono::steady_clock::now();
    const Foam::label size = nCells();
    std::vector<Foam::scalar> dense(static_cast<std::size_t>(size)*size, 0.0);
    for (Foam::label row=0; row<size; ++row)
    {
        dense[static_cast<std::size_t>(row)*size + row] = diag_[row];
        for (Foam::label p=rowStarts_[row]; p<rowStarts_[row + 1]; ++p)
            dense[static_cast<std::size_t>(row)*size + cols_[p]] = coeffs_[p];
    }
    correction = source;
    for (Foam::label pivot=0; pivot<size; ++pivot)
    {
        Foam::label best = pivot;
        for (Foam::label row=pivot + 1; row<size; ++row)
            if
            (
                std::abs(dense[static_cast<std::size_t>(row)*size + pivot])
              > std::abs(dense[static_cast<std::size_t>(best)*size + pivot])
            ) best = row;
        if
        (
            std::abs(dense[static_cast<std::size_t>(best)*size + pivot])
          <= std::numeric_limits<Foam::scalar>::min()
        ) throw std::runtime_error("singular fwGAMG coarsest matrix");
        if (best != pivot)
        {
            for (Foam::label column=pivot; column<size; ++column)
                std::swap
                (
                    dense[static_cast<std::size_t>(pivot)*size + column],
                    dense[static_cast<std::size_t>(best)*size + column]
                );
            std::swap(correction[pivot], correction[best]);
        }
        for (Foam::label row=pivot + 1; row<size; ++row)
        {
            const Foam::scalar factor =
                dense[static_cast<std::size_t>(row)*size + pivot]
               /dense[static_cast<std::size_t>(pivot)*size + pivot];
            dense[static_cast<std::size_t>(row)*size + pivot] = 0;
            for (Foam::label column=pivot + 1; column<size; ++column)
                dense[static_cast<std::size_t>(row)*size + column] -=
                    factor*dense[static_cast<std::size_t>(pivot)*size + column];
            correction[row] -= factor*correction[pivot];
        }
    }
    for (Foam::label row=size; row-- > 0;)
    {
        Foam::scalar value = correction[row];
        for (Foam::label column=row + 1; column<size; ++column)
            value -= dense[static_cast<std::size_t>(row)*size + column]
                *correction[column];
        correction[row] = value/dense[static_cast<std::size_t>(row)*size + row];
    }
    if (timing && timing->enabled)
        add(timing->levels.at(level).coarseSolve, elapsed(begin));
}

} // namespace smootherTest::fwgamg
