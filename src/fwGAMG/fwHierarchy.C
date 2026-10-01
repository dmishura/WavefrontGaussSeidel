#include "fwHierarchy.H"

#include "GAMGMatrix.H"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>

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

gamg::GAMGMatrix makeFineBridge
(
    const Foam::lduMatrix& matrix,
    TimingStats* const timing
)
{
    const auto begin = std::chrono::steady_clock::now();
    gamg::GAMGMatrix result = gamg::GAMGMatrix::fromLduMatrix(matrix);
    if (timing && timing->enabled) timing->lduBridgeSeconds += elapsed(begin);
    return result;
}

gamg::GAMGAgglomeration makeAgglomeration
(
    const gamg::GAMGMatrix& fine,
    const gamg::GAMGControls& controls,
    TimingStats* const timing
)
{
    const auto begin = std::chrono::steady_clock::now();
    gamg::GAMGAgglomeration result
    (
        fine,
        controls,
        nullptr,
        timing ? &timing->coarseLduConstructionSeconds : nullptr
    );
    if (timing && timing->enabled) timing->agglomerationSeconds += elapsed(begin);
    return result;
}

struct LevelNumbering
{
    Foam::labelField nativeToAggCell;
    Foam::labelField aggToNativeCell;
    Foam::labelField nativeFaceToAggFace;
};

struct RenumberedLevel
{
    NativeMatrixData matrix;
    LevelNumbering numbering;
};

LevelNumbering identityNumbering
(
    const Foam::label nCells,
    const Foam::label nFaces
)
{
    LevelNumbering result;
    result.nativeToAggCell.resize(nCells);
    result.aggToNativeCell.resize(nCells);
    result.nativeFaceToAggFace.resize(nFaces);
    std::iota(result.nativeToAggCell.begin(), result.nativeToAggCell.end(), 0);
    std::iota(result.aggToNativeCell.begin(), result.aggToNativeCell.end(), 0);
    std::iota
    (
        result.nativeFaceToAggFace.begin(),
        result.nativeFaceToAggFace.end(),
        0
    );
    return result;
}

NativeMatrixData buildCoarseOperator
(
    const Matrix& fine,
    const LevelNumbering& fineNumbering,
    const gamg::GAMGAgglomerationLevel& addressing
)
{
    NativeMatrixData coarse;
    coarse.nCells = addressing.nCoarseCells;
    coarse.lowerAddr = addressing.coarseLower;
    coarse.upperAddr = addressing.coarseUpper;
    coarse.diag.assign(coarse.nCells, 0.0);
    coarse.upper.assign(coarse.upperAddr.size(), 0.0);

    // Preserve the stock diagonal restriction order (original fine-cell order).
    for (Foam::label aggCell=0; aggCell<addressing.nFineCells; ++aggCell)
    {
        const Foam::label nativeCell = fineNumbering.aggToNativeCell[aggCell];
        const Foam::label fineRow = fine.oldToNew()[nativeCell];
        coarse.diag[addressing.fineToCoarse[aggCell]] += fine.diag()[fineRow];
    }

    // Each symmetric face occurs twice in the packed rows. Recover its value
    // once without constructing an intermediate coarse LDU matrix.
    Foam::scalarField faceCoeffs(fine.nFaces(), 0.0);
    Foam::labelField faceSeen(fine.nFaces(), 0);
    for (Foam::label row=0; row<fine.nCells(); ++row)
        for
        (
            Foam::label p=fine.rowStarts()[row];
            p<fine.rowStarts()[row + 1];
            ++p
        )
        {
            const Foam::label nativeFace = fine.faceIds()[p];
            const Foam::label aggFace =
                fineNumbering.nativeFaceToAggFace[nativeFace];
            if (!faceSeen[aggFace])
            {
                faceCoeffs[aggFace] = fine.coeffs()[p];
                faceSeen[aggFace] = 1;
            }
        }
    for (Foam::label face=0; face<fine.nFaces(); ++face)
    {
        if (!faceSeen[face])
            throw std::runtime_error("fw coarse operator lost a fine face");
        const Foam::label coarseFace = addressing.faceToCoarse[face];
        if (coarseFace >= 0)
            coarse.upper[coarseFace] += faceCoeffs[face];
        else
            coarse.diag[-1 - coarseFace] += 2.0*faceCoeffs[face];
    }
    return coarse;
}

RenumberedLevel renumberCoarse
(
    NativeMatrixData coarse,
    const LevelNumbering& fineNumbering,
    const gamg::GAMGAgglomerationLevel& addressing,
    const CoarseRenumbering mode
)
{
    if (mode == CoarseRenumbering::None)
        return
        {
            std::move(coarse),
            identityNumbering
            (
                addressing.nCoarseCells,
                static_cast<Foam::label>(addressing.coarseUpper.size())
            )
        };

    Foam::labelField keys
    (
        addressing.nCoarseCells,
        std::numeric_limits<Foam::label>::max()
    );
    for
    (
        Foam::label nativeFine=0;
        nativeFine<addressing.nFineCells;
        ++nativeFine
    )
    {
        const Foam::label aggFine = fineNumbering.nativeToAggCell[nativeFine];
        const Foam::label aggCoarse = addressing.fineToCoarse[aggFine];
        keys[aggCoarse] = std::min(keys[aggCoarse], nativeFine);
    }

    Foam::labelField newToOld(addressing.nCoarseCells);
    std::iota(newToOld.begin(), newToOld.end(), 0);
    std::stable_sort
    (
        newToOld.begin(), newToOld.end(),
        [&keys](const Foam::label left, const Foam::label right)
        {
            return std::tie(keys[left], left) < std::tie(keys[right], right);
        }
    );
    Foam::labelField oldToNew(addressing.nCoarseCells);
    for (Foam::label cell=0; cell<addressing.nCoarseCells; ++cell)
        oldToNew[newToOld[cell]] = cell;

    struct Face
    {
        Foam::label owner;
        Foam::label neighbour;
        Foam::label oldFace;
        Foam::scalar coefficient;
    };
    std::vector<Face> faces;
    faces.reserve(coarse.upper.size());
    for
    (
        Foam::label oldFace=0;
        oldFace<static_cast<Foam::label>(coarse.upper.size());
        ++oldFace
    )
    {
        const Foam::label left = oldToNew[coarse.lowerAddr[oldFace]];
        const Foam::label right = oldToNew[coarse.upperAddr[oldFace]];
        faces.push_back
        ({
            std::min(left, right),
            std::max(left, right),
            oldFace,
            coarse.upper[oldFace]
        });
    }
    std::stable_sort
    (
        faces.begin(), faces.end(),
        [](const Face& left, const Face& right)
        {
            return std::tie(left.owner, left.neighbour, left.oldFace)
                 < std::tie(right.owner, right.neighbour, right.oldFace);
        }
    );

    RenumberedLevel result;
    result.matrix.nCells = coarse.nCells;
    result.matrix.diag.resize(coarse.nCells);
    result.matrix.lowerAddr.resize(faces.size());
    result.matrix.upperAddr.resize(faces.size());
    result.matrix.upper.resize(faces.size());
    result.numbering.nativeToAggCell = newToOld;
    result.numbering.aggToNativeCell = oldToNew;
    result.numbering.nativeFaceToAggFace.resize(faces.size());
    for (Foam::label cell=0; cell<coarse.nCells; ++cell)
        result.matrix.diag[cell] = coarse.diag[newToOld[cell]];
    for (Foam::label face=0; face<static_cast<Foam::label>(faces.size()); ++face)
    {
        result.matrix.lowerAddr[face] = faces[face].owner;
        result.matrix.upperAddr[face] = faces[face].neighbour;
        result.matrix.upper[face] = faces[face].coefficient;
        result.numbering.nativeFaceToAggFace[face] = faces[face].oldFace;
    }
    return result;
}

Matrix buildMatrix
(
    const NativeMatrixData& native,
    const HierarchyOptions& options,
    LevelSetupTiming* const timing
)
{
    return options.rowOrdering == RowOrdering::Natural
        ? Matrix::buildNatural
          (
              native, timing, options.buildFlatEdges,
              options.buildDirectFlatEdges,
              options.buildNativeFaceOrderDirectEdges
          )
        : Matrix::build
          (
              native, options.width, timing, options.buildFlatEdges,
              options.buildDirectFlatEdges,
              options.buildNativeFaceOrderDirectEdges
          );
}

} // namespace

Hierarchy::Hierarchy
(
    const Foam::lduMatrix& matrix,
    const gamg::GAMGControls& controls,
    const HierarchyOptions options,
    TimingStats* const timing
)
:
    agglomeration_
    (
        [&]()
        {
            const gamg::GAMGMatrix bridge = makeFineBridge(matrix, timing);
            return makeAgglomeration(bridge, controls, timing);
        }()
    )
{
    const std::size_t count = agglomeration_.size() + 1;
    if (timing) timing->configureLevels(count);
    matrices_.reserve(count);
    restrictionMaps_.reserve(count - 1);
    nativeToAggCells_.reserve(count);
    aggToNativeCells_.reserve(count);

    NativeMatrixData fineNative = nativeMatrixData(matrix);
    LevelNumbering fineNumbering = identityNumbering
    (
        fineNative.nCells,
        static_cast<Foam::label>(fineNative.upper.size())
    );
    nativeToAggCells_.push_back(fineNumbering.nativeToAggCell);
    aggToNativeCells_.push_back(fineNumbering.aggToNativeCell);
    matrices_.push_back(buildMatrix
    (
        fineNative, options, timing ? &timing->levelSetup[0] : nullptr
    ));

    for (std::size_t level=0; level<agglomeration_.size(); ++level)
    {
        const auto coarseBegin = std::chrono::steady_clock::now();
        NativeMatrixData coarse = buildCoarseOperator
        (
            matrices_.back(), fineNumbering, agglomeration_[level]
        );
        if (timing && timing->enabled)
            timing->levelSetup[level + 1].coarseOperatorSeconds +=
                elapsed(coarseBegin);

        const auto renumberBegin = std::chrono::steady_clock::now();
        RenumberedLevel renumbered = renumberCoarse
        (
            std::move(coarse), fineNumbering, agglomeration_[level],
            options.coarseRenumbering
        );
        if
        (
            timing
         && timing->enabled
         && options.coarseRenumbering != CoarseRenumbering::None
        )
            timing->levelSetup[level + 1].localityRenumberSeconds +=
                elapsed(renumberBegin);

        matrices_.push_back(buildMatrix
        (
            renumbered.matrix, options,
            timing ? &timing->levelSetup[level + 1] : nullptr
        ));
        nativeToAggCells_.push_back
        (
            renumbered.numbering.nativeToAggCell
        );
        aggToNativeCells_.push_back
        (
            renumbered.numbering.aggToNativeCell
        );

        const auto mapBegin = std::chrono::steady_clock::now();
        const Matrix& fine = matrices_[level];
        const Matrix& coarseMatrix = matrices_[level + 1];
        Foam::labelField directMap(fine.nCells());
        for (Foam::label fineRow=0; fineRow<fine.nCells(); ++fineRow)
        {
            const Foam::label nativeFine = fine.newToOld()[fineRow];
            const Foam::label aggFine =
                fineNumbering.nativeToAggCell[nativeFine];
            const Foam::label aggCoarse =
                agglomeration_[level].fineToCoarse[aggFine];
            const Foam::label nativeCoarse =
                renumbered.numbering.aggToNativeCell[aggCoarse];
            directMap[fineRow] = coarseMatrix.oldToNew()[nativeCoarse];
        }
        restrictionMaps_.push_back(std::move(directMap));
        if (timing && timing->enabled)
            timing->levelSetup[level + 1].restrictionMapSeconds +=
                elapsed(mapBegin);
        fineNumbering = std::move(renumbered.numbering);
    }

    if (timing)
        for (std::size_t level=0; level<count; ++level)
        {
            timing->levels[level].cells = matrices_[level].nCells();
            timing->levels[level].faces = matrices_[level].nFaces();
        }
}

void Hierarchy::restrictField
(
    Foam::scalarField& coarse,
    const Foam::scalarField& fine,
    const std::size_t fineLevel,
    TimingStats* const timing
) const
{
    const auto begin = std::chrono::steady_clock::now();
    const Foam::labelField& map = restrictionMaps_.at(fineLevel);
    if (fine.size() != map.size())
        throw std::runtime_error("fw restriction field size mismatch");
    coarse.assign(matrices_.at(fineLevel + 1).nCells(), 0.0);
    for (Foam::label row=0; row<static_cast<Foam::label>(fine.size()); ++row)
        coarse[map[row]] += fine[row];
    if (timing && timing->enabled)
        add(timing->transitions.at(fineLevel).restriction, elapsed(begin));
}

void Hierarchy::prolongField
(
    Foam::scalarField& fine,
    const Foam::scalarField& coarse,
    const std::size_t fineLevel,
    TimingStats* const timing
) const
{
    const auto begin = std::chrono::steady_clock::now();
    const Foam::labelField& map = restrictionMaps_.at(fineLevel);
    if (coarse.size() != static_cast<std::size_t>(matrices_.at(fineLevel + 1).nCells()))
        throw std::runtime_error("fw prolongation field size mismatch");
    fine.resize(map.size());
    for (Foam::label row=0; row<static_cast<Foam::label>(fine.size()); ++row)
        fine[row] = coarse[map[row]];
    if (timing && timing->enabled)
        add(timing->transitions.at(fineLevel).prolongation, elapsed(begin));
}

} // namespace smootherTest::fwgamg
