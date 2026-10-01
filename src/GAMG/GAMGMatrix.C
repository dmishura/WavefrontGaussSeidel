/*---------------------------------------------------------------------------*\
  Copyright (C) 2011-2017 OpenFOAM Foundation
  Copyright (C) 2023-2024 OpenCFD Ltd.

  Standalone symmetric LDU storage adapted from OpenFOAM v2606 GAMG.

  OpenFOAM is distributed under the GNU General Public License, version 3
  or later. This adapted file is distributed under the same terms.
\*---------------------------------------------------------------------------*/

#include "GAMGMatrix.H"

#include "GAMGAgglomeration.H"
#include "GAMGTiming.H"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace smootherTest::gamg
{

GAMGMatrix::GAMGMatrix
(
    const Foam::label nCells,
    Foam::labelField lowerAddr,
    Foam::labelField upperAddr,
    Foam::scalarField diag,
    Foam::scalarField upper
)
:
    nCells_(nCells),
    lowerAddr_(std::move(lowerAddr)),
    upperAddr_(std::move(upperAddr)),
    ownerStartAddr_(static_cast<std::size_t>(nCells) + 1, 0),
    diag_(std::move(diag)),
    upper_(std::move(upper))
{
    if (nCells_ < 1 || diag_.size() != static_cast<std::size_t>(nCells_))
        throw std::runtime_error("GAMG matrix has invalid cell/diagonal size");
    if (lowerAddr_.size() != upperAddr_.size() || upper_.size() != upperAddr_.size())
        throw std::runtime_error("GAMG matrix addressing/coefficient size mismatch");

    Foam::label previousOwner = -1;
    for (Foam::label face=0; face<nFaces(); ++face)
    {
        const Foam::label owner = lowerAddr_[face];
        const Foam::label neighbour = upperAddr_[face];
        if (owner < 0 || neighbour <= owner || neighbour >= nCells_)
            throw std::runtime_error("GAMG matrix is not upper-triangular LDU");
        if (owner < previousOwner)
            throw std::runtime_error("GAMG matrix faces are not owner-grouped");
        previousOwner = owner;
        ++ownerStartAddr_[owner + 1];
    }
    for (Foam::label cell=0; cell<nCells_; ++cell)
        ownerStartAddr_[cell + 1] += ownerStartAddr_[cell];
}

GAMGMatrix GAMGMatrix::fromLduMatrix(const Foam::lduMatrix& matrix)
{
    const Foam::label nCells = static_cast<Foam::label>(matrix.diag().size());
    const Foam::labelField& starts = matrix.lduAddr().ownerStartAddr();
    const Foam::labelField& upperAddr = matrix.lduAddr().upperAddr();
    Foam::labelField lowerAddr(upperAddr.size());
    for (Foam::label cell=0; cell<nCells; ++cell)
        for (Foam::label face=starts[cell]; face<starts[cell + 1]; ++face)
            lowerAddr[face] = cell;

    if (matrix.lower().size() != matrix.upper().size())
        throw std::runtime_error("standalone GAMG supports symmetric LDU matrices only");
    for (Foam::label face=0; face<static_cast<Foam::label>(matrix.upper().size()); ++face)
        if (matrix.lower()[face] != matrix.upper()[face])
            throw std::runtime_error("standalone GAMG requires identical lower/upper coefficients");

    return GAMGMatrix
    (
        nCells,
        std::move(lowerAddr),
        Foam::labelField(upperAddr),
        Foam::scalarField(matrix.diag()),
        Foam::scalarField(matrix.upper())
    );
}

GAMGMatrix agglomerateMatrix
(
    const GAMGMatrix& fine,
    const GAMGAgglomerationLevel& addressing
)
{
    Foam::scalarField coarseDiag(addressing.nCoarseCells, 0.0);
    Foam::scalarField coarseUpper(addressing.coarseUpper.size(), 0.0);

    for (Foam::label cell=0; cell<fine.nCells(); ++cell)
        coarseDiag[addressing.fineToCoarse[cell]] += fine.diag()[cell];

    for (Foam::label face=0; face<fine.nFaces(); ++face)
    {
        const Foam::label coarseFace = addressing.faceToCoarse[face];
        if (coarseFace >= 0)
            coarseUpper[coarseFace] += fine.upper()[face];
        else
            coarseDiag[-1 - coarseFace] += 2.0*fine.upper()[face];
    }

    return GAMGMatrix
    (
        addressing.nCoarseCells,
        Foam::labelField(addressing.coarseLower),
        Foam::labelField(addressing.coarseUpper),
        std::move(coarseDiag),
        std::move(coarseUpper)
    );
}

GAMGMatrixHierarchy::GAMGMatrixHierarchy
(
    const GAMGMatrix& finest,
    const std::vector<GAMGAgglomerationLevel>& agglomeration,
    GAMGTimingStats* const timing
)
{
    levels_.reserve(agglomeration.size() + 1);
    levels_.push_back(finest);
    for (std::size_t level=0; level<agglomeration.size(); ++level)
    {
        const GAMGMatrix& fine = levels_.back();
        const auto begin = timing && timing->enabled()
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        GAMGMatrix coarse = agglomerateMatrix(fine, agglomeration[level]);
        if (timing && timing->enabled())
            timing->recordCoarseMatrixBuild
            (
                level,
                fine.nCells(),
                fine.nFaces(),
                coarse.nCells(),
                coarse.nFaces(),
                std::chrono::duration<double>
                (
                    std::chrono::steady_clock::now() - begin
                ).count()
            );
        levels_.push_back(std::move(coarse));
    }
}

} // namespace smootherTest::gamg
