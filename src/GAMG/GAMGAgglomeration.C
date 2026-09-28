/*---------------------------------------------------------------------------*\
  Copyright (C) 2011-2017 OpenFOAM Foundation
  Copyright (C) 2019-2026 OpenCFD Ltd.

  Pairwise agglomeration adapted from OpenFOAM v2606
  pairGAMGAgglomeration and GAMGAgglomerateLduAddressing.

  OpenFOAM is distributed under the GNU General Public License, version 3
  or later. This adapted file is distributed under the same terms.
\*---------------------------------------------------------------------------*/

#include "GAMGAgglomeration.H"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace smootherTest::gamg
{
namespace
{

Foam::labelField pairCells
(
    const GAMGMatrix& fine,
    const Foam::scalarField& faceWeights,
    const bool forward,
    Foam::label& nCoarseCells
)
{
    std::vector<std::vector<Foam::label>> cellFaces(fine.nCells());
    for (Foam::label face=0; face<fine.nFaces(); ++face)
        cellFaces[fine.upperAddr()[face]].push_back(face);
    for (Foam::label face=0; face<fine.nFaces(); ++face)
        cellFaces[fine.lowerAddr()[face]].push_back(face);

    Foam::labelField coarseCellMap(fine.nCells(), -1);
#ifdef _OPENMP
    constexpr Foam::scalar tolerance = 1e-10;
#else
    constexpr Foam::scalar tolerance = 0;
#endif
    nCoarseCells = 0;
    for (Foam::label visit=0; visit<fine.nCells(); ++visit)
    {
        const Foam::label cell = forward ? visit : fine.nCells() - visit - 1;
        if (coarseCellMap[cell] >= 0) continue;

        Foam::label matchFace = -1;
        Foam::scalar maximumWeight = -std::numeric_limits<Foam::scalar>::max();
        for (const Foam::label face : cellFaces[cell])
        {
            const Foam::label owner = fine.lowerAddr()[face];
            const Foam::label neighbour = fine.upperAddr()[face];
            if
            (
                coarseCellMap[owner] < 0
             && coarseCellMap[neighbour] < 0
             && faceWeights[face] > maximumWeight*(1.0 + tolerance)
            )
            {
                matchFace = face;
                maximumWeight = faceWeights[face];
            }
        }

        if (matchFace >= 0)
        {
            coarseCellMap[fine.lowerAddr()[matchFace]] = nCoarseCells;
            coarseCellMap[fine.upperAddr()[matchFace]] = nCoarseCells;
            ++nCoarseCells;
            continue;
        }

        Foam::label clusterFace = -1;
        Foam::scalar clusterWeight = -std::numeric_limits<Foam::scalar>::max();
        for (const Foam::label face : cellFaces[cell])
        {
            if (faceWeights[face] > clusterWeight*(1.0 + tolerance))
            {
                clusterFace = face;
                clusterWeight = faceWeights[face];
            }
        }
        if (clusterFace >= 0)
        {
            coarseCellMap[cell] = std::max
            (
                coarseCellMap[fine.lowerAddr()[clusterFace]],
                coarseCellMap[fine.upperAddr()[clusterFace]]
            );
        }
    }

    for (Foam::label visit=0; visit<fine.nCells(); ++visit)
    {
        const Foam::label cell = forward ? visit : fine.nCells() - visit - 1;
        if (coarseCellMap[cell] < 0) coarseCellMap[cell] = nCoarseCells++;
    }

    if (!forward)
        for (Foam::label cell=0; cell<fine.nCells(); ++cell)
            coarseCellMap[cell] = nCoarseCells - 1 - coarseCellMap[cell];

    return coarseCellMap;
}

GAMGAgglomerationLevel makeCoarseAddressing
(
    const GAMGMatrix& fine,
    Foam::labelField fineToCoarse,
    const Foam::label nCoarseCells
)
{
    GAMGAgglomerationLevel result;
    result.nFineCells = fine.nCells();
    result.nCoarseCells = nCoarseCells;
    result.fineToCoarse = std::move(fineToCoarse);
    result.faceToCoarse.resize(fine.nFaces());
    result.faceFlip.assign(fine.nFaces(), false);

    std::vector<std::vector<Foam::label>> neighbourByOwner(nCoarseCells);
    std::vector<std::vector<Foam::label>> provisionalByOwner(nCoarseCells);
    Foam::label provisionalFaces = 0;
    for (Foam::label face=0; face<fine.nFaces(); ++face)
    {
        const Foam::label mappedUpper = result.fineToCoarse[fine.upperAddr()[face]];
        const Foam::label mappedLower = result.fineToCoarse[fine.lowerAddr()[face]];
        if (mappedUpper == mappedLower)
        {
            result.faceToCoarse[face] = -(mappedUpper + 1);
            continue;
        }

        const Foam::label coarseOwner = std::min(mappedUpper, mappedLower);
        const Foam::label coarseNeighbour = std::max(mappedUpper, mappedLower);
        auto& neighbours = neighbourByOwner[coarseOwner];
        auto found = std::find(neighbours.begin(), neighbours.end(), coarseNeighbour);
        if (found == neighbours.end())
        {
            neighbours.push_back(coarseNeighbour);
            provisionalByOwner[coarseOwner].push_back(provisionalFaces);
            result.faceToCoarse[face] = provisionalFaces++;
        }
        else
        {
            const std::size_t offset = static_cast<std::size_t>(found - neighbours.begin());
            result.faceToCoarse[face] = provisionalByOwner[coarseOwner][offset];
        }
    }

    Foam::labelField provisionalToFinal(provisionalFaces);
    result.coarseLower.resize(provisionalFaces);
    result.coarseUpper.resize(provisionalFaces);
    Foam::label coarseFace = 0;
    for (Foam::label owner=0; owner<nCoarseCells; ++owner)
    {
        for (std::size_t i=0; i<neighbourByOwner[owner].size(); ++i)
        {
            result.coarseLower[coarseFace] = owner;
            result.coarseUpper[coarseFace] = neighbourByOwner[owner][i];
            provisionalToFinal[provisionalByOwner[owner][i]] = coarseFace++;
        }
    }

    for (Foam::label face=0; face<fine.nFaces(); ++face)
    {
        if (result.faceToCoarse[face] < 0) continue;
        result.faceToCoarse[face] = provisionalToFinal[result.faceToCoarse[face]];
        const Foam::label coarseFaceIndex = result.faceToCoarse[face];
        const Foam::label mappedUpper = result.fineToCoarse[fine.upperAddr()[face]];
        const Foam::label mappedLower = result.fineToCoarse[fine.lowerAddr()[face]];
        result.faceFlip[face] =
            result.coarseLower[coarseFaceIndex] == mappedUpper
         && result.coarseUpper[coarseFaceIndex] == mappedLower;
    }
    return result;
}

} // namespace

GAMGAgglomeration::GAMGAgglomeration
(
    const GAMGMatrix& finest,
    const GAMGControls& controls
)
{
    if (controls.maxLevels < 1)
        throw std::runtime_error("GAMG maxLevels must be positive");

    const Foam::label coarsestTarget = std::max
    (
        Foam::label(1),
        std::min(finest.nCells()/2, controls.nCellsInCoarsestLevel)
    );
    GAMGMatrix current = finest;
    Foam::scalarField faceWeights(current.nFaces());
    for (Foam::label face=0; face<current.nFaces(); ++face)
        faceWeights[face] = std::abs(current.upper()[face]);

    bool forward = true;
    for (Foam::label level=0; level<controls.maxLevels - 1; ++level)
    {
        Foam::label nCoarseCells = 0;
        Foam::labelField fineToCoarse = pairCells
        (
            current, faceWeights, forward, nCoarseCells
        );
        forward = !forward;
        if (nCoarseCells < coarsestTarget || nCoarseCells >= current.nCells()) break;

        GAMGAgglomerationLevel addressing = makeCoarseAddressing
        (
            current, std::move(fineToCoarse), nCoarseCells
        );
        Foam::scalarField coarseWeights(addressing.coarseUpper.size(), 0.0);
        for (Foam::label face=0; face<current.nFaces(); ++face)
            if (addressing.faceToCoarse[face] >= 0)
                coarseWeights[addressing.faceToCoarse[face]] += faceWeights[face];

        current = agglomerateMatrix(current, addressing);
        levels_.push_back(std::move(addressing));
        faceWeights = std::move(coarseWeights);
    }

    if (levels_.empty())
        throw std::runtime_error("GAMG pair agglomeration produced no coarse level");
}

} // namespace smootherTest::gamg
