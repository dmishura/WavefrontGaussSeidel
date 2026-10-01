/*---------------------------------------------------------------------------*\
  Copyright (C) 2011-2017 OpenFOAM Foundation
  Copyright (C) 2023-2024 OpenCFD Ltd.

  Standalone symmetric LDU storage for the OpenFOAM v2606 GAMG adaptation.

  OpenFOAM is distributed under the GNU General Public License, version 3
  or later. This adapted file is distributed under the same terms.
\*---------------------------------------------------------------------------*/

#ifndef SmootherTest_GAMGMatrix_H
#define SmootherTest_GAMGMatrix_H

#include "lduMatrix.H"

#include <cstddef>
#include <vector>

namespace smootherTest::gamg
{

struct GAMGAgglomerationLevel;
class GAMGTimingStats;

class GAMGMatrix
{
    Foam::label nCells_ = 0;
    Foam::labelField lowerAddr_;
    Foam::labelField upperAddr_;
    Foam::labelField ownerStartAddr_;
    Foam::scalarField diag_;
    Foam::scalarField upper_;

public:
    GAMGMatrix() = default;

    GAMGMatrix
    (
        Foam::label nCells,
        Foam::labelField lowerAddr,
        Foam::labelField upperAddr,
        Foam::scalarField diag,
        Foam::scalarField upper
    );

    static GAMGMatrix fromLduMatrix(const Foam::lduMatrix& matrix);

    Foam::label nCells() const noexcept { return nCells_; }
    Foam::label nFaces() const noexcept
    {
        return static_cast<Foam::label>(upperAddr_.size());
    }

    const Foam::labelField& lowerAddr() const noexcept { return lowerAddr_; }
    const Foam::labelField& upperAddr() const noexcept { return upperAddr_; }
    const Foam::labelField& ownerStartAddr() const noexcept
    {
        return ownerStartAddr_;
    }
    const Foam::scalarField& diag() const noexcept { return diag_; }
    const Foam::scalarField& upper() const noexcept { return upper_; }
};

class GAMGMatrixHierarchy
{
    std::vector<GAMGMatrix> levels_;

public:
    GAMGMatrixHierarchy
    (
        const GAMGMatrix& finest,
        const std::vector<GAMGAgglomerationLevel>& agglomeration,
        GAMGTimingStats* timing = nullptr
    );

    std::size_t size() const noexcept { return levels_.size(); }
    const GAMGMatrix& operator[](std::size_t level) const
    {
        return levels_.at(level);
    }
    const std::vector<GAMGMatrix>& levels() const noexcept { return levels_; }
};

GAMGMatrix agglomerateMatrix
(
    const GAMGMatrix& fine,
    const GAMGAgglomerationLevel& addressing
);

} // namespace smootherTest::gamg

#endif
