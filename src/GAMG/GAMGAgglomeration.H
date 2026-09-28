/*---------------------------------------------------------------------------*\
  Copyright (C) 2011-2017 OpenFOAM Foundation
  Copyright (C) 2019-2026 OpenCFD Ltd.

  Pairwise GAMG agglomeration adapted from OpenFOAM v2606.

  OpenFOAM is distributed under the GNU General Public License, version 3
  or later. This adapted file is distributed under the same terms.
\*---------------------------------------------------------------------------*/

#ifndef SmootherTest_GAMGAgglomeration_H
#define SmootherTest_GAMGAgglomeration_H

#include "GAMGControls.H"
#include "GAMGMatrix.H"

#include <vector>

namespace smootherTest::gamg
{

struct GAMGAgglomerationLevel
{
    Foam::label nFineCells = 0;
    Foam::label nCoarseCells = 0;
    Foam::labelField fineToCoarse;
    Foam::labelField faceToCoarse;
    std::vector<bool> faceFlip;
    Foam::labelField coarseLower;
    Foam::labelField coarseUpper;
};

class GAMGAgglomeration
{
    std::vector<GAMGAgglomerationLevel> levels_;

public:
    GAMGAgglomeration
    (
        const GAMGMatrix& finest,
        const GAMGControls& controls
    );

    std::size_t size() const noexcept { return levels_.size(); }
    const GAMGAgglomerationLevel& operator[](std::size_t level) const
    {
        return levels_.at(level);
    }
    const std::vector<GAMGAgglomerationLevel>& levels() const noexcept
    {
        return levels_;
    }
};

} // namespace smootherTest::gamg

#endif
