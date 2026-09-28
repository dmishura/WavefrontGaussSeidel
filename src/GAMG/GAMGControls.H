/*---------------------------------------------------------------------------*\
  Copyright (C) 2011-2017 OpenFOAM Foundation
  Copyright (C) 2021-2026 OpenCFD Ltd.

  Standalone adaptation of OpenFOAM v2606 GAMG controls.

  OpenFOAM is distributed under the GNU General Public License, version 3
  or later. This adapted file is distributed under the same terms.
\*---------------------------------------------------------------------------*/

#ifndef SmootherTest_GAMGControls_H
#define SmootherTest_GAMGControls_H

#include "lduMatrix.H"

namespace smootherTest::gamg
{

struct GAMGControls
{
    Foam::label nCellsInCoarsestLevel = 10;
    Foam::label maxLevels = 50;

    Foam::label nPreSweeps = 0;
    Foam::label preSweepsLevelMultiplier = 1;
    Foam::label maxPreSweeps = 4;
    Foam::label nPostSweeps = 2;
    Foam::label postSweepsLevelMultiplier = 1;
    Foam::label maxPostSweeps = 4;
    Foam::label nFinestSweeps = 2;

    bool scaleCorrection = true;
    Foam::scalar tolerance = 1e-8;
    Foam::scalar relativeTolerance = 0.01;
    Foam::label minIterations = 0;
    Foam::label maxIterations = 100;
    bool diagnostics = false;
};

} // namespace smootherTest::gamg

#endif
