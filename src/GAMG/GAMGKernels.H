/*---------------------------------------------------------------------------*\
  Copyright (C) 2011-2017 OpenFOAM Foundation
  Copyright (C) 2017-2025 OpenCFD Ltd.

  Serial GAMG kernels adapted from OpenFOAM v2606.

  OpenFOAM is distributed under the GNU General Public License, version 3
  or later. This adapted file is distributed under the same terms.
\*---------------------------------------------------------------------------*/

#ifndef SmootherTest_GAMGKernels_H
#define SmootherTest_GAMGKernels_H

#include "GAMGAgglomeration.H"

namespace smootherTest::gamg
{

void multiply
(
    Foam::scalarField& result,
    const GAMGMatrix& matrix,
    const Foam::scalarField& field
);

void residual
(
    Foam::scalarField& result,
    const GAMGMatrix& matrix,
    const Foam::scalarField& field,
    const Foam::scalarField& source
);

void restrictField
(
    Foam::scalarField& coarse,
    const Foam::scalarField& fine,
    const GAMGAgglomerationLevel& addressing
);

void prolongField
(
    Foam::scalarField& fine,
    const Foam::scalarField& coarse,
    const GAMGAgglomerationLevel& addressing
);

void gaussSeidelSmooth
(
    Foam::scalarField& psi,
    const GAMGMatrix& matrix,
    const Foam::scalarField& source,
    Foam::label nSweeps
);

void scaleCorrection
(
    Foam::scalarField& field,
    const GAMGMatrix& matrix,
    const Foam::scalarField& source,
    Foam::scalarField& matrixField
);

void solveCoarsest
(
    Foam::scalarField& correction,
    const GAMGMatrix& matrix,
    const Foam::scalarField& source
);

Foam::scalar l1Norm(const Foam::scalarField& field);

} // namespace smootherTest::gamg

#endif
