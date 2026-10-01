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

#include <cstddef>
#include <cstdint>

namespace smootherTest::gamg
{

class GAMGTimingStats;

inline constexpr std::size_t l1EvictionBufferBytes = 256u*1024u;

// Diagnostic-only cache-state equalizer.  The buffer is owned by the solver
// and allocated during setup, never by this helper or inside a timed SpMV.
// Keep this out-of-line so the compiler cannot merge it into the following
// matrix traversal.
std::uint64_t evictL1
(
    const std::uint8_t* buffer,
    std::size_t size
);

// Diagnostic representation used to isolate the matrix-vector traversal from
// the smoother and hierarchy.  Entries retain the native LDU face order.
struct DirectFlatMatrix
{
    Foam::label nCells = 0;
    Foam::scalarField diag;
    Foam::labelField owners;
    Foam::labelField columns;
    Foam::scalarField coefficients;
};

DirectFlatMatrix makeDirectFlatMatrix(const GAMGMatrix& matrix);

void multiply
(
    Foam::scalarField& result,
    const GAMGMatrix& matrix,
    const Foam::scalarField& field,
    GAMGTimingStats* timing = nullptr,
    std::size_t level = 0
);

void multiplyDirectFlat
(
    Foam::scalarField& result,
    const DirectFlatMatrix& matrix,
    const Foam::scalarField& field,
    GAMGTimingStats* timing = nullptr,
    std::size_t level = 0
);

// Experimental LDU edge traversal that retains the owner-side accumulator
// across the contiguous run of faces belonging to an owner.
void multiplyOwnerAccumulated
(
    Foam::scalarField& result,
    const GAMGMatrix& matrix,
    const Foam::scalarField& field,
    GAMGTimingStats* timing = nullptr,
    std::size_t level = 0
);

void residual
(
    Foam::scalarField& result,
    const GAMGMatrix& matrix,
    const Foam::scalarField& field,
    const Foam::scalarField& source,
    GAMGTimingStats* timing = nullptr,
    std::size_t level = 0
);

void residualDirectFlat
(
    Foam::scalarField& result,
    const DirectFlatMatrix& matrix,
    const Foam::scalarField& field,
    const Foam::scalarField& source,
    GAMGTimingStats* timing = nullptr,
    std::size_t level = 0
);

void restrictField
(
    Foam::scalarField& coarse,
    const Foam::scalarField& fine,
    const GAMGAgglomerationLevel& addressing,
    GAMGTimingStats* timing = nullptr,
    std::size_t fineLevel = 0
);

void prolongField
(
    Foam::scalarField& fine,
    const Foam::scalarField& coarse,
    const GAMGAgglomerationLevel& addressing,
    GAMGTimingStats* timing = nullptr,
    std::size_t fineLevel = 0
);

void gaussSeidelSmooth
(
    Foam::scalarField& psi,
    const GAMGMatrix& matrix,
    const Foam::scalarField& source,
    Foam::label nSweeps,
    GAMGTimingStats* timing = nullptr,
    std::size_t level = 0
);

void scaleCorrection
(
    Foam::scalarField& field,
    const GAMGMatrix& matrix,
    const Foam::scalarField& source,
    Foam::scalarField& matrixField,
    GAMGTimingStats* timing = nullptr,
    std::size_t level = 0
);

void scaleCorrectionDirectFlat
(
    Foam::scalarField& field,
    const DirectFlatMatrix& matrix,
    const Foam::scalarField& source,
    Foam::scalarField& matrixField,
    GAMGTimingStats* timing = nullptr,
    std::size_t level = 0
);

void solveCoarsest
(
    Foam::scalarField& correction,
    const GAMGMatrix& matrix,
    const Foam::scalarField& source,
    GAMGTimingStats* timing = nullptr,
    std::size_t level = 0
);

Foam::scalar l1Norm(const Foam::scalarField& field);

} // namespace smootherTest::gamg

#endif
