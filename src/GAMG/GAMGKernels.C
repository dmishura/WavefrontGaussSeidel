/*---------------------------------------------------------------------------*\
  Copyright (C) 2011-2018 OpenFOAM Foundation
  Copyright (C) 2017-2025 OpenCFD Ltd.

  Serial kernels adapted from OpenFOAM v2606 GAMGSolver, lduMatrix and
  GaussSeidelSmoother implementations.

  OpenFOAM is distributed under the GNU General Public License, version 3
  or later. This adapted file is distributed under the same terms.
\*---------------------------------------------------------------------------*/

#include "GAMGKernels.H"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace smootherTest::gamg
{

void multiply
(
    Foam::scalarField& result,
    const GAMGMatrix& matrix,
    const Foam::scalarField& field
)
{
    if (field.size() != static_cast<std::size_t>(matrix.nCells()))
        throw std::runtime_error("GAMG multiply field size mismatch");
    result.resize(field.size());
    for (Foam::label cell=0; cell<matrix.nCells(); ++cell)
        result[cell] = matrix.diag()[cell]*field[cell];
    for (Foam::label face=0; face<matrix.nFaces(); ++face)
    {
        const Foam::label owner = matrix.lowerAddr()[face];
        const Foam::label neighbour = matrix.upperAddr()[face];
        const Foam::scalar coefficient = matrix.upper()[face];
        result[owner] += coefficient*field[neighbour];
        result[neighbour] += coefficient*field[owner];
    }
}

void residual
(
    Foam::scalarField& result,
    const GAMGMatrix& matrix,
    const Foam::scalarField& field,
    const Foam::scalarField& source
)
{
    multiply(result, matrix, field);
    for (Foam::label cell=0; cell<matrix.nCells(); ++cell)
        result[cell] = source[cell] - result[cell];
}

void restrictField
(
    Foam::scalarField& coarse,
    const Foam::scalarField& fine,
    const GAMGAgglomerationLevel& addressing
)
{
    if (fine.size() != addressing.fineToCoarse.size())
        throw std::runtime_error("GAMG restriction field size mismatch");
    coarse.assign(addressing.nCoarseCells, 0.0);
    for (Foam::label cell=0; cell<addressing.nFineCells; ++cell)
        coarse[addressing.fineToCoarse[cell]] += fine[cell];
}

void prolongField
(
    Foam::scalarField& fine,
    const Foam::scalarField& coarse,
    const GAMGAgglomerationLevel& addressing
)
{
    if (coarse.size() != static_cast<std::size_t>(addressing.nCoarseCells))
        throw std::runtime_error("GAMG prolongation field size mismatch");
    fine.resize(addressing.nFineCells);
    for (Foam::label cell=0; cell<addressing.nFineCells; ++cell)
        fine[cell] = coarse[addressing.fineToCoarse[cell]];
}

void gaussSeidelSmooth
(
    Foam::scalarField& psi,
    const GAMGMatrix& matrix,
    const Foam::scalarField& source,
    const Foam::label nSweeps
)
{
    if
    (
        psi.size() != static_cast<std::size_t>(matrix.nCells())
     || source.size() != psi.size()
    )
        throw std::runtime_error("GAMG smoother field size mismatch");

    Foam::scalarField bPrime(psi.size());
    const Foam::labelField& starts = matrix.ownerStartAddr();
    for (Foam::label sweep=0; sweep<nSweeps; ++sweep)
    {
        bPrime = source;
        for (Foam::label cell=0; cell<matrix.nCells(); ++cell)
        {
            Foam::scalar value = bPrime[cell];
            for (Foam::label face=starts[cell]; face<starts[cell + 1]; ++face)
                value -= matrix.upper()[face]*psi[matrix.upperAddr()[face]];
            value /= matrix.diag()[cell];
            for (Foam::label face=starts[cell]; face<starts[cell + 1]; ++face)
                bPrime[matrix.upperAddr()[face]] -= matrix.upper()[face]*value;
            psi[cell] = value;
        }
    }
}

void scaleCorrection
(
    Foam::scalarField& field,
    const GAMGMatrix& matrix,
    const Foam::scalarField& source,
    Foam::scalarField& matrixField
)
{
    multiply(matrixField, matrix, field);
    Foam::scalar numerator = 0;
    Foam::scalar denominator = 0;
    for (Foam::label cell=0; cell<matrix.nCells(); ++cell)
    {
        numerator += field[cell]*source[cell];
        denominator += field[cell]*matrixField[cell];
    }
    constexpr Foam::scalar small = 1e-300;
    const Foam::scalar stableDenominator =
        std::abs(denominator) < small
      ? std::copysign(small, denominator == 0 ? 1.0 : denominator)
      : denominator;
    const Foam::scalar factor = numerator/stableDenominator;
    for (Foam::label cell=0; cell<matrix.nCells(); ++cell)
        field[cell] = factor*field[cell]
            + (source[cell] - factor*matrixField[cell])/matrix.diag()[cell];
}

void solveCoarsest
(
    Foam::scalarField& correction,
    const GAMGMatrix& matrix,
    const Foam::scalarField& source
)
{
    const Foam::label size = matrix.nCells();
    std::vector<Foam::scalar> dense
    (
        static_cast<std::size_t>(size)*size,
        0.0
    );
    for (Foam::label cell=0; cell<size; ++cell)
        dense[static_cast<std::size_t>(cell)*size + cell] = matrix.diag()[cell];
    for (Foam::label face=0; face<matrix.nFaces(); ++face)
    {
        const Foam::label owner = matrix.lowerAddr()[face];
        const Foam::label neighbour = matrix.upperAddr()[face];
        dense[static_cast<std::size_t>(owner)*size + neighbour] = matrix.upper()[face];
        dense[static_cast<std::size_t>(neighbour)*size + owner] = matrix.upper()[face];
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
        )
            throw std::runtime_error("singular GAMG coarsest matrix");
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
}

Foam::scalar l1Norm(const Foam::scalarField& field)
{
    Foam::scalar result = 0;
    for (const Foam::scalar value : field) result += std::abs(value);
    return result;
}

} // namespace smootherTest::gamg
