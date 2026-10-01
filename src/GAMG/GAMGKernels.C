/*---------------------------------------------------------------------------*\
  Copyright (C) 2011-2018 OpenFOAM Foundation
  Copyright (C) 2017-2025 OpenCFD Ltd.

  Serial kernels adapted from OpenFOAM v2606 GAMGSolver, lduMatrix and
  GaussSeidelSmoother implementations.

  OpenFOAM is distributed under the GNU General Public License, version 3
  or later. This adapted file is distributed under the same terms.
\*---------------------------------------------------------------------------*/

#include "GAMGKernels.H"
#include "GAMGTiming.H"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace smootherTest::gamg
{
#if defined(__GNUC__)
__attribute__((noinline))
#endif
std::uint64_t evictL1
(
    const std::uint8_t* const buffer,
    const std::size_t size
)
{
    std::uint64_t sum = 0;
    for (std::size_t offset=0; offset<size; offset += 64)
        sum += buffer[offset];
    return sum;
}

DirectFlatMatrix makeDirectFlatMatrix(const GAMGMatrix& matrix)
{
    DirectFlatMatrix result;
    result.nCells = matrix.nCells();
    result.diag = matrix.diag();
    result.owners = matrix.lowerAddr();
    result.columns = matrix.upperAddr();
    result.coefficients = matrix.upper();
    return result;
}

void multiply
(
    Foam::scalarField& result,
    const GAMGMatrix& matrix,
    const Foam::scalarField& field,
    GAMGTimingStats* const timing,
    const std::size_t level
)
{
    if (field.size() != static_cast<std::size_t>(matrix.nCells()))
        throw std::runtime_error("GAMG multiply field size mismatch");
    const auto begin = timing && timing->enabled()
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
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
    if (timing && timing->enabled())
        timing->recordAmul
        (
            level,
            std::chrono::duration<double>
            (
                std::chrono::steady_clock::now() - begin
            ).count()
        );
}

void multiplyDirectFlat
(
    Foam::scalarField& result,
    const DirectFlatMatrix& matrix,
    const Foam::scalarField& field,
    GAMGTimingStats* const timing,
    const std::size_t level
)
{
    if (field.size() != static_cast<std::size_t>(matrix.nCells))
        throw std::runtime_error("GAMG direct-flat multiply field size mismatch");
    if
    (
        matrix.diag.size() != field.size()
     || matrix.owners.size() != matrix.columns.size()
     || matrix.owners.size() != matrix.coefficients.size()
    ) throw std::runtime_error("invalid GAMG direct-flat matrix");
    const auto begin = timing && timing->enabled()
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    result.resize(field.size());
    for (Foam::label cell=0; cell<matrix.nCells; ++cell)
        result[cell] = matrix.diag[cell]*field[cell];
    for
    (
        Foam::label face=0;
        face<static_cast<Foam::label>(matrix.coefficients.size());
        ++face
    )
    {
        const Foam::label owner = matrix.owners[face];
        const Foam::label neighbour = matrix.columns[face];
        const Foam::scalar coefficient = matrix.coefficients[face];
        result[owner] += coefficient*field[neighbour];
        result[neighbour] += coefficient*field[owner];
    }
    if (timing && timing->enabled())
        timing->recordAmul
        (
            level,
            std::chrono::duration<double>
            (
                std::chrono::steady_clock::now() - begin
            ).count()
        );
}

void multiplyOwnerAccumulated
(
    Foam::scalarField& result,
    const GAMGMatrix& matrix,
    const Foam::scalarField& field,
    GAMGTimingStats* const timing,
    const std::size_t level
)
{
    if (field.size() != static_cast<std::size_t>(matrix.nCells()))
        throw std::runtime_error
        (
            "GAMG owner-accumulated multiply field size mismatch"
        );
    const auto begin = timing && timing->enabled()
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    result.resize(field.size());
    for (Foam::label cell=0; cell<matrix.nCells(); ++cell)
        result[cell] = matrix.diag()[cell]*field[cell];

    const Foam::label* const ownerStarts = matrix.ownerStartAddr().data();
    const Foam::label* const columns = matrix.upperAddr().data();
    const Foam::scalar* const coefficients = matrix.upper().data();
    const Foam::scalar* const x = field.data();
    Foam::scalar* const y = result.data();
    for (Foam::label owner=0; owner<matrix.nCells(); ++owner)
    {
        Foam::scalar ownerResult = y[owner];
        const Foam::scalar xOwner = x[owner];
        for
        (
            Foam::label face=ownerStarts[owner];
            face<ownerStarts[owner + 1];
            ++face
        )
        {
            const Foam::label column = columns[face];
            const Foam::scalar coefficient = coefficients[face];
            ownerResult += coefficient*x[column];
            y[column] += coefficient*xOwner;
        }
        y[owner] = ownerResult;
    }
    if (timing && timing->enabled())
        timing->recordAmul
        (
            level,
            std::chrono::duration<double>
            (
                std::chrono::steady_clock::now() - begin
            ).count()
        );
}

void residual
(
    Foam::scalarField& result,
    const GAMGMatrix& matrix,
    const Foam::scalarField& field,
    const Foam::scalarField& source,
    GAMGTimingStats* const timing,
    const std::size_t level
)
{
    multiply(result, matrix, field, timing, level);
    const auto begin = timing && timing->enabled()
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    for (Foam::label cell=0; cell<matrix.nCells(); ++cell)
        result[cell] = source[cell] - result[cell];
    if (timing && timing->enabled())
        timing->recordResidual
        (
            level,
            std::chrono::duration<double>
            (
                std::chrono::steady_clock::now() - begin
            ).count()
        );
}

void residualDirectFlat
(
    Foam::scalarField& result,
    const DirectFlatMatrix& matrix,
    const Foam::scalarField& field,
    const Foam::scalarField& source,
    GAMGTimingStats* const timing,
    const std::size_t level
)
{
    multiplyDirectFlat(result, matrix, field, timing, level);
    const auto begin = timing && timing->enabled()
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    for (Foam::label cell=0; cell<matrix.nCells; ++cell)
        result[cell] = source[cell] - result[cell];
    if (timing && timing->enabled())
        timing->recordResidual
        (
            level,
            std::chrono::duration<double>
            (
                std::chrono::steady_clock::now() - begin
            ).count()
        );
}

void restrictField
(
    Foam::scalarField& coarse,
    const Foam::scalarField& fine,
    const GAMGAgglomerationLevel& addressing,
    GAMGTimingStats* const timing,
    const std::size_t fineLevel
)
{
    if (fine.size() != addressing.fineToCoarse.size())
        throw std::runtime_error("GAMG restriction field size mismatch");
    const auto begin = timing && timing->enabled()
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    coarse.assign(addressing.nCoarseCells, 0.0);
    for (Foam::label cell=0; cell<addressing.nFineCells; ++cell)
        coarse[addressing.fineToCoarse[cell]] += fine[cell];
    if (timing && timing->enabled())
        timing->recordRestriction
        (
            fineLevel,
            std::chrono::duration<double>
            (
                std::chrono::steady_clock::now() - begin
            ).count()
        );
}

void prolongField
(
    Foam::scalarField& fine,
    const Foam::scalarField& coarse,
    const GAMGAgglomerationLevel& addressing,
    GAMGTimingStats* const timing,
    const std::size_t fineLevel
)
{
    if (coarse.size() != static_cast<std::size_t>(addressing.nCoarseCells))
        throw std::runtime_error("GAMG prolongation field size mismatch");
    const auto begin = timing && timing->enabled()
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    fine.resize(addressing.nFineCells);
    for (Foam::label cell=0; cell<addressing.nFineCells; ++cell)
        fine[cell] = coarse[addressing.fineToCoarse[cell]];
    if (timing && timing->enabled())
        timing->recordProlongation
        (
            fineLevel,
            std::chrono::duration<double>
            (
                std::chrono::steady_clock::now() - begin
            ).count()
        );
}

void gaussSeidelSmooth
(
    Foam::scalarField& psi,
    const GAMGMatrix& matrix,
    const Foam::scalarField& source,
    const Foam::label nSweeps,
    GAMGTimingStats* const timing,
    const std::size_t level
)
{
    if
    (
        psi.size() != static_cast<std::size_t>(matrix.nCells())
     || source.size() != psi.size()
    )
        throw std::runtime_error("GAMG smoother field size mismatch");

    const auto begin = timing && timing->enabled()
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
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
    if (timing && timing->enabled())
        timing->recordSmoothing
        (
            level,
            nSweeps,
            std::chrono::duration<double>
            (
                std::chrono::steady_clock::now() - begin
            ).count()
        );
}

void scaleCorrection
(
    Foam::scalarField& field,
    const GAMGMatrix& matrix,
    const Foam::scalarField& source,
    Foam::scalarField& matrixField,
    GAMGTimingStats* const timing,
    const std::size_t level
)
{
    const auto begin = timing && timing->enabled()
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    multiply(matrixField, matrix, field, timing, level);
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
    if (timing && timing->enabled())
        timing->recordScaleCorrection
        (
            level,
            std::chrono::duration<double>
            (
                std::chrono::steady_clock::now() - begin
            ).count()
        );
}

void scaleCorrectionDirectFlat
(
    Foam::scalarField& field,
    const DirectFlatMatrix& matrix,
    const Foam::scalarField& source,
    Foam::scalarField& matrixField,
    GAMGTimingStats* const timing,
    const std::size_t level
)
{
    const auto begin = timing && timing->enabled()
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    multiplyDirectFlat(matrixField, matrix, field, timing, level);
    Foam::scalar numerator = 0;
    Foam::scalar denominator = 0;
    for (Foam::label cell=0; cell<matrix.nCells; ++cell)
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
    for (Foam::label cell=0; cell<matrix.nCells; ++cell)
        field[cell] = factor*field[cell]
            + (source[cell] - factor*matrixField[cell])/matrix.diag[cell];
    if (timing && timing->enabled())
        timing->recordScaleCorrection
        (
            level,
            std::chrono::duration<double>
            (
                std::chrono::steady_clock::now() - begin
            ).count()
        );
}

void solveCoarsest
(
    Foam::scalarField& correction,
    const GAMGMatrix& matrix,
    const Foam::scalarField& source,
    GAMGTimingStats* const timing,
    const std::size_t level
)
{
    const auto begin = timing && timing->enabled()
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
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
    if (timing && timing->enabled())
        timing->recordCoarsestSolve
        (
            level,
            std::chrono::duration<double>
            (
                std::chrono::steady_clock::now() - begin
            ).count()
        );
}

Foam::scalar l1Norm(const Foam::scalarField& field)
{
    Foam::scalar result = 0;
    for (const Foam::scalar value : field) result += std::abs(value);
    return result;
}

} // namespace smootherTest::gamg
