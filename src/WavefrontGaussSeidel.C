#include "WavefrontGaussSeidel.H"

#include <limits>
#include <omp.h>

namespace smootherTest
{

using Foam::label;
using Foam::scalar;

void serialGatherSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps
)
{
    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const cols = schedule.cols.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
            {
                const label cell = waveCells[row];
                scalar psii = source[cell];
                for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                    psii -= coeffs[p]*psi[cols[p]];
                psi[cell] = psii/diag[row];
            }
        }
    }
}

void serialWholeRowInt16Smooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WholeRowInt16Schedule& schedule,
    const label nSweeps
)
{
    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const compactStarts = schedule.compactStarts.data();
    const label* const fallbackStarts = schedule.fallbackStarts.data();
    const label* const fallbackCols = schedule.fallbackCols.data();
    const std::int16_t* const offsets = schedule.offsets.data();
    const unsigned char* const compactRows = schedule.compactRows.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
            {
                const label cell = waveCells[row];
                scalar psii = source[cell];
                const label coefficientBegin = rowStarts[row];
                if (compactRows[row])
                {
                    const label begin = compactStarts[row];
                    const label end = compactStarts[row + 1];
                    for (label q=begin; q<end; ++q)
                        psii -= coeffs[coefficientBegin + q - begin]*psi[cell + offsets[q]];
                }
                else
                {
                    const label begin = fallbackStarts[row];
                    const label end = fallbackStarts[row + 1];
                    for (label q=begin; q<end; ++q)
                        psii -= coeffs[coefficientBegin + q - begin]*psi[fallbackCols[q]];
                }
                psi[cell] = psii/diag[row];
            }
        }
    }
}

void serialDeltaEscapeInt16Smooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const DeltaEscapeInt16Schedule& schedule,
    const label nSweeps
)
{
    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const firstCols = schedule.firstCols.data();
    const label* const deltaStarts = schedule.deltaStarts.data();
    const label* const escapeStarts = schedule.escapeStarts.data();
    const label* const escapeCols = schedule.escapeCols.data();
    const std::int16_t* const deltas = schedule.deltas.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::int16_t escape = std::numeric_limits<std::int16_t>::min();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
            {
                const label cell = waveCells[row];
                scalar psii = source[cell];
                const label begin = rowStarts[row];
                const label end = rowStarts[row + 1];
                if (begin < end)
                {
                    label col = firstCols[row];
                    psii -= coeffs[begin]*psi[col];
                    label deltaIndex = deltaStarts[row];
                    label escapeIndex = escapeStarts[row];
                    for (label p=begin + 1; p<end; ++p)
                    {
                        const std::int16_t delta = deltas[deltaIndex++];
                        col = delta == escape ? escapeCols[escapeIndex++] : col + delta;
                        psii -= coeffs[p]*psi[col];
                    }
                }
                psi[cell] = psii/diag[row];
            }
        }
    }
}

void serialPackedUint24Smooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const PackedUint24Schedule& schedule,
    const label nSweeps
)
{
    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const std::uint8_t* const cols = schedule.cols.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
            {
                const label cell = waveCells[row];
                scalar psii = source[cell];
                for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                {
                    const std::size_t byte = 3*std::size_t(p);
                    const std::uint32_t col =
                        std::uint32_t(cols[byte])
                      | (std::uint32_t(cols[byte + 1]) << 8)
                      | (std::uint32_t(cols[byte + 2]) << 16);
                    psii -= coeffs[p]*psi[col];
                }
                psi[cell] = psii/diag[row];
            }
        }
    }
}

void serialGatherPrefetchSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps,
    const label rowDistance,
    const label neighboursToPrefetch
)
{
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const cols = schedule.cols.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const label nRows = static_cast<label>(schedule.waveCells.size());
    const label prefetchEnd = std::max<label>(0, nRows - rowDistance);

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (label row=0; row<prefetchEnd; ++row)
        {
            const label futureRow = row + rowDistance;
            const label futureBegin = rowStarts[futureRow];
            const label futureCount = std::min
            (
                neighboursToPrefetch,
                rowStarts[futureRow + 1] - futureBegin
            );
            for (label neighbour=0; neighbour<futureCount; ++neighbour)
            {
                __builtin_prefetch
                (
                    psi + cols[futureBegin + neighbour], 0, 3
                );
            }

            const label cell = waveCells[row];
            scalar psii = source[cell];
            for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                psii -= coeffs[p]*psi[cols[p]];
            psi[cell] = psii/diag[row];
        }
        for (label row=prefetchEnd; row<nRows; ++row)
        {
            const label cell = waveCells[row];
            scalar psii = source[cell];
            for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                psii -= coeffs[p]*psi[cols[p]];
            psi[cell] = psii/diag[row];
        }
    }
}

void serialGatherInterleavedRowsSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps
)
{
    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const cols = schedule.cols.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            const label levelBegin = levelStarts[level];
            const label levelEnd = levelStarts[level + 1];
            const label groupEnd = levelEnd - (levelEnd - levelBegin)%4;
            label row = levelBegin;
            for (; row<groupEnd; row += 4)
            {
                const label cell0 = waveCells[row];
                const label cell1 = waveCells[row + 1];
                const label cell2 = waveCells[row + 2];
                const label cell3 = waveCells[row + 3];
                scalar s0 = source[cell0];
                scalar s1 = source[cell1];
                scalar s2 = source[cell2];
                scalar s3 = source[cell3];
                label p0 = rowStarts[row];
                label p1 = rowStarts[row + 1];
                label p2 = rowStarts[row + 2];
                label p3 = rowStarts[row + 3];
                const label end0 = rowStarts[row + 1];
                const label end1 = rowStarts[row + 2];
                const label end2 = rowStarts[row + 3];
                const label end3 = rowStarts[row + 4];

                while (p0<end0 || p1<end1 || p2<end2 || p3<end3)
                {
                    if (p0<end0) { s0 -= coeffs[p0]*psi[cols[p0]]; ++p0; }
                    if (p1<end1) { s1 -= coeffs[p1]*psi[cols[p1]]; ++p1; }
                    if (p2<end2) { s2 -= coeffs[p2]*psi[cols[p2]]; ++p2; }
                    if (p3<end3) { s3 -= coeffs[p3]*psi[cols[p3]]; ++p3; }
                }

                psi[cell0] = s0/diag[row];
                psi[cell1] = s1/diag[row + 1];
                psi[cell2] = s2/diag[row + 2];
                psi[cell3] = s3/diag[row + 3];
            }
            for (; row<levelEnd; ++row)
            {
                const label cell = waveCells[row];
                scalar psii = source[cell];
                for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                    psii -= coeffs[p]*psi[cols[p]];
                psi[cell] = psii/diag[row];
            }
        }
    }
}

void serialGatherDegree6Smooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps
)
{
    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const cols = schedule.cols.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
            {
                const label cell = waveCells[row];
                scalar psii = source[cell];
                const label begin = rowStarts[row];
                if (rowStarts[row + 1] - begin == 6)
                {
                    psii -= coeffs[begin]*psi[cols[begin]];
                    psii -= coeffs[begin + 1]*psi[cols[begin + 1]];
                    psii -= coeffs[begin + 2]*psi[cols[begin + 2]];
                    psii -= coeffs[begin + 3]*psi[cols[begin + 3]];
                    psii -= coeffs[begin + 4]*psi[cols[begin + 4]];
                    psii -= coeffs[begin + 5]*psi[cols[begin + 5]];
                }
                else
                {
                    for (label p=begin; p<rowStarts[row + 1]; ++p)
                        psii -= coeffs[p]*psi[cols[p]];
                }
                psi[cell] = psii/diag[row];
            }
        }
    }
}

void serialGatherDegree6InterleavedRowsSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps
)
{
    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const cols = schedule.cols.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            const label levelEnd = levelStarts[level + 1];
            label row = levelStarts[level];
            while (row < levelEnd)
            {
                const bool fourDegree6 = row + 3 < levelEnd
                    && rowStarts[row + 1] - rowStarts[row] == 6
                    && rowStarts[row + 2] - rowStarts[row + 1] == 6
                    && rowStarts[row + 3] - rowStarts[row + 2] == 6
                    && rowStarts[row + 4] - rowStarts[row + 3] == 6;
                if (fourDegree6)
                {
                    const label cell0 = waveCells[row];
                    const label cell1 = waveCells[row + 1];
                    const label cell2 = waveCells[row + 2];
                    const label cell3 = waveCells[row + 3];
                    const label p0 = rowStarts[row];
                    const label p1 = rowStarts[row + 1];
                    const label p2 = rowStarts[row + 2];
                    const label p3 = rowStarts[row + 3];
                    scalar s0 = source[cell0];
                    scalar s1 = source[cell1];
                    scalar s2 = source[cell2];
                    scalar s3 = source[cell3];

                    s0 -= coeffs[p0]*psi[cols[p0]];
                    s1 -= coeffs[p1]*psi[cols[p1]];
                    s2 -= coeffs[p2]*psi[cols[p2]];
                    s3 -= coeffs[p3]*psi[cols[p3]];
                    s0 -= coeffs[p0 + 1]*psi[cols[p0 + 1]];
                    s1 -= coeffs[p1 + 1]*psi[cols[p1 + 1]];
                    s2 -= coeffs[p2 + 1]*psi[cols[p2 + 1]];
                    s3 -= coeffs[p3 + 1]*psi[cols[p3 + 1]];
                    s0 -= coeffs[p0 + 2]*psi[cols[p0 + 2]];
                    s1 -= coeffs[p1 + 2]*psi[cols[p1 + 2]];
                    s2 -= coeffs[p2 + 2]*psi[cols[p2 + 2]];
                    s3 -= coeffs[p3 + 2]*psi[cols[p3 + 2]];
                    s0 -= coeffs[p0 + 3]*psi[cols[p0 + 3]];
                    s1 -= coeffs[p1 + 3]*psi[cols[p1 + 3]];
                    s2 -= coeffs[p2 + 3]*psi[cols[p2 + 3]];
                    s3 -= coeffs[p3 + 3]*psi[cols[p3 + 3]];
                    s0 -= coeffs[p0 + 4]*psi[cols[p0 + 4]];
                    s1 -= coeffs[p1 + 4]*psi[cols[p1 + 4]];
                    s2 -= coeffs[p2 + 4]*psi[cols[p2 + 4]];
                    s3 -= coeffs[p3 + 4]*psi[cols[p3 + 4]];
                    s0 -= coeffs[p0 + 5]*psi[cols[p0 + 5]];
                    s1 -= coeffs[p1 + 5]*psi[cols[p1 + 5]];
                    s2 -= coeffs[p2 + 5]*psi[cols[p2 + 5]];
                    s3 -= coeffs[p3 + 5]*psi[cols[p3 + 5]];

                    psi[cell0] = s0/diag[row];
                    psi[cell1] = s1/diag[row + 1];
                    psi[cell2] = s2/diag[row + 2];
                    psi[cell3] = s3/diag[row + 3];
                    row += 4;
                }
                else
                {
                    const label cell = waveCells[row];
                    scalar psii = source[cell];
                    for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                        psii -= coeffs[p]*psi[cols[p]];
                    psi[cell] = psii/diag[row];
                    ++row;
                }
            }
        }
    }
}

void serialReorderedPsiSmooth
(
    Foam::scalarField& originalPsiField,
    Foam::scalarField& reorderedPsiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps
)
{
    if (nSweeps <= 1)
    {
        serialGatherSmooth
        (
            originalPsiField, sourceField, schedule, nSweeps
        );
        return;
    }

    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const cols = schedule.cols.data();
    const label* const localCols = schedule.localCols.data();
    const label* const incomingCounts = schedule.incomingCounts.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const originalPsi = originalPsiField.data();
    scalar* const reorderedPsi = reorderedPsiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    // First sweep: updated dependencies are already in reorderedPsi, while
    // not-yet-updated upper neighbours still live in the original field.
    for (std::size_t level=0; level<nLevels; ++level)
    {
        for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
        {
            const label cell = waveCells[row];
            scalar psii = source[cell];
            const label incomingEnd = rowStarts[row] + incomingCounts[row];
            for (label p=rowStarts[row]; p<incomingEnd; ++p)
                psii -= coeffs[p]*reorderedPsi[localCols[p]];
            for (label p=incomingEnd; p<rowStarts[row + 1]; ++p)
                psii -= coeffs[p]*originalPsi[cols[p]];
            reorderedPsi[row] = psii/diag[row];
        }
    }

    // Middle sweeps remain entirely in wavefront-row numbering.
    for (label sweep=1; sweep<nSweeps - 1; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
            {
                scalar psii = source[waveCells[row]];
                for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                    psii -= coeffs[p]*reorderedPsi[localCols[p]];
                reorderedPsi[row] = psii/diag[row];
            }
        }
    }

    // Last sweep: updated dependencies are written in original numbering;
    // upper neighbours retain their previous-sweep reordered values.
    for (std::size_t level=0; level<nLevels; ++level)
    {
        for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
        {
            const label cell = waveCells[row];
            scalar psii = source[cell];
            const label incomingEnd = rowStarts[row] + incomingCounts[row];
            for (label p=rowStarts[row]; p<incomingEnd; ++p)
                psii -= coeffs[p]*originalPsi[cols[p]];
            for (label p=incomingEnd; p<rowStarts[row + 1]; ++p)
                psii -= coeffs[p]*reorderedPsi[localCols[p]];
            originalPsi[cell] = psii/diag[row];
        }
    }
}

void perLevelOpenMpSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps,
    int& detectedThreads
)
{
    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const cols = schedule.cols.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            #pragma omp parallel
            {
                #pragma omp single
                {
                    if (sweep == 0 && level == 0)
                        detectedThreads = omp_get_num_threads();
                }
                #pragma omp for schedule(static)
                for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
                {
                    const label cell = waveCells[row];
                    scalar psii = source[cell];
                    for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                        psii -= coeffs[p]*psi[cols[p]];
                    psi[cell] = psii/diag[row];
                }
            }
        }
    }
}

void persistentOpenMpSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps,
    int& detectedThreads
)
{
    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const cols = schedule.cols.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    #pragma omp parallel
    {
        #pragma omp single
        detectedThreads = omp_get_num_threads();

        for (label sweep=0; sweep<nSweeps; ++sweep)
        {
            for (std::size_t level=0; level<nLevels; ++level)
            {
                #pragma omp for schedule(static)
                for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
                {
                    const label cell = waveCells[row];
                    scalar psii = source[cell];
                    for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                        psii -= coeffs[p]*psi[cols[p]];
                    psi[cell] = psii/diag[row];
                }
                // No nowait: the implicit barrier orders dependency levels.
            }
        }
    }
}

}
