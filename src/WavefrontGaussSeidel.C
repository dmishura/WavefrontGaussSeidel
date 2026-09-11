#include "WavefrontGaussSeidel.H"

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
