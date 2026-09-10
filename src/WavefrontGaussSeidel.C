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
