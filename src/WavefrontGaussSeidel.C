#include "WavefrontGaussSeidel.H"

#include <omp.h>

namespace smootherTest
{

using Foam::label;
using Foam::scalar;

void serialGatherSmooth
(
    Foam::scalarField& psiField,
    const Foam::lduMatrix& matrix,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps
)
{
    const label* const ownerStarts = matrix.lduAddr().ownerStartAddr().data();
    const label* const neighbours = matrix.lduAddr().upperAddr().data();
    const scalar* const diag = matrix.diag().data();
    const scalar* const upper = matrix.upper().data();
    const scalar* const lower = matrix.lower().data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const label* const incomingStarts = schedule.incomingStarts.data();
    const label* const incomingFaces = schedule.incomingFaces.data();
    const label* const incomingOwners = schedule.incomingOwners.data();

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (const std::vector<label>& levelCells : schedule.cellsByLevel)
        {
            const label* const cells = levelCells.data();
            for (std::size_t index=0; index<levelCells.size(); ++index)
            {
                const label cell = cells[index];
                scalar psii = source[cell];
                for
                (
                    label in=incomingStarts[cell];
                    in<incomingStarts[cell + 1];
                    ++in
                )
                {
                    const label face = incomingFaces[in];
                    psii -= lower[face]*psi[incomingOwners[in]];
                }
                for
                (
                    label face=ownerStarts[cell];
                    face<ownerStarts[cell + 1];
                    ++face
                )
                {
                    psii -= upper[face]*psi[neighbours[face]];
                }
                psi[cell] = psii/diag[cell];
            }
        }
    }
}

void perLevelOpenMpSmooth
(
    Foam::scalarField& psiField,
    const Foam::lduMatrix& matrix,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps,
    int& detectedThreads
)
{
    const label* const ownerStarts = matrix.lduAddr().ownerStartAddr().data();
    const label* const neighbours = matrix.lduAddr().upperAddr().data();
    const scalar* const diag = matrix.diag().data();
    const scalar* const upper = matrix.upper().data();
    const scalar* const lower = matrix.lower().data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const label* const incomingStarts = schedule.incomingStarts.data();
    const label* const incomingFaces = schedule.incomingFaces.data();
    const label* const incomingOwners = schedule.incomingOwners.data();

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (std::size_t level=0; level<schedule.cellsByLevel.size(); ++level)
        {
            const std::vector<label>& levelCells = schedule.cellsByLevel[level];
            const label* const cells = levelCells.data();
            // This intentionally creates a new OpenMP team for every level.
            #pragma omp parallel
            {
                #pragma omp single
                {
                    if (sweep == 0 && level == 0)
                        detectedThreads = omp_get_num_threads();
                }
                #pragma omp for schedule(static)
                for (std::size_t index=0; index<levelCells.size(); ++index)
                {
                    const label cell = cells[index];
                    scalar psii = source[cell];
                    for
                    (
                        label in=incomingStarts[cell];
                        in<incomingStarts[cell + 1];
                        ++in
                    )
                    {
                        const label face = incomingFaces[in];
                        psii -= lower[face]*psi[incomingOwners[in]];
                    }
                    for
                    (
                        label face=ownerStarts[cell];
                        face<ownerStarts[cell + 1];
                        ++face
                    )
                    {
                        psii -= upper[face]*psi[neighbours[face]];
                    }
                    psi[cell] = psii/diag[cell];
                }
            }
        }
    }
}

void persistentOpenMpSmooth
(
    Foam::scalarField& psiField,
    const Foam::lduMatrix& matrix,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps,
    int& detectedThreads
)
{
    const label* const ownerStarts = matrix.lduAddr().ownerStartAddr().data();
    const label* const neighbours = matrix.lduAddr().upperAddr().data();
    const scalar* const diag = matrix.diag().data();
    const scalar* const upper = matrix.upper().data();
    const scalar* const lower = matrix.lower().data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const label* const incomingStarts = schedule.incomingStarts.data();
    const label* const incomingFaces = schedule.incomingFaces.data();
    const label* const incomingOwners = schedule.incomingOwners.data();

    // Read-only arrays are shared, loop indices and psii are private, and
    // every iteration writes only psi[cell]. Same-level cells have no edges.
    #pragma omp parallel
    {
        #pragma omp single
        detectedThreads = omp_get_num_threads();

        for (label sweep=0; sweep<nSweeps; ++sweep)
        {
            for (std::size_t level=0; level<schedule.cellsByLevel.size(); ++level)
            {
                const std::vector<label>& levelCells =
                    schedule.cellsByLevel[level];
                const label* const cells = levelCells.data();
                #pragma omp for schedule(static)
                for (std::size_t index=0; index<levelCells.size(); ++index)
                {
                    const label cell = cells[index];
                    scalar psii = source[cell];
                    for
                    (
                        label in=incomingStarts[cell];
                        in<incomingStarts[cell + 1];
                        ++in
                    )
                    {
                        const label face = incomingFaces[in];
                        psii -= lower[face]*psi[incomingOwners[in]];
                    }
                    for
                    (
                        label face=ownerStarts[cell];
                        face<ownerStarts[cell + 1];
                        ++face
                    )
                    {
                        psii -= upper[face]*psi[neighbours[face]];
                    }
                    psi[cell] = psii/diag[cell];
                }
                // No nowait: the implicit barrier orders dependency levels.
            }
        }
    }
}

}
