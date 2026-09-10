#ifndef WavefrontGaussSeidel_H
#define WavefrontGaussSeidel_H

#include "lduMatrix.H"

#include <vector>

namespace smootherTest
{

struct WavefrontSchedule
{
    std::vector<std::vector<Foam::label>> cellsByLevel;
    std::vector<Foam::label> incomingStarts;
    std::vector<Foam::label> incomingFaces;
    std::vector<Foam::label> incomingOwners;
};

void serialGatherSmooth
(
    Foam::scalarField& psi,
    const Foam::lduMatrix& matrix,
    const Foam::scalarField& source,
    const WavefrontSchedule& schedule,
    Foam::label nSweeps
);

void perLevelOpenMpSmooth
(
    Foam::scalarField& psi,
    const Foam::lduMatrix& matrix,
    const Foam::scalarField& source,
    const WavefrontSchedule& schedule,
    Foam::label nSweeps,
    int& detectedThreads
);

void persistentOpenMpSmooth
(
    Foam::scalarField& psi,
    const Foam::lduMatrix& matrix,
    const Foam::scalarField& source,
    const WavefrontSchedule& schedule,
    Foam::label nSweeps,
    int& detectedThreads
);

}

#endif
