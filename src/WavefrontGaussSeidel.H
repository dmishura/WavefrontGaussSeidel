#ifndef WavefrontGaussSeidel_H
#define WavefrontGaussSeidel_H

#include "lduMatrix.H"

#include <vector>

namespace smootherTest
{

struct WavefrontSchedule
{
    std::vector<Foam::label> levelStarts;
    std::vector<Foam::label> waveCells;
    std::vector<Foam::label> rowStarts;
    std::vector<Foam::label> cols;
    std::vector<Foam::label> localCols;
    std::vector<Foam::label> incomingCounts;
    std::vector<Foam::scalar> coeffs;
    std::vector<Foam::scalar> diag;
};

void serialGatherSmooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const WavefrontSchedule& schedule,
    Foam::label nSweeps
);

void serialReorderedPsiSmooth
(
    Foam::scalarField& originalPsi,
    Foam::scalarField& reorderedPsi,
    const Foam::scalarField& source,
    const WavefrontSchedule& schedule,
    Foam::label nSweeps
);

bool avx512GatherAvailable();

bool avx2GatherAvailable();

void serialGatherAvx2Smooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const WavefrontSchedule& schedule,
    Foam::label nSweeps
);

void serialGatherAvx2AcrossRowsSmooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const WavefrontSchedule& schedule,
    Foam::label nSweeps
);

void serialGatherAvx512Smooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const WavefrontSchedule& schedule,
    Foam::label nSweeps
);

void serialGatherAvx512AcrossRowsSmooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const WavefrontSchedule& schedule,
    Foam::label nSweeps
);

void perLevelOpenMpSmooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const WavefrontSchedule& schedule,
    Foam::label nSweeps,
    int& detectedThreads
);

void persistentOpenMpSmooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const WavefrontSchedule& schedule,
    Foam::label nSweeps,
    int& detectedThreads
);

}

#endif
