#ifndef WavefrontGaussSeidel_H
#define WavefrontGaussSeidel_H

#include "lduMatrix.H"

#include <cstdint>
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

struct HybridWavefrontSchedule
{
    const WavefrontSchedule* packed = nullptr;
    const Foam::scalarField* lowerCoeffs = nullptr;
    std::vector<Foam::label> incomingStarts;
    std::vector<Foam::label> incomingFaces;
    std::vector<Foam::label> outgoingFaceStarts;
    std::vector<Foam::label> outgoingFaceEnds;
    Foam::label nInternalFaces = 0;
};

struct RowDegreeSchedule
{
    const WavefrontSchedule* packed = nullptr;
    std::vector<std::uint8_t> degrees;
};

struct BlockedRowDegreeSchedule
{
    const WavefrontSchedule* packed = nullptr;
    std::vector<std::uint8_t> degrees;
    std::vector<Foam::label> levelBlockStarts;
    std::vector<Foam::label> blockRowStarts;
    std::vector<Foam::label> blockRowEnds;
    std::vector<Foam::label> blockPStarts;
};

// Topology-only packed rows. Coefficients and diagonal remain in the live
// lduMatrix, so this representation needs no per-matrix coefficient copy.
struct DirectCoefficientSchedule
{
    std::vector<Foam::label> levelStarts;
    std::vector<Foam::label> waveCells;
    std::vector<Foam::label> cols;
    std::vector<Foam::label> faceIds;
    std::vector<std::uint8_t> degrees;
    std::vector<std::uint8_t> incomingDegrees;
    std::vector<Foam::label> levelBlockStarts;
    std::vector<Foam::label> blockRowStarts;
    std::vector<Foam::label> blockRowEnds;
    std::vector<Foam::label> blockPStarts;
};

struct WholeRowInt16Schedule
{
    std::vector<Foam::label> levelStarts, waveCells, rowStarts;
    std::vector<Foam::label> compactStarts, fallbackStarts, fallbackCols;
    std::vector<std::int16_t> offsets;
    std::vector<unsigned char> compactRows;
    std::vector<Foam::scalar> coeffs, diag;
};

struct DeltaEscapeInt16Schedule
{
    std::vector<Foam::label> levelStarts, waveCells, rowStarts;
    std::vector<Foam::label> firstCols, deltaStarts, escapeStarts, escapeCols;
    std::vector<std::int16_t> deltas;
    std::vector<Foam::scalar> coeffs, diag;
};

struct PackedUint24Schedule
{
    std::vector<Foam::label> levelStarts, waveCells, rowStarts;
    std::vector<std::uint8_t> cols;
    std::vector<Foam::scalar> coeffs, diag;
};

void serialGatherSmooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const WavefrontSchedule& schedule,
    Foam::label nSweeps
);

void serialRowDegreeSmooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const RowDegreeSchedule& schedule,
    Foam::label nSweeps
);

void serialDirectCoefficientSmooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const Foam::lduMatrix& matrix,
    const DirectCoefficientSchedule& schedule,
    Foam::label nSweeps
);

void persistentOpenMpDirectCoefficientSmooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const Foam::lduMatrix& matrix,
    const DirectCoefficientSchedule& schedule,
    Foam::label nSweeps,
    int& detectedThreads
);

void serialRowDegreeUnrolled8Smooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const RowDegreeSchedule& schedule,
    Foam::label nSweeps
);

void serialRowDegreeAvx512DivideSmooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const RowDegreeSchedule& schedule,
    Foam::label nSweeps
);

void persistentOpenMpBlockedRowDegreeSmooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const BlockedRowDegreeSchedule& schedule,
    Foam::label nSweeps,
    int& detectedThreads
);

void serialHybridSmooth
(
    Foam::scalarField& psi,
    Foam::scalarField& edgeValues,
    const Foam::scalarField& source,
    const HybridWavefrontSchedule& schedule,
    Foam::label nSweeps
);

void persistentOpenMpHybridSmooth
(
    Foam::scalarField& psi,
    Foam::scalarField& edgeValues,
    const Foam::scalarField& source,
    const HybridWavefrontSchedule& schedule,
    Foam::label nSweeps,
    int& detectedThreads
);

void serialWholeRowInt16Smooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const WholeRowInt16Schedule& schedule,
    Foam::label nSweeps
);

void serialDeltaEscapeInt16Smooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const DeltaEscapeInt16Schedule& schedule,
    Foam::label nSweeps
);

void serialPackedUint24Smooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const PackedUint24Schedule& schedule,
    Foam::label nSweeps
);

void serialGatherInterleavedRowsSmooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const WavefrontSchedule& schedule,
    Foam::label nSweeps
);

void serialGatherDegree6Smooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const WavefrontSchedule& schedule,
    Foam::label nSweeps
);

void serialGatherDegree6InterleavedRowsSmooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const WavefrontSchedule& schedule,
    Foam::label nSweeps
);

void serialGatherPrefetchSmooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const WavefrontSchedule& schedule,
    Foam::label nSweeps,
    Foam::label rowDistance,
    Foam::label neighboursToPrefetch
);

void serialGatherFirstPsiPrefetchSmooth
(
    Foam::scalarField& psi,
    const Foam::scalarField& source,
    const WavefrontSchedule& schedule,
    Foam::label nSweeps,
    Foam::label rowDistance
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

bool avx512FAvailable();

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
