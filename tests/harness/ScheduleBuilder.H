#ifndef SmootherTest_ScheduleBuilder_H
#define SmootherTest_ScheduleBuilder_H

#include "PolyMeshReader.H"
#include "WavefrontGaussSeidel.H"
#include "lduMatrix.H"

#include <cstddef>
#include <vector>

namespace smootherTest::harness
{

struct WavefrontStatistics
{
    std::vector<Foam::label> levels;
    std::vector<std::size_t> widths;
    std::size_t minimumWidth = 0;
    std::size_t maximumWidth = 0;
    Foam::scalar meanWidth = 0;
    Foam::scalar medianWidth = 0;
    std::size_t p90Width = 0;
    std::size_t p95Width = 0;
};

Foam::lduMatrix makeMatrix(const PolyMeshTopology& mesh);

Foam::scalarField multiply
(
    const Foam::lduMatrix& matrix,
    const Foam::scalarField& psi
);

Foam::scalar relativeResidual
(
    const Foam::lduMatrix& matrix,
    const Foam::scalarField& psi,
    const Foam::scalarField& source
);

Foam::scalar maxAbsDifference
(
    const Foam::scalarField& left,
    const Foam::scalarField& right
);

WavefrontStatistics constructMinimalWavefronts(const PolyMeshTopology& mesh);

WavefrontStatistics constructGeometryWavefronts
(
    const PolyMeshTopology& mesh,
    Foam::label xSlabs,
    Foam::label targetWidth
);

WavefrontStatistics constructIndexWavefronts
(
    const PolyMeshTopology& mesh,
    const Foam::lduMatrix& matrix,
    Foam::label targetWidth,
    bool window
);

bool validateDependencies
(
    const PolyMeshTopology& mesh,
    const WavefrontStatistics& statistics
);

WavefrontSchedule makeWavefrontSchedule
(
    const PolyMeshTopology& mesh,
    const Foam::lduMatrix& matrix,
    const WavefrontStatistics& statistics
);

WavefrontSchedule makeXSortedSchedule
(
    const WavefrontSchedule& original,
    const std::vector<double>& cellCentreX
);

WavefrontSchedule makeMedianRowSchedule(const WavefrontSchedule& original);
WavefrontSchedule makeLocalitySchedule(const WavefrontSchedule& original);
WholeRowInt16Schedule makeWholeRowInt16Schedule(const WavefrontSchedule& original);
DeltaEscapeInt16Schedule makeDeltaEscapeInt16Schedule(const WavefrontSchedule& original);
PackedUint24Schedule makePackedUint24Schedule(const WavefrontSchedule& original);
RowDegreeSchedule makeRowDegreeSchedule(const WavefrontSchedule& original);
BlockedRowDegreeSchedule makeBlockedRowDegreeSchedule
(
    const WavefrontSchedule& original,
    Foam::label blockSize
);

AdaptiveBlockedRowDegreeSchedule makeAdaptiveBlockedRowDegreeSchedule
(
    const BlockedRowDegreeSchedule& blocked,
    int configuredThreads,
    Foam::label minWidth2Threads,
    Foam::label minWidthFullTeam
);

HybridWavefrontSchedule makeHybridSchedule
(
    const PolyMeshTopology& mesh,
    const Foam::lduMatrix& matrix,
    const WavefrontSchedule& packed
);

DirectCoefficientSchedule makeDirectCoefficientSchedule
(
    const PolyMeshTopology& mesh,
    const Foam::lduMatrix& matrix,
    const WavefrontStatistics& statistics,
    Foam::label blockSize
);

} // namespace smootherTest::harness

#endif
