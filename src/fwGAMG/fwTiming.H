#ifndef SmootherTest_fwTiming_H
#define SmootherTest_fwTiming_H

#include "lduMatrix.H"

#include <cstddef>
#include <iosfwd>
#include <vector>

namespace smootherTest::fwgamg
{

struct OperationTiming
{
    std::size_t calls = 0;
    double seconds = 0;
};

struct LevelSetupTiming
{
    Foam::label cells = 0;
    Foam::label faces = 0;
    Foam::label dependencyLevels = 0;
    double scheduleSeconds = 0;
    double permutationSeconds = 0;
    double inversePermutationSeconds = 0;
    double matrixPackingSeconds = 0;
    double flatEdgeMetadataSeconds = 0;
    double directFlatEdgeMetadataSeconds = 0;
    double coarseOperatorSeconds = 0;
    double localityRenumberSeconds = 0;
    double restrictionMapSeconds = 0;
};

struct LevelSolveTiming
{
    Foam::label cells = 0;
    Foam::label faces = 0;
    OperationTiming preSmooth;
    std::size_t preSweeps = 0;
    OperationTiming postSmooth;
    std::size_t postSweeps = 0;
    // Inclusive timing for the experimental final IH1 sweep that also
    // produces A*psi.  This replaces one postSmooth call and one SpMV call.
    OperationTiming fusedSmoothSpmv;
    OperationTiming spmv;
    OperationTiming residual;
    OperationTiming scaleCorrection;
    OperationTiming coarseSolve;
};

struct TransitionTiming
{
    OperationTiming restriction;
    OperationTiming prolongation;
};

class TimingStats
{
public:
    bool enabled = true;
    double setupWallSeconds = 0;
    double lduBridgeSeconds = 0;
    double agglomerationSeconds = 0;
    double coarseLduConstructionSeconds = 0;
    double coarseFwPackingSeconds = 0;
    double coarseConversionSeconds = 0;
    OperationTiming inputPermutation;
    OperationTiming outputPermutation;
    std::vector<LevelSetupTiming> levelSetup;
    std::vector<LevelSolveTiming> levels;
    std::vector<TransitionTiming> transitions;
    std::vector<double> cycleDurations;
    std::vector<double> solveDurations;

    explicit TimingStats(bool enable = true) : enabled(enable) {}

    void configureLevels(std::size_t count);
    void resetRuntime();
    void print(std::ostream& output) const;
};

void add(OperationTiming& timing, double seconds);

} // namespace smootherTest::fwgamg

#endif
