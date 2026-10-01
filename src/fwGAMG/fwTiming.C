#include "fwTiming.H"

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <ostream>

namespace smootherTest::fwgamg
{
namespace
{

double total(const std::vector<double>& values)
{
    return std::accumulate(values.begin(), values.end(), 0.0);
}

double median(std::vector<double> values)
{
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size()/2;
    return values.size()%2
        ? values[middle]
        : 0.5*(values[middle - 1] + values[middle]);
}

double percentage(const double value, const double reference)
{
    return reference > 0 ? 100.0*value/reference : 0;
}

} // namespace

void add(OperationTiming& timing, const double seconds)
{
    ++timing.calls;
    timing.seconds += seconds;
}

void TimingStats::configureLevels(const std::size_t count)
{
    levelSetup.resize(count);
    levels.resize(count);
    if (count) transitions.resize(count - 1);
}

void TimingStats::resetRuntime()
{
    for (LevelSolveTiming& level : levels)
    {
        const Foam::label cells = level.cells;
        const Foam::label faces = level.faces;
        level = {};
        level.cells = cells;
        level.faces = faces;
    }
    for (TransitionTiming& transition : transitions) transition = {};
    inputPermutation = {};
    outputPermutation = {};
    cycleDurations.clear();
    solveDurations.clear();
}

void TimingStats::print(std::ostream& output) const
{
    double preSmooth = 0;
    double postSmooth = 0;
    double fusedSmoothSpmv = 0;
    double spmv = 0;
    double residual = 0;
    double scale = 0;
    double coarse = 0;
    double restriction = 0;
    double prolongation = 0;
    for (const LevelSolveTiming& level : levels)
    {
        preSmooth += level.preSmooth.seconds;
        postSmooth += level.postSmooth.seconds;
        fusedSmoothSpmv += level.fusedSmoothSpmv.seconds;
        spmv += level.spmv.seconds;
        residual += level.residual.seconds;
        scale += level.scaleCorrection.seconds;
        coarse += level.coarseSolve.seconds;
    }
    for (const TransitionTiming& transition : transitions)
    {
        restriction += transition.restriction.seconds;
        prolongation += transition.prolongation.seconds;
    }
    const double solve = total(solveDurations);
    const double smoothing = preSmooth + postSmooth + fusedSmoothSpmv;
    const double setupCategorized = lduBridgeSeconds + agglomerationSeconds;
    const double cycles = total(cycleDurations);

    output << std::fixed << std::setprecision(6)
        << "\nfwGAMG timing summary\n"
        << "  solve/cycle/scaleCorrection are inclusive; SpMV inside "
           "scaleCorrection is also reported separately\n"
        << "Setup:\n"
        << "  total wall: " << 1e3*setupWallSeconds << " ms\n"
        << "  fine LDU bridge: " << 1e3*lduBridgeSeconds << " ms\n"
        << "  stock-compatible agglomeration metadata: "
        << 1e3*agglomerationSeconds << " ms\n"
        << "  intermediate coarse LDU construction: "
        << 1e3*coarseLduConstructionSeconds << " ms\n"
        << "  coarse LDU->fw repacking: " << 1e3*coarseFwPackingSeconds
        << " ms\n"
        << "  coarsest fw->LDU conversion: " << 1e3*coarseConversionSeconds
        << " ms\n"
        << "  other fw setup: " << 1e3*std::max
        (
            0.0, setupWallSeconds - setupCategorized
        ) << " ms\n"
        << "Solve:\n"
        << "  solves=" << solveDurations.size()
        << " total=" << 1e3*solve
        << " ms mean=" << (solveDurations.empty()
            ? 0 : 1e3*solve/solveDurations.size())
        << " ms median=" << 1e3*median(solveDurations) << " ms\n"
        << "  cycles=" << cycleDurations.size()
        << " total=" << 1e3*cycles
        << " ms median=" << 1e3*median(cycleDurations) << " ms\n"
        << "  pre-smoothing=" << 1e3*preSmooth << " ms\n"
        << "  post-smoothing=" << 1e3*postSmooth << " ms\n"
        << "  fused finest IH1+SpMV=" << 1e3*fusedSmoothSpmv
        << " ms (inclusive combined kernel)\n"
        << "  smoothing total=" << 1e3*smoothing << " ms ("
        << percentage(smoothing, solve) << "%)\n"
        << "  SpMV=" << 1e3*spmv << " ms (" << percentage(spmv, solve)
        << "%)\n"
        << "  residual excluding SpMV=" << 1e3*residual << " ms\n"
        << "  scaleCorrection inclusive=" << 1e3*scale << " ms\n"
        << "  restriction=" << 1e3*restriction << " ms\n"
        << "  prolongation=" << 1e3*prolongation << " ms\n"
        << "  coarse solve=" << 1e3*coarse << " ms\n"
        << "  input permutation=" << 1e3*inputPermutation.seconds << " ms\n"
        << "  output permutation=" << 1e3*outputPermutation.seconds << " ms\n";

    output << "Per-level setup:\n";
    for (std::size_t level=0; level<levelSetup.size(); ++level)
    {
        const LevelSetupTiming& value = levelSetup[level];
        output << "  L" << level << " cells=" << value.cells
            << " faces=" << value.faces
            << " dependencyLevels=" << value.dependencyLevels
            << " schedule=" << 1e3*value.scheduleSeconds
            << " ms permutation=" << 1e3*value.permutationSeconds
            << " ms inverse=" << 1e3*value.inversePermutationSeconds
            << " ms packing=" << 1e3*value.matrixPackingSeconds
            << " ms flatEdgeMetadata="
            << 1e3*value.flatEdgeMetadataSeconds
            << " ms directFlatEdgeMetadata="
            << 1e3*value.directFlatEdgeMetadataSeconds
            << " ms coarseOperator=" << 1e3*value.coarseOperatorSeconds
            << " ms localityRenumber=" << 1e3*value.localityRenumberSeconds
            << " ms restrictionMap=" << 1e3*value.restrictionMapSeconds
            << " ms\n";
    }

    output << "Per-level solve:\n";
    for (std::size_t level=0; level<levels.size(); ++level)
    {
        const LevelSolveTiming& value = levels[level];
        const std::size_t sweeps = value.preSweeps + value.postSweeps;
        const double smoothSeconds = value.preSmooth.seconds
            + value.postSmooth.seconds + value.fusedSmoothSpmv.seconds;
        output << "  L" << level << " cells=" << value.cells
            << " faces=" << value.faces
            << " preCalls=" << value.preSmooth.calls
            << " preSweeps=" << value.preSweeps
            << " postCalls=" << value.postSmooth.calls
            << " postSweeps=" << value.postSweeps
            << " fusedCalls=" << value.fusedSmoothSpmv.calls
            << " fused=" << 1e3*value.fusedSmoothSpmv.seconds << " ms"
            << " smooth=" << 1e3*smoothSeconds << " ms"
            << " ns/cell/sweep=" << (sweeps && value.cells
                ? 1e9*smoothSeconds/(static_cast<double>(sweeps)*value.cells)
                : 0)
            << " SpMVcalls=" << value.spmv.calls
            << " SpMV=" << 1e3*value.spmv.seconds << " ms"
            << " residualCalls=" << value.residual.calls
            << " residual=" << 1e3*value.residual.seconds << " ms";
        if (value.coarseSolve.calls)
            output << " coarseCalls=" << value.coarseSolve.calls
                << " coarse=" << 1e3*value.coarseSolve.seconds << " ms";
        output << '\n';
    }
}

} // namespace smootherTest::fwgamg
