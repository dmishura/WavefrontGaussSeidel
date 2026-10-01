#include "GAMGTiming.H"

#include "GAMGAgglomeration.H"
#include "GAMGMatrix.H"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <stdexcept>

namespace smootherTest::gamg
{
namespace
{

double total(const std::vector<GAMGLevelTiming>& levels, const int operation)
{
    double result = 0;
    for (const GAMGLevelTiming& level : levels)
    {
        switch (operation)
        {
            case 0: result += level.smoothing.seconds; break;
            case 1: result += level.amul.seconds; break;
            case 2: result += level.residual.seconds; break;
            case 3: result += level.scaleCorrection.seconds; break;
            case 4: result += level.coarsestSolve.seconds; break;
        }
    }
    return result;
}

double transitionTotal
(
    const std::vector<GAMGTransitionTiming>& transitions,
    const bool restriction
)
{
    double result = 0;
    for (const GAMGTransitionTiming& transition : transitions)
        result += restriction
            ? transition.restriction.seconds
            : transition.prolongation.seconds;
    return result;
}

double sum(const std::vector<double>& values)
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

double percent(const double value, const double reference)
{
    return reference > 0 ? 100.0*value/reference : 0;
}

void add(GAMGOperationTiming& timing, const double seconds)
{
    ++timing.calls;
    timing.seconds += seconds;
}

} // namespace

void GAMGTimingStats::configure
(
    const GAMGMatrixHierarchy& matrices,
    const GAMGAgglomeration& agglomeration
)
{
    levels.resize(matrices.size());
    for (std::size_t level=0; level<matrices.size(); ++level)
    {
        levels[level].cells = matrices[level].nCells();
        levels[level].faces = matrices[level].nFaces();
    }
    transitions.resize(agglomeration.size());
    for (std::size_t level=0; level<agglomeration.size(); ++level)
    {
        transitions[level].fineCells = matrices[level].nCells();
        transitions[level].fineFaces = matrices[level].nFaces();
        transitions[level].coarseCells = matrices[level + 1].nCells();
        transitions[level].coarseFaces = matrices[level + 1].nFaces();
    }
}

void GAMGTimingStats::resetRuntime()
{
    for (GAMGLevelTiming& level : levels)
    {
        const Foam::label cells = level.cells;
        const Foam::label faces = level.faces;
        level = {};
        level.cells = cells;
        level.faces = faces;
    }
    for (GAMGTransitionTiming& transition : transitions)
    {
        transition.restriction = {};
        transition.prolongation = {};
    }
    vcycleDurations.clear();
    solveDurations.clear();
}

void GAMGTimingStats::recordSmoothing
(
    const std::size_t level,
    const Foam::label sweeps,
    const double seconds
)
{
    add(levels.at(level).smoothing, seconds);
    levels.at(level).smoothingSweeps += static_cast<std::size_t>(sweeps);
}

void GAMGTimingStats::recordAmul(const std::size_t level, const double seconds)
{
    add(levels.at(level).amul, seconds);
}

void GAMGTimingStats::recordResidual
(
    const std::size_t level,
    const double seconds
)
{
    add(levels.at(level).residual, seconds);
}

void GAMGTimingStats::recordScaleCorrection
(
    const std::size_t level,
    const double seconds
)
{
    add(levels.at(level).scaleCorrection, seconds);
}

void GAMGTimingStats::recordCoarsestSolve
(
    const std::size_t level,
    const double seconds
)
{
    add(levels.at(level).coarsestSolve, seconds);
}

void GAMGTimingStats::recordRestriction
(
    const std::size_t fineLevel,
    const double seconds
)
{
    add(transitions.at(fineLevel).restriction, seconds);
}

void GAMGTimingStats::recordProlongation
(
    const std::size_t fineLevel,
    const double seconds
)
{
    add(transitions.at(fineLevel).prolongation, seconds);
}

void GAMGTimingStats::recordCoarseMatrixBuild
(
    const std::size_t fineLevel,
    const Foam::label fineCells,
    const Foam::label fineFaces,
    const Foam::label coarseCells,
    const Foam::label coarseFaces,
    const double seconds
)
{
    if (transitions.size() <= fineLevel) transitions.resize(fineLevel + 1);
    GAMGTransitionTiming& transition = transitions[fineLevel];
    transition.fineCells = fineCells;
    transition.fineFaces = fineFaces;
    transition.coarseCells = coarseCells;
    transition.coarseFaces = coarseFaces;
    add(transition.coarseMatrixBuild, seconds);
    coarseMatrixBuildSeconds += seconds;
}

void GAMGTimingStats::print(std::ostream& output) const
{
    const double solve = sum(solveDurations);
    const double smoothing = total(levels, 0);
    const double amul = total(levels, 1);
    const double residual = total(levels, 2);
    const double scaling = total(levels, 3);
    const double coarsest = total(levels, 4);
    const double restriction = transitionTotal(transitions, true);
    const double prolongation = transitionTotal(transitions, false);
    const double remaining = std::max(0.0, solve - smoothing);
    const double maxSpeedup = remaining > 0 ? solve/remaining : 0;

    output << std::fixed << std::setprecision(6)
        << "\nGAMG timing summary\n"
        << "  convention: hierarchy/coarse-build and leaf kernel timers are "
           "exclusive operation timers\n"
        << "  convention: residual excludes Amul; scaleCorrection is inclusive "
           "of its internally timed Amul\n"
        << "  convention: vcycle and solve are inclusive wall timers; "
           "overlapping categories must not be summed\n"
        << "\nSetup:\n"
        << "  total setup wall: " << 1e3*setupWallSeconds << " ms\n"
        << "  hierarchyBuild: " << 1e3*hierarchyBuildSeconds << " ms\n"
        << "  coarseMatrixBuild: " << 1e3*coarseMatrixBuildSeconds << " ms\n"
        << "  uncategorized setup: " << 1e3*std::max
        (
            0.0,
            setupWallSeconds - hierarchyBuildSeconds - coarseMatrixBuildSeconds
        ) << " ms\n"
        << "\nSolve:\n"
        << "  solves: " << solveDurations.size() << '\n'
        << "  solve total: " << 1e3*solve << " ms"
        << " mean: " << (solveDurations.empty() ? 0 : 1e3*solve/solveDurations.size())
        << " ms median: " << 1e3*median(solveDurations) << " ms\n"
        << "  Vcycles: " << vcycleDurations.size();
    if (!vcycleDurations.empty())
    {
        output << " total: " << 1e3*sum(vcycleDurations)
            << " ms min/median/mean/max: "
            << 1e3**std::min_element(vcycleDurations.begin(), vcycleDurations.end())
            << " / " << 1e3*median(vcycleDurations)
            << " / " << 1e3*sum(vcycleDurations)/vcycleDurations.size()
            << " / "
            << 1e3**std::max_element(vcycleDurations.begin(), vcycleDurations.end())
            << " ms";
    }
    output << '\n'
        << "  smoothing: " << 1e3*smoothing << " ms ("
        << percent(smoothing, solve) << "% of solve)\n"
        << "  Amul: " << 1e3*amul << " ms (" << percent(amul, solve)
        << "% of solve)\n"
        << "  residual excluding Amul: " << 1e3*residual << " ms\n"
        << "  scaleCorrection inclusive: " << 1e3*scaling << " ms\n"
        << "  restriction: " << 1e3*restriction << " ms ("
        << percent(restriction, solve) << "% of solve)\n"
        << "  prolongation: " << 1e3*prolongation << " ms ("
        << percent(prolongation, solve) << "% of solve)\n"
        << "  coarsestSolve: " << 1e3*coarsest << " ms ("
        << percent(coarsest, solve) << "% of solve)\n"
        << "  smoothing-only Amdahl upper bound: " << maxSpeedup << "x\n";

    output << "\nCoarse matrix construction:\n";
    for (std::size_t level=0; level<transitions.size(); ++level)
    {
        const GAMGTransitionTiming& transition = transitions[level];
        output << "  L" << level << "->L" << level + 1
            << " fineCells=" << transition.fineCells
            << " fineFaces=" << transition.fineFaces
            << " coarseCells=" << transition.coarseCells
            << " coarseFaces=" << transition.coarseFaces
            << " time=" << 1e3*transition.coarseMatrixBuild.seconds << " ms\n";
    }

    output << "\nPer level:\n";
    for (std::size_t index=0; index<levels.size(); ++index)
    {
        const GAMGLevelTiming& level = levels[index];
        const double millisecondsPerCall = level.smoothing.calls
            ? 1e3*level.smoothing.seconds/level.smoothing.calls : 0;
        const double millisecondsPerSweep = level.smoothingSweeps
            ? 1e3*level.smoothing.seconds/level.smoothingSweeps : 0;
        const double nsPerCellSweep = level.smoothingSweeps && level.cells
            ? 1e9*level.smoothing.seconds
                /(static_cast<double>(level.smoothingSweeps)*level.cells)
            : 0;
        const double amulNsPerCell = level.amul.calls && level.cells
            ? 1e9*level.amul.seconds
                /(static_cast<double>(level.amul.calls)*level.cells)
            : 0;
        output << "  L" << index << " cells=" << level.cells
            << " faces=" << level.faces << '\n'
            << "    GS calls=" << level.smoothing.calls
            << " sweeps=" << level.smoothingSweeps
            << " total=" << 1e3*level.smoothing.seconds
            << " ms ms/call=" << millisecondsPerCall
            << " ms/sweep=" << millisecondsPerSweep
            << " ns/cell/sweep=" << nsPerCellSweep << '\n'
            << "    Amul calls=" << level.amul.calls
            << " total=" << 1e3*level.amul.seconds
            << " ms ns/cell/call=" << amulNsPerCell
            << " residual calls=" << level.residual.calls
            << " total=" << 1e3*level.residual.seconds
            << " ms scale calls=" << level.scaleCorrection.calls
            << " total=" << 1e3*level.scaleCorrection.seconds << " ms";
        if (level.coarsestSolve.calls)
            output << " coarsest calls=" << level.coarsestSolve.calls
                << " total=" << 1e3*level.coarsestSolve.seconds << " ms";
        output << '\n';
    }

    output << "\nTransitions:\n";
    for (std::size_t level=0; level<transitions.size(); ++level)
    {
        const GAMGTransitionTiming& transition = transitions[level];
        output << "  L" << level << "->L" << level + 1
            << " restrict calls=" << transition.restriction.calls
            << " total=" << 1e3*transition.restriction.seconds
            << " ms; prolong L" << level + 1 << "->L" << level
            << " calls=" << transition.prolongation.calls
            << " total=" << 1e3*transition.prolongation.seconds << " ms\n";
    }
}

void GAMGTimingStats::printCsv(std::ostream& output) const
{
    output << "level,cells,faces,smooth_calls,sweeps,smooth_ms,amul_calls,"
              "amul_ms,residual_calls,residual_ms,scale_calls,scale_ms\n";
    for (std::size_t index=0; index<levels.size(); ++index)
    {
        const GAMGLevelTiming& level = levels[index];
        output << index << ',' << level.cells << ',' << level.faces << ','
            << level.smoothing.calls << ',' << level.smoothingSweeps << ','
            << 1e3*level.smoothing.seconds << ',' << level.amul.calls << ','
            << 1e3*level.amul.seconds << ',' << level.residual.calls << ','
            << 1e3*level.residual.seconds << ','
            << level.scaleCorrection.calls << ','
            << 1e3*level.scaleCorrection.seconds << '\n';
    }
}

} // namespace smootherTest::gamg
