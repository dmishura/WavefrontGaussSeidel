#ifndef SmootherTest_GAMGTiming_H
#define SmootherTest_GAMGTiming_H

#include "lduMatrix.H"

#include <cstddef>
#include <iosfwd>
#include <vector>

namespace smootherTest::gamg
{

class GAMGMatrixHierarchy;
class GAMGAgglomeration;

struct GAMGOperationTiming
{
    std::size_t calls = 0;
    double seconds = 0;
};

struct GAMGLevelTiming
{
    Foam::label cells = 0;
    Foam::label faces = 0;

    GAMGOperationTiming smoothing;
    std::size_t smoothingSweeps = 0;
    GAMGOperationTiming amul;
    GAMGOperationTiming residual;
    GAMGOperationTiming scaleCorrection;
    GAMGOperationTiming coarsestSolve;
};

struct GAMGTransitionTiming
{
    Foam::label fineCells = 0;
    Foam::label fineFaces = 0;
    Foam::label coarseCells = 0;
    Foam::label coarseFaces = 0;

    GAMGOperationTiming coarseMatrixBuild;
    GAMGOperationTiming restriction;
    GAMGOperationTiming prolongation;
};

class GAMGTimingStats
{
    bool enabled_ = true;

public:
    double setupWallSeconds = 0;
    double hierarchyBuildSeconds = 0;
    double coarseMatrixBuildSeconds = 0;
    std::vector<GAMGLevelTiming> levels;
    std::vector<GAMGTransitionTiming> transitions;
    std::vector<double> vcycleDurations;
    std::vector<double> solveDurations;

    explicit GAMGTimingStats(bool enabled = true) : enabled_(enabled) {}

    bool enabled() const noexcept { return enabled_; }
    void enabled(bool value) noexcept { enabled_ = value; }

    void configure
    (
        const GAMGMatrixHierarchy& matrices,
        const GAMGAgglomeration& agglomeration
    );

    void resetRuntime();
    double setupSeconds() const noexcept
    {
        return setupWallSeconds;
    }

    void recordSmoothing(std::size_t level, Foam::label sweeps, double seconds);
    void recordAmul(std::size_t level, double seconds);
    void recordResidual(std::size_t level, double seconds);
    void recordScaleCorrection(std::size_t level, double seconds);
    void recordCoarsestSolve(std::size_t level, double seconds);
    void recordRestriction(std::size_t fineLevel, double seconds);
    void recordProlongation(std::size_t fineLevel, double seconds);
    void recordCoarseMatrixBuild
    (
        std::size_t fineLevel,
        Foam::label fineCells,
        Foam::label fineFaces,
        Foam::label coarseCells,
        Foam::label coarseFaces,
        double seconds
    );

    void print(std::ostream& output) const;
    void printCsv(std::ostream& output) const;
};

} // namespace smootherTest::gamg

#endif
