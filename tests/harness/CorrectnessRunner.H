#ifndef SmootherTest_CorrectnessRunner_H
#define SmootherTest_CorrectnessRunner_H

#include "SmootherRegistry.H"

#include <string>
#include <vector>

namespace smootherTest::harness
{

struct CorrectnessResult
{
    std::string workload;
    std::string solver;
    Foam::label sweeps = 0;
    Foam::scalar residual = 0;
    Foam::scalar referenceResidual = 0;
    Foam::scalar residualDifference = 0;
    Foam::scalar maxReferenceDifference = 0;
    Foam::scalar maxSolutionError = 0;
    bool exact = false;
    bool passed = false;
    std::string failure;
};

CorrectnessResult runCorrectness
(
    const Workload& workload,
    const SmootherVariant& variant,
    Foam::label nSweeps
);

std::vector<CorrectnessResult> runCorrectnessRegistry
(
    const Workload& workload,
    const SmootherRegistry& registry,
    Foam::label nSweeps
);

CorrectnessResult runConvergence
(
    const Workload& workload,
    const SmootherVariant& variant,
    Foam::label nSweeps,
    Foam::scalar tolerance
);

} // namespace smootherTest::harness

#endif
