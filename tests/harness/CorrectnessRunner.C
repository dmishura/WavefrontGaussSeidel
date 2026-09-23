#include "CorrectnessRunner.H"
#include "GaussSeidelSmoother.H"

#include <cmath>

namespace smootherTest::harness
{

CorrectnessResult runCorrectness
(
    const Workload& workload,
    const SmootherVariant& variant,
    const Foam::label nSweeps
)
{
    CorrectnessResult result;
    result.workload = workload.name;
    result.solver = variant.meta.name;
    result.sweeps = nSweeps;

    Foam::scalarField reference = workload.initialPsi;
    Foam::GaussSeidelSmoother::smooth
    (
        "psi", reference, *workload.matrix, workload.source,
        workload.interfaceCoeffs, workload.interfaces, 0, nSweeps
    );
    Foam::scalarField candidate = workload.initialPsi;
    variant.run(candidate, nSweeps);

    result.referenceResidual = relativeResidual
    (
        *workload.matrix, reference, workload.source
    );
    result.residual = relativeResidual
    (
        *workload.matrix, candidate, workload.source
    );
    result.residualDifference = std::abs
    (
        result.residual - result.referenceResidual
    );
    result.maxReferenceDifference = maxAbsDifference(reference, candidate);
    result.maxSolutionError = maxAbsDifference(workload.exact, candidate);
    result.exact = result.maxReferenceDifference == 0
        && result.residualDifference == 0;

    switch (variant.correctness)
    {
        case CorrectnessPolicy::Reference:
            result.passed = result.exact;
            break;
        case CorrectnessPolicy::ExactReference:
            result.passed = result.exact;
            if (!result.passed) result.failure = "not exactly equal to Reference GS";
            break;
        case CorrectnessPolicy::ToleranceReference:
            result.passed =
                result.maxReferenceDifference <= variant.tolerance
             && result.residualDifference <= variant.tolerance;
            if (!result.passed) result.failure = "reference tolerance exceeded";
            break;
        case CorrectnessPolicy::ResidualReduction:
        {
            const Foam::scalar initialResidual = relativeResidual
            (
                *workload.matrix, workload.initialPsi, workload.source
            );
            result.passed = std::isfinite(result.residual)
                && result.residual < initialResidual;
            if (!result.passed) result.failure = "residual did not decrease";
            break;
        }
    }
    return result;
}

std::vector<CorrectnessResult> runCorrectnessRegistry
(
    const Workload& workload,
    const SmootherRegistry& registry,
    const Foam::label nSweeps
)
{
    std::vector<CorrectnessResult> results;
    for (const SmootherVariant& variant : registry.variants())
        if (variant.enableCorrectness)
            results.push_back(runCorrectness(workload, variant, nSweeps));
    return results;
}

CorrectnessResult runConvergence
(
    const Workload& workload,
    const SmootherVariant& variant,
    const Foam::label nSweeps,
    const Foam::scalar tolerance
)
{
    CorrectnessResult result = runCorrectness(workload, variant, nSweeps);
    result.passed = std::isfinite(result.residual)
        && std::isfinite(result.maxSolutionError)
        && result.residual <= tolerance
        && result.maxSolutionError <= tolerance;
    if (!result.passed) result.failure = "convergence tolerance exceeded";
    return result;
}

} // namespace smootherTest::harness
