#ifndef SmootherTest_SmootherRegistry_H
#define SmootherTest_SmootherRegistry_H

#include "VariantSelection.H"
#include "Workload.H"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace smootherTest::harness
{

using SolverFunction = std::function<void
(
    Foam::scalarField& psi,
    Foam::label nSweeps
)>;

enum class CorrectnessPolicy
{
    Reference,
    ExactReference,
    ExactSymmetricReference,
    ToleranceReference,
    ResidualReduction
};

struct SmootherVariant
{
    VariantMetadata meta;
    SolverFunction run;
    CorrectnessPolicy correctness = CorrectnessPolicy::ExactReference;
    Foam::scalar tolerance = 0;
    bool enableCorrectness = true;
    bool enableBenchmark = true;
    Foam::label benchmarkSweepsPerCall = 3;
    bool parallel = false;

    SmootherVariant
    (
        std::string name,
        std::string id,
        SolverFunction function,
        CorrectnessPolicy correctnessPolicy,
        Foam::scalar correctnessTolerance,
        bool correctnessEnabled,
        bool benchmarkEnabled,
        Foam::label sweepsPerCall,
        bool isParallel,
        VariantTier tier,
        std::vector<std::string> tags
    )
    :
        meta
        {
            std::move(name), std::move(id), tier, std::move(tags)
        },
        run(std::move(function)),
        correctness(correctnessPolicy),
        tolerance(correctnessTolerance),
        enableCorrectness(correctnessEnabled),
        enableBenchmark(benchmarkEnabled),
        benchmarkSweepsPerCall(sweepsPerCall),
        parallel(isParallel)
    {}
};

class SmootherRegistry
{
    std::vector<SmootherVariant> variants_;

public:
    void add(SmootherVariant variant);
    const std::vector<SmootherVariant>& variants() const noexcept;
    const SmootherVariant& at(const std::string& id) const;
    std::vector<VariantMetadata> metadata() const;
};

SolverFunction makeBatchedSolver
(
    SolverFunction solver,
    Foam::label sweepsPerCall
);

SmootherRegistry createSmootherRegistry(Workload& workload);

} // namespace smootherTest::harness

#endif
