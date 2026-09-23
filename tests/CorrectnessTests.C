#include "CorrectnessRunner.H"

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace harness = smootherTest::harness;

namespace
{

class SolverCorrectnessTest : public ::testing::Test
{
    const harness::Workload* workload_;
    const harness::SmootherVariant* variant_;
    Foam::label sweeps_;
    bool convergence_;

public:
    SolverCorrectnessTest
    (
        const harness::Workload* workload,
        const harness::SmootherVariant* variant,
        const Foam::label sweeps,
        const bool convergence
    )
    :
        workload_(workload),
        variant_(variant),
        sweeps_(sweeps),
        convergence_(convergence)
    {}

    void TestBody() override
    {
        const harness::CorrectnessResult result = convergence_
            ? harness::runConvergence(*workload_, *variant_, sweeps_, 1e-10)
            : harness::runCorrectness(*workload_, *variant_, sweeps_);
        SCOPED_TRACE("workload=" + result.workload + " solver=" + result.solver);
        EXPECT_TRUE(result.passed)
            << result.failure
            << "\nresidual=" << result.residual
            << " referenceResidual=" << result.referenceResidual
            << " residualDifference=" << result.residualDifference
            << " maxReferenceDifference=" << result.maxReferenceDifference
            << " maxSolutionError=" << result.maxSolutionError;
    }
};

class DependencyValidationTest : public ::testing::Test
{
    const harness::Workload* workload_;

public:
    explicit DependencyValidationTest(const harness::Workload* workload)
    : workload_(workload)
    {}

    void TestBody() override
    {
        EXPECT_TRUE(workload_->dependenciesValid);
        EXPECT_EQ
        (
            workload_->packed.waveCells.size(),
            workload_->mesh.nCells
        );
    }
};

const char* suiteFor(const harness::CorrectnessPolicy policy)
{
    switch (policy)
    {
        case harness::CorrectnessPolicy::Reference:
            return "ReferenceCorrectness";
        case harness::CorrectnessPolicy::ExactReference:
            return "OneSweepExactEquality";
        case harness::CorrectnessPolicy::ToleranceReference:
            return "OneSweepToleranceEquality";
        case harness::CorrectnessPolicy::ResidualReduction:
            return "OneSweepResidualReduction";
    }
    return "Correctness";
}

void registerCorrectnessTests
(
    const harness::Workload& workload,
    const harness::SmootherRegistry& registry,
    const harness::VariantSelection& selection
)
{
    ::testing::RegisterTest
    (
        "DependencyValidation", workload.id.c_str(), nullptr, nullptr,
        __FILE__, __LINE__,
        [&workload]() -> DependencyValidationTest*
        {
            return new DependencyValidationTest(&workload);
        }
    );

    for (const harness::SmootherVariant& variant : registry.variants())
    {
        if
        (
            !variant.enableCorrectness
         || !harness::matchesSelection(variant.meta, selection)
        ) continue;
        const harness::SmootherVariant* const selected = &variant;
        ::testing::RegisterTest
        (
            suiteFor(variant.correctness), variant.meta.id.c_str(),
            nullptr, nullptr,
            __FILE__, __LINE__,
            [&workload, selected]() -> SolverCorrectnessTest*
            {
                return new SolverCorrectnessTest
                (
                    &workload, selected, 1, false
                );
            }
        );
        if
        (
            variant.correctness == harness::CorrectnessPolicy::ExactReference
         || variant.correctness == harness::CorrectnessPolicy::ToleranceReference
        )
        {
            ::testing::RegisterTest
            (
                "ThreeSweepReferenceComparison", variant.meta.id.c_str(),
                nullptr, nullptr, __FILE__, __LINE__,
                [&workload, selected]() -> SolverCorrectnessTest*
                {
                    return new SolverCorrectnessTest
                    (
                        &workload, selected, 3, false
                    );
                }
            );
        }
        if
        (
            variant.correctness == harness::CorrectnessPolicy::Reference
         || variant.correctness == harness::CorrectnessPolicy::ResidualReduction
        )
        {
            ::testing::RegisterTest
            (
                "Convergence", variant.meta.id.c_str(), nullptr, nullptr,
                __FILE__, __LINE__,
                [&workload, selected]() -> SolverCorrectnessTest*
                {
                    return new SolverCorrectnessTest
                    (
                        &workload, selected, 200, true
                    );
                }
            );
        }
    }
}

std::string consumeMeshArgument(int& argc, char** argv)
{
#ifdef SMOOTHER_TEST_DEFAULT_MESH
    std::string mesh = SMOOTHER_TEST_DEFAULT_MESH;
#else
    std::string mesh;
#endif
    int destination = 1;
    for (int source=1; source<argc; ++source)
    {
        const std::string argument = argv[source];
        if (argument.rfind("--mesh=", 0) == 0)
            mesh = argument.substr(7);
        else argv[destination++] = argv[source];
    }
    argc = destination;
    return mesh;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const harness::VariantSelection selection =
            harness::consumeVariantSelectionArguments(argc, argv);
        const std::string meshDirectory = consumeMeshArgument(argc, argv);
        if (meshDirectory.empty())
        {
            std::cerr << "Usage: smoother_tests [--mesh=POLYMESH_DIR] "
                         "[--tier core|experimental|legacy|all] "
                         "[--tag TAG] [--list-variants] "
                         "[GoogleTest options]\n";
            return 2;
        }
        ::testing::InitGoogleTest(&argc, argv);
        harness::WorkloadRegistry workloads =
            harness::createWorkloadRegistry(meshDirectory);
        std::unique_ptr<harness::Workload> workload =
            workloads.create("MotorBike");
        harness::SmootherRegistry solvers =
            harness::createSmootherRegistry(*workload);
        const std::vector<harness::VariantMetadata> metadata =
            solvers.metadata();
        if (selection.listVariants)
        {
            harness::printVariantListing(std::cout, metadata);
            return 0;
        }
        const std::size_t selectedCount =
            harness::matchingVariantCount(metadata, selection);
        if (!selectedCount)
            throw std::runtime_error("variant selection matched no smoothers");
        registerCorrectnessTests(*workload, solvers, selection);
        std::cout << "Registry: workload=" << workload->name
            << " cells=" << workload->mesh.nCells
            << " selected="
            << selectedCount
            << " total=" << solvers.variants().size()
            << " preprocessing=" << workload->preprocessingSeconds << " s\n";
        return RUN_ALL_TESTS();
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
