#include "SmootherRegistry.H"

#include <benchmark/benchmark.h>
#include <omp.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace harness = smootherTest::harness;

namespace
{

struct BenchmarkOptions
{
#ifdef SMOOTHER_TEST_DEFAULT_MESH
    std::string mesh = SMOOTHER_TEST_DEFAULT_MESH;
#endif
    Foam::label sweeps = 250;
    Foam::label samples = 7;
    Foam::label warmupSweeps = 10;
    bool googleRepetitionsSpecified = false;
};

int detectOpenMpThreads()
{
    int detected = 1;
    #pragma omp parallel
    {
        #pragma omp single
        detected = omp_get_num_threads();
    }
    return detected;
}

Foam::label parsePositive(const std::string& text, const std::string& option)
{
    std::size_t parsed = 0;
    const int value = std::stoi(text, &parsed);
    if (parsed != text.size() || value < 1)
        throw std::runtime_error(option + " must be a positive integer");
    return value;
}

BenchmarkOptions consumeHarnessArguments(int& argc, char** argv)
{
    BenchmarkOptions options;
    int destination = 1;
    for (int source=1; source<argc; ++source)
    {
        const std::string argument = argv[source];
        if (argument.rfind("--mesh=", 0) == 0)
            options.mesh = argument.substr(7);
        else if (argument.rfind("--smoother_sweeps=", 0) == 0)
            options.sweeps = parsePositive(argument.substr(18), "smoother_sweeps");
        else if (argument.rfind("--smoother_samples=", 0) == 0)
            options.samples = parsePositive(argument.substr(19), "smoother_samples");
        else if (argument.rfind("--smoother_warmup=", 0) == 0)
            options.warmupSweeps = parsePositive
            (
                argument.substr(18), "smoother_warmup"
            );
        else if (argument.rfind("--benchmark_repetitions=", 0) == 0)
        {
            options.googleRepetitionsSpecified = true;
            argv[destination++] = argv[source];
        }
        else argv[destination++] = argv[source];
    }
    argc = destination;
    return options;
}

void registerBenchmarks
(
    harness::Workload& workload,
    const harness::SmootherRegistry& registry,
    const BenchmarkOptions& options,
    const harness::VariantSelection& selection
)
{
    const int detectedThreads = detectOpenMpThreads();
    for (const harness::SmootherVariant& variant : registry.variants())
    {
        if
        (
            !variant.enableBenchmark
         || !harness::matchesSelection(variant.meta, selection)
        ) continue;
        const harness::SmootherVariant* const selected = &variant;
        const harness::SolverFunction batched = harness::makeBatchedSolver
        (
            variant.run, variant.benchmarkSweepsPerCall
        );

        const std::string benchmarkName =
            workload.id + "/" + variant.meta.id;
        auto* const registered = benchmark::RegisterBenchmark
        (
            benchmarkName.c_str(),
            [&, selected, batched, detectedThreads](benchmark::State& state)
            {
                Foam::scalarField warmup = workload.initialPsi;
                batched(warmup, options.warmupSweeps);
                Foam::scalarField psi(workload.initialPsi.size());
                double accumulatedSecondsPerSweep = 0;
                std::size_t measuredSamples = 0;
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    state.PauseTiming();
                    psi = workload.initialPsi;
                    state.ResumeTiming();
                    const auto begin = std::chrono::steady_clock::now();
                    batched(psi, options.sweeps);
                    const auto end = std::chrono::steady_clock::now();
                    const double secondsPerSweep =
                        std::chrono::duration<double>(end - begin).count()
                       /options.sweeps;
                    state.SetIterationTime(secondsPerSweep);
                    accumulatedSecondsPerSweep += secondsPerSweep;
                    ++measuredSamples;
                    benchmark::DoNotOptimize(psi.data());
                }
                const double meanSecondsPerSweep =
                    accumulatedSecondsPerSweep/measuredSamples;
                state.counters["cells"] = static_cast<double>
                (
                    workload.mesh.nCells
                );
                state.counters["sweeps"] = options.sweeps;
                state.counters["threads"] = detectedThreads;
                state.counters["ns_per_cell"] =
                    1e9*meanSecondsPerSweep/workload.mesh.nCells;
                state.SetLabel(selected->meta.name);
            }
        )
        ->UseManualTime()
        ->Iterations(1)
        ->ReportAggregatesOnly(true);
        if (!options.googleRepetitionsSpecified)
            registered->Repetitions(options.samples);
    }
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const harness::VariantSelection selection =
            harness::consumeVariantSelectionArguments(argc, argv);
        const BenchmarkOptions options = consumeHarnessArguments(argc, argv);
        if (options.mesh.empty())
        {
            std::cerr << "Usage: smoother_bench [--mesh=POLYMESH_DIR] "
                         "[--smoother_sweeps=N] [--smoother_samples=N] "
                         "[--smoother_warmup=N] "
                         "[--tier core|experimental|legacy|all] "
                         "[--tag TAG] [--list-variants] "
                         "[Google Benchmark options]\n";
            return 2;
        }
        benchmark::Initialize(&argc, argv);
        if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 2;

        harness::WorkloadRegistry workloads =
            harness::createWorkloadRegistry(options.mesh);
        std::unique_ptr<harness::Workload> workload =
            workloads.create("MotorBike");
        harness::SmootherRegistry solvers =
            harness::createSmootherRegistry(*workload);
        const std::vector<harness::VariantMetadata> metadata =
            solvers.metadata();
        if (selection.listVariants)
        {
            harness::printVariantListing(std::cout, metadata);
            benchmark::Shutdown();
            return 0;
        }
        const std::size_t selectedCount =
            harness::matchingVariantCount(metadata, selection);
        if (!selectedCount)
            throw std::runtime_error("variant selection matched no smoothers");
        const int detectedThreads = detectOpenMpThreads();
        registerBenchmarks(*workload, solvers, options, selection);
        std::cout << "Registry: workload=" << workload->name
            << " cells=" << workload->mesh.nCells
            << " selected="
            << selectedCount
            << " total=" << solvers.variants().size()
            << " preprocessing=" << workload->preprocessingSeconds << " s"
            << " OpenMP_threads=" << detectedThreads << '\n';
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
