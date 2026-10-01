#include "GAMGSolver.H"
#include "GAMGTiming.H"
#include "fwGAMG.H"
#include "PolyMeshReader.H"
#include "ScheduleBuilder.H"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HAVE_ITTNOTIFY
#include <ittnotify.h>
#endif

namespace gamg = smootherTest::gamg;
namespace fwgamg = smootherTest::fwgamg;
namespace harness = smootherTest::harness;

namespace
{

struct Options
{
#ifdef SMOOTHER_TEST_DEFAULT_MESH
    std::string mesh = SMOOTHER_TEST_DEFAULT_MESH;
#endif
    Foam::label warmup = 1;
    Foam::label samples = 7;
    bool csv = false;
    bool flatEdgeOnly = false;
    bool productionCompareOnly = false;
    bool fusionCompareOnly = false;
    bool deltaCandidateOnly = false;
    bool deltaCandidateSolveOnly = false;
    bool l1EvictionCompareOnly = false;
    std::string deltaCandidateSolveVariant = "both";
    std::string vtuneProfileVariant;
    bool googleRepetitionsSpecified = false;
};

Foam::label parsePositive(const std::string& text, const std::string& option)
{
    std::size_t parsed = 0;
    const int value = std::stoi(text, &parsed);
    if (parsed != text.size() || value < 1)
        throw std::runtime_error(option + " must be a positive integer");
    return value;
}

Options consumeArguments(int& argc, char** argv)
{
    Options options;
    int destination = 1;
    for (int source=1; source<argc; ++source)
    {
        const std::string argument = argv[source];
        if (argument.rfind("--mesh=", 0) == 0)
            options.mesh = argument.substr(7);
        else if (argument.rfind("--gamg_warmup=", 0) == 0)
            options.warmup = parsePositive(argument.substr(14), "gamg_warmup");
        else if (argument.rfind("--gamg_samples=", 0) == 0)
            options.samples = parsePositive(argument.substr(15), "gamg_samples");
        else if (argument == "--gamg_csv") options.csv = true;
        else if (argument == "--gamg_flat_edge_only")
            options.flatEdgeOnly = true;
        else if (argument == "--gamg_production_compare_only")
            options.productionCompareOnly = true;
        else if (argument == "--gamg_fusion_compare_only")
            options.fusionCompareOnly = true;
        else if (argument == "--gamg_delta_candidate_only")
            options.deltaCandidateOnly = true;
        else if (argument == "--gamg_delta_candidate_solve_only")
            options.deltaCandidateSolveOnly = true;
        else if (argument == "--gamg_l1_eviction_compare_only")
            options.l1EvictionCompareOnly = true;
        else if
        (
            argument.rfind("--gamg_delta_candidate_solve_variant=", 0) == 0
        )
            options.deltaCandidateSolveVariant = argument.substr(37);
        else if (argument.rfind("--gamg_vtune_profile=", 0) == 0)
            options.vtuneProfileVariant = argument.substr(21);
        else
        {
            if (argument.rfind("--benchmark_repetitions=", 0) == 0)
                options.googleRepetitionsSpecified = true;
            argv[destination++] = argv[source];
        }
    }
    argc = destination;
    if
    (
        options.deltaCandidateSolveVariant != "both"
     && options.deltaCandidateSolveVariant != "direct"
     && options.deltaCandidateSolveVariant != "cleaned"
    ) throw std::runtime_error
    (
        "gamg_delta_candidate_solve_variant must be direct, cleaned, or both"
    );
    if
    (
        !options.vtuneProfileVariant.empty()
     && options.vtuneProfileVariant != "reference"
     && options.vtuneProfileVariant != "native-gs-direct"
     && options.vtuneProfileVariant != "native-gs-direct-evict"
     && options.vtuneProfileVariant != "direct"
     && options.vtuneProfileVariant != "direct-evict"
    ) throw std::runtime_error
    (
        "gamg_vtune_profile must be reference, native-gs-direct, "
        "native-gs-direct-evict, direct, or direct-evict"
    );
    return options;
}

gamg::GAMGControls controls()
{
    gamg::GAMGControls result;
    result.relativeTolerance = 0;
    result.tolerance = 1e-8;
    result.maxIterations = 100;
    result.diagnostics = false;
    return result;
}

fwgamg::Controls fwControls()
{
    fwgamg::Controls result;
    result.gamg = controls();
    result.width = 1024;
    return result;
}

fwgamg::Controls flatEdgeSpmvControls()
{
    fwgamg::Controls result = fwControls();
    result.spmvTraversal = fwgamg::SpmvTraversal::FlatEdgeWise;
    return result;
}

fwgamg::Controls rowSpmvControls()
{
    fwgamg::Controls result = fwControls();
    result.spmvTraversal = fwgamg::SpmvTraversal::RowWise;
    return result;
}

fwgamg::Controls csrDerivedDirectControls()
{
    fwgamg::Controls result = fwControls();
    result.spmvTraversal = fwgamg::SpmvTraversal::DirectFlatEdgeWise;
    return result;
}

fwgamg::Controls evictedDirectFlatControls()
{
    fwgamg::Controls result = csrDerivedDirectControls();
    result.evictL1BeforeDirectFlat = true;
    return result;
}

fwgamg::Controls fusedSpmvControls()
{
    fwgamg::Controls result = fwControls();
    result.finestSpmvFusion = fwgamg::FinestSpmvFusion::IncomingHeld;
    return result;
}

fwgamg::Controls deltaFusedSpmvControls()
{
    fwgamg::Controls result = fwControls();
    result.finestSpmvFusion = fwgamg::FinestSpmvFusion::DeltaCorrection;
    return result;
}

fwgamg::Controls currentDeltaFusedSpmvControls()
{
    fwgamg::Controls result = fwControls();
    result.finestSpmvFusion = fwgamg::FinestSpmvFusion::DeltaCorrectionCurrent;
    return result;
}

fwgamg::Controls edgeRowsDeltaFusedSpmvControls()
{
    fwgamg::Controls result = fwControls();
    result.finestSpmvFusion = fwgamg::FinestSpmvFusion::DeltaCorrectionEdgeRows;
    return result;
}

fwgamg::Controls packedBaselineDeltaFusedSpmvControls()
{
    fwgamg::Controls result = fwControls();
    result.finestSpmvFusion =
        fwgamg::FinestSpmvFusion::DeltaCorrectionPackedBaseline;
    return result;
}

fwgamg::Controls experimentalControls
(
    const fwgamg::RowOrdering ordering,
    const fwgamg::CoarseRenumbering renumbering
)
{
    fwgamg::Controls result = fwControls();
    result.rowOrdering = ordering;
    result.coarseRenumbering = renumbering;
    return result;
}

struct Problem
{
    smootherTest::PolyMeshTopology mesh;
    Foam::lduMatrix matrix;
    Foam::scalarField exact;
    Foam::scalarField source;

    explicit Problem(const std::string& meshDirectory)
    :
        mesh(smootherTest::PolyMeshReader::read(meshDirectory)),
        matrix(harness::makeMatrix(mesh)),
        exact(mesh.nCells),
        source()
    {
        const Foam::scalar pi = std::acos(-1.0);
        for (Foam::label cell=0; cell<static_cast<Foam::label>(mesh.nCells); ++cell)
        {
            const Foam::scalar position = Foam::scalar(cell)
                /Foam::scalar(mesh.nCells - 1);
            exact[cell] = 1.0 + 0.25*std::sin(2.0*pi*position)
                + 0.1*std::cos(10.0*pi*position);
        }
        source = harness::multiply(matrix, exact);
    }
};

void runVtuneProfile(const Problem& problem, const Options& options)
{
#ifndef HAVE_ITTNOTIFY
    static_cast<void>(problem);
    static_cast<void>(options);
    throw std::runtime_error("this build has no VTune ITT support");
#else
    // VTune is launched with -start-paused. Hierarchy construction, warm-up,
    // and resetting psi all remain outside the event-based sampling ROI.
    __itt_pause();

    if (options.vtuneProfileVariant == "reference")
    {
        gamg::GAMGSolver solver(problem.matrix, controls());
        gamg::GAMGSolverPerformance result;
        for (Foam::label warmup=0; warmup<options.warmup; ++warmup)
        {
            Foam::scalarField psi(problem.mesh.nCells, 0.0);
            solver.solve(psi, problem.source);
        }
        Foam::scalarField psi(problem.mesh.nCells, 0.0);
        for (Foam::label sample=0; sample<options.samples; ++sample)
        {
            psi.assign(problem.mesh.nCells, 0.0);
            __itt_resume();
            result = solver.solve(psi, problem.source);
            __itt_pause();
            benchmark::DoNotOptimize(psi.data());
        }
        std::cout << "VTune GAMG ROI variant=reference"
            << " samples=" << options.samples
            << " warmup=" << options.warmup
            << " vcycles=" << result.iterations
            << " finalResidual=" << std::scientific << result.finalResidual
            << std::defaultfloat << '\n';
    }
    else if
    (
        options.vtuneProfileVariant == "native-gs-direct"
     || options.vtuneProfileVariant == "native-gs-direct-evict"
    )
    {
        const bool evict =
            options.vtuneProfileVariant == "native-gs-direct-evict";
        gamg::GAMGSolver solver
        (
            problem.matrix,
            controls(),
            nullptr,
            gamg::GAMGSpmvTraversal::DirectFlat,
            evict
        );
        gamg::GAMGSolverPerformance result;
        for (Foam::label warmup=0; warmup<options.warmup; ++warmup)
        {
            Foam::scalarField psi(problem.mesh.nCells, 0.0);
            solver.solve(psi, problem.source);
        }
        Foam::scalarField psi(problem.mesh.nCells, 0.0);
        for (Foam::label sample=0; sample<options.samples; ++sample)
        {
            psi.assign(problem.mesh.nCells, 0.0);
            __itt_resume();
            result = solver.solve(psi, problem.source);
            __itt_pause();
            benchmark::DoNotOptimize(psi.data());
        }
        std::cout << "VTune GAMG ROI variant="
            << options.vtuneProfileVariant
            << " samples=" << options.samples
            << " warmup=" << options.warmup
            << " vcycles=" << result.iterations
            << " finalResidual=" << std::scientific << result.finalResidual
            << std::defaultfloat << '\n';
    }
    else
    {
        fwgamg::Controls profileControls = fwControls();
        if (options.vtuneProfileVariant == "direct-evict")
            profileControls.evictL1BeforeDirectFlat = true;
        fwgamg::Solver solver(problem.matrix, profileControls);
        fwgamg::Performance result;
        for (Foam::label warmup=0; warmup<options.warmup; ++warmup)
        {
            Foam::scalarField psi(problem.mesh.nCells, 0.0);
            solver.solve(psi, problem.source);
        }
        Foam::scalarField psi(problem.mesh.nCells, 0.0);
        for (Foam::label sample=0; sample<options.samples; ++sample)
        {
            psi.assign(problem.mesh.nCells, 0.0);
            __itt_resume();
            result = solver.solve(psi, problem.source);
            __itt_pause();
            benchmark::DoNotOptimize(psi.data());
        }
        std::cout << "VTune GAMG ROI variant="
            << options.vtuneProfileVariant
            << " samples=" << options.samples
            << " warmup=" << options.warmup
            << " width=1024"
            << " vcycles=" << result.iterations
            << " finalResidual=" << std::scientific << result.finalResidual
            << std::defaultfloat << '\n';
    }
#endif
}

void runTimingSummary(const Problem& problem, const Options& options)
{
    for (Foam::label warmup=0; warmup<options.warmup; ++warmup)
    {
        gamg::GAMGSolver setupWarmup(problem.matrix, controls());
        benchmark::DoNotOptimize(&setupWarmup);
    }
    std::vector<gamg::GAMGTimingStats> setupSamples;
    setupSamples.reserve(options.samples);
    for (Foam::label sample=0; sample<options.samples; ++sample)
    {
        gamg::GAMGTimingStats setupTiming;
        gamg::GAMGSolver setupSolver
        (
            problem.matrix, controls(), &setupTiming
        );
        benchmark::DoNotOptimize(&setupSolver);
        setupSamples.push_back(std::move(setupTiming));
    }
    std::sort
    (
        setupSamples.begin(),
        setupSamples.end(),
        [](const auto& left, const auto& right)
        {
            return left.setupWallSeconds < right.setupWallSeconds;
        }
    );
    const gamg::GAMGTimingStats& representativeSetup =
        setupSamples[setupSamples.size()/2];
    double setupMean = 0;
    for (const auto& sample : setupSamples)
        setupMean += sample.setupWallSeconds;
    setupMean /= setupSamples.size();
    double setupVariance = 0;
    for (const auto& sample : setupSamples)
        setupVariance += (sample.setupWallSeconds - setupMean)
            *(sample.setupWallSeconds - setupMean);
    setupVariance /= setupSamples.size();

    gamg::GAMGTimingStats timing;
    gamg::GAMGSolver solver(problem.matrix, controls(), &timing);
    for (Foam::label warmup=0; warmup<options.warmup; ++warmup)
    {
        Foam::scalarField psi(problem.mesh.nCells, 0.0);
        solver.solve(psi, problem.source);
    }
    timing.resetRuntime();
    timing.setupWallSeconds = representativeSetup.setupWallSeconds;
    timing.hierarchyBuildSeconds = representativeSetup.hierarchyBuildSeconds;
    timing.coarseMatrixBuildSeconds =
        representativeSetup.coarseMatrixBuildSeconds;
    for (std::size_t level=0; level<timing.transitions.size(); ++level)
        timing.transitions[level].coarseMatrixBuild =
            representativeSetup.transitions[level].coarseMatrixBuild;

    gamg::GAMGSolverPerformance last;
    for (Foam::label sample=0; sample<options.samples; ++sample)
    {
        Foam::scalarField psi(problem.mesh.nCells, 0.0);
        last = solver.solve(psi, problem.source);
        benchmark::DoNotOptimize(psi.data());
    }
    std::cout << std::fixed << std::setprecision(6)
        << "GAMG setup samples: count=" << setupSamples.size()
        << " min=" << 1e3*setupSamples.front().setupWallSeconds
        << " ms median=" << 1e3*representativeSetup.setupWallSeconds
        << " ms mean=" << 1e3*setupMean
        << " ms stddev=" << 1e3*std::sqrt(setupVariance)
        << " ms CV=" << (setupMean > 0
            ? 100.0*std::sqrt(setupVariance)/setupMean : 0.0) << "%\n";
    timing.print(std::cout);
    if (options.csv)
    {
        std::cout << "\nGAMG timing CSV\n";
        timing.printCsv(std::cout);
    }
    std::cout << "Timing workload: samples=" << options.samples
        << " warmup=" << options.warmup
        << " iterations/solve=" << last.iterations
        << " finalResidual=" << std::scientific << last.finalResidual
        << std::defaultfloat << "\n\n";
}

struct L1EvictionResult
{
    std::string path;
    std::vector<double> spmvSeconds;
    double cellVisits = 0;
    double faceVisits = 0;
    Foam::label iterations = 0;
    Foam::scalar finalResidual = 0;
    Foam::scalarField finalPsi;
    std::vector<Foam::scalar> residualHistory;
};

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size()/2];
}

double coefficientOfVariation(const std::vector<double>& values)
{
    const double mean = std::accumulate(values.begin(), values.end(), 0.0)
        /static_cast<double>(values.size());
    double variance = 0;
    for (const double value : values)
        variance += (value - mean)*(value - mean);
    variance /= static_cast<double>(values.size());
    return mean > 0 ? std::sqrt(variance)/mean : 0;
}

bool exactlyEqual
(
    const Foam::scalarField& left,
    const Foam::scalarField& right
)
{
    return left.size() == right.size()
        && std::equal(left.begin(), left.end(), right.begin());
}

void runL1EvictionComparison(const Problem& problem, const Options& options)
{
    gamg::GAMGTimingStats gsTiming;
    gamg::GAMGTimingStats gsEvictedTiming;
    fwgamg::TimingStats ih1Timing;
    fwgamg::TimingStats ih1EvictedTiming;
    gamg::GAMGSolver gs
    (
        problem.matrix, controls(), &gsTiming,
        gamg::GAMGSpmvTraversal::DirectFlat, false
    );
    gamg::GAMGSolver gsEvicted
    (
        problem.matrix, controls(), &gsEvictedTiming,
        gamg::GAMGSpmvTraversal::DirectFlat, true
    );
    fwgamg::Solver ih1(problem.matrix, fwControls(), &ih1Timing);
    fwgamg::Solver ih1Evicted
    (
        problem.matrix, evictedDirectFlatControls(), &ih1EvictedTiming
    );

    for (Foam::label warmup=0; warmup<options.warmup; ++warmup)
    {
        for (int variant=0; variant<4; ++variant)
        {
            Foam::scalarField psi(problem.mesh.nCells, 0.0);
            if (variant == 0) gs.solve(psi, problem.source);
            else if (variant == 1) gsEvicted.solve(psi, problem.source);
            else if (variant == 2) ih1.solve(psi, problem.source);
            else ih1Evicted.solve(psi, problem.source);
            benchmark::DoNotOptimize(psi.data());
        }
    }

    std::array<L1EvictionResult, 4> results{};
    results[0].path = "GS -> Direct-flat";
    results[1].path = "GS -> eviction -> Direct-flat";
    results[2].path = "IH1 -> Direct-flat";
    results[3].path = "IH1 -> eviction -> Direct-flat";
    std::mt19937 random(0x1a2b3c4dU);
    for (Foam::label sample=0; sample<options.samples; ++sample)
    {
        std::array<int, 4> order{{0, 1, 2, 3}};
        std::shuffle(order.begin(), order.end(), random);
        for (const int variant : order)
        {
            Foam::scalarField psi(problem.mesh.nCells, 0.0);
            if (variant < 2)
            {
                gamg::GAMGTimingStats& timing = variant == 0
                    ? gsTiming : gsEvictedTiming;
                timing.resetRuntime();
                const gamg::GAMGSolverPerformance performance = variant == 0
                    ? gs.solve(psi, problem.source)
                    : gsEvicted.solve(psi, problem.source);
                double seconds = 0;
                double cells = 0;
                double faces = 0;
                for (const gamg::GAMGLevelTiming& level : timing.levels)
                {
                    seconds += level.amul.seconds;
                    cells += static_cast<double>(level.cells)*level.amul.calls;
                    faces += static_cast<double>(level.faces)*level.amul.calls;
                }
                results[variant].spmvSeconds.push_back(seconds);
                results[variant].cellVisits = cells;
                results[variant].faceVisits = faces;
                results[variant].iterations = performance.iterations;
                results[variant].finalResidual = performance.finalResidual;
                results[variant].residualHistory = performance.residualHistory;
            }
            else
            {
                fwgamg::TimingStats& timing = variant == 2
                    ? ih1Timing : ih1EvictedTiming;
                timing.resetRuntime();
                const fwgamg::Performance performance = variant == 2
                    ? ih1.solve(psi, problem.source)
                    : ih1Evicted.solve(psi, problem.source);
                double seconds = 0;
                double cells = 0;
                double faces = 0;
                for (const fwgamg::LevelSolveTiming& level : timing.levels)
                {
                    seconds += level.spmv.seconds;
                    cells += static_cast<double>(level.cells)*level.spmv.calls;
                    faces += static_cast<double>(level.faces)*level.spmv.calls;
                }
                results[variant].spmvSeconds.push_back(seconds);
                results[variant].cellVisits = cells;
                results[variant].faceVisits = faces;
                results[variant].iterations = performance.iterations;
                results[variant].finalResidual = performance.finalResidual;
                results[variant].residualHistory = performance.residualHistory;
            }
            results[variant].finalPsi = std::move(psi);
            benchmark::DoNotOptimize(results[variant].finalPsi.data());
        }
    }

    if
    (
        !exactlyEqual(results[0].finalPsi, results[1].finalPsi)
     || results[0].residualHistory != results[1].residualHistory
     || !exactlyEqual(results[2].finalPsi, results[3].finalPsi)
     || results[2].residualHistory != results[3].residualHistory
    ) throw std::runtime_error
    (
        "L1 eviction changed a GAMG numerical result"
    );

    std::array<double, 4> medians{};
    std::cout << "\nL1 eviction Direct-flat comparison\n"
        << "Path\tmedian_ms\tCV_pct\tns_per_cell\t"
           "relative_to_non_evicted\tvcycles\tfinalResidual\n";
    for (std::size_t variant=0; variant<results.size(); ++variant)
    {
        medians[variant] = median(results[variant].spmvSeconds);
        const double baseline = variant == 1 ? medians[0]
            : variant == 3 ? medians[2] : medians[variant];
        std::cout << results[variant].path << '\t'
            << std::fixed << std::setprecision(6)
            << 1e3*medians[variant] << '\t'
            << 100.0*coefficientOfVariation(results[variant].spmvSeconds)
            << '\t' << 1e9*medians[variant]/results[variant].cellVisits
            << '\t' << medians[variant]/baseline
            << '\t' << results[variant].iterations
            << '\t' << std::scientific << results[variant].finalResidual
            << std::defaultfloat << '\n';
    }
    std::cout << "GS eviction delta ms="
        << 1e3*(medians[1] - medians[0])
        << "\nIH1 eviction delta ms="
        << 1e3*(medians[3] - medians[2])
        << "\nEvicted IH1/GS Direct-flat ratio="
        << medians[3]/medians[1]
        << "\nExact GS pair=PASS\nExact IH1 pair=PASS\n\n";
}

void runFwTimingSummary
(
    const Problem& problem,
    const Options& options,
    const fwgamg::Controls& solverControls,
    const char* const variant
)
{
    for (Foam::label warmup=0; warmup<options.warmup; ++warmup)
    {
        fwgamg::Solver setupWarmup(problem.matrix, solverControls);
        benchmark::DoNotOptimize(&setupWarmup);
    }
    std::vector<fwgamg::TimingStats> setupSamples;
    setupSamples.reserve(options.samples);
    for (Foam::label sample=0; sample<options.samples; ++sample)
    {
        fwgamg::TimingStats timing;
        fwgamg::Solver solver(problem.matrix, solverControls, &timing);
        benchmark::DoNotOptimize(&solver);
        setupSamples.push_back(std::move(timing));
    }
    std::sort
    (
        setupSamples.begin(), setupSamples.end(),
        [](const auto& left, const auto& right)
        {
            return left.setupWallSeconds < right.setupWallSeconds;
        }
    );
    const fwgamg::TimingStats& setup = setupSamples[setupSamples.size()/2];

    fwgamg::TimingStats timing;
    fwgamg::Solver solver(problem.matrix, solverControls, &timing);
    for (Foam::label warmup=0; warmup<options.warmup; ++warmup)
    {
        Foam::scalarField psi(problem.mesh.nCells, 0.0);
        solver.solve(psi, problem.source);
    }
    timing.resetRuntime();
    timing.setupWallSeconds = setup.setupWallSeconds;
    timing.lduBridgeSeconds = setup.lduBridgeSeconds;
    timing.agglomerationSeconds = setup.agglomerationSeconds;
    timing.coarseLduConstructionSeconds = setup.coarseLduConstructionSeconds;
    timing.coarseFwPackingSeconds = setup.coarseFwPackingSeconds;
    timing.coarseConversionSeconds = setup.coarseConversionSeconds;
    timing.levelSetup = setup.levelSetup;

    fwgamg::Performance last;
    for (Foam::label sample=0; sample<options.samples; ++sample)
    {
        Foam::scalarField psi(problem.mesh.nCells, 0.0);
        last = solver.solve(psi, problem.source);
        benchmark::DoNotOptimize(psi.data());
    }
    std::cout << "\nfwGAMG timing variant=" << variant << '\n';
    timing.print(std::cout);
    std::cout << "fwGAMG workload variant=" << variant
        << " samples=" << options.samples
        << " warmup=" << options.warmup
        << " width=1024 iterations/solve=" << last.iterations
        << " finalResidual=" << std::scientific << last.finalResidual
        << std::defaultfloat << "\n\n";
}

void runFwEdgeTimingSummary(const Problem& problem, const Options& options)
{
    fwgamg::TimingStats timing;
    fwgamg::Solver solver(problem.matrix, flatEdgeSpmvControls(), &timing);
    for (Foam::label warmup=0; warmup<options.warmup; ++warmup)
    {
        Foam::scalarField psi(problem.mesh.nCells, 0.0);
        solver.solve(psi, problem.source);
    }
    timing.resetRuntime();
    fwgamg::Performance last;
    for (Foam::label sample=0; sample<options.samples; ++sample)
    {
        Foam::scalarField psi(problem.mesh.nCells, 0.0);
        last = solver.solve(psi, problem.source);
        benchmark::DoNotOptimize(psi.data());
    }
    std::cout << "\nfwGAMG flat edge-wise SpMV timing\n";
    timing.print(std::cout);
    std::cout << "fwGAMG flat edge-wise workload: samples=" << options.samples
        << " warmup=" << options.warmup
        << " iterations/solve=" << last.iterations
        << " finalResidual=" << std::scientific << last.finalResidual
        << std::defaultfloat << "\n\n";
}

void registerFlatEdgeComparison
(
    const Problem& problem,
    const Options& options
)
{
    const auto registerVariant =
    [&problem, &options]
    (
        const std::string& name,
        const fwgamg::Controls controls
    )
    {
        auto* setup = benchmark::RegisterBenchmark
        (
            ("fwGAMG/MotorBike/" + name + "/Setup").c_str(),
            [&problem, controls](benchmark::State& state)
            {
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    fwgamg::Solver solver(problem.matrix, controls);
                    benchmark::DoNotOptimize(&solver);
                }
            }
        )->ReportAggregatesOnly(true);
        auto solver = std::make_shared<fwgamg::Solver>
        (
            problem.matrix, controls
        );
        auto* solve = benchmark::RegisterBenchmark
        (
            ("fwGAMG/MotorBike/" + name + "/Solve").c_str(),
            [&problem, solver, warmupCount=options.warmup]
            (benchmark::State& state)
            {
                for (Foam::label warmup=0; warmup<warmupCount; ++warmup)
                {
                    Foam::scalarField psi(problem.mesh.nCells, 0.0);
                    solver->solve(psi, problem.source);
                }
                Foam::scalarField psi(problem.mesh.nCells, 0.0);
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    state.PauseTiming();
                    psi.assign(problem.mesh.nCells, 0.0);
                    state.ResumeTiming();
                    const fwgamg::Performance result =
                        solver->solve(psi, problem.source);
                    benchmark::DoNotOptimize(psi.data());
                    state.counters["vcycles"] = result.iterations;
                }
                state.counters["cells"] = problem.mesh.nCells;
            }
        )->Iterations(1)->ReportAggregatesOnly(true);
        if (!options.googleRepetitionsSpecified)
        {
            setup->Repetitions(options.samples);
            solve->Repetitions(options.samples);
        }
    };
    registerVariant("RowSpMV", rowSpmvControls());
    registerVariant("FlatEdgeSpMV", flatEdgeSpmvControls());
}

void registerProductionComparison
(
    const Problem& problem,
    const Options& options
)
{
    const auto configure = [&options](auto* const registered)
    {
        registered->Iterations(1)->ReportAggregatesOnly(true);
        if (!options.googleRepetitionsSpecified)
            registered->Repetitions(options.samples);
    };
    configure
    (
        benchmark::RegisterBenchmark
        (
            "GAMG/Production/ReferenceGS/Setup",
            [&problem](benchmark::State& state)
            {
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    gamg::GAMGSolver solver(problem.matrix, controls());
                    benchmark::DoNotOptimize(&solver);
                }
            }
        )
    );

    auto reference = std::make_shared<gamg::GAMGSolver>
    (
        problem.matrix, controls()
    );
    configure
    (
        benchmark::RegisterBenchmark
        (
            "GAMG/Production/ReferenceGS/Solve",
            [&problem, reference, warmupCount=options.warmup]
            (benchmark::State& state)
            {
                for (Foam::label warmup=0; warmup<warmupCount; ++warmup)
                {
                    Foam::scalarField psi(problem.mesh.nCells, 0.0);
                    reference->solve(psi, problem.source);
                }
                Foam::scalarField psi(problem.mesh.nCells, 0.0);
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    state.PauseTiming();
                    psi.assign(problem.mesh.nCells, 0.0);
                    state.ResumeTiming();
                    const auto result = reference->solve(psi, problem.source);
                    benchmark::DoNotOptimize(psi.data());
                    state.counters["vcycles"] = result.iterations;
                    state.counters["finalResidual"] = result.finalResidual;
                }
            }
        )
    );

    const auto registerFw =
    [&problem, &options, &configure]
    (
        const std::string& name,
        const fwgamg::Controls variantControls
    )
    {
        configure
        (
            benchmark::RegisterBenchmark
            (
                ("fwGAMG/Production/" + name + "/Setup").c_str(),
                [&problem, variantControls](benchmark::State& state)
                {
                    for (auto unused : state)
                    {
                        static_cast<void>(unused);
                        fwgamg::Solver solver(problem.matrix, variantControls);
                        benchmark::DoNotOptimize(&solver);
                    }
                }
            )
        );
        auto solver = std::make_shared<fwgamg::Solver>
        (
            problem.matrix, variantControls
        );
        configure
        (
            benchmark::RegisterBenchmark
            (
                ("fwGAMG/Production/" + name + "/Solve").c_str(),
                [&problem, solver, warmupCount=options.warmup]
                (benchmark::State& state)
                {
                    for (Foam::label warmup=0; warmup<warmupCount; ++warmup)
                    {
                        Foam::scalarField psi(problem.mesh.nCells, 0.0);
                        solver->solve(psi, problem.source);
                    }
                    Foam::scalarField psi(problem.mesh.nCells, 0.0);
                    for (auto unused : state)
                    {
                        static_cast<void>(unused);
                        state.PauseTiming();
                        psi.assign(problem.mesh.nCells, 0.0);
                        state.ResumeTiming();
                        const auto result = solver->solve(psi, problem.source);
                        benchmark::DoNotOptimize(psi.data());
                        state.counters["vcycles"] = result.iterations;
                        state.counters["finalResidual"] = result.finalResidual;
                    }
                }
            )
        );
    };
    registerFw("RowCSR", rowSpmvControls());
    registerFw("CSRDerivedDirectFlat", csrDerivedDirectControls());
    registerFw("DirectFlat", fwControls());
    registerFw("FusedFinestIH1SpMV", fusedSpmvControls());
    registerFw("FusedDeltaIH1SpMV", deltaFusedSpmvControls());
}

void runFwRuntimeSummary
(
    const Problem& problem,
    const Options& options,
    const fwgamg::Controls& solverControls,
    const char* const variant
)
{
    fwgamg::TimingStats timing;
    fwgamg::Solver solver(problem.matrix, solverControls, &timing);
    for (Foam::label warmup=0; warmup<options.warmup; ++warmup)
    {
        Foam::scalarField psi(problem.mesh.nCells, 0.0);
        solver.solve(psi, problem.source);
    }
    timing.resetRuntime();
    fwgamg::Performance last;
    for (Foam::label sample=0; sample<options.samples; ++sample)
    {
        Foam::scalarField psi(problem.mesh.nCells, 0.0);
        last = solver.solve(psi, problem.source);
        benchmark::DoNotOptimize(psi.data());
    }
    std::cout << "\nfwGAMG fusion runtime variant=" << variant << '\n';
    timing.print(std::cout);
    std::cout << "fwGAMG fusion workload variant=" << variant
        << " samples=" << options.samples
        << " warmup=" << options.warmup
        << " width=1024 iterations/solve=" << last.iterations
        << " finalResidual=" << std::scientific << last.finalResidual
        << std::defaultfloat << "\n\n";
}

void registerFusionComparison
(
    const Problem& problem,
    const Options& options
)
{
    const auto registerVariant = [&problem, &options]
    (
        const std::string& name,
        const fwgamg::Controls variantControls
    )
    {
        auto solver = std::make_shared<fwgamg::Solver>
        (
            problem.matrix, variantControls
        );
        auto* registered = benchmark::RegisterBenchmark
        (
            ("fwGAMG/Fusion/" + name + "/Solve").c_str(),
            [&problem, solver, warmupCount=options.warmup]
            (benchmark::State& state)
            {
                for (Foam::label warmup=0; warmup<warmupCount; ++warmup)
                {
                    Foam::scalarField psi(problem.mesh.nCells, 0.0);
                    solver->solve(psi, problem.source);
                }
                Foam::scalarField psi(problem.mesh.nCells, 0.0);
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    state.PauseTiming();
                    psi.assign(problem.mesh.nCells, 0.0);
                    state.ResumeTiming();
                    const auto result = solver->solve(psi, problem.source);
                    benchmark::DoNotOptimize(psi.data());
                    state.counters["vcycles"] = result.iterations;
                    state.counters["finalResidual"] = result.finalResidual;
                }
            }
        )->Iterations(1)->ReportAggregatesOnly(true);
        if (!options.googleRepetitionsSpecified)
            registered->Repetitions(options.samples);
    };

    if (options.deltaCandidateOnly)
    {
        // The focused pair/instrumentation path uses one shared hierarchy
        // registered below and intentionally avoids full-solve solver copies.
    }
    else if (options.deltaCandidateSolveOnly)
    {
        if
        (
            options.deltaCandidateSolveVariant == "both"
         || options.deltaCandidateSolveVariant == "direct"
        )
            registerVariant("DirectFlat", fwControls());
        if
        (
            options.deltaCandidateSolveVariant == "both"
         || options.deltaCandidateSolveVariant == "cleaned"
        )
            registerVariant("DeltaCleaned", deltaFusedSpmvControls());
    }
    else
    {
        registerVariant("RowCSR", rowSpmvControls());
        registerVariant("DirectFlat", fwControls());
        registerVariant("FusedFinestIH1SpMV", fusedSpmvControls());
        registerVariant("DeltaCurrent", currentDeltaFusedSpmvControls());
        registerVariant("DeltaEdgeRowStarts", edgeRowsDeltaFusedSpmvControls());
        registerVariant("DeltaPackedCSR", packedBaselineDeltaFusedSpmvControls());
        registerVariant("DeltaCleaned", deltaFusedSpmvControls());
    }
    if (options.deltaCandidateSolveOnly) return;

    auto finestSolver = std::make_shared<fwgamg::Solver>
    (
        problem.matrix, fwControls()
    );
    auto finestSource = std::make_shared<Foam::scalarField>
    (
        finestSolver->toFw(problem.source)
    );
    const auto registerPair = [&options, finestSolver, finestSource]
    (
        const std::string& name,
        const fwgamg::FinestSpmvFusion fusion
    )
    {
        auto* registered = benchmark::RegisterBenchmark
        (
            ("fwGAMG/Fusion/" + name + "/FinestPair").c_str(),
            [finestSolver, finestSource, fusion](benchmark::State& state)
            {
                const fwgamg::Matrix& matrix = finestSolver->hierarchy()[0];
                Foam::scalarField psi(matrix.nCells(), 0.0);
                Foam::scalarField matrixPsi(matrix.nCells());
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    state.PauseTiming();
                    psi.assign(matrix.nCells(), 0.0);
                    state.ResumeTiming();
                    if (fusion == fwgamg::FinestSpmvFusion::None)
                    {
                        matrix.smooth(psi, *finestSource, 2, false);
                        matrix.multiplyDirectFlatEdgeWise(matrixPsi, psi);
                    }
                    else if
                    (
                        fusion == fwgamg::FinestSpmvFusion::IncomingHeld
                    )
                        matrix.smoothAndMultiply
                        (
                            psi, *finestSource, matrixPsi, 2, false
                        );
                    else if
                    (
                        fusion == fwgamg::FinestSpmvFusion::DeltaCorrection
                    )
                        matrix.smoothAndMultiplyDelta
                        (
                            psi, *finestSource, matrixPsi, 2, false
                        );
                    else if
                    (
                        fusion
                            == fwgamg::FinestSpmvFusion::DeltaCorrectionCurrent
                    )
                        matrix.smoothAndMultiplyDeltaCurrent
                        (
                            psi, *finestSource, matrixPsi, 2, false
                        );
                    else if
                    (
                        fusion ==
                            fwgamg::FinestSpmvFusion::DeltaCorrectionEdgeRows
                    )
                        matrix.smoothAndMultiplyDeltaEdgeRows
                        (
                            psi, *finestSource, matrixPsi, 2, false
                        );
                    else
                        matrix.smoothAndMultiplyDeltaPackedBaseline
                        (
                            psi, *finestSource, matrixPsi, 2, false
                        );
                    benchmark::DoNotOptimize(psi.data());
                    benchmark::DoNotOptimize(matrixPsi.data());
                }
                state.counters["cells"] = matrix.nCells();
                state.counters["faces"] = matrix.nFaces();
            }
        )->ReportAggregatesOnly(true);
        if (!options.googleRepetitionsSpecified)
            registered->Repetitions(options.samples);
    };
    registerPair("DirectFlat", fwgamg::FinestSpmvFusion::None);
    registerPair("DeltaCleaned", fwgamg::FinestSpmvFusion::DeltaCorrection);
    if (!options.deltaCandidateOnly)
    {
        registerPair
        (
            "FusedFinestIH1SpMV", fwgamg::FinestSpmvFusion::IncomingHeld
        );
        registerPair
        (
            "DeltaCurrent",
            fwgamg::FinestSpmvFusion::DeltaCorrectionCurrent
        );
        registerPair
        (
            "DeltaEdgeRowStarts",
            fwgamg::FinestSpmvFusion::DeltaCorrectionEdgeRows
        );
        registerPair
        (
            "DeltaPackedCSR",
            fwgamg::FinestSpmvFusion::DeltaCorrectionPackedBaseline
        );
    }

    auto* ih1Only = benchmark::RegisterBenchmark
    (
        "fwGAMG/Fusion/IH1Only/FinestPair",
        [finestSolver, finestSource](benchmark::State& state)
        {
            const fwgamg::Matrix& matrix = finestSolver->hierarchy()[0];
            Foam::scalarField psi(matrix.nCells(), 0.0);
            for (auto unused : state)
            {
                static_cast<void>(unused);
                state.PauseTiming();
                psi.assign(matrix.nCells(), 0.0);
                state.ResumeTiming();
                matrix.smooth(psi, *finestSource, 2, false);
                benchmark::DoNotOptimize(psi.data());
            }
        }
    )->ReportAggregatesOnly(true);
    if (!options.googleRepetitionsSpecified)
        ih1Only->Repetitions(options.samples);

    Foam::scalarField initialPsi(finestSolver->hierarchy()[0].nCells(), 0.0);
    Foam::scalarField postPsi(initialPsi);
    finestSolver->hierarchy()[0].smooth(postPsi, *finestSource, 1, false);
    auto delta = std::make_shared<Foam::scalarField>(postPsi.size());
    for (Foam::label row=0; row<static_cast<Foam::label>(postPsi.size()); ++row)
        (*delta)[row] = postPsi[row] - initialPsi[row];
    const auto registerCorrection =
        [&options, finestSolver, finestSource, delta]
        (const std::string& name, const int variant)
    {
        auto* registered = benchmark::RegisterBenchmark
        (
            ("fwGAMG/Fusion/" + name + "/CorrectionOnly").c_str(),
            [finestSolver, finestSource, delta, variant](benchmark::State& state)
            {
                const fwgamg::Matrix& matrix = finestSolver->hierarchy()[0];
                Foam::scalarField work(*delta);
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    state.PauseTiming();
                    work = *delta;
                    state.ResumeTiming();
                    if (variant == 0)
                        matrix.deltaCorrectionCurrent(*finestSource, work);
                    else if (variant == 1)
                        matrix.deltaCorrectionEdgeRows(*finestSource, work);
                    else
                        matrix.deltaCorrectionPacked(*finestSource, work);
                    benchmark::DoNotOptimize(work.data());
                }
            }
        )->ReportAggregatesOnly(true);
        if (!options.googleRepetitionsSpecified)
            registered->Repetitions(options.samples);
    };
    if (options.deltaCandidateOnly)
        registerCorrection("DeltaEdgeRowStarts", 1);
    else
    {
        registerCorrection("DeltaCurrent", 0);
        registerCorrection("DeltaEdgeRowStarts", 1);
        registerCorrection("DeltaPackedCSR", 2);
    }
}

void printLocalityRenumberSetup(const Problem& problem)
{
    for (const auto& variant : std::vector<std::pair<std::string, fwgamg::Controls>>
    {
        {
            "LocalNatural",
            experimentalControls
            (
                fwgamg::RowOrdering::Natural,
                fwgamg::CoarseRenumbering::MinimumFineIndex
            )
        },
        {
            "LocalIndexKahn",
            experimentalControls
            (
                fwgamg::RowOrdering::IndexKahn,
                fwgamg::CoarseRenumbering::MinimumFineIndex
            )
        }
    })
    {
        fwgamg::TimingStats timing;
        fwgamg::Solver solver(problem.matrix, variant.second, &timing);
        benchmark::DoNotOptimize(&solver);
        double total = 0;
        std::cout << "fwGAMG locality renumber setup variant="
            << variant.first << '\n';
        for (std::size_t level=1; level<timing.levelSetup.size(); ++level)
        {
            const double seconds =
                timing.levelSetup[level].localityRenumberSeconds;
            total += seconds;
            std::cout << "  L" << level << " cells="
                << timing.levelSetup[level].cells
                << " renumber=" << 1e3*seconds << " ms\n";
        }
        std::cout << "  total=" << 1e3*total << " ms\n";
    }
}

void registerBenchmarks(const Problem& problem, const Options& options)
{
    auto* setup = benchmark::RegisterBenchmark
    (
        "GAMG/MotorBike/Setup",
        [&problem](benchmark::State& state)
        {
            for (auto unused : state)
            {
                static_cast<void>(unused);
                gamg::GAMGSolver solver(problem.matrix, controls());
                benchmark::DoNotOptimize(&solver);
            }
        }
    )->Iterations(1)->ReportAggregatesOnly(true);

    auto cycleSolver = std::make_shared<gamg::GAMGSolver>
    (
        problem.matrix, controls()
    );
    auto* cycle = benchmark::RegisterBenchmark
    (
        "GAMG/MotorBike/Vcycle",
        [&problem, cycleSolver, warmupCount=options.warmup]
        (benchmark::State& state)
        {
            Foam::scalarField warmPsi(problem.mesh.nCells, 0.0);
            for (Foam::label warmup=0; warmup<warmupCount; ++warmup)
                cycleSolver->applyVCycle
                (
                    warmPsi, problem.source, problem.source
                );
            Foam::scalarField psi(problem.mesh.nCells, 0.0);
            Foam::scalarField residual(problem.source);
            for (auto unused : state)
            {
                static_cast<void>(unused);
                state.PauseTiming();
                psi.assign(problem.mesh.nCells, 0.0);
                residual = problem.source;
                state.ResumeTiming();
                cycleSolver->applyVCycle(psi, problem.source, residual);
                benchmark::DoNotOptimize(psi.data());
            }
            state.counters["cells"] = problem.mesh.nCells;
        }
    )->Iterations(1)->ReportAggregatesOnly(true);

    auto solveSolver = std::make_shared<gamg::GAMGSolver>
    (
        problem.matrix, controls()
    );
    auto* solve = benchmark::RegisterBenchmark
    (
        "GAMG/MotorBike/Solve",
        [&problem, solveSolver, warmupCount=options.warmup]
        (benchmark::State& state)
        {
            for (Foam::label warmup=0; warmup<warmupCount; ++warmup)
            {
                Foam::scalarField warmPsi(problem.mesh.nCells, 0.0);
                solveSolver->solve(warmPsi, problem.source);
            }
            Foam::scalarField psi(problem.mesh.nCells, 0.0);
            for (auto unused : state)
            {
                static_cast<void>(unused);
                state.PauseTiming();
                psi.assign(problem.mesh.nCells, 0.0);
                state.ResumeTiming();
                const gamg::GAMGSolverPerformance result =
                    solveSolver->solve(psi, problem.source);
                benchmark::DoNotOptimize(psi.data());
                state.counters["vcycles"] = result.iterations;
            }
            state.counters["cells"] = problem.mesh.nCells;
        }
    )->Iterations(1)->ReportAggregatesOnly(true);

    auto* fwSetup = benchmark::RegisterBenchmark
    (
        "fwGAMG/MotorBike/Setup",
        [&problem](benchmark::State& state)
        {
            for (auto unused : state)
            {
                static_cast<void>(unused);
                fwgamg::Solver solver(problem.matrix, fwControls());
                benchmark::DoNotOptimize(&solver);
            }
        }
    )->Iterations(1)->ReportAggregatesOnly(true);

    auto fwCycleSolver = std::make_shared<fwgamg::Solver>
    (
        problem.matrix, fwControls()
    );
    auto* fwCycle = benchmark::RegisterBenchmark
    (
        "fwGAMG/MotorBike/Vcycle",
        [&problem, fwCycleSolver, warmupCount=options.warmup]
        (benchmark::State& state)
        {
            Foam::scalarField warmPsi(problem.mesh.nCells, 0.0);
            for (Foam::label warmup=0; warmup<warmupCount; ++warmup)
                fwCycleSolver->applyVCycle
                (
                    warmPsi, problem.source, problem.source
                );
            Foam::scalarField psi(problem.mesh.nCells, 0.0);
            Foam::scalarField residual(problem.source);
            for (auto unused : state)
            {
                static_cast<void>(unused);
                state.PauseTiming();
                psi.assign(problem.mesh.nCells, 0.0);
                residual = problem.source;
                state.ResumeTiming();
                fwCycleSolver->applyVCycle(psi, problem.source, residual);
                benchmark::DoNotOptimize(psi.data());
            }
            state.counters["cells"] = problem.mesh.nCells;
        }
    )->Iterations(1)->ReportAggregatesOnly(true);

    auto fwSolveSolver = std::make_shared<fwgamg::Solver>
    (
        problem.matrix, fwControls()
    );
    auto* fwSolve = benchmark::RegisterBenchmark
    (
        "fwGAMG/MotorBike/Solve",
        [&problem, fwSolveSolver, warmupCount=options.warmup]
        (benchmark::State& state)
        {
            for (Foam::label warmup=0; warmup<warmupCount; ++warmup)
            {
                Foam::scalarField warmPsi(problem.mesh.nCells, 0.0);
                fwSolveSolver->solve(warmPsi, problem.source);
            }
            Foam::scalarField psi(problem.mesh.nCells, 0.0);
            for (auto unused : state)
            {
                static_cast<void>(unused);
                state.PauseTiming();
                psi.assign(problem.mesh.nCells, 0.0);
                state.ResumeTiming();
                const fwgamg::Performance result =
                    fwSolveSolver->solve(psi, problem.source);
                benchmark::DoNotOptimize(psi.data());
                state.counters["vcycles"] = result.iterations;
            }
            state.counters["cells"] = problem.mesh.nCells;
        }
    )->Iterations(1)->ReportAggregatesOnly(true);

    const auto registerExperimental =
    [&problem, &options]
    (
        const std::string& name,
        const fwgamg::Controls variantControls
    )
    {
        auto* setupBenchmark = benchmark::RegisterBenchmark
            (
                ("fwGAMG/MotorBike/" + name + "/Setup").c_str(),
                [&problem, variantControls](benchmark::State& state)
                {
                    for (auto unused : state)
                    {
                        static_cast<void>(unused);
                        fwgamg::Solver solver
                        (
                            problem.matrix, variantControls
                        );
                        benchmark::DoNotOptimize(&solver);
                    }
                }
            )->Iterations(1)->ReportAggregatesOnly(true);
        auto solver = std::make_shared<fwgamg::Solver>
        (
            problem.matrix, variantControls
        );
        auto* solveBenchmark = benchmark::RegisterBenchmark
            (
                ("fwGAMG/MotorBike/" + name + "/Solve").c_str(),
                [&problem, solver, warmupCount=options.warmup]
                (benchmark::State& state)
                {
                    for (Foam::label warmup=0; warmup<warmupCount; ++warmup)
                    {
                        Foam::scalarField warmPsi(problem.mesh.nCells, 0.0);
                        solver->solve(warmPsi, problem.source);
                    }
                    Foam::scalarField psi(problem.mesh.nCells, 0.0);
                    for (auto unused : state)
                    {
                        static_cast<void>(unused);
                        state.PauseTiming();
                        psi.assign(problem.mesh.nCells, 0.0);
                        state.ResumeTiming();
                        const fwgamg::Performance result =
                            solver->solve(psi, problem.source);
                        benchmark::DoNotOptimize(psi.data());
                        state.counters["vcycles"] = result.iterations;
                    }
                    state.counters["cells"] = problem.mesh.nCells;
                }
            )->Iterations(1)->ReportAggregatesOnly(true);
        if (!options.googleRepetitionsSpecified)
        {
            setupBenchmark->Repetitions(options.samples);
            solveBenchmark->Repetitions(options.samples);
        }
    };
    registerExperimental
    (
        "FlatEdgeSpMV",
        flatEdgeSpmvControls()
    );
    registerExperimental
    (
        "Natural",
        experimentalControls
        (
            fwgamg::RowOrdering::Natural,
            fwgamg::CoarseRenumbering::None
        )
    );
    registerExperimental
    (
        "LocalNatural",
        experimentalControls
        (
            fwgamg::RowOrdering::Natural,
            fwgamg::CoarseRenumbering::MinimumFineIndex
        )
    );
    registerExperimental
    (
        "LocalIndexKahn",
        experimentalControls
        (
            fwgamg::RowOrdering::IndexKahn,
            fwgamg::CoarseRenumbering::MinimumFineIndex
        )
    );

    if (!options.googleRepetitionsSpecified)
    {
        setup->Repetitions(options.samples);
        cycle->Repetitions(options.samples);
        solve->Repetitions(options.samples);
        fwSetup->Repetitions(options.samples);
        fwCycle->Repetitions(options.samples);
        fwSolve->Repetitions(options.samples);
    }
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Options options = consumeArguments(argc, argv);
        if (options.mesh.empty())
        {
            std::cerr << "Usage: gamg_bench [--mesh=POLYMESH_DIR] "
                         "[--gamg_warmup=N] [--gamg_samples=N] [--gamg_csv] "
                         "[--gamg_production_compare_only] "
                         "[--gamg_fusion_compare_only] "
                         "[--gamg_delta_candidate_only] "
                         "[--gamg_delta_candidate_solve_only] "
                         "[--gamg_delta_candidate_solve_variant=direct|cleaned] "
                         "[--gamg_l1_eviction_compare_only] "
                         "[--gamg_vtune_profile="
                         "reference|native-gs-direct|native-gs-direct-evict|"
                         "direct|direct-evict] "
                         "[Google Benchmark options]\n";
            return 2;
        }
        benchmark::Initialize(&argc, argv);
        if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 2;

        const Problem problem(options.mesh);
        if (!options.vtuneProfileVariant.empty())
        {
            runVtuneProfile(problem, options);
            benchmark::Shutdown();
            return 0;
        }
        if (options.l1EvictionCompareOnly)
        {
            runL1EvictionComparison(problem, options);
            benchmark::Shutdown();
            return 0;
        }
        if (options.deltaCandidateOnly || options.deltaCandidateSolveOnly)
        {
            registerFusionComparison(problem, options);
        }
        else if (options.fusionCompareOnly)
        {
            runFwRuntimeSummary
            (
                problem, options, rowSpmvControls(), "RowCSR"
            );
            runFwRuntimeSummary
            (
                problem, options, fwControls(), "DirectFlat"
            );
            runFwRuntimeSummary
            (
                problem, options, fusedSpmvControls(),
                "FusedFinestIH1SpMV"
            );
            runFwRuntimeSummary
            (
                problem, options, deltaFusedSpmvControls(),
                "FusedDeltaIH1SpMV"
            );
            registerFusionComparison(problem, options);
        }
        else if (options.flatEdgeOnly)
        {
            runFwEdgeTimingSummary(problem, options);
            registerFlatEdgeComparison(problem, options);
        }
        else if (options.productionCompareOnly)
        {
            runTimingSummary(problem, options);
            runFwTimingSummary
            (
                problem, options, rowSpmvControls(), "RowCSR"
            );
            runFwTimingSummary
            (
                problem, options, csrDerivedDirectControls(),
                "CSRDerivedDirectFlat"
            );
            runFwTimingSummary
            (
                problem, options, fwControls(), "DirectFlat"
            );
            runFwTimingSummary
            (
                problem, options, fusedSpmvControls(),
                "FusedFinestIH1SpMV"
            );
            registerProductionComparison(problem, options);
        }
        else
        {
            runTimingSummary(problem, options);
            runFwTimingSummary
            (
                problem, options, fwControls(), "DirectFlat"
            );
            runFwEdgeTimingSummary(problem, options);
            printLocalityRenumberSetup(problem);
            registerBenchmarks(problem, options);
        }
        std::cout << std::flush;
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
