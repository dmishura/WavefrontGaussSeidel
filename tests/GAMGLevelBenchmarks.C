#include "GAMGKernels.H"
#include "GAMGSolver.H"
#include "PolyMeshReader.H"
#include "ScheduleBuilder.H"
#include "fwGAMG.H"
#include "fwMatrix.H"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace gamg = smootherTest::gamg;
namespace fwgamg = smootherTest::fwgamg;
namespace harness = smootherTest::harness;

namespace
{

struct Locality
{
    double colMedian = 0;
    double colP90 = 0;
    double colP99 = 0;
    double successiveMedian = 0;
    double successiveP90 = 0;
    double successiveP99 = 0;
    double sameCacheLine = 0;
    double samePage = 0;
    std::size_t dependencyLevels = 0;
    double meanWidth = 0;
    std::size_t maxWidth = 0;
};

struct PageAlignedScalars
{
    std::vector<Foam::scalar> storage;
    Foam::scalar* data = nullptr;

    explicit PageAlignedScalars(const std::size_t count)
    :
        storage(count + 1024)
    {
        const std::uintptr_t address = reinterpret_cast<std::uintptr_t>
        (
            storage.data()
        );
        const std::uintptr_t aligned = (address + 4095u) & ~std::uintptr_t(4095u);
        data = reinterpret_cast<Foam::scalar*>(aligned);
    }
};

double percentile(std::vector<Foam::label>& values, const double fraction)
{
    if (values.empty()) return 0;
    const std::size_t index = static_cast<std::size_t>
    (
        fraction*static_cast<double>(values.size() - 1)
    );
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}

std::vector<std::size_t> naturalDependencyWidths(const gamg::GAMGMatrix& matrix)
{
    std::vector<Foam::label> levels(matrix.nCells(), 0);
    for (Foam::label face=0; face<matrix.nFaces(); ++face)
        levels[matrix.upperAddr()[face]] = std::max
        (
            levels[matrix.upperAddr()[face]],
            levels[matrix.lowerAddr()[face]] + 1
        );
    const Foam::label count = *std::max_element(levels.begin(), levels.end()) + 1;
    std::vector<std::size_t> widths(count, 0);
    for (const Foam::label level : levels) ++widths[level];
    return widths;
}

std::vector<std::size_t> naturalDependencyWidths(const fwgamg::Matrix& matrix)
{
    std::vector<Foam::label> levels(matrix.nCells(), 0);
    for (Foam::label row=0; row<matrix.nCells(); ++row)
        for
        (
            Foam::label p=matrix.rowStarts()[row];
            p<matrix.rowStarts()[row + 1];
            ++p
        )
            if (matrix.cols()[p] > row)
                levels[matrix.cols()[p]] = std::max
                (
                    levels[matrix.cols()[p]], levels[row] + 1
                );
    const Foam::label count = *std::max_element(levels.begin(), levels.end()) + 1;
    std::vector<std::size_t> widths(count, 0);
    for (const Foam::label level : levels) ++widths[level];
    return widths;
}

Locality locality
(
    const fwgamg::Matrix& matrix,
    const std::vector<std::size_t>& dependencyWidths
)
{
    std::vector<Foam::label> colDistances;
    std::vector<Foam::label> successiveDistances;
    colDistances.reserve(matrix.cols().size());
    successiveDistances.reserve(matrix.cols().size());
    std::size_t sameLine = 0;
    std::size_t samePage = 0;
    for (Foam::label row=0; row<matrix.nCells(); ++row)
    {
        const Foam::label begin = matrix.rowStarts()[row];
        const Foam::label end = matrix.rowStarts()[row + 1];
        for (Foam::label p=begin; p<end; ++p)
            colDistances.push_back(std::abs(matrix.cols()[p] - row));
        for (Foam::label p=begin + 1; p<end; ++p)
        {
            const Foam::label previous = matrix.cols()[p - 1];
            const Foam::label current = matrix.cols()[p];
            successiveDistances.push_back(std::abs(current - previous));
            if (previous/8 == current/8) ++sameLine;
            if (previous/512 == current/512) ++samePage;
        }
    }
    Locality result;
    result.colMedian = percentile(colDistances, 0.50);
    result.colP90 = percentile(colDistances, 0.90);
    result.colP99 = percentile(colDistances, 0.99);
    result.successiveMedian = percentile(successiveDistances, 0.50);
    result.successiveP90 = percentile(successiveDistances, 0.90);
    result.successiveP99 = percentile(successiveDistances, 0.99);
    if (!successiveDistances.empty())
    {
        result.sameCacheLine = 100.0*sameLine/successiveDistances.size();
        result.samePage = 100.0*samePage/successiveDistances.size();
    }
    result.dependencyLevels = dependencyWidths.size();
    if (!dependencyWidths.empty())
    {
        result.meanWidth = static_cast<double>(matrix.nCells())
            /dependencyWidths.size();
        result.maxWidth = *std::max_element
        (
            dependencyWidths.begin(), dependencyWidths.end()
        );
    }
    return result;
}

gamg::GAMGControls controls()
{
    gamg::GAMGControls result;
    result.relativeTolerance = 0;
    result.tolerance = 1e-8;
    result.maxIterations = 100;
    return result;
}

fwgamg::Controls fwControls()
{
    fwgamg::Controls result;
    result.gamg = controls();
    result.width = 1024;
    return result;
}

fwgamg::Controls hierarchyControls
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

fwgamg::NativeMatrixData nativeData(const gamg::GAMGMatrix& matrix)
{
    fwgamg::NativeMatrixData result;
    result.nCells = matrix.nCells();
    result.lowerAddr = matrix.lowerAddr();
    result.upperAddr = matrix.upperAddr();
    result.diag = matrix.diag();
    result.upper = matrix.upper();
    return result;
}

struct Problem
{
    smootherTest::PolyMeshTopology mesh;
    Foam::lduMatrix fineMatrix;
    gamg::GAMGSolver baseline;
    fwgamg::Solver reordered;
    fwgamg::Solver localNatural;
    fwgamg::Solver localIndex;
    std::vector<fwgamg::Matrix> natural;
    std::vector<fwgamg::LevelSetupTiming> naturalSetup;
    std::vector<Foam::scalarField> naturalSource;
    std::vector<Foam::scalarField> reorderedSource;
    std::vector<Foam::scalarField> localNaturalSource;
    std::vector<Foam::scalarField> localIndexSource;

    explicit Problem(const std::string& meshDirectory)
    :
        mesh(smootherTest::PolyMeshReader::read(meshDirectory)),
        fineMatrix(harness::makeMatrix(mesh)),
        baseline(fineMatrix, controls()),
        reordered(fineMatrix, fwControls()),
        localNatural
        (
            fineMatrix,
            hierarchyControls
            (
                fwgamg::RowOrdering::Natural,
                fwgamg::CoarseRenumbering::MinimumFineIndex
            )
        ),
        localIndex
        (
            fineMatrix,
            hierarchyControls
            (
                fwgamg::RowOrdering::IndexKahn,
                fwgamg::CoarseRenumbering::MinimumFineIndex
            )
        )
    {
        natural.reserve(baseline.matrices().size());
        naturalSetup.reserve(baseline.matrices().size());
        naturalSource.reserve(baseline.matrices().size());
        reorderedSource.reserve(baseline.matrices().size());
        localNaturalSource.reserve(baseline.matrices().size());
        localIndexSource.reserve(baseline.matrices().size());
        for (std::size_t level=0; level<baseline.matrices().size(); ++level)
        {
            const gamg::GAMGMatrix& matrix = baseline.matrices()[level];
            fwgamg::LevelSetupTiming setup;
            natural.push_back
            (
                fwgamg::Matrix::buildNatural
                (
                    nativeData(matrix), &setup, true, true, true
                )
            );
            naturalSetup.push_back(setup);
            Foam::scalarField source(matrix.nCells());
            for (Foam::label cell=0; cell<matrix.nCells(); ++cell)
                source[cell] = 0.75 + std::sin(0.019*cell)
                    + 0.2*std::cos(0.071*cell);
            naturalSource.push_back(source);
            reorderedSource.push_back(reordered.toFw(source, level));
            Foam::scalarField localNaturalField(matrix.nCells());
            Foam::scalarField localIndexField(matrix.nCells());
            for (Foam::label row=0; row<matrix.nCells(); ++row)
            {
                localNaturalField[row] = 0.75 + std::sin(0.019*row)
                    + 0.2*std::cos(0.071*row);
                localIndexField[row] = localNaturalField[row];
            }
            localNaturalSource.push_back(std::move(localNaturalField));
            localIndexSource.push_back(std::move(localIndexField));
        }
    }
};

std::string consumeMesh(int& argc, char** argv)
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
        if (argument.rfind("--mesh=", 0) == 0) mesh = argument.substr(7);
        else argv[destination++] = argv[source];
    }
    argc = destination;
    if (mesh.empty()) throw std::runtime_error("--mesh=POLYMESH_DIR is required");
    return mesh;
}

void addCounters
(
    benchmark::State& state,
    const Foam::label cells,
    const Foam::label faces
)
{
    const double nnz = static_cast<double>(cells) + 2.0*faces;
    state.counters["cells"] = cells;
    state.counters["nnz"] = nnz;
    state.counters["avgDegree"] = 2.0*faces/cells;
    state.counters["ns/cell"] = benchmark::Counter
    (
        cells,
        benchmark::Counter::kIsIterationInvariantRate
      | benchmark::Counter::kInvert
    );
    state.counters["ns/nnz"] = benchmark::Counter
    (
        nnz,
        benchmark::Counter::kIsIterationInvariantRate
      | benchmark::Counter::kInvert
    );
    state.counters["ns/internal-face"] = benchmark::Counter
    (
        faces,
        benchmark::Counter::kIsIterationInvariantRate
      | benchmark::Counter::kInvert
    );
}

void registerLevelBenchmarks(const Problem& problem)
{
    for (std::size_t level=0; level<problem.baseline.matrices().size(); ++level)
    {
        const gamg::GAMGMatrix* const ldu = &problem.baseline.matrices()[level];
        const fwgamg::Matrix* const natural = &problem.natural[level];
        const fwgamg::Matrix* const reordered = &problem.reordered.hierarchy()[level];
        const fwgamg::Matrix* const localNatural =
            &problem.localNatural.hierarchy()[level];
        const fwgamg::Matrix* const localIndex =
            &problem.localIndex.hierarchy()[level];
        const Foam::scalarField* const source = &problem.naturalSource[level];
        const Foam::scalarField* const fwSource = &problem.reorderedSource[level];
        const Foam::scalarField* const localNaturalSource =
            &problem.localNaturalSource[level];
        const Foam::scalarField* const localIndexSource =
            &problem.localIndexSource[level];
        const std::string prefix = "GAMGLevel/L" +
            (level < 10 ? std::string("0") : std::string()) +
            std::to_string(level) + '/';
        const auto finish = [ldu](benchmark::State& state)
        {
            addCounters(state, ldu->nCells(), ldu->nFaces());
        };

        benchmark::RegisterBenchmark
        (
            (prefix + "A_LDU/GS").c_str(),
            [ldu, source, finish](benchmark::State& state)
            {
                Foam::scalarField psi(ldu->nCells(), 0.0);
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    gamg::gaussSeidelSmooth(psi, *ldu, *source, 1);
                    benchmark::DoNotOptimize(psi.data());
                }
                finish(state);
            }
        )->MinTime(0.05);
        benchmark::RegisterBenchmark
        (
            (prefix + "B_CSRNatural/GS").c_str(),
            [natural, source, finish](benchmark::State& state)
            {
                Foam::scalarField psi(natural->nCells(), 0.0);
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    natural->smooth(psi, *source, 1, false);
                    benchmark::DoNotOptimize(psi.data());
                }
                finish(state);
            }
        )->MinTime(0.05);
        benchmark::RegisterBenchmark
        (
            (prefix + "C_CSRIndexKahn/GS").c_str(),
            [reordered, fwSource, finish](benchmark::State& state)
            {
                Foam::scalarField psi(reordered->nCells(), 0.0);
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    reordered->smooth(psi, *fwSource, 1, false);
                    benchmark::DoNotOptimize(psi.data());
                }
                finish(state);
            }
        )->MinTime(0.05);

        const auto registerSpmv = [&](const std::string& variant, auto operation)
        {
            benchmark::RegisterBenchmark
            (
                (prefix + variant + "/SpMV").c_str(), operation
            )->MinTime(0.05);
        };
        registerSpmv
        (
            "A_LDU",
            [ldu, source, finish](benchmark::State& state)
            {
                Foam::scalarField result;
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    gamg::multiply(result, *ldu, *source);
                    benchmark::DoNotOptimize(result.data());
                }
                finish(state);
            }
        );

        if (level == 0)
        {
            const auto registerColourSweep =
            [finish]
            (
                const std::string& variant,
                const fwgamg::Matrix* const matrix,
                const Foam::scalarField* const input
            )
            {
                benchmark::RegisterBenchmark
                (
                    ("GAMGColour/L00/" + variant).c_str(),
                    [matrix, input, finish](benchmark::State& state)
                    {
                        PageAlignedScalars field(matrix->nCells());
                        PageAlignedScalars result(matrix->nCells() + 512);
                        std::copy(input->begin(), input->end(), field.data);
                        Foam::scalar* const colouredResult =
                            result.data + 8*state.range(0);
                        for (auto unused : state)
                        {
                            static_cast<void>(unused);
                            matrix->multiplyLduOrderDirectFlatEdgeWiseRaw
                            (
                                colouredResult, field.data
                            );
                            benchmark::DoNotOptimize(colouredResult);
                        }
                        state.counters["resultOffsetBytes"] =
                            64*state.range(0);
                        finish(state);
                    }
                )->DenseRange(0, 63)->MinTime(0.02);
            };
            registerColourSweep("Natural", natural, source);
            registerColourSweep("IndexKahn", reordered, fwSource);
        }
        registerSpmv
        (
            "A2_LDUOwnerAccumulated",
            [ldu, source, finish](benchmark::State& state)
            {
                Foam::scalarField result;
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    gamg::multiplyOwnerAccumulated(result, *ldu, *source);
                    benchmark::DoNotOptimize(result.data());
                }
                finish(state);
            }
        );
        registerSpmv
        (
            "B_CSRNatural",
            [natural, source, finish](benchmark::State& state)
            {
                Foam::scalarField result;
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    natural->multiply(result, *source);
                    benchmark::DoNotOptimize(result.data());
                }
                finish(state);
            }
        );
        registerSpmv
        (
            "B2_CSRFlatEdgeNatural",
            [natural, source, finish](benchmark::State& state)
            {
                Foam::scalarField result;
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    natural->multiplyFlatEdgeWise(result, *source);
                    benchmark::DoNotOptimize(result.data());
                }
                finish(state);
            }
        );
        registerSpmv
        (
            "B3_CSRDirectFlatEdgeNatural",
            [natural, source, finish](benchmark::State& state)
            {
                Foam::scalarField result;
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    natural->multiplyDirectFlatEdgeWise(result, *source);
                    benchmark::DoNotOptimize(result.data());
                }
                finish(state);
            }
        );
        registerSpmv
        (
            "B3A_CSRDirectFlatOwnerAccumulatedNatural",
            [natural, source, finish](benchmark::State& state)
            {
                Foam::scalarField result;
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    natural->multiplyDirectFlatOwnerAccumulated
                    (
                        result, *source
                    );
                    benchmark::DoNotOptimize(result.data());
                }
                finish(state);
            }
        );
        registerSpmv
        (
            "B3B_CSRDirectFlatOwnerRunAccumulatedNatural",
            [natural, source, finish](benchmark::State& state)
            {
                Foam::scalarField result;
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    natural->multiplyDirectFlatOwnerRunAccumulated
                    (
                        result, *source
                    );
                    benchmark::DoNotOptimize(result.data());
                }
                finish(state);
            }
        );
        registerSpmv
        (
            "B4_LDUOrderDirectFlatEdgeNatural",
            [natural, source, finish](benchmark::State& state)
            {
                Foam::scalarField result;
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    natural->multiplyLduOrderDirectFlatEdgeWise
                    (
                        result, *source
                    );
                    benchmark::DoNotOptimize(result.data());
                }
                finish(state);
            }
        );
        registerSpmv
        (
            "C_CSRIndexKahn",
            [reordered, fwSource, finish](benchmark::State& state)
            {
                Foam::scalarField result;
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    reordered->multiply(result, *fwSource);
                    benchmark::DoNotOptimize(result.data());
                }
                finish(state);
            }
        );
        registerSpmv
        (
            "C2_LDUOrderDirectFlatIndexKahn",
            [reordered, fwSource, finish](benchmark::State& state)
            {
                Foam::scalarField result;
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    reordered->multiplyLduOrderDirectFlatEdgeWise
                    (
                        result, *fwSource
                    );
                    benchmark::DoNotOptimize(result.data());
                }
                finish(state);
            }
        );
        registerSpmv
        (
            "C3_CSRDirectFlatIndexKahn",
            [reordered, fwSource, finish](benchmark::State& state)
            {
                Foam::scalarField result;
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    reordered->multiplyDirectFlatEdgeWise(result, *fwSource);
                    benchmark::DoNotOptimize(result.data());
                }
                finish(state);
            }
        );

        const auto registerResidual = [&](const std::string& variant, auto operation)
        {
            benchmark::RegisterBenchmark
            (
                (prefix + variant + "/Residual").c_str(), operation
            )->MinTime(0.05);
        };
        registerResidual
        (
            "A_LDU",
            [ldu, source, finish](benchmark::State& state)
            {
                Foam::scalarField result;
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    gamg::residual(result, *ldu, *source, *source);
                    benchmark::DoNotOptimize(result.data());
                }
                finish(state);
            }
        );
        registerResidual
        (
            "B_CSRNatural",
            [natural, source, finish](benchmark::State& state)
            {
                Foam::scalarField result;
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    natural->residual(result, *source, *source);
                    benchmark::DoNotOptimize(result.data());
                }
                finish(state);
            }
        );
        registerResidual
        (
            "C_CSRIndexKahn",
            [reordered, fwSource, finish](benchmark::State& state)
            {
                Foam::scalarField result;
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    reordered->residual(result, *fwSource, *fwSource);
                    benchmark::DoNotOptimize(result.data());
                }
                finish(state);
            }
        );

        const auto registerScale = [&](const std::string& variant, auto operation)
        {
            benchmark::RegisterBenchmark
            (
                (prefix + variant + "/ScaleCorrection").c_str(), operation
            )->MinTime(0.05);
        };
        registerScale
        (
            "A_LDU",
            [ldu, source, finish](benchmark::State& state)
            {
                Foam::scalarField field(*source), work;
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    gamg::scaleCorrection(field, *ldu, *source, work);
                    benchmark::DoNotOptimize(field.data());
                }
                finish(state);
            }
        );
        registerScale
        (
            "B_CSRNatural",
            [natural, source, finish](benchmark::State& state)
            {
                Foam::scalarField field(*source), work;
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    natural->scaleCorrection(field, *source, work);
                    benchmark::DoNotOptimize(field.data());
                }
                finish(state);
            }
        );
        registerScale
        (
            "C_CSRIndexKahn",
            [reordered, fwSource, finish](benchmark::State& state)
            {
                Foam::scalarField field(*fwSource), work;
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    reordered->scaleCorrection(field, *fwSource, work);
                    benchmark::DoNotOptimize(field.data());
                }
                finish(state);
            }
        );

        const auto registerPackedVariant =
        [prefix, finish]
        (
            const std::string& name,
            const fwgamg::Matrix* const matrix,
            const Foam::scalarField* const variantSource
        )
        {
            benchmark::RegisterBenchmark
            (
                (prefix + name + "/GS").c_str(),
                [matrix, variantSource, finish](benchmark::State& state)
                {
                    Foam::scalarField psi(matrix->nCells(), 0.0);
                    for (auto unused : state)
                    {
                        static_cast<void>(unused);
                        matrix->smooth(psi, *variantSource, 1, false);
                        benchmark::DoNotOptimize(psi.data());
                    }
                    finish(state);
                }
            )->MinTime(0.05);
            benchmark::RegisterBenchmark
            (
                (prefix + name + "/SpMV").c_str(),
                [matrix, variantSource, finish](benchmark::State& state)
                {
                    Foam::scalarField result;
                    for (auto unused : state)
                    {
                        static_cast<void>(unused);
                        matrix->multiply(result, *variantSource);
                        benchmark::DoNotOptimize(result.data());
                    }
                    finish(state);
                }
            )->MinTime(0.05);
            benchmark::RegisterBenchmark
            (
                (prefix + name + "/Residual").c_str(),
                [matrix, variantSource, finish](benchmark::State& state)
                {
                    Foam::scalarField result;
                    for (auto unused : state)
                    {
                        static_cast<void>(unused);
                        matrix->residual
                        (
                            result, *variantSource, *variantSource
                        );
                        benchmark::DoNotOptimize(result.data());
                    }
                    finish(state);
                }
            )->MinTime(0.05);
            benchmark::RegisterBenchmark
            (
                (prefix + name + "/ScaleCorrection").c_str(),
                [matrix, variantSource, finish](benchmark::State& state)
                {
                    Foam::scalarField field(*variantSource), work;
                    for (auto unused : state)
                    {
                        static_cast<void>(unused);
                        matrix->scaleCorrection
                        (
                            field, *variantSource, work
                        );
                        benchmark::DoNotOptimize(field.data());
                    }
                    finish(state);
                }
            )->MinTime(0.05);
        };
        registerPackedVariant
        (
            "D_CSRLocalNatural", localNatural, localNaturalSource
        );
        registerPackedVariant
        (
            "E_CSRLocalIndexKahn", localIndex, localIndexSource
        );
    }
}

void printLocality(const Problem& problem)
{
    std::cout << "GAMG coarse-level locality diagnostics\n"
        << "level variant cells nnz avgDegree colMedian colP90 colP99 "
           "successiveMedian successiveP90 successiveP99 same64B% same4KiB% "
           "dependencyLevels meanWidth maxWidth\n";
    for (std::size_t level=0; level<problem.natural.size(); ++level)
    {
        const gamg::GAMGMatrix& ldu = problem.baseline.matrices()[level];
        const std::vector<std::size_t> naturalWidths = naturalDependencyWidths(ldu);
        const fwgamg::Matrix& localNatural =
            problem.localNatural.hierarchy()[level];
        const fwgamg::Matrix& localIndex =
            problem.localIndex.hierarchy()[level];
        const std::vector<std::size_t> localNaturalWidths =
            naturalDependencyWidths(localNatural);
        std::vector<std::size_t> localIndexWidths;
        const auto& starts = localIndex.levelStarts();
        for (std::size_t i=0; i + 1<starts.size(); ++i)
            localIndexWidths.push_back(starts[i + 1] - starts[i]);
        for (const auto& entry : std::vector<std::pair<const char*, Locality>>
        {
            {"B", locality(problem.natural[level], naturalWidths)},
            {"D", locality(localNatural, localNaturalWidths)},
            {"E", locality(localIndex, localIndexWidths)}
        })
        {
            const Locality& value = entry.second;
            std::cout << level << ' ' << entry.first << ' '
                << ldu.nCells() << ' ' << ldu.nCells() + 2*ldu.nFaces() << ' '
                << 2.0*ldu.nFaces()/ldu.nCells() << ' '
                << value.colMedian << ' ' << value.colP90 << ' '
                << value.colP99 << ' ' << value.successiveMedian << ' '
                << value.successiveP90 << ' ' << value.successiveP99 << ' '
                << value.sameCacheLine << ' ' << value.samePage << ' '
                << value.dependencyLevels << ' ' << value.meanWidth << ' '
                << value.maxWidth << '\n';
        }
    }
    std::cout << std::flush;
}

void printFlatEdgeSetup(const Problem& problem)
{
    double indirectSeconds = 0;
    double directSeconds = 0;
    std::size_t indirectBytes = 0;
    std::size_t directBytes = 0;
    std::cout << "GAMG flat-edge metadata diagnostics\n"
        << "level faces indirectMs directMs indirectBytes directBytes\n";
    for (std::size_t level=0; level<problem.natural.size(); ++level)
    {
        const auto& matrix = problem.natural[level];
        const auto& timing = problem.naturalSetup[level];
        const std::size_t levelIndirectBytes =
            matrix.edgeOwners().size()*sizeof(Foam::label)
          + matrix.upperEntries().size()*sizeof(Foam::label);
        const std::size_t levelDirectBytes =
            matrix.directEdgeOwners().size()*sizeof(Foam::label)
          + matrix.directEdgeColumns().size()*sizeof(Foam::label)
          + matrix.directEdgeCoeffs().size()*sizeof(Foam::scalar);
        indirectSeconds += timing.flatEdgeMetadataSeconds;
        directSeconds += timing.directFlatEdgeMetadataSeconds;
        indirectBytes += levelIndirectBytes;
        directBytes += levelDirectBytes;
        std::cout << level << ' ' << matrix.nFaces() << ' '
            << 1e3*timing.flatEdgeMetadataSeconds << ' '
            << 1e3*timing.directFlatEdgeMetadataSeconds << ' '
            << levelIndirectBytes << ' ' << levelDirectBytes << '\n';
    }
    std::cout << "total " << indirectBytes/sizeof(Foam::label)/2 << ' '
        << 1e3*indirectSeconds << ' ' << 1e3*directSeconds << ' '
        << indirectBytes << ' ' << directBytes << '\n' << std::flush;
}

double maximumDifference
(
    const Foam::scalarField& left,
    const Foam::scalarField& right
)
{
    if (left.size() != right.size())
        throw std::runtime_error("SpMV validation field size mismatch");
    double result = 0;
    for (std::size_t i=0; i<left.size(); ++i)
        result = std::max(result, std::abs(left[i] - right[i]));
    return result;
}

void validateSpmv(const Problem& problem)
{
    double rowDifference = 0;
    double flatDifference = 0;
    double directDifference = 0;
    double lduOrderDirectDifference = 0;
    double reorderedLduOrderDirectDifference = 0;
    double reorderedDirectDifference = 0;
    double rawLduOrderDirectDifference = 0;
    bool reorderedLduOrderDirectExact = true;
    double lduOwnerAccumulatedDifference = 0;
    double directOwnerAccumulatedDifference = 0;
    double directOwnerRunAccumulatedDifference = 0;
    std::size_t directOwnerOrderMismatches = 0;
    std::size_t directColumnOrderMismatches = 0;
    std::size_t directCoefficientOrderMismatches = 0;
    for (std::size_t level=0; level<problem.natural.size(); ++level)
    {
        const fwgamg::Matrix& natural = problem.natural[level];
        for (Foam::label face=0; face<natural.nFaces(); ++face)
        {
            directOwnerOrderMismatches +=
                natural.directEdgeOwners()[face]
             != natural.lduOrderDirectEdgeOwners()[face];
            directColumnOrderMismatches +=
                natural.directEdgeColumns()[face]
             != natural.lduOrderDirectEdgeColumns()[face];
            directCoefficientOrderMismatches +=
                natural.directEdgeCoeffs()[face]
             != natural.lduOrderDirectEdgeCoeffs()[face];
        }
        Foam::scalarField expected, actual;
        gamg::multiply
        (
            expected,
            problem.baseline.matrices()[level],
            problem.naturalSource[level]
        );
        problem.natural[level].multiply
        (
            actual, problem.naturalSource[level]
        );
        rowDifference = std::max
        (
            rowDifference, maximumDifference(expected, actual)
        );
        problem.natural[level].multiplyFlatEdgeWise
        (
            actual, problem.naturalSource[level]
        );
        flatDifference = std::max
        (
            flatDifference, maximumDifference(expected, actual)
        );
        problem.natural[level].multiplyDirectFlatEdgeWise
        (
            actual, problem.naturalSource[level]
        );
        directDifference = std::max
        (
            directDifference, maximumDifference(expected, actual)
        );
        problem.natural[level].multiplyLduOrderDirectFlatEdgeWise
        (
            actual, problem.naturalSource[level]
        );
        lduOrderDirectDifference = std::max
        (
            lduOrderDirectDifference, maximumDifference(expected, actual)
        );
        const Foam::scalarField expectedReordered =
            problem.reordered.toFw(expected, level);
        problem.reordered.hierarchy()[level].multiplyLduOrderDirectFlatEdgeWise
        (
            actual, problem.reorderedSource[level]
        );
        reorderedLduOrderDirectDifference = std::max
        (
            reorderedLduOrderDirectDifference,
            maximumDifference(expectedReordered, actual)
        );
        reorderedLduOrderDirectExact =
            reorderedLduOrderDirectExact && actual == expectedReordered;
        PageAlignedScalars rawField(actual.size());
        PageAlignedScalars rawResult(actual.size());
        std::copy
        (
            problem.reorderedSource[level].begin(),
            problem.reorderedSource[level].end(),
            rawField.data
        );
        problem.reordered.hierarchy()[level]
            .multiplyLduOrderDirectFlatEdgeWiseRaw
            (
                rawResult.data, rawField.data
            );
        Foam::scalarField rawActual(actual.size());
        std::copy
        (
            rawResult.data, rawResult.data + rawActual.size(),
            rawActual.begin()
        );
        rawLduOrderDirectDifference = std::max
        (
            rawLduOrderDirectDifference,
            maximumDifference(expectedReordered, rawActual)
        );
        problem.reordered.hierarchy()[level].multiplyDirectFlatEdgeWise
        (
            actual, problem.reorderedSource[level]
        );
        reorderedDirectDifference = std::max
        (
            reorderedDirectDifference,
            maximumDifference(expectedReordered, actual)
        );
        gamg::multiplyOwnerAccumulated
        (
            actual,
            problem.baseline.matrices()[level],
            problem.naturalSource[level]
        );
        lduOwnerAccumulatedDifference = std::max
        (
            lduOwnerAccumulatedDifference,
            maximumDifference(expected, actual)
        );
        problem.natural[level].multiplyDirectFlatOwnerAccumulated
        (
            actual, problem.naturalSource[level]
        );
        directOwnerAccumulatedDifference = std::max
        (
            directOwnerAccumulatedDifference,
            maximumDifference(expected, actual)
        );
        problem.natural[level].multiplyDirectFlatOwnerRunAccumulated
        (
            actual, problem.naturalSource[level]
        );
        directOwnerRunAccumulatedDifference = std::max
        (
            directOwnerRunAccumulatedDifference,
            maximumDifference(expected, actual)
        );
    }
    constexpr double tolerance = 1e-12;
    std::cout << "GAMG SpMV validation: rowMax=" << rowDifference
        << " flatMax=" << flatDifference
        << " directMax=" << directDifference
        << " lduOrderDirectMax=" << lduOrderDirectDifference
        << " reorderedLduOrderDirectMax="
        << reorderedLduOrderDirectDifference
        << " reorderedDirectMax=" << reorderedDirectDifference
        << " rawLduOrderDirectMax=" << rawLduOrderDirectDifference
        << " reorderedLduOrderDirectExact="
        << (reorderedLduOrderDirectExact ? "PASS" : "FAIL")
        << " lduOwnerAccumulatedMax=" << lduOwnerAccumulatedDifference
        << " directOwnerAccumulatedMax="
        << directOwnerAccumulatedDifference
        << " directOwnerRunAccumulatedMax="
        << directOwnerRunAccumulatedDifference
        << " directVsLduOrderMismatches(owner/column/coeff)="
        << directOwnerOrderMismatches << '/'
        << directColumnOrderMismatches << '/'
        << directCoefficientOrderMismatches
        << " tolerance=" << tolerance << '\n';
    if
    (
        rowDifference > tolerance
     || flatDifference > tolerance
     || directDifference > tolerance
     || lduOrderDirectDifference > tolerance
     || reorderedLduOrderDirectDifference > tolerance
     || rawLduOrderDirectDifference > tolerance
     // reorderedDirectDifference is reported, not gated: the production
     // CSR-derived edge stream intentionally changes accumulation order.
     || lduOwnerAccumulatedDifference > tolerance
     || directOwnerAccumulatedDifference > tolerance
     || directOwnerRunAccumulatedDifference > tolerance
    ) throw std::runtime_error("GAMG SpMV validation failed");
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const std::string mesh = consumeMesh(argc, argv);
        benchmark::Initialize(&argc, argv);
        if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 2;
        const Problem problem(mesh);
        validateSpmv(problem);
        printLocality(problem);
        printFlatEdgeSetup(problem);
        registerLevelBenchmarks(problem);
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
