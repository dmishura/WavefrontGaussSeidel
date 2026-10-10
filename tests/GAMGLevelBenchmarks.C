#include "GAMGKernels.H"
#include "GAMGSolver.H"
#include "PolyMeshReader.H"
#include "ScheduleBuilder.H"
#include "WavefrontGaussSeidel.H"
#include "fwGAMG.H"
#include "fwMatrix.H"

#include <benchmark/benchmark.h>

#ifdef HAVE_ITTNOTIFY
#include <ittnotify.h>
#endif

#ifdef HAVE_AMDUPROF_CONTROLLER
#include <AMDProfileController.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <type_traits>
#include <unistd.h>
#include <vector>

#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif

namespace gamg = smootherTest::gamg;
namespace fwgamg = smootherTest::fwgamg;
namespace harness = smootherTest::harness;

namespace
{

bool explicitHugePages = false;
bool amdProfileRoi = false;
bool amdPcmSignalRoi = false;
bool productionIh1Only = false;
bool ih1w2WidthSweep = false;
bool ih1w3Experiment = false;
bool ih1w4Experiment = false;
bool ih1w5Experiment = false;
bool releaseImpactExperiment = false;
std::string ih1w4Batch = "all";
std::string ih1w5W4Profile = "IH1W4_A";
std::string ih1w5TraceDirectory;
std::string releaseImpactTraceDirectory;
bool permutationDumpOnly = false;
std::string benchmarkMeshName;
std::string permutationDumpDirectory;
std::string vtuneSmoother;
Foam::label vtuneSweeps = 64;
Foam::label vtuneWarmupSweeps = 4;

std::vector<Foam::label> boundedWidths()
{
    if (ih1w5Experiment) return {1024};
    if (ih1w4Experiment)
        return ih1w4Batch == "adaptive"
            ? std::vector<Foam::label>{1024}
            : std::vector<Foam::label>{1024, 2048, 4096, 8192};
    if (ih1w2WidthSweep) return {64, 128, 256, 512, 1024, 2048};
    if (productionIh1Only) return {1024};
    return {64, 128, 256, 512, 1024};
}

std::vector<Foam::label> orientedWidths()
{
    return ih1w2WidthSweep
        ? boundedWidths()
        : std::vector<Foam::label>{1024};
}

std::string safeFileName(std::string value)
{
    for (char& character : value)
    {
        const unsigned char byte = static_cast<unsigned char>(character);
        if (!std::isalnum(byte) && character != '-' && character != '_')
            character = '_';
    }
    return value.empty() ? "mesh" : value;
}

template<class UInt>
void writeLittleEndian(std::ostream& output, UInt value)
{
    static_assert(std::is_unsigned_v<UInt>);
    for (std::size_t byte=0; byte<sizeof(UInt); ++byte)
    {
        output.put(static_cast<char>(value & 0xffu));
        value >>= 8;
    }
}

void writePermutationLabels
(
    std::ostream& output,
    const std::vector<Foam::label>& labels
)
{
    constexpr std::size_t chunkSize = 1u << 20;
    std::vector<std::int32_t> buffer;
    buffer.reserve(std::min(chunkSize, labels.size()));
    for (std::size_t begin=0; begin<labels.size(); begin += chunkSize)
    {
        const std::size_t end = std::min(begin + chunkSize, labels.size());
        buffer.clear();
        for (std::size_t index=begin; index<end; ++index)
            buffer.push_back(static_cast<std::int32_t>(labels[index]));
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        for (std::int32_t& value : buffer)
            value = static_cast<std::int32_t>(__builtin_bswap32(value));
#endif
        output.write
        (
            reinterpret_cast<const char*>(buffer.data()),
            static_cast<std::streamsize>(buffer.size()*sizeof(buffer.front()))
        );
    }
}

void dumpPermutation
(
    const std::filesystem::path& directory,
    const std::string& variant,
    const smootherTest::IndexKahnOrdering& ordering
)
{
    const std::size_t nCells = ordering.newToOld.size();
    if (ordering.oldToNew.size() != nCells)
        throw std::runtime_error("permutation inverse size mismatch");
    for (std::size_t row=0; row<nCells; ++row)
    {
        const Foam::label oldCell = ordering.newToOld[row];
        if (oldCell < 0 || static_cast<std::size_t>(oldCell) >= nCells)
            throw std::runtime_error("newToOld value outside cell range");
        if (ordering.oldToNew[oldCell] != static_cast<Foam::label>(row))
            throw std::runtime_error("newToOld/oldToNew are not inverse");
    }

    std::filesystem::create_directories(directory);
    const std::filesystem::path path =
        directory/(safeFileName(variant) + ".ihperm");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create " + path.string());

    output.write("IH1PERM", 8);
    writeLittleEndian<std::uint32_t>(output, 2); // format version
    writeLittleEndian<std::uint32_t>(output, 4); // stored label bytes
    writeLittleEndian<std::uint64_t>(output, nCells);
    writeLittleEndian<std::uint32_t>(output, 0); // GAMG level: fine mesh
    writeLittleEndian<std::uint32_t>(output, variant.size());
    writeLittleEndian<std::uint64_t>(output, ordering.levelStarts.size());
    output.write(variant.data(), static_cast<std::streamsize>(variant.size()));
    writePermutationLabels(output, ordering.levelStarts);
    writePermutationLabels(output, ordering.newToOld);
    writePermutationLabels(output, ordering.oldToNew);
    if (!output) throw std::runtime_error("failed writing " + path.string());

    std::cout << "GAMG permutation dump variant=" << variant
        << " level=0 cells=" << nCells << " path=" << path << '\n';
}

void dumpFinePermutations
(
    const std::string& meshDirectory,
    const std::filesystem::path& outputRoot
)
{
    const smootherTest::PolyMeshTopology mesh =
        smootherTest::PolyMeshReader::read(meshDirectory);
    const std::vector<int> ownerStartsRaw = mesh.ownerStartAddressing();
    Foam::labelField ownerStarts(ownerStartsRaw.begin(), ownerStartsRaw.end());
    Foam::labelField upperAddr(mesh.neighbour.begin(), mesh.neighbour.end());
    const Foam::label nCells = static_cast<Foam::label>(mesh.nCells);
    const std::filesystem::path directory =
        outputRoot/safeFileName(benchmarkMeshName);

    for (const Foam::label width : boundedWidths())
    {
        dumpPermutation
        (
            directory,
            "IH1W" + std::to_string(width),
            smootherTest::makeIndexKahnOrdering
            (nCells, ownerStarts, upperAddr, width)
        );
    }
    dumpPermutation
    (
        directory, "IH1U",
        smootherTest::makeIndexKahnOrderingUnbounded
        (nCells, ownerStarts, upperAddr)
    );
    for (const Foam::label width : orientedWidths())
    {
        dumpPermutation
        (
            directory, "IH1W2_" + std::to_string(width),
            smootherTest::makeIndexKahnOrderingOriented
            (nCells, ownerStarts, upperAddr, width)
        );
    }
    if (ih1w3Experiment)
    {
        for (const Foam::label step : {64, 128, 256})
        {
            dumpPermutation
            (
                directory,
                "IH1W3_W1024_S" + std::to_string(step),
                smootherTest::makeIndexKahnOrderingSlidingWindow
                (nCells, ownerStarts, upperAddr, 1024, step)
            );
        }
    }
    if (ih1w4Experiment && ih1w4Batch != "fixed")
    {
        struct Controls
        {
            const char* name;
            Foam::label minWidth, targetWidth, maxWidth;
        };
        for
        (
            const Controls controls :
            {
                Controls{"IH1W4_A", 768, 1024, 1536},
                Controls{"IH1W4_B", 512, 1024, 2048},
                Controls{"IH1W4_C", 1024, 1024, 2048}
            }
        )
        {
            dumpPermutation
            (
                directory,
                controls.name,
                smootherTest::makeIndexKahnOrderingAdaptiveGap
                (
                    nCells,
                    ownerStarts,
                    upperAddr,
                    controls.minWidth,
                    controls.targetWidth,
                    controls.maxWidth
                )
            );
        }
    }
    if (ih1w5Experiment)
    {
        for (const Foam::label divisor : {4, 8})
        {
            dumpPermutation
            (
                directory,
                "IH1W5_Q" + std::to_string(divisor),
                smootherTest::makeIndexKahnOrderingReadyFractionAdaptiveGap
                (
                    nCells, ownerStarts, upperAddr, divisor
                )
            );
        }
    }
    dumpPermutation
    (
        directory, "IH1S1024",
        smootherTest::makeIndexKahnOrderingSliced
        (nCells, ownerStarts, upperAddr, 1024)
    );
    dumpPermutation
    (
        directory, "IH1C1024",
        smootherTest::makeIndexKahnOrderingContinuing
        (nCells, ownerStarts, upperAddr, 1024)
    );
    dumpPermutation
    (
        directory, "IH1C2_1024",
        smootherTest::makeIndexKahnOrderingFrontierContinuing
        (nCells, ownerStarts, upperAddr, 1024)
    );
}

void beginAmdProfileRoi()
{
    if (amdPcmSignalRoi)
    {
        if (::kill(::getppid(), SIGUSR1) != 0)
        {
            const int error = errno;
            throw std::runtime_error
            (
                "failed to signal AMDuProfPcm: "
              + std::string(std::strerror(error))
            );
        }
        // AMDuProfPcm has already created the target and counters.  Give its
        // signal handler a short interval to enable collection before the ROI.
        ::usleep(100000);
    }
    if (!amdProfileRoi) return;
#ifdef HAVE_AMDUPROF_CONTROLLER
    if (!amdProfileStrictResume())
        throw std::runtime_error
        (
            "amdProfileStrictResume failed: "
          + std::to_string(amdGetLastProfileError())
        );
#else
    throw std::runtime_error
    (
        "--gamg_amd_profile_roi requires AMDProfileController"
    );
#endif
}

void endAmdProfileRoi()
{
    if (!amdProfileRoi) return;
#ifdef HAVE_AMDUPROF_CONTROLLER
    if (!amdProfileStrictPause())
        throw std::runtime_error
        (
            "amdProfileStrictPause failed: "
          + std::to_string(amdGetLastProfileError())
        );
#endif
}

std::size_t adviseAndCollapseHugePages
(
    const std::string& name,
    const void* const data,
    const std::size_t bytes,
    const bool report
)
{
    if (!data || !bytes) return 0;

    const long pageSizeValue = ::sysconf(_SC_PAGESIZE);
    if (pageSizeValue <= 0)
        throw std::runtime_error("sysconf(_SC_PAGESIZE) failed");
    const std::uintptr_t pageSize =
        static_cast<std::uintptr_t>(pageSizeValue);
    const std::uintptr_t first = reinterpret_cast<std::uintptr_t>(data);
    const std::uintptr_t last = first + bytes;
    const std::uintptr_t begin = (first + pageSize - 1) & ~(pageSize - 1);
    const std::uintptr_t end = last & ~(pageSize - 1);
    if (end <= begin) return 0;

    void* const address = reinterpret_cast<void*>(begin);
    const std::size_t length = static_cast<std::size_t>(end - begin);
    if (::madvise(address, length, MADV_HUGEPAGE) != 0)
    {
        const int error = errno;
        throw std::runtime_error
        (
            "MADV_HUGEPAGE failed for " + name + ": "
          + std::strerror(error) + " (errno=" + std::to_string(error) + ')'
        );
    }
    constexpr std::uintptr_t hugePageSize = 2u*1024u*1024u;
    const std::uintptr_t collapseBegin =
        (first + hugePageSize - 1) & ~(hugePageSize - 1);
    const std::uintptr_t collapseEnd = last & ~(hugePageSize - 1);
    const std::size_t collapseLength = collapseEnd > collapseBegin
        ? static_cast<std::size_t>(collapseEnd - collapseBegin) : 0;
    if
    (
        collapseLength
     && ::madvise
        (
            reinterpret_cast<void*>(collapseBegin),
            collapseLength,
            MADV_COLLAPSE
        ) != 0
    )
    {
        const int error = errno;
        throw std::runtime_error
        (
            "MADV_COLLAPSE failed for " + name + ": "
          + std::strerror(error) + " (errno=" + std::to_string(error) + ')'
        );
    }
    if (report)
        std::cout << "ExplicitTHP " << name << " bytes=" << bytes
            << " advisedBytes=" << length
            << " collapseBytes=" << collapseLength
            << " MADV_HUGEPAGE=PASS MADV_COLLAPSE="
            << (collapseLength ? "PASS" : "SKIP") << '\n';
    return length;
}

template<class Container>
std::size_t adviseAndCollapseHugePages
(
    const std::string& name,
    const Container& values,
    const bool report = true
)
{
    using Value = typename Container::value_type;
    return adviseAndCollapseHugePages
    (
        name, values.data(), values.size()*sizeof(Value), report
    );
}

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

double percentileDouble(std::vector<double> values, const double fraction)
{
    if (values.empty()) return 0;
    const std::size_t index = static_cast<std::size_t>
    (
        fraction*static_cast<double>(values.size() - 1)
    );
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}

struct ChunkJumpStats
{
    double median = 0;
    double p90 = 0;
    double p99 = 0;
};

ChunkJumpStats chunkJumpStats(const fwgamg::Matrix& matrix)
{
    std::vector<Foam::label> jumps;
    const std::vector<Foam::label>& starts = matrix.levelStarts();
    const std::vector<Foam::label>& cells = matrix.newToOld();
    jumps.reserve(starts.size() > 2 ? starts.size() - 2 : 0);
    for (std::size_t chunk=1; chunk + 1<starts.size(); ++chunk)
    {
        const Foam::label previous = cells[starts[chunk] - 1];
        const Foam::label next = cells[starts[chunk]];
        jumps.push_back(previous > next ? previous - next : next - previous);
    }
    ChunkJumpStats result;
    result.median = percentile(jumps, 0.50);
    result.p90 = percentile(jumps, 0.90);
    result.p99 = percentile(jumps, 0.99);
    return result;
}

struct OrderingDiagnosticStats
{
    std::size_t chunks = 0;
    double chunkMin = 0;
    double chunkP10 = 0;
    double chunkP50 = 0;
    double chunkP90 = 0;
    double chunkMean = 0;
    double chunkMax = 0;
    double targetP10 = 0;
    double targetP50 = 0;
    double targetP90 = 0;
    double readyP10 = 0;
    double readyP50 = 0;
    double readyP90 = 0;
    std::uint64_t rowJumpTotal = 0;
    double rowJumpP90 = 0;
    double rowJumpP99 = 0;
    std::size_t largeJumps = 0;
};

OrderingDiagnosticStats orderingDiagnosticStats
(
    const fwgamg::Matrix& matrix,
    const std::vector<smootherTest::AdaptiveReadyChunkDiagnostic>* diagnostics,
    const double largeJumpThreshold
)
{
    OrderingDiagnosticStats result;
    std::vector<Foam::label> chunkSizes;
    std::vector<Foam::label> targets;
    std::vector<Foam::label> readySizes;
    if (diagnostics)
    {
        chunkSizes.reserve(diagnostics->size());
        targets.reserve(diagnostics->size());
        readySizes.reserve(diagnostics->size());
        for (const auto& diagnostic : *diagnostics)
        {
            chunkSizes.push_back(diagnostic.chosenChunkSize);
            targets.push_back(diagnostic.targetWidth);
            readySizes.push_back(diagnostic.readySize);
        }
    }
    else
    {
        const std::vector<Foam::label>& starts = matrix.levelStarts();
        chunkSizes.reserve(starts.empty() ? 0 : starts.size() - 1);
        for (std::size_t chunk=0; chunk + 1<starts.size(); ++chunk)
            chunkSizes.push_back(starts[chunk + 1] - starts[chunk]);
    }
    result.chunks = chunkSizes.size();
    if (!chunkSizes.empty())
    {
        result.chunkMin = *std::min_element(chunkSizes.begin(), chunkSizes.end());
        result.chunkMax = *std::max_element(chunkSizes.begin(), chunkSizes.end());
        result.chunkMean = std::accumulate
        (
            chunkSizes.begin(), chunkSizes.end(), 0.0
        )/chunkSizes.size();
        result.chunkP10 = percentile(chunkSizes, 0.10);
        result.chunkP50 = percentile(chunkSizes, 0.50);
        result.chunkP90 = percentile(chunkSizes, 0.90);
    }
    result.targetP10 = percentile(targets, 0.10);
    result.targetP50 = percentile(targets, 0.50);
    result.targetP90 = percentile(targets, 0.90);
    result.readyP10 = percentile(readySizes, 0.10);
    result.readyP50 = percentile(readySizes, 0.50);
    result.readyP90 = percentile(readySizes, 0.90);

    std::vector<Foam::label> rowJumps;
    const std::vector<Foam::label>& cells = matrix.newToOld();
    rowJumps.reserve(cells.empty() ? 0 : cells.size() - 1);
    for (std::size_t row=1; row<cells.size(); ++row)
    {
        const Foam::label jump = cells[row] > cells[row - 1]
            ? cells[row] - cells[row - 1]
            : cells[row - 1] - cells[row];
        rowJumps.push_back(jump);
        result.rowJumpTotal += static_cast<std::uint64_t>(jump);
        result.largeJumps += jump >= largeJumpThreshold;
    }
    result.rowJumpP90 = percentile(rowJumps, 0.90);
    result.rowJumpP99 = percentile(rowJumps, 0.99);
    return result;
}

double setupTotalMilliseconds(const fwgamg::LevelSetupTiming& timing)
{
    return 1e3*
    (
        timing.scheduleSeconds
      + timing.permutationSeconds
      + timing.inversePermutationSeconds
      + timing.matrixPackingSeconds
      + timing.flatEdgeMetadataSeconds
      + timing.directFlatEdgeMetadataSeconds
    );
}

struct OrientedChunkStats
{
    std::size_t total = 0;
    std::size_t reversed = 0;
    bool valid = true;
};

OrientedChunkStats orientedChunkStats
(
    const fwgamg::Matrix& bounded,
    const fwgamg::Matrix& oriented
)
{
    OrientedChunkStats result;
    if (bounded.levelStarts() != oriented.levelStarts())
    {
        result.valid = false;
        return result;
    }
    const auto& starts = bounded.levelStarts();
    const auto& left = bounded.newToOld();
    const auto& right = oriented.newToOld();
    if (left.size() != right.size())
    {
        result.valid = false;
        return result;
    }
    result.total = starts.empty() ? 0 : starts.size() - 1;
    for (std::size_t chunk=0; chunk<result.total; ++chunk)
    {
        const std::size_t begin = starts[chunk];
        const std::size_t end = starts[chunk + 1];
        const bool ascending = std::equal
        (
            left.begin() + begin, left.begin() + end,
            right.begin() + begin
        );
        const bool descending = std::equal
        (
            left.begin() + begin, left.begin() + end,
            std::make_reverse_iterator(right.begin() + end)
        );
        if (!ascending && !descending)
        {
            result.valid = false;
            return result;
        }
        if (!ascending && descending) ++result.reversed;
    }
    return result;
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

struct SmootherProfileProblem
{
    smootherTest::PolyMeshTopology mesh;
    Foam::lduMatrix fineMatrix;
    gamg::GAMGMatrix ldu;
    fwgamg::Matrix ih1;
    fwgamg::Matrix ih1w2;
    Foam::scalarField naturalSource;
    Foam::scalarField ih1Source;
    Foam::scalarField ih1w2Source;

    explicit SmootherProfileProblem(const std::string& meshDirectory)
    :
        mesh(smootherTest::PolyMeshReader::read(meshDirectory)),
        fineMatrix(harness::makeMatrix(mesh)),
        ldu(gamg::GAMGMatrix::fromLduMatrix(fineMatrix)),
        ih1(fwgamg::Matrix::build(nativeData(ldu), 1024)),
        ih1w2(fwgamg::Matrix::buildOriented(nativeData(ldu), 1024)),
        naturalSource(ldu.nCells()),
        ih1Source(ldu.nCells()),
        ih1w2Source(ldu.nCells())
    {
        for (Foam::label cell=0; cell<ldu.nCells(); ++cell)
        {
            naturalSource[cell] = 0.75 + std::sin(0.019*cell)
                + 0.2*std::cos(0.071*cell);
        }
        for (Foam::label row=0; row<ih1.nCells(); ++row)
            ih1Source[row] = naturalSource[ih1.newToOld()[row]];
        for (Foam::label row=0; row<ih1w2.nCells(); ++row)
            ih1w2Source[row] = naturalSource[ih1w2.newToOld()[row]];
    }
};

struct Problem
{
    struct WidthSweep
    {
        Foam::label width = 0;
        std::vector<fwgamg::Matrix> matrices;
        std::vector<Foam::scalarField> sources;
        std::vector<fwgamg::LevelSetupTiming> setup;
    };

    struct AdaptiveGapSweep
    {
        std::string name;
        Foam::label minWidth = 0;
        Foam::label targetWidth = 0;
        Foam::label maxWidth = 0;
        std::vector<fwgamg::Matrix> matrices;
        std::vector<Foam::scalarField> sources;
        std::vector<fwgamg::LevelSetupTiming> setup;
    };

    struct ReadyFractionSweep
    {
        std::string name;
        Foam::label divisor = 0;
        fwgamg::Matrix matrix;
        Foam::scalarField source;
        fwgamg::LevelSetupTiming setup;
        std::vector<smootherTest::AdaptiveReadyChunkDiagnostic> diagnostics;
        bool deterministic = false;
    };

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
    std::vector<WidthSweep> widthSweeps;
    std::vector<fwgamg::Matrix> unboundedMatrices;
    std::vector<Foam::scalarField> unboundedSources;
    std::vector<WidthSweep> orientedWidthSweeps;
    std::vector<WidthSweep> slidingWindowSweeps;
    std::vector<AdaptiveGapSweep> adaptiveGapSweeps;
    std::vector<ReadyFractionSweep> readyFractionSweeps;
    std::vector<fwgamg::Matrix> slicedMatrices;
    std::vector<Foam::scalarField> slicedSources;
    std::vector<fwgamg::Matrix> continuingMatrices;
    std::vector<Foam::scalarField> continuingSources;
    std::vector<fwgamg::Matrix> frontierContinuingMatrices;
    std::vector<Foam::scalarField> frontierContinuingSources;

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

        for (const Foam::label width : boundedWidths())
        {
            WidthSweep variant;
            variant.width = width;
            variant.matrices.reserve(baseline.matrices().size());
            variant.sources.reserve(baseline.matrices().size());
            variant.setup.reserve(baseline.matrices().size());
            for
            (
                std::size_t level=0;
                level<baseline.matrices().size();
                ++level
            )
            {
                fwgamg::LevelSetupTiming setup;
                variant.matrices.push_back
                (
                    fwgamg::Matrix::build
                    (
                        nativeData(baseline.matrices()[level]), width, &setup
                    )
                );
                variant.setup.push_back(setup);
                const fwgamg::Matrix& matrix = variant.matrices.back();
                Foam::scalarField source(matrix.nCells());
                for (Foam::label row=0; row<matrix.nCells(); ++row)
                {
                    source[row] =
                        naturalSource[level][matrix.newToOld()[row]];
                }
                variant.sources.push_back(std::move(source));
            }
            widthSweeps.push_back(std::move(variant));
        }

        unboundedMatrices.reserve(baseline.matrices().size());
        unboundedSources.reserve(baseline.matrices().size());
        for (std::size_t level=0; level<baseline.matrices().size(); ++level)
        {
            unboundedMatrices.push_back
            (
                fwgamg::Matrix::buildUnbounded
                (
                    nativeData(baseline.matrices()[level])
                )
            );
            const fwgamg::Matrix& matrix = unboundedMatrices.back();
            Foam::scalarField source(matrix.nCells());
            for (Foam::label row=0; row<matrix.nCells(); ++row)
                source[row] = naturalSource[level][matrix.newToOld()[row]];
            unboundedSources.push_back(std::move(source));
        }

        for (const Foam::label width : orientedWidths())
        {
            WidthSweep variant;
            variant.width = width;
            variant.matrices.reserve(baseline.matrices().size());
            variant.sources.reserve(baseline.matrices().size());
            for
            (
                std::size_t level=0;
                level<baseline.matrices().size();
                ++level
            )
            {
                variant.matrices.push_back
                (
                    fwgamg::Matrix::buildOriented
                    (
                        nativeData(baseline.matrices()[level]), width
                    )
                );
                const fwgamg::Matrix& matrix = variant.matrices.back();
                Foam::scalarField source(matrix.nCells());
                for (Foam::label row=0; row<matrix.nCells(); ++row)
                    source[row] = naturalSource[level][matrix.newToOld()[row]];
                variant.sources.push_back(std::move(source));
            }
            orientedWidthSweeps.push_back(std::move(variant));
        }

        if (ih1w3Experiment)
        {
            for (const Foam::label step : {64, 128, 256})
            {
                WidthSweep variant;
                variant.width = step;
                variant.matrices.reserve(baseline.matrices().size());
                variant.sources.reserve(baseline.matrices().size());
                for
                (
                    std::size_t level=0;
                    level<baseline.matrices().size();
                    ++level
                )
                {
                    variant.matrices.push_back
                    (
                        fwgamg::Matrix::buildSlidingWindow
                        (
                            nativeData(baseline.matrices()[level]),
                            1024,
                            step
                        )
                    );
                    const fwgamg::Matrix& matrix = variant.matrices.back();
                    Foam::scalarField source(matrix.nCells());
                    for (Foam::label row=0; row<matrix.nCells(); ++row)
                    {
                        source[row] =
                            naturalSource[level][matrix.newToOld()[row]];
                    }
                    variant.sources.push_back(std::move(source));
                }
                slidingWindowSweeps.push_back(std::move(variant));
            }
        }

        if
        (
            (ih1w4Experiment && ih1w4Batch != "fixed")
         || ih1w5Experiment
        )
        {
            struct Controls
            {
                const char* name;
                Foam::label minWidth, targetWidth, maxWidth;
            };
            const std::vector<Controls> allConfigurations
            {
                Controls{"IH1W4_A", 768, 1024, 1536},
                Controls{"IH1W4_B", 512, 1024, 2048},
                Controls{"IH1W4_C", 1024, 1024, 2048}
            };
            std::vector<Controls> configurations;
            if (ih1w5Experiment)
            {
                for (const Controls controls : allConfigurations)
                    if (controls.name == ih1w5W4Profile)
                        configurations.push_back(controls);
                if (configurations.empty())
                    throw std::runtime_error("invalid IH1W5 W4 profile");
            }
            else
                configurations = allConfigurations;
            for
            (
                const Controls controls : configurations
            )
            {
                AdaptiveGapSweep variant;
                variant.name = controls.name;
                variant.minWidth = controls.minWidth;
                variant.targetWidth = controls.targetWidth;
                variant.maxWidth = controls.maxWidth;
                variant.matrices.reserve(baseline.matrices().size());
                variant.sources.reserve(baseline.matrices().size());
                variant.setup.reserve(baseline.matrices().size());
                const std::size_t buildLevels = ih1w5Experiment
                    ? 1 : baseline.matrices().size();
                for
                (
                    std::size_t level=0;
                    level<buildLevels;
                    ++level
                )
                {
                    fwgamg::LevelSetupTiming setup;
                    variant.matrices.push_back
                    (
                        fwgamg::Matrix::buildAdaptiveGap
                        (
                            nativeData(baseline.matrices()[level]),
                            controls.minWidth,
                            controls.targetWidth,
                            controls.maxWidth,
                            &setup
                        )
                    );
                    variant.setup.push_back(setup);
                    const fwgamg::Matrix& matrix = variant.matrices.back();
                    Foam::scalarField source(matrix.nCells());
                    for (Foam::label row=0; row<matrix.nCells(); ++row)
                        source[row] =
                            naturalSource[level][matrix.newToOld()[row]];
                    variant.sources.push_back(std::move(source));
                }
                adaptiveGapSweeps.push_back(std::move(variant));
            }
        }

        if (ih1w5Experiment)
        {
            const gamg::GAMGMatrix& level0 = baseline.matrices()[0];
            const fwgamg::NativeMatrixData native = nativeData(level0);
            Foam::labelField ownerStarts(level0.nCells() + 1, 0);
            for (const Foam::label owner : level0.lowerAddr())
                ++ownerStarts[owner + 1];
            for (Foam::label cell=0; cell<level0.nCells(); ++cell)
                ownerStarts[cell + 1] += ownerStarts[cell];
            for (const Foam::label divisor : {4, 8})
            {
                ReadyFractionSweep variant;
                variant.name = "IH1W5_Q" + std::to_string(divisor);
                variant.divisor = divisor;
                variant.matrix = fwgamg::Matrix::buildReadyFractionAdaptiveGap
                (
                    native, divisor, &variant.diagnostics, &variant.setup
                );
                variant.source.resize(variant.matrix.nCells());
                for (Foam::label row=0; row<variant.matrix.nCells(); ++row)
                    variant.source[row] =
                        naturalSource.front()[variant.matrix.newToOld()[row]];
                const smootherTest::IndexKahnOrdering repeated =
                    smootherTest::makeIndexKahnOrderingReadyFractionAdaptiveGap
                    (
                        level0.nCells(), ownerStarts, level0.upperAddr(), divisor
                    );
                variant.deterministic =
                    repeated.newToOld == variant.matrix.newToOld()
                 && repeated.levelStarts == variant.matrix.levelStarts();
                readyFractionSweeps.push_back(std::move(variant));
            }
        }

        slicedMatrices.reserve(baseline.matrices().size());
        slicedSources.reserve(baseline.matrices().size());
        for (std::size_t level=0; level<baseline.matrices().size(); ++level)
        {
            slicedMatrices.push_back
            (
                fwgamg::Matrix::buildSliced
                (
                    nativeData(baseline.matrices()[level]), 1024
                )
            );
            const fwgamg::Matrix& matrix = slicedMatrices.back();
            Foam::scalarField source(matrix.nCells());
            for (Foam::label row=0; row<matrix.nCells(); ++row)
                source[row] = naturalSource[level][matrix.newToOld()[row]];
            slicedSources.push_back(std::move(source));
        }

        continuingMatrices.reserve(baseline.matrices().size());
        continuingSources.reserve(baseline.matrices().size());
        for (std::size_t level=0; level<baseline.matrices().size(); ++level)
        {
            continuingMatrices.push_back
            (
                fwgamg::Matrix::buildContinuing
                (
                    nativeData(baseline.matrices()[level]), 1024
                )
            );
            const fwgamg::Matrix& matrix = continuingMatrices.back();
            Foam::scalarField source(matrix.nCells());
            for (Foam::label row=0; row<matrix.nCells(); ++row)
                source[row] = naturalSource[level][matrix.newToOld()[row]];
            continuingSources.push_back(std::move(source));
        }

        frontierContinuingMatrices.reserve(baseline.matrices().size());
        frontierContinuingSources.reserve(baseline.matrices().size());
        for (std::size_t level=0; level<baseline.matrices().size(); ++level)
        {
            frontierContinuingMatrices.push_back
            (
                fwgamg::Matrix::buildFrontierContinuing
                (
                    nativeData(baseline.matrices()[level]), 1024
                )
            );
            const fwgamg::Matrix& matrix =
                frontierContinuingMatrices.back();
            Foam::scalarField source(matrix.nCells());
            for (Foam::label row=0; row<matrix.nCells(); ++row)
                source[row] = naturalSource[level][matrix.newToOld()[row]];
            frontierContinuingSources.push_back(std::move(source));
        }
    }
};

// Memory-bounded L0-only fixture for the IH1W5 portability sweep.  Keeping
// this separate avoids constructing unrelated full GAMG hierarchy views in
// every process-isolated runner task.
struct Ih1w5FocusedProblem
{
    struct Variant
    {
        std::string name;
        fwgamg::Matrix matrix;
        Foam::scalarField source;
        fwgamg::LevelSetupTiming setup;
        std::vector<smootherTest::AdaptiveReadyChunkDiagnostic> diagnostics;
        bool deterministic = true;
    };

    smootherTest::PolyMeshTopology mesh;
    Foam::lduMatrix fineMatrix;
    gamg::GAMGMatrix ldu;
    Foam::scalarField naturalSource;
    std::vector<std::size_t> naturalWidths;
    std::vector<Variant> variants;

    explicit Ih1w5FocusedProblem(const std::string& meshDirectory)
    :
        mesh(smootherTest::PolyMeshReader::read(meshDirectory)),
        fineMatrix(harness::makeMatrix(mesh)),
        ldu(gamg::GAMGMatrix::fromLduMatrix(fineMatrix)),
        naturalSource(ldu.nCells()),
        naturalWidths(naturalDependencyWidths(ldu))
    {
        for (Foam::label cell=0; cell<ldu.nCells(); ++cell)
            naturalSource[cell] = 0.75 + std::sin(0.019*cell)
                + 0.2*std::cos(0.071*cell);

        const fwgamg::NativeMatrixData native = nativeData(ldu);
        const auto add = [&](Variant&& variant)
        {
            variant.source.resize(variant.matrix.nCells());
            for (Foam::label row=0; row<variant.matrix.nCells(); ++row)
                variant.source[row] =
                    naturalSource[variant.matrix.newToOld()[row]];
            variants.push_back(std::move(variant));
        };

        Variant bounded;
        bounded.name = "IH1W1024";
        bounded.matrix = fwgamg::Matrix::build
        (
            native, 1024, &bounded.setup
        );
        add(std::move(bounded));

        Foam::label minWidth = 0, targetWidth = 1024, maxWidth = 0;
        if (ih1w5W4Profile == "IH1W4_A")
            minWidth = 768, maxWidth = 1536;
        else if (ih1w5W4Profile == "IH1W4_B")
            minWidth = 512, maxWidth = 2048;
        else if (ih1w5W4Profile == "IH1W4_C")
            minWidth = 1024, maxWidth = 2048;
        else
            throw std::runtime_error("invalid IH1W5 W4 profile");
        Variant w4;
        w4.name = ih1w5W4Profile;
        w4.matrix = fwgamg::Matrix::buildAdaptiveGap
        (
            native, minWidth, targetWidth, maxWidth, &w4.setup
        );
        add(std::move(w4));

        Foam::labelField ownerStarts(ldu.nCells() + 1, 0);
        for (const Foam::label owner : ldu.lowerAddr())
            ++ownerStarts[owner + 1];
        for (Foam::label cell=0; cell<ldu.nCells(); ++cell)
            ownerStarts[cell + 1] += ownerStarts[cell];
        for (const Foam::label divisor : {4, 8})
        {
            Variant adaptive;
            adaptive.name = "IH1W5_Q" + std::to_string(divisor);
            adaptive.matrix = fwgamg::Matrix::buildReadyFractionAdaptiveGap
            (
                native, divisor, &adaptive.diagnostics, &adaptive.setup
            );
            const smootherTest::IndexKahnOrdering repeated =
                smootherTest::makeIndexKahnOrderingReadyFractionAdaptiveGap
                (
                    ldu.nCells(), ownerStarts, ldu.upperAddr(), divisor
                );
            adaptive.deterministic =
                repeated.newToOld == adaptive.matrix.newToOld()
             && repeated.levelStarts == adaptive.matrix.levelStarts();
            add(std::move(adaptive));
        }
    }
};

struct ReleaseImpactFocusedProblem
{
    struct Variant
    {
        Foam::label width = 0;
        fwgamg::Matrix matrix;
        Foam::scalarField source;
        fwgamg::LevelSetupTiming setup;
        std::vector<smootherTest::ReleaseImpactChunkDiagnostic> diagnostics;
        bool deterministic = false;
    };

    smootherTest::PolyMeshTopology mesh;
    Foam::lduMatrix fineMatrix;
    gamg::GAMGMatrix ldu;
    Foam::scalarField naturalSource;
    std::vector<std::size_t> naturalWidths;
    std::vector<Variant> variants;

    explicit ReleaseImpactFocusedProblem(const std::string& meshDirectory)
    :
        mesh(smootherTest::PolyMeshReader::read(meshDirectory)),
        fineMatrix(harness::makeMatrix(mesh)),
        ldu(gamg::GAMGMatrix::fromLduMatrix(fineMatrix)),
        naturalSource(ldu.nCells()),
        naturalWidths(naturalDependencyWidths(ldu))
    {
        for (Foam::label cell=0; cell<ldu.nCells(); ++cell)
            naturalSource[cell] = 0.75 + std::sin(0.019*cell)
                + 0.2*std::cos(0.071*cell);

        Foam::labelField ownerStarts(ldu.nCells() + 1, 0);
        for (const Foam::label owner : ldu.lowerAddr())
            ++ownerStarts[owner + 1];
        for (Foam::label cell=0; cell<ldu.nCells(); ++cell)
            ownerStarts[cell + 1] += ownerStarts[cell];

        const fwgamg::NativeMatrixData native = nativeData(ldu);
        for (const Foam::label width : {256, 512, 1024, 2048, 4096, 8192})
        {
            Variant variant;
            variant.width = width;
            variant.matrix = fwgamg::Matrix::buildReleaseDiagnostic
            (
                native, width, &variant.diagnostics, &variant.setup
            );
            variant.source.resize(variant.matrix.nCells());
            for (Foam::label row=0; row<variant.matrix.nCells(); ++row)
                variant.source[row] =
                    naturalSource[variant.matrix.newToOld()[row]];
            const smootherTest::IndexKahnOrdering ordinary =
                smootherTest::makeIndexKahnOrdering
                (
                    ldu.nCells(), ownerStarts, ldu.upperAddr(), width
                );
            const smootherTest::IndexKahnOrdering repeated =
                smootherTest::makeIndexKahnOrderingReleaseDiagnostic
                (
                    ldu.nCells(), ownerStarts, ldu.upperAddr(), width, nullptr
                );
            variant.deterministic =
                repeated.newToOld == variant.matrix.newToOld()
             && repeated.levelStarts == variant.matrix.levelStarts()
             && ordinary.newToOld == variant.matrix.newToOld()
             && ordinary.levelStarts == variant.matrix.levelStarts();
            variants.push_back(std::move(variant));
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
        else if (argument == "--gamg_explicit_thp") explicitHugePages = true;
        else if (argument == "--gamg_production_ih1_only")
            productionIh1Only = true;
        else if (argument == "--gamg_ih1w2_width_sweep")
            ih1w2WidthSweep = true;
        else if (argument == "--gamg_ih1w3") ih1w3Experiment = true;
        else if (argument == "--gamg_ih1w4") ih1w4Experiment = true;
        else if (argument == "--gamg_ih1w5") ih1w5Experiment = true;
        else if (argument == "--gamg_release_impact")
            releaseImpactExperiment = true;
        else if (argument.rfind("--gamg_ih1w5_w4=", 0) == 0)
        {
            ih1w5Experiment = true;
            ih1w5W4Profile = argument.substr
            (
                sizeof("--gamg_ih1w5_w4=") - 1
            );
        }
        else if (argument.rfind("--gamg_ih1w5_trace_dir=", 0) == 0)
        {
            ih1w5Experiment = true;
            ih1w5TraceDirectory = argument.substr
            (
                sizeof("--gamg_ih1w5_trace_dir=") - 1
            );
        }
        else if (argument.rfind("--gamg_release_impact_trace_dir=", 0) == 0)
        {
            releaseImpactExperiment = true;
            releaseImpactTraceDirectory = argument.substr
            (
                sizeof("--gamg_release_impact_trace_dir=") - 1
            );
        }
        else if (argument.rfind("--gamg_ih1w4_batch=", 0) == 0)
        {
            ih1w4Experiment = true;
            ih1w4Batch = argument.substr(19);
            if (ih1w4Batch != "all" && ih1w4Batch != "fixed"
             && ih1w4Batch != "adaptive")
                throw std::runtime_error
                (
                    "--gamg_ih1w4_batch must be all, fixed or adaptive"
                );
        }
        else if (argument.rfind("--gamg_dump_permutations=", 0) == 0)
            permutationDumpDirectory = argument.substr(25);
        else if (argument == "--gamg_dump_permutations_only")
            permutationDumpOnly = true;
        else if (argument.rfind("--gamg_mesh_name=", 0) == 0)
            benchmarkMeshName = argument.substr(17);
        else if (argument.rfind("--gamg_vtune_smoother=", 0) == 0)
            vtuneSmoother = argument.substr(22);
        else if (argument.rfind("--gamg_vtune_sweeps=", 0) == 0)
            vtuneSweeps = std::stoi(argument.substr(20));
        else if (argument.rfind("--gamg_vtune_warmup_sweeps=", 0) == 0)
            vtuneWarmupSweeps = std::stoi(argument.substr(27));
        else if (argument == "--gamg_amd_profile_roi") amdProfileRoi = true;
        else if (argument == "--gamg_amd_pcm_signal_roi")
            amdPcmSignalRoi = true;
        else argv[destination++] = argv[source];
    }
    argc = destination;
    if (mesh.empty()) throw std::runtime_error("--mesh=POLYMESH_DIR is required");
    return mesh;
}

void prepareHugePages(const Problem& problem)
{
    if (!explicitHugePages) return;

    for (std::size_t level=0; level<problem.natural.size(); ++level)
    {
        // The focused IH1W4 experiment registers only L0.  Avoid consuming
        // explicit collapse capacity for coarse-level arrays that never enter
        // the timed benchmark.
        if ((ih1w4Experiment || ih1w5Experiment) && level != 0) continue;
        const gamg::GAMGMatrix& ldu = problem.baseline.matrices()[level];
        const fwgamg::Matrix& natural = problem.natural[level];
        const std::string prefix = "L" + std::to_string(level) + '.';

        adviseAndCollapseHugePages
        (
            prefix + "LDU.lowerAddr", ldu.lowerAddr()
        );
        adviseAndCollapseHugePages
        (
            prefix + "LDU.upperAddr", ldu.upperAddr()
        );
        adviseAndCollapseHugePages(prefix + "LDU.upper", ldu.upper());
        adviseAndCollapseHugePages(prefix + "LDU.diag", ldu.diag());
        adviseAndCollapseHugePages
        (
            prefix + "LDU.field", problem.naturalSource[level]
        );

        if (!(ih1w4Experiment || ih1w5Experiment))
        {
            adviseAndCollapseHugePages
            (
                prefix + "fw.directEdgeOwners", natural.directEdgeOwners()
            );
            adviseAndCollapseHugePages
            (
                prefix + "fw.directEdgeColumns", natural.directEdgeColumns()
            );
            adviseAndCollapseHugePages
            (
                prefix + "fw.directEdgeCoeffs", natural.directEdgeCoeffs()
            );
            adviseAndCollapseHugePages
            (
                prefix + "fw.lduEdgeOwners",
                natural.lduOrderDirectEdgeOwners()
            );
            adviseAndCollapseHugePages
            (
                prefix + "fw.lduEdgeColumns",
                natural.lduOrderDirectEdgeColumns()
            );
            adviseAndCollapseHugePages
            (
                prefix + "fw.lduEdgeCoeffs",
                natural.lduOrderDirectEdgeCoeffs()
            );
            adviseAndCollapseHugePages(prefix + "fw.diag", natural.diag());
        }

        for (const Problem::WidthSweep& variant : problem.widthSweeps)
        {
            const fwgamg::Matrix& matrix = variant.matrices[level];
            const std::string widthPrefix = prefix + "IH1W"
                + std::to_string(variant.width) + '.';
            adviseAndCollapseHugePages
            (
                widthPrefix + "levelStarts", matrix.levelStarts()
            );
            adviseAndCollapseHugePages
            (
                widthPrefix + "rowStarts", matrix.rowStarts()
            );
            adviseAndCollapseHugePages(widthPrefix + "cols", matrix.cols());
            adviseAndCollapseHugePages
            (
                widthPrefix + "coeffs", matrix.coeffs()
            );
            adviseAndCollapseHugePages(widthPrefix + "diag", matrix.diag());
            adviseAndCollapseHugePages
            (
                widthPrefix + "source", variant.sources[level]
            );
        }

        for (const Problem::WidthSweep& variant : problem.orientedWidthSweeps)
        {
            const fwgamg::Matrix& oriented = variant.matrices[level];
            const std::string orientedPrefix = prefix + "IH1W2_"
                + std::to_string(variant.width) + '.';
            adviseAndCollapseHugePages
            (
                orientedPrefix + "levelStarts", oriented.levelStarts()
            );
            adviseAndCollapseHugePages
            (
                orientedPrefix + "rowStarts", oriented.rowStarts()
            );
            adviseAndCollapseHugePages
            (
                orientedPrefix + "cols", oriented.cols()
            );
            adviseAndCollapseHugePages
            (
                orientedPrefix + "coeffs", oriented.coeffs()
            );
            adviseAndCollapseHugePages
            (
                orientedPrefix + "diag", oriented.diag()
            );
            adviseAndCollapseHugePages
            (
                orientedPrefix + "source", variant.sources[level]
            );
        }

        for (const Problem::WidthSweep& variant : problem.slidingWindowSweeps)
        {
            const fwgamg::Matrix& sliding = variant.matrices[level];
            const std::string slidingPrefix = prefix + "IH1W3_W1024_S"
                + std::to_string(variant.width) + '.';
            adviseAndCollapseHugePages
            (
                slidingPrefix + "levelStarts", sliding.levelStarts()
            );
            adviseAndCollapseHugePages
            (
                slidingPrefix + "rowStarts", sliding.rowStarts()
            );
            adviseAndCollapseHugePages
            (
                slidingPrefix + "cols", sliding.cols()
            );
            adviseAndCollapseHugePages
            (
                slidingPrefix + "coeffs", sliding.coeffs()
            );
            adviseAndCollapseHugePages
            (
                slidingPrefix + "diag", sliding.diag()
            );
            adviseAndCollapseHugePages
            (
                slidingPrefix + "source", variant.sources[level]
            );
        }

        for
        (
            const Problem::AdaptiveGapSweep& variant
          : problem.adaptiveGapSweeps
        )
        {
            const fwgamg::Matrix& adaptive = variant.matrices[level];
            const std::string adaptivePrefix = prefix + variant.name + '.';
            adviseAndCollapseHugePages
            (
                adaptivePrefix + "levelStarts", adaptive.levelStarts()
            );
            adviseAndCollapseHugePages
            (
                adaptivePrefix + "rowStarts", adaptive.rowStarts()
            );
            adviseAndCollapseHugePages
            (
                adaptivePrefix + "cols", adaptive.cols()
            );
            adviseAndCollapseHugePages
            (
                adaptivePrefix + "coeffs", adaptive.coeffs()
            );
            adviseAndCollapseHugePages
            (
                adaptivePrefix + "diag", adaptive.diag()
            );
            adviseAndCollapseHugePages
            (
                adaptivePrefix + "source", variant.sources[level]
            );
        }

        for (const Problem::ReadyFractionSweep& variant : problem.readyFractionSweeps)
        {
            const fwgamg::Matrix& adaptive = variant.matrix;
            const std::string adaptivePrefix = prefix + variant.name + '.';
            adviseAndCollapseHugePages
            (
                adaptivePrefix + "levelStarts", adaptive.levelStarts()
            );
            adviseAndCollapseHugePages
            (
                adaptivePrefix + "rowStarts", adaptive.rowStarts()
            );
            adviseAndCollapseHugePages
            (
                adaptivePrefix + "cols", adaptive.cols()
            );
            adviseAndCollapseHugePages
            (
                adaptivePrefix + "coeffs", adaptive.coeffs()
            );
            adviseAndCollapseHugePages
            (
                adaptivePrefix + "diag", adaptive.diag()
            );
            adviseAndCollapseHugePages
            (
                adaptivePrefix + "source", variant.source
            );
        }

        // The focused IH1W4 benchmark never executes the older U/S/C/W2
        // schedules.  Leaving those allocations on ordinary pages avoids
        // exhausting explicit huge-page collapse capacity on large meshes.
        if (ih1w4Experiment || ih1w5Experiment) continue;

        const fwgamg::Matrix& unbounded = problem.unboundedMatrices[level];
        const std::string unboundedPrefix = prefix + "IH1U.";
        adviseAndCollapseHugePages
        (
            unboundedPrefix + "levelStarts", unbounded.levelStarts()
        );
        adviseAndCollapseHugePages
        (
            unboundedPrefix + "rowStarts", unbounded.rowStarts()
        );
        adviseAndCollapseHugePages
        (
            unboundedPrefix + "cols", unbounded.cols()
        );
        adviseAndCollapseHugePages
        (
            unboundedPrefix + "coeffs", unbounded.coeffs()
        );
        adviseAndCollapseHugePages
        (
            unboundedPrefix + "diag", unbounded.diag()
        );
        adviseAndCollapseHugePages
        (
            unboundedPrefix + "source", problem.unboundedSources[level]
        );

        const fwgamg::Matrix& sliced = problem.slicedMatrices[level];
        const std::string slicedPrefix = prefix + "IH1S1024.";
        adviseAndCollapseHugePages
        (
            slicedPrefix + "levelStarts", sliced.levelStarts()
        );
        adviseAndCollapseHugePages
        (
            slicedPrefix + "rowStarts", sliced.rowStarts()
        );
        adviseAndCollapseHugePages(slicedPrefix + "cols", sliced.cols());
        adviseAndCollapseHugePages(slicedPrefix + "coeffs", sliced.coeffs());
        adviseAndCollapseHugePages(slicedPrefix + "diag", sliced.diag());
        adviseAndCollapseHugePages
        (
            slicedPrefix + "source", problem.slicedSources[level]
        );

        const fwgamg::Matrix& continuing = problem.continuingMatrices[level];
        const std::string continuingPrefix = prefix + "IH1C1024.";
        adviseAndCollapseHugePages
        (
            continuingPrefix + "levelStarts", continuing.levelStarts()
        );
        adviseAndCollapseHugePages
        (
            continuingPrefix + "rowStarts", continuing.rowStarts()
        );
        adviseAndCollapseHugePages
        (
            continuingPrefix + "cols", continuing.cols()
        );
        adviseAndCollapseHugePages
        (
            continuingPrefix + "coeffs", continuing.coeffs()
        );
        adviseAndCollapseHugePages
        (
            continuingPrefix + "diag", continuing.diag()
        );
        adviseAndCollapseHugePages
        (
            continuingPrefix + "source", problem.continuingSources[level]
        );

        const fwgamg::Matrix& frontierContinuing =
            problem.frontierContinuingMatrices[level];
        const std::string frontierContinuingPrefix =
            prefix + "IH1C2_1024.";
        adviseAndCollapseHugePages
        (
            frontierContinuingPrefix + "levelStarts",
            frontierContinuing.levelStarts()
        );
        adviseAndCollapseHugePages
        (
            frontierContinuingPrefix + "rowStarts",
            frontierContinuing.rowStarts()
        );
        adviseAndCollapseHugePages
        (
            frontierContinuingPrefix + "cols", frontierContinuing.cols()
        );
        adviseAndCollapseHugePages
        (
            frontierContinuingPrefix + "coeffs", frontierContinuing.coeffs()
        );
        adviseAndCollapseHugePages
        (
            frontierContinuingPrefix + "diag", frontierContinuing.diag()
        );
        adviseAndCollapseHugePages
        (
            frontierContinuingPrefix + "source",
            problem.frontierContinuingSources[level]
        );
    }
    std::cout << std::flush;
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

void addDependencyCounters
(
    benchmark::State& state,
    const fwgamg::Matrix& schedule,
    const fwgamg::Matrix& naturalLevels
)
{
    const auto maximumLevelWidth = [](const fwgamg::Matrix& matrix)
    {
        std::size_t maximum = 0;
        for
        (
            std::size_t level=0;
            level + 1<matrix.levelStarts().size();
            ++level
        )
        {
            maximum = std::max
            (
                maximum,
                static_cast<std::size_t>
                (
                    matrix.levelStarts()[level + 1]
                  - matrix.levelStarts()[level]
                )
            );
        }
        return maximum;
    };
    const std::size_t dependencyLevels = schedule.levelStarts().size() - 1;
    const std::size_t naturalDependencyLevels =
        naturalLevels.levelStarts().size() - 1;
    state.counters["dependencyLevels"] = dependencyLevels;
    state.counters["naturalDependencyLevels"] = naturalDependencyLevels;
    state.counters["meanNaturalLevelWidth"] =
        static_cast<double>(naturalLevels.nCells())/naturalDependencyLevels;
    state.counters["maxNaturalLevelWidth"] =
        maximumLevelWidth(naturalLevels);
}

void addFocusedDependencyCounters
(
    benchmark::State& state,
    const fwgamg::Matrix& schedule,
    const std::vector<std::size_t>& naturalWidths
)
{
    const std::size_t naturalLevels = naturalWidths.size();
    state.counters["dependencyLevels"] = schedule.levelStarts().size() - 1;
    state.counters["naturalDependencyLevels"] = naturalLevels;
    state.counters["meanNaturalLevelWidth"] =
        static_cast<double>(schedule.nCells())/naturalLevels;
    state.counters["maxNaturalLevelWidth"] =
        *std::max_element(naturalWidths.begin(), naturalWidths.end());
}

void validateIh1w5Focused(const Ih1w5FocusedProblem& problem)
{
    Foam::scalarField reference(problem.ldu.nCells(), 0.0);
    gamg::gaussSeidelSmooth
    (
        reference, problem.ldu, problem.naturalSource, 1
    );
    bool allExact = true;
    bool allDeterministic = true;
    for (const Ih1w5FocusedProblem::Variant& variant : problem.variants)
    {
        Foam::scalarField psi(variant.matrix.nCells(), 0.0);
        variant.matrix.smooth(psi, variant.source, 1, false);
        Foam::scalarField naturalPsi(variant.matrix.nCells());
        for (Foam::label row=0; row<variant.matrix.nCells(); ++row)
            naturalPsi[variant.matrix.newToOld()[row]] = psi[row];
        const bool exact = naturalPsi == reference;
        allExact = allExact && exact;
        allDeterministic = allDeterministic && variant.deterministic;
        std::cout << "GAMG IH1W5 focused validation variant="
            << variant.name << " exact=" << (exact ? "PASS" : "FAIL")
            << " deterministic="
            << (variant.deterministic ? "PASS" : "FAIL")
            << " chunks=" << variant.matrix.levelStarts().size() - 1
            << '\n';
    }
    if (!allExact || !allDeterministic)
        throw std::runtime_error("IH1W5 focused validation failed");
}

void writeIh1w5FocusedChunkTraces(const Ih1w5FocusedProblem& problem)
{
    if (ih1w5TraceDirectory.empty()) return;
    const std::filesystem::path directory(ih1w5TraceDirectory);
    std::filesystem::create_directories(directory);
    const std::string meshName = safeFileName
    (
        benchmarkMeshName.empty() ? "mesh" : benchmarkMeshName
    );
    for (const Ih1w5FocusedProblem::Variant& variant : problem.variants)
    {
        if (variant.diagnostics.empty()) continue;
        const std::filesystem::path path = directory/
        (
            meshName + '_' + safeFileName(variant.name) + "_chunks.csv"
        );
        std::ofstream output(path);
        if (!output)
            throw std::runtime_error("cannot open IH1W5 chunk trace: " + path.string());
        output << "chunkIndex,readySize,targetWidth,searchMinWidth,"
            << "searchMaxWidth,chosenChunkSize,largestGap\n";
        for (std::size_t chunk=0; chunk<variant.diagnostics.size(); ++chunk)
        {
            const auto& diagnostic = variant.diagnostics[chunk];
            output << chunk << ',' << diagnostic.readySize << ','
                << diagnostic.targetWidth << ','
                << diagnostic.searchMinWidth << ','
                << diagnostic.searchMaxWidth << ','
                << diagnostic.chosenChunkSize << ','
                << diagnostic.largestGap << '\n';
        }
        if (!output)
            throw std::runtime_error("failed writing IH1W5 chunk trace: " + path.string());
    }
}

void printIh1w5FocusedDiagnostics(const Ih1w5FocusedProblem& problem)
{
    const Ih1w5FocusedProblem::Variant& baseline = problem.variants.front();
    const OrderingDiagnosticStats preliminary = orderingDiagnosticStats
    (
        baseline.matrix, nullptr, std::numeric_limits<double>::infinity()
    );
    const double threshold = preliminary.rowJumpP99;
    std::cout << "IH1W5 ordering diagnostics"
        << " baselineLargeJumpThreshold=" << threshold << '\n'
        << "variant orderingMs setupMs chunks chunkMin chunkP10 "
        << "chunkP50 chunkP90 chunkMean chunkMax targetP10 targetP50 "
        << "targetP90 readyP10 readyP50 readyP90 rowJumpTotal rowP90 "
        << "rowP99 largeJumps deterministic\n";
    for (const Ih1w5FocusedProblem::Variant& variant : problem.variants)
    {
        const auto* diagnostics = variant.diagnostics.empty()
            ? nullptr : &variant.diagnostics;
        const OrderingDiagnosticStats stats = orderingDiagnosticStats
        (
            variant.matrix, diagnostics, threshold
        );
        std::cout << variant.name << ' '
            << 1e3*variant.setup.scheduleSeconds << ' '
            << setupTotalMilliseconds(variant.setup) << ' '
            << stats.chunks << ' ' << stats.chunkMin << ' '
            << stats.chunkP10 << ' ' << stats.chunkP50 << ' '
            << stats.chunkP90 << ' ' << stats.chunkMean << ' '
            << stats.chunkMax << ' ' << stats.targetP10 << ' '
            << stats.targetP50 << ' ' << stats.targetP90 << ' '
            << stats.readyP10 << ' ' << stats.readyP50 << ' '
            << stats.readyP90 << ' ' << stats.rowJumpTotal << ' '
            << stats.rowJumpP90 << ' ' << stats.rowJumpP99 << ' '
            << stats.largeJumps << ' '
            << (variant.deterministic ? "PASS" : "FAIL") << '\n';
    }
}

void prepareIh1w5FocusedHugePages(const Ih1w5FocusedProblem& problem)
{
    if (!explicitHugePages) return;
    adviseAndCollapseHugePages("L0.LDU.lowerAddr", problem.ldu.lowerAddr());
    adviseAndCollapseHugePages("L0.LDU.upperAddr", problem.ldu.upperAddr());
    adviseAndCollapseHugePages("L0.LDU.upper", problem.ldu.upper());
    adviseAndCollapseHugePages("L0.LDU.diag", problem.ldu.diag());
    adviseAndCollapseHugePages("L0.LDU.source", problem.naturalSource);
    for (const Ih1w5FocusedProblem::Variant& variant : problem.variants)
    {
        const std::string prefix = "L0." + variant.name + '.';
        adviseAndCollapseHugePages(prefix + "levelStarts", variant.matrix.levelStarts());
        adviseAndCollapseHugePages(prefix + "rowStarts", variant.matrix.rowStarts());
        adviseAndCollapseHugePages(prefix + "cols", variant.matrix.cols());
        adviseAndCollapseHugePages(prefix + "coeffs", variant.matrix.coeffs());
        adviseAndCollapseHugePages(prefix + "diag", variant.matrix.diag());
        adviseAndCollapseHugePages(prefix + "source", variant.source);
    }
}

void registerIh1w5FocusedBenchmarks(const Ih1w5FocusedProblem& problem)
{
    const std::string prefix = "GAMGLevel/"
        + (benchmarkMeshName.empty() ? std::string() : benchmarkMeshName + '/')
        + "L00/";
    for (const Ih1w5FocusedProblem::Variant& variant : problem.variants)
    {
        const fwgamg::Matrix* const matrix = &variant.matrix;
        const Foam::scalarField* const source = &variant.source;
        const std::vector<std::size_t>* const naturalWidths =
            &problem.naturalWidths;
        benchmark::RegisterBenchmark
        (
            (prefix + variant.name + "/GS").c_str(),
            [matrix, source, naturalWidths](benchmark::State& state)
            {
                Foam::scalarField psi(matrix->nCells(), 0.0);
                if (explicitHugePages)
                    adviseAndCollapseHugePages("IH1W5.psi", psi, false);
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    matrix->smooth(psi, *source, 1, false);
                    benchmark::DoNotOptimize(psi.data());
                }
                addCounters(state, matrix->nCells(), matrix->nFaces());
                addFocusedDependencyCounters(state, *matrix, *naturalWidths);
            }
        )->MinTime(0.05);
    }
}

struct ReleaseImpactStats
{
    std::size_t chunks = 0;
    double chunkMean = 0;
    double chunkP50 = 0;
    double chunkP90 = 0;
    std::uint64_t releasedCount = 0;
    std::uint64_t releasedBeforeHead = 0;
    std::uint64_t releasedBelowCut = 0;
    double headChangedPercent = 0;
    double beforeHeadPerChunkWorkMean = 0;
    double beforeHeadPerChunkWorkMedian = 0;
    double beforeHeadFractionMean = 0;
    double beforeHeadFractionMedian = 0;
    double headBackstepP50 = 0;
    double headBackstepP90 = 0;
    double headBackstepP99 = 0;
};

ReleaseImpactStats releaseImpactStats
(
    const std::vector<smootherTest::ReleaseImpactChunkDiagnostic>& diagnostics
)
{
    ReleaseImpactStats result;
    result.chunks = diagnostics.size();
    std::vector<Foam::label> chunkSizes;
    std::vector<Foam::label> backsteps;
    std::vector<double> perChunkWork;
    std::vector<double> releasedFractions;
    chunkSizes.reserve(diagnostics.size());
    backsteps.reserve(diagnostics.size());
    perChunkWork.reserve(diagnostics.size());
    releasedFractions.reserve(diagnostics.size());
    std::size_t changed = 0;
    for (const auto& diagnostic : diagnostics)
    {
        chunkSizes.push_back(diagnostic.chosenChunkSize);
        backsteps.push_back(diagnostic.headBackstep);
        result.releasedCount += diagnostic.releasedCount;
        result.releasedBeforeHead += diagnostic.releasedBeforeHead;
        result.releasedBelowCut += diagnostic.releasedBelowCut;
        changed += diagnostic.headChanged;
        perChunkWork.push_back
        (
            diagnostic.chosenChunkSize
          ? static_cast<double>(diagnostic.releasedBeforeHead)
           /diagnostic.chosenChunkSize
          : 0.0
        );
        releasedFractions.push_back
        (
            diagnostic.releasedCount
          ? static_cast<double>(diagnostic.releasedBeforeHead)
           /diagnostic.releasedCount
          : 0.0
        );
    }
    if (!diagnostics.empty())
    {
        result.chunkMean = std::accumulate
        (
            chunkSizes.begin(), chunkSizes.end(), 0.0
        )/chunkSizes.size();
        result.headChangedPercent = 100.0*changed/diagnostics.size();
        result.beforeHeadPerChunkWorkMean = std::accumulate
        (
            perChunkWork.begin(), perChunkWork.end(), 0.0
        )/perChunkWork.size();
        result.beforeHeadFractionMean = std::accumulate
        (
            releasedFractions.begin(), releasedFractions.end(), 0.0
        )/releasedFractions.size();
    }
    result.chunkP50 = percentile(chunkSizes, 0.50);
    result.chunkP90 = percentile(chunkSizes, 0.90);
    result.beforeHeadPerChunkWorkMedian =
        percentileDouble(perChunkWork, 0.50);
    result.beforeHeadFractionMedian =
        percentileDouble(releasedFractions, 0.50);
    result.headBackstepP50 = percentile(backsteps, 0.50);
    result.headBackstepP90 = percentile(backsteps, 0.90);
    result.headBackstepP99 = percentile(backsteps, 0.99);
    return result;
}

void validateReleaseImpactFocused(const ReleaseImpactFocusedProblem& problem)
{
    Foam::scalarField reference(problem.ldu.nCells(), 0.0);
    gamg::gaussSeidelSmooth
    (
        reference, problem.ldu, problem.naturalSource, 1
    );
    bool allExact = true;
    bool allDeterministic = true;
    for (const ReleaseImpactFocusedProblem::Variant& variant : problem.variants)
    {
        Foam::scalarField psi(variant.matrix.nCells(), 0.0);
        variant.matrix.smooth(psi, variant.source, 1, false);
        Foam::scalarField naturalPsi(variant.matrix.nCells());
        for (Foam::label row=0; row<variant.matrix.nCells(); ++row)
            naturalPsi[variant.matrix.newToOld()[row]] = psi[row];
        const bool exact = naturalPsi == reference;
        allExact = allExact && exact;
        allDeterministic = allDeterministic && variant.deterministic;
        std::cout << "GAMG release-impact validation width="
            << variant.width << " exact=" << (exact ? "PASS" : "FAIL")
            << " deterministic="
            << (variant.deterministic ? "PASS" : "FAIL") << '\n';
    }
    if (!allExact || !allDeterministic)
        throw std::runtime_error("release-impact validation failed");
}

void writeReleaseImpactTraces(const ReleaseImpactFocusedProblem& problem)
{
    if (releaseImpactTraceDirectory.empty()) return;
    const std::filesystem::path directory(releaseImpactTraceDirectory);
    std::filesystem::create_directories(directory);
    const std::string meshName = safeFileName
    (
        benchmarkMeshName.empty() ? "mesh" : benchmarkMeshName
    );
    for (const ReleaseImpactFocusedProblem::Variant& variant : problem.variants)
    {
        const std::filesystem::path path = directory/
        (
            meshName + "_W" + std::to_string(variant.width)
          + "_release_trace.csv"
        );
        std::ofstream output(path);
        if (!output)
            throw std::runtime_error
            (
                "cannot open release-impact trace: " + path.string()
            );
        output << "chunkIndex,chosenChunkSize,headBefore,headAfter,cut,"
            << "releasedCount,releasedBelowCut,releasedBeforeHead,"
            << "headBackstep,headChanged,releasedBeforeHeadPerChunk,"
            << "releasedBeforeHeadFraction\n";
        for (std::size_t chunk=0; chunk<variant.diagnostics.size(); ++chunk)
        {
            const auto& diagnostic = variant.diagnostics[chunk];
            const double perChunk = diagnostic.chosenChunkSize
                ? static_cast<double>(diagnostic.releasedBeforeHead)
                    /diagnostic.chosenChunkSize
                : 0.0;
            const double fraction = diagnostic.releasedCount
                ? static_cast<double>(diagnostic.releasedBeforeHead)
                    /diagnostic.releasedCount
                : 0.0;
            output << chunk << ',' << diagnostic.chosenChunkSize << ','
                << diagnostic.headBefore << ',' << diagnostic.headAfter << ','
                << diagnostic.cut << ',' << diagnostic.releasedCount << ','
                << diagnostic.releasedBelowCut << ','
                << diagnostic.releasedBeforeHead << ','
                << diagnostic.headBackstep << ','
                << static_cast<int>(diagnostic.headChanged) << ','
                << std::setprecision(17) << perChunk << ',' << fraction << '\n';
        }
        if (!output)
            throw std::runtime_error
            (
                "failed writing release-impact trace: " + path.string()
            );
        std::cout << "release-impact trace width=" << variant.width
            << " path=\"" << path.string() << "\" rows="
            << variant.diagnostics.size() << '\n';
    }
}

void printReleaseImpactDiagnostics(const ReleaseImpactFocusedProblem& problem)
{
    std::cout << "Release-impact diagnostics\n"
        << "width setupMs chunks chunkMean chunkP50 chunkP90 releasedCount "
        << "releasedBeforeHead releasedBelowCut headChangedPercent "
        << "beforeHeadPerChunkWorkMean beforeHeadPerChunkWorkMedian "
        << "beforeHeadFractionMean beforeHeadFractionMedian "
        << "headBackstepP50 headBackstepP90 headBackstepP99\n";
    for (const ReleaseImpactFocusedProblem::Variant& variant : problem.variants)
    {
        const ReleaseImpactStats stats = releaseImpactStats(variant.diagnostics);
        std::cout << variant.width << ' '
            << setupTotalMilliseconds(variant.setup) << ' '
            << stats.chunks << ' ' << stats.chunkMean << ' '
            << stats.chunkP50 << ' ' << stats.chunkP90 << ' '
            << stats.releasedCount << ' ' << stats.releasedBeforeHead << ' '
            << stats.releasedBelowCut << ' ' << stats.headChangedPercent << ' '
            << stats.beforeHeadPerChunkWorkMean << ' '
            << stats.beforeHeadPerChunkWorkMedian << ' '
            << stats.beforeHeadFractionMean << ' '
            << stats.beforeHeadFractionMedian << ' '
            << stats.headBackstepP50 << ' ' << stats.headBackstepP90 << ' '
            << stats.headBackstepP99 << '\n';
    }
}

void prepareReleaseImpactHugePages(const ReleaseImpactFocusedProblem& problem)
{
    if (!explicitHugePages) return;
    adviseAndCollapseHugePages("release.LDU.lowerAddr", problem.ldu.lowerAddr());
    adviseAndCollapseHugePages("release.LDU.upperAddr", problem.ldu.upperAddr());
    adviseAndCollapseHugePages("release.LDU.upper", problem.ldu.upper());
    adviseAndCollapseHugePages("release.LDU.diag", problem.ldu.diag());
    adviseAndCollapseHugePages("release.LDU.source", problem.naturalSource);
    for (const ReleaseImpactFocusedProblem::Variant& variant : problem.variants)
    {
        const std::string prefix = "release.W" + std::to_string(variant.width) + '.';
        adviseAndCollapseHugePages(prefix + "levelStarts", variant.matrix.levelStarts());
        adviseAndCollapseHugePages(prefix + "rowStarts", variant.matrix.rowStarts());
        adviseAndCollapseHugePages(prefix + "cols", variant.matrix.cols());
        adviseAndCollapseHugePages(prefix + "coeffs", variant.matrix.coeffs());
        adviseAndCollapseHugePages(prefix + "diag", variant.matrix.diag());
        adviseAndCollapseHugePages(prefix + "source", variant.source);
    }
}

void registerReleaseImpactBenchmarks(const ReleaseImpactFocusedProblem& problem)
{
    const std::string prefix = "GAMGLevel/"
        + (benchmarkMeshName.empty() ? std::string() : benchmarkMeshName + '/')
        + "L00/ReleaseImpact/";
    for (const ReleaseImpactFocusedProblem::Variant& variant : problem.variants)
    {
        const fwgamg::Matrix* const matrix = &variant.matrix;
        const Foam::scalarField* const source = &variant.source;
        const std::vector<std::size_t>* const naturalWidths = &problem.naturalWidths;
        const ReleaseImpactStats stats = releaseImpactStats(variant.diagnostics);
        const double setupMs = setupTotalMilliseconds(variant.setup);
        const Foam::label width = variant.width;
        benchmark::RegisterBenchmark
        (
            (prefix + "W" + std::to_string(width) + "/GS").c_str(),
            [matrix, source, naturalWidths, stats, setupMs]
            (benchmark::State& state)
            {
                Foam::scalarField psi(matrix->nCells(), 0.0);
                if (explicitHugePages)
                    adviseAndCollapseHugePages("release.psi", psi, false);
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    matrix->smooth(psi, *source, 1, false);
                    benchmark::DoNotOptimize(psi.data());
                }
                addCounters(state, matrix->nCells(), matrix->nFaces());
                addFocusedDependencyCounters(state, *matrix, *naturalWidths);
                state.counters["setupMs"] = setupMs;
                state.counters["chunks"] = stats.chunks;
                state.counters["chunkMean"] = stats.chunkMean;
                state.counters["chunkP50"] = stats.chunkP50;
                state.counters["chunkP90"] = stats.chunkP90;
                state.counters["releasedCount"] = stats.releasedCount;
                state.counters["releasedBeforeHead"] = stats.releasedBeforeHead;
                state.counters["releasedBelowCut"] = stats.releasedBelowCut;
                state.counters["headChangedPercent"] = stats.headChangedPercent;
                state.counters["beforeHeadPerChunkWorkMean"] =
                    stats.beforeHeadPerChunkWorkMean;
                state.counters["beforeHeadPerChunkWorkMedian"] =
                    stats.beforeHeadPerChunkWorkMedian;
                state.counters["beforeHeadFractionMean"] =
                    stats.beforeHeadFractionMean;
                state.counters["beforeHeadFractionMedian"] =
                    stats.beforeHeadFractionMedian;
                state.counters["headBackstepP50"] = stats.headBackstepP50;
                state.counters["headBackstepP90"] = stats.headBackstepP90;
                state.counters["headBackstepP99"] = stats.headBackstepP99;
            }
        )->MinTime(0.05);
    }
}

void validateSmootherProfile(const SmootherProfileProblem& problem)
{
    Foam::scalarField reference(problem.ldu.nCells(), 0.0);
    Foam::scalarField actual(problem.ih1.nCells(), 0.0);
    gamg::gaussSeidelSmooth(reference, problem.ldu, problem.naturalSource, 1);
    problem.ih1.smooth(actual, problem.ih1Source, 1, false);
    Foam::scalarField naturalActual(problem.ih1.nCells());
    for (Foam::label row=0; row<problem.ih1.nCells(); ++row)
        naturalActual[problem.ih1.newToOld()[row]] = actual[row];
    const bool exact = naturalActual == reference;
    Foam::scalarField oriented(problem.ih1w2.nCells(), 0.0);
    problem.ih1w2.smooth(oriented, problem.ih1w2Source, 1, false);
    Foam::scalarField naturalOriented(problem.ih1w2.nCells());
    for (Foam::label row=0; row<problem.ih1w2.nCells(); ++row)
        naturalOriented[problem.ih1w2.newToOld()[row]] = oriented[row];
    const bool orientedExact = naturalOriented == reference;
    std::cout << "VTune smoother correctness GS=PASS IH1W1024Exact="
        << (exact ? "PASS" : "FAIL") << " IH1W2_1024Exact="
        << (orientedExact ? "PASS" : "FAIL") << '\n';
    if (!exact || !orientedExact)
        throw std::runtime_error("VTune smoother correctness failed");
}

void prepareProfileHugePages(const SmootherProfileProblem& problem)
{
    if (!explicitHugePages) return;
    adviseAndCollapseHugePages("profile.LDU.lowerAddr", problem.ldu.lowerAddr());
    adviseAndCollapseHugePages("profile.LDU.upperAddr", problem.ldu.upperAddr());
    adviseAndCollapseHugePages("profile.LDU.upper", problem.ldu.upper());
    adviseAndCollapseHugePages("profile.LDU.diag", problem.ldu.diag());
    adviseAndCollapseHugePages("profile.LDU.source", problem.naturalSource);
    adviseAndCollapseHugePages("profile.IH1.rowStarts", problem.ih1.rowStarts());
    adviseAndCollapseHugePages("profile.IH1.cols", problem.ih1.cols());
    adviseAndCollapseHugePages("profile.IH1.coeffs", problem.ih1.coeffs());
    adviseAndCollapseHugePages("profile.IH1.diag", problem.ih1.diag());
    adviseAndCollapseHugePages("profile.IH1.source", problem.ih1Source);
    adviseAndCollapseHugePages
    (
        "profile.IH1W2.rowStarts", problem.ih1w2.rowStarts()
    );
    adviseAndCollapseHugePages("profile.IH1W2.cols", problem.ih1w2.cols());
    adviseAndCollapseHugePages
    (
        "profile.IH1W2.coeffs", problem.ih1w2.coeffs()
    );
    adviseAndCollapseHugePages("profile.IH1W2.diag", problem.ih1w2.diag());
    adviseAndCollapseHugePages("profile.IH1W2.source", problem.ih1w2Source);
    std::cout << std::flush;
}

void runVtuneSmootherProfile(const SmootherProfileProblem& problem)
{
    if (vtuneSmoother.empty()) return;
#ifndef HAVE_ITTNOTIFY
    static_cast<void>(problem);
    throw std::runtime_error("this build has no VTune ITT support");
#else
    if
    (
        vtuneSweeps < 1
     || vtuneWarmupSweeps < 0
     || (vtuneSmoother != "gs"
      && vtuneSmoother != "ih1"
      && vtuneSmoother != "ih1w2")
    ) throw std::runtime_error("invalid VTune smoother profile options");

    __itt_pause();
    const gamg::GAMGMatrix& ldu = problem.ldu;
    const Foam::scalarField& naturalSource = problem.naturalSource;
    const fwgamg::Matrix& ih1 = problem.ih1;
    const Foam::scalarField& ih1Source = problem.ih1Source;
    const bool useOriented = vtuneSmoother == "ih1w2";
    const fwgamg::Matrix& selected = useOriented ? problem.ih1w2 : ih1;
    const Foam::scalarField& selectedSource =
        useOriented ? problem.ih1w2Source : ih1Source;
    Foam::scalarField psi(ldu.nCells(), 0.0);
    if (explicitHugePages)
        adviseAndCollapseHugePages("profile.psi", psi, true);

    if (vtuneSmoother == "gs")
        gamg::gaussSeidelSmooth
        (
            psi, ldu, naturalSource, vtuneWarmupSweeps
        );
    else
        selected.smooth(psi, selectedSource, vtuneWarmupSweeps, false);
    std::fill(psi.begin(), psi.end(), 0.0);

    const auto begin = std::chrono::steady_clock::now();
    __itt_resume();
    if (vtuneSmoother == "gs")
        gamg::gaussSeidelSmooth(psi, ldu, naturalSource, vtuneSweeps);
    else
        selected.smooth(psi, selectedSource, vtuneSweeps, false);
    __itt_pause();
    const double seconds = std::chrono::duration<double>
    (
        std::chrono::steady_clock::now() - begin
    ).count();
    double checksum = 0;
    for (const Foam::scalar value : psi) checksum += value;
    std::cout << "VTune smoother ROI"
        << " variant=" << vtuneSmoother
        << " mesh=" << benchmarkMeshName
        << " level=0"
        << " cells=" << ldu.nCells()
        << " faces=" << ldu.nFaces()
        << " nnz=" << ldu.nCells() + 2*ldu.nFaces()
        << " warmupSweeps=" << vtuneWarmupSweeps
        << " sweeps=" << vtuneSweeps
        << " elapsed=" << std::setprecision(17) << seconds
        << " checksum=" << checksum << '\n';
#endif
}

void registerLevelBenchmarks(const Problem& problem)
{
    const std::size_t registeredLevels = ih1w5Experiment
        ? std::size_t(1) : problem.baseline.matrices().size();
    for (std::size_t level=0; level<registeredLevels; ++level)
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
        const std::string prefix = "GAMGLevel/"
            + (benchmarkMeshName.empty() ? std::string() : benchmarkMeshName + '/')
            + "L" +
            (level < 10 ? std::string("0") : std::string()) +
            std::to_string(level) + '/';
        const auto finish = [ldu](benchmark::State& state)
        {
            addCounters(state, ldu->nCells(), ldu->nFaces());
        };
        const fwgamg::Matrix* const unbounded =
            &problem.unboundedMatrices[level];

        benchmark::RegisterBenchmark
        (
            (prefix + "A_LDU/GS").c_str(),
            [ldu, source, finish, unbounded](benchmark::State& state)
            {
                Foam::scalarField psi(ldu->nCells(), 0.0);
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    gamg::gaussSeidelSmooth(psi, *ldu, *source, 1);
                    benchmark::DoNotOptimize(psi.data());
                }
                finish(state);
                addDependencyCounters(state, *unbounded, *unbounded);
            }
        )->MinTime(0.05);

        for (const Problem::WidthSweep& variant : problem.widthSweeps)
        {
            const fwgamg::Matrix* const matrix = &variant.matrices[level];
            const Foam::scalarField* const widthSource =
                &variant.sources[level];
            const std::string name = prefix + "IH1W"
                + std::to_string(variant.width) + "/GS";
            benchmark::RegisterBenchmark
            (
                name.c_str(),
                [matrix, widthSource, finish, unbounded]
                (benchmark::State& state)
                {
                    Foam::scalarField psi(matrix->nCells(), 0.0);
                    if (explicitHugePages)
                    {
                        adviseAndCollapseHugePages
                        (
                            "IH1.psi", psi, false
                        );
                    }
                    for (auto unused : state)
                    {
                        static_cast<void>(unused);
                        matrix->smooth(psi, *widthSource, 1, false);
                        benchmark::DoNotOptimize(psi.data());
                    }
                    finish(state);
                    addDependencyCounters(state, *matrix, *unbounded);
                }
            )->MinTime(0.05);
        }
        for (const Problem::WidthSweep& variant : problem.orientedWidthSweeps)
        {
            const fwgamg::Matrix* const oriented = &variant.matrices[level];
            const Foam::scalarField* const orientedSource =
                &variant.sources[level];
            const std::string name = prefix + "IH1W2_"
                + std::to_string(variant.width) + "/GS";
            benchmark::RegisterBenchmark
            (
                name.c_str(),
                [oriented, orientedSource, finish, unbounded]
                (benchmark::State& state)
                {
                    Foam::scalarField psi(oriented->nCells(), 0.0);
                    if (explicitHugePages)
                        adviseAndCollapseHugePages
                        (
                            "IH1W2.psi", psi, false
                        );
                    for (auto unused : state)
                    {
                        static_cast<void>(unused);
                        oriented->smooth(psi, *orientedSource, 1, false);
                        benchmark::DoNotOptimize(psi.data());
                    }
                    finish(state);
                    addDependencyCounters(state, *oriented, *unbounded);
                }
            )->MinTime(0.05);
        }
        for
        (
            const Problem::WidthSweep& variant
          : problem.slidingWindowSweeps
        )
        {
            const fwgamg::Matrix* const sliding = &variant.matrices[level];
            const Foam::scalarField* const slidingSource =
                &variant.sources[level];
            const std::string name = prefix + "IH1W3_W1024_S"
                + std::to_string(variant.width) + "/GS";
            benchmark::RegisterBenchmark
            (
                name.c_str(),
                [sliding, slidingSource, finish, unbounded]
                (benchmark::State& state)
                {
                    Foam::scalarField psi(sliding->nCells(), 0.0);
                    if (explicitHugePages)
                        adviseAndCollapseHugePages("IH1W3.psi", psi, false);
                    for (auto unused : state)
                    {
                        static_cast<void>(unused);
                        sliding->smooth(psi, *slidingSource, 1, false);
                        benchmark::DoNotOptimize(psi.data());
                    }
                    finish(state);
                    addDependencyCounters(state, *sliding, *unbounded);
                }
            )->MinTime(0.05);
        }
        for
        (
            const Problem::AdaptiveGapSweep& variant
          : problem.adaptiveGapSweeps
        )
        {
            const fwgamg::Matrix* const adaptive =
                &variant.matrices[level];
            const Foam::scalarField* const adaptiveSource =
                &variant.sources[level];
            const std::string name = prefix + variant.name + "/GS";
            benchmark::RegisterBenchmark
            (
                name.c_str(),
                [adaptive, adaptiveSource, finish, unbounded]
                (benchmark::State& state)
                {
                    Foam::scalarField psi(adaptive->nCells(), 0.0);
                    if (explicitHugePages)
                        adviseAndCollapseHugePages("IH1W4.psi", psi, false);
                    for (auto unused : state)
                    {
                        static_cast<void>(unused);
                        adaptive->smooth(psi, *adaptiveSource, 1, false);
                        benchmark::DoNotOptimize(psi.data());
                    }
                    finish(state);
                    addDependencyCounters(state, *adaptive, *unbounded);
                }
            )->MinTime(0.05);
        }
        for
        (
            const Problem::ReadyFractionSweep& variant
          : problem.readyFractionSweeps
        )
        {
            const fwgamg::Matrix* const adaptive = &variant.matrix;
            const Foam::scalarField* const adaptiveSource = &variant.source;
            const std::string name = prefix + variant.name + "/GS";
            benchmark::RegisterBenchmark
            (
                name.c_str(),
                [adaptive, adaptiveSource, finish, unbounded]
                (benchmark::State& state)
                {
                    Foam::scalarField psi(adaptive->nCells(), 0.0);
                    if (explicitHugePages)
                        adviseAndCollapseHugePages("IH1W5.psi", psi, false);
                    for (auto unused : state)
                    {
                        static_cast<void>(unused);
                        adaptive->smooth(psi, *adaptiveSource, 1, false);
                        benchmark::DoNotOptimize(psi.data());
                    }
                    finish(state);
                    addDependencyCounters(state, *adaptive, *unbounded);
                }
            )->MinTime(0.05);
        }
        benchmark::RegisterBenchmark
        (
            (prefix + "IH1U/GS").c_str(),
            [unbounded, &problem, level, finish](benchmark::State& state)
            {
                const Foam::scalarField& source =
                    problem.unboundedSources[level];
                Foam::scalarField psi(unbounded->nCells(), 0.0);
                if (explicitHugePages)
                    adviseAndCollapseHugePages("IH1U.psi", psi, false);
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    unbounded->smooth(psi, source, 1, false);
                    benchmark::DoNotOptimize(psi.data());
                }
                finish(state);
                addDependencyCounters(state, *unbounded, *unbounded);
            }
        )->MinTime(0.05);
        const fwgamg::Matrix* const sliced =
            &problem.slicedMatrices[level];
        benchmark::RegisterBenchmark
        (
            (prefix + "IH1S1024/GS").c_str(),
            [sliced, &problem, level, finish, unbounded]
            (benchmark::State& state)
            {
                const Foam::scalarField& source =
                    problem.slicedSources[level];
                Foam::scalarField psi(sliced->nCells(), 0.0);
                if (explicitHugePages)
                    adviseAndCollapseHugePages("IH1S1024.psi", psi, false);
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    sliced->smooth(psi, source, 1, false);
                    benchmark::DoNotOptimize(psi.data());
                }
                finish(state);
                addDependencyCounters(state, *sliced, *unbounded);
            }
        )->MinTime(0.05);
        const fwgamg::Matrix* const continuing =
            &problem.continuingMatrices[level];
        benchmark::RegisterBenchmark
        (
            (prefix + "IH1C1024/GS").c_str(),
            [continuing, &problem, level, finish, unbounded]
            (benchmark::State& state)
            {
                const Foam::scalarField& source =
                    problem.continuingSources[level];
                Foam::scalarField psi(continuing->nCells(), 0.0);
                if (explicitHugePages)
                    adviseAndCollapseHugePages("IH1C1024.psi", psi, false);
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    continuing->smooth(psi, source, 1, false);
                    benchmark::DoNotOptimize(psi.data());
                }
                finish(state);
                addDependencyCounters(state, *continuing, *unbounded);
            }
        )->MinTime(0.05);
        const fwgamg::Matrix* const frontierContinuing =
            &problem.frontierContinuingMatrices[level];
        benchmark::RegisterBenchmark
        (
            (prefix + "IH1C2_1024/GS").c_str(),
            [frontierContinuing, &problem, level, finish, unbounded]
            (benchmark::State& state)
            {
                const Foam::scalarField& source =
                    problem.frontierContinuingSources[level];
                Foam::scalarField psi(frontierContinuing->nCells(), 0.0);
                if (explicitHugePages)
                {
                    adviseAndCollapseHugePages
                    (
                        "IH1C2_1024.psi", psi, false
                    );
                }
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    frontierContinuing->smooth(psi, source, 1, false);
                    benchmark::DoNotOptimize(psi.data());
                }
                finish(state);
                addDependencyCounters
                (
                    state, *frontierContinuing, *unbounded
                );
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
            )->MinTime((amdProfileRoi || amdPcmSignalRoi) ? 10.0 : 0.05);
        };
        registerSpmv
        (
            "A_LDU",
            [ldu, source, finish](benchmark::State& state)
            {
                Foam::scalarField result(ldu->nCells(), 0.0);
                if (explicitHugePages)
                    adviseAndCollapseHugePages
                    (
                        "LDU.result", result, false
                    );
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
                Foam::scalarField result(natural->nCells(), 0.0);
                if (explicitHugePages)
                    adviseAndCollapseHugePages
                    (
                        "fw.directResult", result, false
                    );
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
                Foam::scalarField result(natural->nCells(), 0.0);
                if (explicitHugePages)
                    adviseAndCollapseHugePages
                    (
                        "fw.lduOrderResult", result, false
                    );
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
                Foam::scalarField result(reordered->nCells(), 0.0);
                if (explicitHugePages)
                    adviseAndCollapseHugePages
                    (
                        "fw.result", result, false
                    );
                beginAmdProfileRoi();
                for (auto unused : state)
                {
                    static_cast<void>(unused);
                    reordered->multiplyLduOrderDirectFlatEdgeWise
                    (
                        result, *fwSource
                    );
                    benchmark::DoNotOptimize(result.data());
                }
                endAmdProfileRoi();
                finish(state);
            }
        );
        const std::array<std::pair<fwgamg::DirectFlatPrefetch, const char*>, 3>
            prefetchModes{{
                {fwgamg::DirectFlatPrefetch::Field, "Field"},
                {fwgamg::DirectFlatPrefetch::Result, "Result"},
                {fwgamg::DirectFlatPrefetch::FieldAndResult, "Both"}
            }};
        for (const auto& [mode, modeName] : prefetchModes)
        {
            for (const Foam::label distance : {64, 128, 256, 512})
            {
                registerSpmv
                (
                    "C2P_L2_" + std::string(modeName) + "_D"
                  + std::to_string(distance),
                    [reordered, fwSource, finish, mode, distance]
                    (benchmark::State& state)
                    {
                        Foam::scalarField result(reordered->nCells(), 0.0);
                        if (explicitHugePages)
                            adviseAndCollapseHugePages
                            (
                                "fw.prefetchResult", result, false
                            );
                        beginAmdProfileRoi();
                        for (auto unused : state)
                        {
                            static_cast<void>(unused);
                            reordered->multiplyLduOrderDirectFlatEdgeWisePrefetch
                            (
                                result, *fwSource, mode, distance
                            );
                            benchmark::DoNotOptimize(result.data());
                        }
                        endAmdProfileRoi();
                        finish(state);
                    }
                );
            }
        }
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

void printIh1w5Diagnostics(const Problem& problem)
{
    if (!ih1w5Experiment) return;

    const Problem::WidthSweep* bounded1024 = nullptr;
    for (const Problem::WidthSweep& variant : problem.widthSweeps)
        if (variant.width == 1024) bounded1024 = &variant;
    if (!bounded1024 || bounded1024->matrices.empty())
        throw std::runtime_error("IH1W5 diagnostics require IH1W1024");

    const OrderingDiagnosticStats preliminary = orderingDiagnosticStats
    (
        bounded1024->matrices.front(), nullptr,
        std::numeric_limits<double>::infinity()
    );
    const double largeJumpThreshold = preliminary.rowJumpP99;
    std::cout << "IH1W5 ordering diagnostics"
        << " baselineLargeJumpThreshold=" << largeJumpThreshold << '\n';
    std::cout << "variant orderingMs setupMs chunks chunkMin chunkP10 "
        << "chunkP50 chunkP90 chunkMean chunkMax targetP10 targetP50 "
        << "targetP90 readyP10 readyP50 readyP90 rowJumpTotal rowP90 "
        << "rowP99 largeJumps deterministic\n";

    const auto print = [largeJumpThreshold]
    (
        const std::string& name,
        const fwgamg::Matrix& matrix,
        const fwgamg::LevelSetupTiming& setup,
        const std::vector<smootherTest::AdaptiveReadyChunkDiagnostic>* diagnostics,
        const bool deterministic
    )
    {
        const OrderingDiagnosticStats stats = orderingDiagnosticStats
        (
            matrix, diagnostics, largeJumpThreshold
        );
        std::cout << name << ' ' << 1e3*setup.scheduleSeconds << ' '
            << setupTotalMilliseconds(setup) << ' ' << stats.chunks << ' '
            << stats.chunkMin << ' ' << stats.chunkP10 << ' '
            << stats.chunkP50 << ' ' << stats.chunkP90 << ' '
            << stats.chunkMean << ' ' << stats.chunkMax << ' '
            << stats.targetP10 << ' ' << stats.targetP50 << ' '
            << stats.targetP90 << ' ' << stats.readyP10 << ' '
            << stats.readyP50 << ' ' << stats.readyP90 << ' '
            << stats.rowJumpTotal << ' ' << stats.rowJumpP90 << ' '
            << stats.rowJumpP99 << ' ' << stats.largeJumps << ' '
            << (deterministic ? "PASS" : "N/A") << '\n';
    };

    print
    (
        "IH1W1024", bounded1024->matrices.front(),
        bounded1024->setup.front(), nullptr, true
    );
    for (const Problem::AdaptiveGapSweep& variant : problem.adaptiveGapSweeps)
    {
        print
        (
            variant.name, variant.matrices.front(), variant.setup.front(),
            nullptr, true
        );
    }
    for (const Problem::ReadyFractionSweep& variant : problem.readyFractionSweeps)
    {
        print
        (
            variant.name, variant.matrix, variant.setup,
            &variant.diagnostics, variant.deterministic
        );
        if (ih1w5TraceDirectory.empty()) for
        (
            std::size_t chunk=0;
            chunk<variant.diagnostics.size();
            ++chunk
        )
        {
            const auto& diagnostic = variant.diagnostics[chunk];
            std::cout << "IH1W5 chunk variant=" << variant.name
                << " index=" << chunk
                << " readySize=" << diagnostic.readySize
                << " targetWidth=" << diagnostic.targetWidth
                << " searchMinWidth=" << diagnostic.searchMinWidth
                << " searchMaxWidth=" << diagnostic.searchMaxWidth
                << " chosenChunkSize=" << diagnostic.chosenChunkSize
                << " largestGap=" << diagnostic.largestGap << '\n';
        }
    }
    std::cout << std::flush;
}

void writeIh1w5ChunkTraces(const Problem& problem)
{
    if (!ih1w5Experiment || ih1w5TraceDirectory.empty()) return;
    const std::filesystem::path directory(ih1w5TraceDirectory);
    std::filesystem::create_directories(directory);
    const std::string meshName = safeFileName
    (
        benchmarkMeshName.empty() ? "mesh" : benchmarkMeshName
    );
    for (const Problem::ReadyFractionSweep& variant : problem.readyFractionSweeps)
    {
        const std::filesystem::path path = directory/
        (
            meshName + '_' + safeFileName(variant.name) + "_chunks.csv"
        );
        std::ofstream output(path);
        if (!output)
            throw std::runtime_error("cannot open IH1W5 chunk trace: " + path.string());
        output << "chunkIndex,readySize,targetWidth,searchMinWidth,"
            << "searchMaxWidth,chosenChunkSize,largestGap\n";
        for (std::size_t chunk=0; chunk<variant.diagnostics.size(); ++chunk)
        {
            const auto& diagnostic = variant.diagnostics[chunk];
            output << chunk << ',' << diagnostic.readySize << ','
                << diagnostic.targetWidth << ','
                << diagnostic.searchMinWidth << ','
                << diagnostic.searchMaxWidth << ','
                << diagnostic.chosenChunkSize << ','
                << diagnostic.largestGap << '\n';
        }
        if (!output)
            throw std::runtime_error("failed writing IH1W5 chunk trace: " + path.string());
        std::cout << "IH1W5 chunk trace variant=" << variant.name
            << " path=\"" << path.string() << "\" rows="
            << variant.diagnostics.size() << '\n';
    }
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
    bool directExact = true;
    bool lduOrderDirectExact = true;
    bool reorderedLduOrderDirectExact = true;
    double prefetchDifference = 0;
    double lduOwnerAccumulatedDifference = 0;
    double directOwnerAccumulatedDifference = 0;
    double directOwnerRunAccumulatedDifference = 0;
    std::size_t directOwnerOrderMismatches = 0;
    std::size_t directColumnOrderMismatches = 0;
    std::size_t directCoefficientOrderMismatches = 0;
    std::vector<bool> widthSweepExact(problem.widthSweeps.size(), true);
    bool unboundedSweepExact = true;
    bool slicedSweepExact = true;
    bool continuingSweepExact = true;
    bool frontierContinuingSweepExact = true;
    bool orientedSweepExact = true;
    bool slidingWindowSweepExact = true;
    std::vector<bool> adaptiveGapSweepExact
    (
        problem.adaptiveGapSweeps.size(), true
    );
    std::vector<bool> readyFractionSweepExact
    (
        problem.readyFractionSweeps.size(), true
    );
    bool readyFractionDeterministic = true;
    bool orientedChunkMembershipExact = true;
    std::size_t orientedChunks = 0;
    std::size_t orientedReversedChunks = 0;
    bool slicedPermutationExact = true;
    bool slicedLevelStartsValid = true;
    bool frontierContinuingNarrowExact = true;
    std::size_t frontierContinuingNarrowLevels = 0;
    std::size_t boundedPermutationMismatches = 0;
    std::size_t continuingPermutationMismatches = 0;
    std::size_t frontierContinuingPermutationMismatches = 0;
    for (std::size_t level=0; level<problem.natural.size(); ++level)
    {
        Foam::scalarField referencePsi
        (
            problem.baseline.matrices()[level].nCells(), 0.0
        );
        gamg::gaussSeidelSmooth
        (
            referencePsi,
            problem.baseline.matrices()[level],
            problem.naturalSource[level],
            1
        );
        for
        (
            std::size_t variantIndex=0;
            variantIndex<problem.widthSweeps.size();
            ++variantIndex
        )
        {
            const Problem::WidthSweep& variant =
                problem.widthSweeps[variantIndex];
            const fwgamg::Matrix& matrix = variant.matrices[level];
            Foam::scalarField actualPsi(matrix.nCells(), 0.0);
            matrix.smooth(actualPsi, variant.sources[level], 1, false);
            Foam::scalarField naturalPsi(matrix.nCells());
            for (Foam::label row=0; row<matrix.nCells(); ++row)
                naturalPsi[matrix.newToOld()[row]] = actualPsi[row];
            widthSweepExact[variantIndex] =
                widthSweepExact[variantIndex]
             && naturalPsi == referencePsi;
        }
        const fwgamg::Matrix& unbounded = problem.unboundedMatrices[level];
        Foam::scalarField unboundedPsi(unbounded.nCells(), 0.0);
        unbounded.smooth
        (
            unboundedPsi, problem.unboundedSources[level], 1, false
        );
        Foam::scalarField naturalUnboundedPsi(unbounded.nCells());
        for (Foam::label row=0; row<unbounded.nCells(); ++row)
        {
            naturalUnboundedPsi[unbounded.newToOld()[row]] =
                unboundedPsi[row];
        }
        unboundedSweepExact =
            unboundedSweepExact && naturalUnboundedPsi == referencePsi;

        const fwgamg::Matrix& sliced = problem.slicedMatrices[level];
        Foam::scalarField slicedPsi(sliced.nCells(), 0.0);
        sliced.smooth(slicedPsi, problem.slicedSources[level], 1, false);
        Foam::scalarField naturalSlicedPsi(sliced.nCells());
        for (Foam::label row=0; row<sliced.nCells(); ++row)
            naturalSlicedPsi[sliced.newToOld()[row]] = slicedPsi[row];
        slicedSweepExact =
            slicedSweepExact && naturalSlicedPsi == referencePsi;

        const bool permutationExact =
            sliced.newToOld() == unbounded.newToOld();
        slicedPermutationExact = slicedPermutationExact && permutationExact;
        std::vector<Foam::label> expectedSlicedStarts{0};
        for
        (
            std::size_t naturalLevel=0;
            naturalLevel + 1<unbounded.levelStarts().size();
            ++naturalLevel
        )
        {
            const Foam::label begin =
                unbounded.levelStarts()[naturalLevel];
            const Foam::label end =
                unbounded.levelStarts()[naturalLevel + 1];
            for (Foam::label split=begin + 1024; split<end; split += 1024)
                expectedSlicedStarts.push_back(split);
            expectedSlicedStarts.push_back(end);
        }
        const bool levelStartsValid =
            sliced.levelStarts() == expectedSlicedStarts;
        slicedLevelStartsValid =
            slicedLevelStartsValid && levelStartsValid;

        const Problem::WidthSweep* bounded1024 = nullptr;
        for (const Problem::WidthSweep& variant : problem.widthSweeps)
            if (variant.width == 1024) bounded1024 = &variant;
        if (!bounded1024)
            throw std::runtime_error("IH1W1024 validation matrix is missing");
        const std::vector<Foam::label>& boundedPermutation =
            bounded1024->matrices[level].newToOld();
        std::size_t levelMismatches = 0;
        for (std::size_t row=0; row<boundedPermutation.size(); ++row)
            levelMismatches +=
                boundedPermutation[row] != unbounded.newToOld()[row];
        boundedPermutationMismatches += levelMismatches;

        for
        (
            const Problem::WidthSweep& orientedVariant
          : problem.orientedWidthSweeps
        )
        {
            const Problem::WidthSweep* boundedVariant = nullptr;
            for (const Problem::WidthSweep& variant : problem.widthSweeps)
                if (variant.width == orientedVariant.width)
                    boundedVariant = &variant;
            if (!boundedVariant)
                throw std::runtime_error
                (
                    "matching IH1W validation matrix is missing"
                );
            const fwgamg::Matrix& oriented =
                orientedVariant.matrices[level];
            Foam::scalarField orientedPsi(oriented.nCells(), 0.0);
            oriented.smooth
            (
                orientedPsi, orientedVariant.sources[level], 1, false
            );
            Foam::scalarField naturalOrientedPsi(oriented.nCells());
            for (Foam::label row=0; row<oriented.nCells(); ++row)
                naturalOrientedPsi[oriented.newToOld()[row]] =
                    orientedPsi[row];
            const bool levelExact = naturalOrientedPsi == referencePsi;
            orientedSweepExact = orientedSweepExact && levelExact;
            const OrientedChunkStats stats = orientedChunkStats
            (
                boundedVariant->matrices[level], oriented
            );
            orientedChunkMembershipExact =
                orientedChunkMembershipExact && stats.valid;
            orientedChunks += stats.total;
            orientedReversedChunks += stats.reversed;
            std::cout << "GAMG IH1W2 permutation L" << level
                << " width=" << orientedVariant.width
                << " exact=" << (levelExact ? "PASS" : "FAIL")
                << " chunkMembership="
                << (stats.valid ? "PASS" : "FAIL")
                << " reversedChunks=" << stats.reversed
                << '/' << stats.total << '\n';
        }

        for
        (
            const Problem::WidthSweep& slidingVariant
          : problem.slidingWindowSweeps
        )
        {
            const fwgamg::Matrix& sliding =
                slidingVariant.matrices[level];
            Foam::scalarField slidingPsi(sliding.nCells(), 0.0);
            sliding.smooth
            (
                slidingPsi, slidingVariant.sources[level], 1, false
            );
            Foam::scalarField naturalSlidingPsi(sliding.nCells());
            for (Foam::label row=0; row<sliding.nCells(); ++row)
                naturalSlidingPsi[sliding.newToOld()[row]] = slidingPsi[row];
            const bool levelExact = naturalSlidingPsi == referencePsi;
            slidingWindowSweepExact =
                slidingWindowSweepExact && levelExact;
            std::cout << "GAMG IH1W3 permutation L" << level
                << " W=1024 S=" << slidingVariant.width
                << " exact=" << (levelExact ? "PASS" : "FAIL")
                << " executionLevels="
                << sliding.levelStarts().size() - 1 << '\n';
        }

        for
        (
            std::size_t variantIndex=0;
            variantIndex<problem.adaptiveGapSweeps.size();
            ++variantIndex
        )
        {
            const Problem::AdaptiveGapSweep& variant =
                problem.adaptiveGapSweeps[variantIndex];
            if (level >= variant.matrices.size()) continue;
            const fwgamg::Matrix& adaptive = variant.matrices[level];
            Foam::scalarField adaptivePsi(adaptive.nCells(), 0.0);
            adaptive.smooth
            (
                adaptivePsi, variant.sources[level], 1, false
            );
            Foam::scalarField naturalAdaptivePsi(adaptive.nCells());
            for (Foam::label row=0; row<adaptive.nCells(); ++row)
                naturalAdaptivePsi[adaptive.newToOld()[row]] =
                    adaptivePsi[row];
            const bool levelExact = naturalAdaptivePsi == referencePsi;
            adaptiveGapSweepExact[variantIndex] =
                adaptiveGapSweepExact[variantIndex] && levelExact;
            std::cout << "GAMG IH1W4 permutation L" << level
                << " variant=" << variant.name
                << " min=" << variant.minWidth
                << " target=" << variant.targetWidth
                << " max=" << variant.maxWidth
                << " exact=" << (levelExact ? "PASS" : "FAIL")
                << " chunks=" << adaptive.levelStarts().size() - 1
                << '\n';
        }

        if (level == 0)
        {
            for
            (
                std::size_t variantIndex=0;
                variantIndex<problem.readyFractionSweeps.size();
                ++variantIndex
            )
            {
                const Problem::ReadyFractionSweep& variant =
                    problem.readyFractionSweeps[variantIndex];
                Foam::scalarField adaptivePsi(variant.matrix.nCells(), 0.0);
                variant.matrix.smooth
                (
                    adaptivePsi, variant.source, 1, false
                );
                Foam::scalarField naturalAdaptivePsi
                (
                    variant.matrix.nCells()
                );
                for
                (
                    Foam::label row=0;
                    row<variant.matrix.nCells();
                    ++row
                )
                {
                    naturalAdaptivePsi[variant.matrix.newToOld()[row]] =
                        adaptivePsi[row];
                }
                const bool exact = naturalAdaptivePsi == referencePsi;
                readyFractionSweepExact[variantIndex] = exact;
                readyFractionDeterministic =
                    readyFractionDeterministic && variant.deterministic;
                std::cout << "GAMG IH1W5 permutation L0 variant="
                    << variant.name
                    << " exact=" << (exact ? "PASS" : "FAIL")
                    << " deterministic="
                    << (variant.deterministic ? "PASS" : "FAIL")
                    << " chunks="
                    << variant.matrix.levelStarts().size() - 1 << '\n';
            }
        }

        const fwgamg::Matrix& continuing =
            problem.continuingMatrices[level];
        Foam::scalarField continuingPsi(continuing.nCells(), 0.0);
        continuing.smooth
        (
            continuingPsi, problem.continuingSources[level], 1, false
        );
        Foam::scalarField naturalContinuingPsi(continuing.nCells());
        for (Foam::label row=0; row<continuing.nCells(); ++row)
        {
            naturalContinuingPsi[continuing.newToOld()[row]] =
                continuingPsi[row];
        }
        continuingSweepExact =
            continuingSweepExact && naturalContinuingPsi == referencePsi;
        std::size_t continuingLevelMismatches = 0;
        for (std::size_t row=0; row<continuing.newToOld().size(); ++row)
        {
            continuingLevelMismatches +=
                continuing.newToOld()[row] != unbounded.newToOld()[row];
        }
        continuingPermutationMismatches += continuingLevelMismatches;
        const fwgamg::Matrix& frontierContinuing =
            problem.frontierContinuingMatrices[level];
        Foam::scalarField frontierContinuingPsi
        (
            frontierContinuing.nCells(), 0.0
        );
        frontierContinuing.smooth
        (
            frontierContinuingPsi,
            problem.frontierContinuingSources[level],
            1,
            false
        );
        Foam::scalarField naturalFrontierContinuingPsi
        (
            frontierContinuing.nCells()
        );
        for (Foam::label row=0; row<frontierContinuing.nCells(); ++row)
        {
            naturalFrontierContinuingPsi
                [frontierContinuing.newToOld()[row]] =
                frontierContinuingPsi[row];
        }
        frontierContinuingSweepExact =
            frontierContinuingSweepExact
         && naturalFrontierContinuingPsi == referencePsi;
        std::size_t frontierContinuingLevelMismatches = 0;
        for
        (
            std::size_t row=0;
            row<frontierContinuing.newToOld().size();
            ++row
        )
        {
            frontierContinuingLevelMismatches +=
                frontierContinuing.newToOld()[row]
             != unbounded.newToOld()[row];
        }
        frontierContinuingPermutationMismatches +=
            frontierContinuingLevelMismatches;
        Foam::label maxNaturalLevelWidth = 0;
        for
        (
            std::size_t naturalLevel=0;
            naturalLevel + 1<unbounded.levelStarts().size();
            ++naturalLevel
        )
        {
            maxNaturalLevelWidth = std::max
            (
                maxNaturalLevelWidth,
                unbounded.levelStarts()[naturalLevel + 1]
              - unbounded.levelStarts()[naturalLevel]
            );
        }
        const bool narrowPermutationExact =
            maxNaturalLevelWidth > 1024
         || frontierContinuingLevelMismatches == 0;
        if (maxNaturalLevelWidth <= 1024)
            ++frontierContinuingNarrowLevels;
        frontierContinuingNarrowExact =
            frontierContinuingNarrowExact && narrowPermutationExact;
        const ChunkJumpStats boundedJumps =
            chunkJumpStats(bounded1024->matrices[level]);
        const ChunkJumpStats continuingJumps = chunkJumpStats(continuing);
        const ChunkJumpStats frontierContinuingJumps =
            chunkJumpStats(frontierContinuing);
        std::cout << "GAMG IH1 permutation L" << level
            << " IH1S1024EqIH1U="
            << (permutationExact ? "PASS" : "FAIL")
            << " IH1S1024LevelStarts="
            << (levelStartsValid ? "PASS" : "FAIL")
            << " IH1W1024VsIH1UPositionMismatches="
            << levelMismatches
            << " IH1C1024VsIH1UPositionMismatches="
            << continuingLevelMismatches
            << " IH1W1024ChunkJumpMedian/P90/P99="
            << boundedJumps.median << '/'
            << boundedJumps.p90 << '/'
            << boundedJumps.p99
            << " IH1C1024ChunkJumpMedian/P90/P99="
            << continuingJumps.median << '/'
            << continuingJumps.p90 << '/'
            << continuingJumps.p99
            << " IH1C2_1024VsIH1UPositionMismatches="
            << frontierContinuingLevelMismatches
            << " IH1C2_1024ChunkJumpMedian/P90/P99="
            << frontierContinuingJumps.median << '/'
            << frontierContinuingJumps.p90 << '/'
            << frontierContinuingJumps.p99
            << " IH1C2EqIH1UWhenMaxNaturalLE1024="
            << (maxNaturalLevelWidth <= 1024
                ? (narrowPermutationExact ? "PASS" : "FAIL")
                : "N/A")
            << '\n';

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
        directExact = directExact && actual == expected;
        directDifference = std::max
        (
            directDifference, maximumDifference(expected, actual)
        );
        problem.natural[level].multiplyLduOrderDirectFlatEdgeWise
        (
            actual, problem.naturalSource[level]
        );
        lduOrderDirectExact = lduOrderDirectExact && actual == expected;
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
        for
        (
            const fwgamg::DirectFlatPrefetch mode :
            {
                fwgamg::DirectFlatPrefetch::Field,
                fwgamg::DirectFlatPrefetch::Result,
                fwgamg::DirectFlatPrefetch::FieldAndResult
            }
        )
        {
            for (const Foam::label distance : {64, 128, 256, 512})
            {
                problem.reordered.hierarchy()[level]
                    .multiplyLduOrderDirectFlatEdgeWisePrefetch
                    (
                        actual,
                        problem.reorderedSource[level],
                        mode,
                        distance
                    );
                prefetchDifference = std::max
                (
                    prefetchDifference,
                    maximumDifference(expectedReordered, actual)
                );
            }
        }
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
        // Production Index-Kahn matrices no longer construct the obsolete
        // CSR-derived Direct-flat view.  Validate it only when a diagnostic
        // hierarchy explicitly contains that metadata.
        if
        (
            problem.reordered.hierarchy()[level].directEdgeCoeffs().size()
         == static_cast<std::size_t>
            (problem.reordered.hierarchy()[level].nFaces())
        )
        {
            problem.reordered.hierarchy()[level].multiplyDirectFlatEdgeWise
            (
                actual, problem.reorderedSource[level]
            );
            reorderedDirectDifference = std::max
            (
                reorderedDirectDifference,
                maximumDifference(expectedReordered, actual)
            );
        }
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
    std::cout << "GAMG IH1 width validation:";
    bool allWidthSweepsExact = true;
    for
    (
        std::size_t variantIndex=0;
        variantIndex<problem.widthSweeps.size();
        ++variantIndex
    )
    {
        allWidthSweepsExact =
            allWidthSweepsExact && widthSweepExact[variantIndex];
        std::cout << " W" << problem.widthSweeps[variantIndex].width
            << "Exact="
            << (widthSweepExact[variantIndex] ? "PASS" : "FAIL");
    }
    allWidthSweepsExact =
        allWidthSweepsExact
     && unboundedSweepExact
     && slicedSweepExact
     && continuingSweepExact
     && frontierContinuingSweepExact;
    std::cout << " IH1UExact="
        << (unboundedSweepExact ? "PASS" : "FAIL")
        << " IH1S1024Exact="
        << (slicedSweepExact ? "PASS" : "FAIL")
        << " IH1C1024Exact="
        << (continuingSweepExact ? "PASS" : "FAIL")
        << " IH1C2_1024Exact="
        << (frontierContinuingSweepExact ? "PASS" : "FAIL")
        << " IH1W2Exact="
        << (orientedSweepExact ? "PASS" : "FAIL")
        << " IH1W3Exact="
        << (slidingWindowSweepExact ? "PASS" : "FAIL");
    for
    (
        std::size_t variantIndex=0;
        variantIndex<problem.adaptiveGapSweeps.size();
        ++variantIndex
    )
    {
        allWidthSweepsExact =
            allWidthSweepsExact && adaptiveGapSweepExact[variantIndex];
        std::cout << ' ' << problem.adaptiveGapSweeps[variantIndex].name
            << "Exact="
            << (adaptiveGapSweepExact[variantIndex] ? "PASS" : "FAIL");
    }
    for
    (
        std::size_t variantIndex=0;
        variantIndex<problem.readyFractionSweeps.size();
        ++variantIndex
    )
    {
        allWidthSweepsExact =
            allWidthSweepsExact && readyFractionSweepExact[variantIndex];
        std::cout << ' ' << problem.readyFractionSweeps[variantIndex].name
            << "Exact="
            << (readyFractionSweepExact[variantIndex] ? "PASS" : "FAIL");
    }
    allWidthSweepsExact =
        allWidthSweepsExact && readyFractionDeterministic;
    if (!problem.readyFractionSweeps.empty())
        std::cout << " IH1W5Deterministic="
            << (readyFractionDeterministic ? "PASS" : "FAIL");
    std::cout << '\n';
    std::cout << "GAMG IH1 permutation total: IH1S1024EqIH1U="
        << (slicedPermutationExact ? "PASS" : "FAIL")
        << " IH1S1024LevelStarts="
        << (slicedLevelStartsValid ? "PASS" : "FAIL")
        << " IH1W1024VsIH1UPositionMismatches="
        << boundedPermutationMismatches
        << " IH1C1024VsIH1UPositionMismatches="
        << continuingPermutationMismatches
        << " IH1C2_1024VsIH1UPositionMismatches="
        << frontierContinuingPermutationMismatches
        << " IH1C2EqIH1UWhenMaxNaturalLE1024="
        << (frontierContinuingNarrowExact ? "PASS" : "FAIL")
        << " IH1C2NarrowLevels="
        << frontierContinuingNarrowLevels
        << " IH1W2ChunkMembership="
        << (orientedChunkMembershipExact ? "PASS" : "FAIL")
        << " IH1W2ReversedChunks="
        << orientedReversedChunks << '/' << orientedChunks
        << " IH1W2FractionReversed="
        << (orientedChunks
            ? static_cast<double>(orientedReversedChunks)/orientedChunks
            : 0.0)
        << '\n';
    std::cout << "GAMG SpMV validation: rowMax=" << rowDifference
        << " flatMax=" << flatDifference
        << " directMax=" << directDifference
        << " directExact=" << (directExact ? "PASS" : "FAIL")
        << " lduOrderDirectMax=" << lduOrderDirectDifference
        << " lduOrderDirectExact="
        << (lduOrderDirectExact ? "PASS" : "FAIL")
        << " reorderedLduOrderDirectMax="
        << reorderedLduOrderDirectDifference
        << " reorderedDirectMax=" << reorderedDirectDifference
        << " rawLduOrderDirectMax=" << rawLduOrderDirectDifference
        << " reorderedLduOrderDirectExact="
        << (reorderedLduOrderDirectExact ? "PASS" : "FAIL")
        << " prefetchMax=" << prefetchDifference
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
        !allWidthSweepsExact
     || !orientedSweepExact
     || !slidingWindowSweepExact
     || !orientedChunkMembershipExact
     || !slicedPermutationExact
     || !slicedLevelStartsValid
     || !frontierContinuingNarrowExact
     || rowDifference > tolerance
     || flatDifference > tolerance
     || directDifference > tolerance
     || lduOrderDirectDifference > tolerance
     || reorderedLduOrderDirectDifference > tolerance
     || rawLduOrderDirectDifference > tolerance
     || prefetchDifference > tolerance
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
        if (!permutationDumpDirectory.empty())
        {
            dumpFinePermutations(mesh, permutationDumpDirectory);
            if (permutationDumpOnly)
            {
                benchmark::Shutdown();
                return 0;
            }
        }
        if (!vtuneSmoother.empty())
        {
            const SmootherProfileProblem problem(mesh);
            validateSmootherProfile(problem);
            prepareProfileHugePages(problem);
            runVtuneSmootherProfile(problem);
            return 0;
        }
        if (releaseImpactExperiment)
        {
            const ReleaseImpactFocusedProblem problem(mesh);
            validateReleaseImpactFocused(problem);
            writeReleaseImpactTraces(problem);
            printReleaseImpactDiagnostics(problem);
            prepareReleaseImpactHugePages(problem);
            registerReleaseImpactBenchmarks(problem);
            benchmark::RunSpecifiedBenchmarks();
            benchmark::Shutdown();
            return 0;
        }
        if (ih1w5Experiment)
        {
            const Ih1w5FocusedProblem problem(mesh);
            validateIh1w5Focused(problem);
            writeIh1w5FocusedChunkTraces(problem);
            printIh1w5FocusedDiagnostics(problem);
            prepareIh1w5FocusedHugePages(problem);
            registerIh1w5FocusedBenchmarks(problem);
            benchmark::RunSpecifiedBenchmarks();
            benchmark::Shutdown();
            return 0;
        }
        const Problem problem(mesh);
        validateSpmv(problem);
        writeIh1w5ChunkTraces(problem);
        printLocality(problem);
        printIh1w5Diagnostics(problem);
        printFlatEdgeSetup(problem);
        prepareHugePages(problem);
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
