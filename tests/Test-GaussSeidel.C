#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include "GaussSeidelSmoother.H"
#include "PolyMeshReader.H"
#include "WavefrontGaussSeidel.H"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <vector>

using namespace Foam;
using smootherTest::PolyMeshReader;
using smootherTest::PolyMeshTopology;
using smootherTest::WavefrontSchedule;
using smootherTest::avx2GatherAvailable;
using smootherTest::avx512GatherAvailable;
using smootherTest::perLevelOpenMpSmooth;
using smootherTest::persistentOpenMpSmooth;
using smootherTest::serialGatherAvx512Smooth;
using smootherTest::serialGatherAvx512AcrossRowsSmooth;
using smootherTest::serialGatherAvx2AcrossRowsSmooth;
using smootherTest::serialGatherAvx2Smooth;
using smootherTest::serialGatherSmooth;
using smootherTest::serialGatherInterleavedRowsSmooth;
using smootherTest::serialGatherDegree6Smooth;
using smootherTest::serialGatherDegree6InterleavedRowsSmooth;
using smootherTest::serialGatherPrefetchSmooth;
using smootherTest::serialReorderedPsiSmooth;

namespace
{

lduMatrix makeMatrix(const PolyMeshTopology& mesh)
{
    labelField upperAddr(mesh.neighbour.begin(), mesh.neighbour.end());
    const std::vector<int> starts = mesh.ownerStartAddressing();
    labelField ownerStart(starts.begin(), starts.end());
    scalarField upper(mesh.nInternalFaces());
    scalarField lower(mesh.nInternalFaces());
    scalarField diag(mesh.nCells, 0.0);

    // Use the real motorBike topology. Coefficients are deterministic and
    // mildly non-uniform; a reaction term makes the matrix strictly dominant.
    for (std::size_t face=0; face<mesh.nInternalFaces(); ++face)
    {
        const label own = mesh.owner[face];
        const label nei = mesh.neighbour[face];
        const scalar coefficient =
            1.0 + 0.001*((17*own + 13*nei) % 101);
        upper[face] = -coefficient;
        lower[face] = -coefficient;
        diag[own] += coefficient;
        diag[nei] += coefficient;
    }
    const scalar meanDiag =
        std::accumulate(diag.begin(), diag.end(), 0.0)/mesh.nCells;
    for (scalar& value : diag) value += 0.25*meanDiag;

    return lduMatrix
    (
        lduAddressing(std::move(upperAddr), std::move(ownerStart)),
        std::move(diag), std::move(upper), std::move(lower)
    );
}

scalarField multiply(const lduMatrix& matrix, const scalarField& psi)
{
    scalarField result(psi.size(), 0.0);
    const auto& starts = matrix.lduAddr().ownerStartAddr();
    const auto& neighbours = matrix.lduAddr().upperAddr();
    for (label cell=0; cell<label(psi.size()); ++cell)
    {
        result[cell] += matrix.diag()[cell]*psi[cell];
        for (label face=starts[cell]; face<starts[cell + 1]; ++face)
        {
            const label neighbour = neighbours[face];
            result[cell] += matrix.upper()[face]*psi[neighbour];
            result[neighbour] += matrix.lower()[face]*psi[cell];
        }
    }
    return result;
}

scalar relativeResidual
(
    const lduMatrix& matrix,
    const scalarField& psi,
    const scalarField& source
)
{
    const scalarField product = multiply(matrix, psi);
    scalar numerator = 0, denominator = 0;
    for (label i=0; i<label(psi.size()); ++i)
    {
        numerator += std::abs(source[i] - product[i]);
        denominator += std::abs(source[i]);
    }
    return numerator/std::max(denominator, 1e-300);
}

struct SweepMeasurements
{
    std::vector<scalar> residuals;
    std::chrono::nanoseconds sweepTime{0};
};

SweepMeasurements measureSweeps
(
    scalarField& psi,
    const lduMatrix& matrix,
    const scalarField& source,
    const label nSweeps,
    const FieldField<Field, scalar>& interfaceCoeffs,
    const lduInterfaceFieldPtrsList& interfaces
)
{
    SweepMeasurements result;
    result.residuals.reserve(static_cast<std::size_t>(nSweeps) + 1);
    result.residuals.push_back(relativeResidual(matrix, psi, source));

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        const auto begin = std::chrono::steady_clock::now();
        GaussSeidelSmoother::smooth
        (
            "psi", psi, matrix, source,
            interfaceCoeffs, interfaces, 0, 1
        );
        const auto end = std::chrono::steady_clock::now();
        result.sweepTime +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

        // Deliberately outside the timed interval.
        result.residuals.push_back(relativeResidual(matrix, psi, source));
    }
    return result;
}

struct WavefrontStatistics
{
    std::vector<label> levels;
    std::vector<std::size_t> widths;
    std::size_t minimumWidth = 0;
    std::size_t maximumWidth = 0;
    scalar meanWidth = 0;
    scalar medianWidth = 0;
    std::size_t p90Width = 0;
    std::size_t p95Width = 0;
};

std::size_t percentile
(
    const std::vector<std::size_t>& sortedValues,
    const scalar fraction
)
{
    const std::size_t index = static_cast<std::size_t>
    (
        std::ceil(fraction*sortedValues.size()) - 1
    );
    return sortedValues[std::min(index, sortedValues.size() - 1)];
}

WavefrontStatistics constructWavefronts(const PolyMeshTopology& mesh)
{
    WavefrontStatistics result;
    result.levels.assign(mesh.nCells, 0);

    // OpenFOAM internal faces are upper-triangular: owner < neighbour.
    // In a forward sweep, neighbour depends on its already-updated owner.
    for (std::size_t face=0; face<mesh.nInternalFaces(); ++face)
    {
        const label dependency = mesh.owner[face];
        const label cell = mesh.neighbour[face];
        result.levels[cell] = std::max
        (
            result.levels[cell], result.levels[dependency] + 1
        );
    }

    const label maximumLevel =
        *std::max_element(result.levels.begin(), result.levels.end());
    result.widths.assign(static_cast<std::size_t>(maximumLevel) + 1, 0);
    for (const label level : result.levels) ++result.widths[level];

    // Validate the exact dependency invariant used to construct the levels.
    for (std::size_t face=0; face<mesh.nInternalFaces(); ++face)
    {
        const label dependency = mesh.owner[face];
        const label cell = mesh.neighbour[face];
        if (result.levels[dependency] >= result.levels[cell])
        {
            throw std::runtime_error
            (
                "wavefront dependency validation failed at internal face "
              + std::to_string(face)
            );
        }
    }

    const auto bounds =
        std::minmax_element(result.widths.begin(), result.widths.end());
    result.minimumWidth = *bounds.first;
    result.maximumWidth = *bounds.second;
    result.meanWidth = scalar(mesh.nCells)/result.widths.size();

    std::vector<std::size_t> sortedWidths = result.widths;
    std::sort(sortedWidths.begin(), sortedWidths.end());
    const std::size_t middle = sortedWidths.size()/2;
    result.medianWidth = sortedWidths.size() % 2
        ? scalar(sortedWidths[middle])
        : 0.5*scalar(sortedWidths[middle - 1] + sortedWidths[middle]);
    result.p90Width = percentile(sortedWidths, 0.90);
    result.p95Width = percentile(sortedWidths, 0.95);
    return result;
}

WavefrontSchedule makeWavefrontSchedule
(
    const PolyMeshTopology& mesh,
    const lduMatrix& matrix,
    const WavefrontStatistics& statistics
)
{
    WavefrontSchedule schedule;
    schedule.levelStarts.resize(statistics.widths.size() + 1, 0);
    for (std::size_t level=0; level<statistics.widths.size(); ++level)
        schedule.levelStarts[level + 1] =
            schedule.levelStarts[level] + statistics.widths[level];

    schedule.waveCells.resize(mesh.nCells);
    std::vector<label> levelCursor = schedule.levelStarts;
    for (label cell=0; cell<label(mesh.nCells); ++cell)
        schedule.waveCells[levelCursor[statistics.levels[cell]]++] = cell;
    std::vector<label> cellToWaveRow(mesh.nCells);
    for (label row=0; row<label(mesh.nCells); ++row)
        cellToWaveRow[schedule.waveCells[row]] = row;

    // This temporary incoming-face index is used only while packing rows.
    // Global face insertion order preserves the reference subtraction order.
    std::vector<label> incomingStarts(mesh.nCells + 1, 0);
    for (const int neighbour : mesh.neighbour)
        ++incomingStarts[static_cast<std::size_t>(neighbour) + 1];
    for (std::size_t cell=0; cell<mesh.nCells; ++cell)
        incomingStarts[cell + 1] += incomingStarts[cell];

    std::vector<label> incomingFaces(mesh.nInternalFaces());
    std::vector<label> cursor = incomingStarts;
    for (label face=0; face<label(mesh.nInternalFaces()); ++face)
    {
        const label neighbour = mesh.neighbour[face];
        incomingFaces[cursor[neighbour]++] = face;
    }

    const labelField& ownerStarts = matrix.lduAddr().ownerStartAddr();
    const labelField& neighbours = matrix.lduAddr().upperAddr();
    const scalarField& lower = matrix.lower();
    const scalarField& upper = matrix.upper();
    schedule.rowStarts.reserve(mesh.nCells + 1);
    schedule.cols.reserve(2*mesh.nInternalFaces());
    schedule.coeffs.reserve(2*mesh.nInternalFaces());
    schedule.diag.reserve(mesh.nCells);
    schedule.incomingCounts.reserve(mesh.nCells);
    schedule.rowStarts.push_back(0);
    for (const label cell : schedule.waveCells)
    {
        // Pack incoming/lower first, then outgoing/upper, without reordering.
        schedule.incomingCounts.push_back
        (
            incomingStarts[cell + 1] - incomingStarts[cell]
        );
        for (label in=incomingStarts[cell]; in<incomingStarts[cell + 1]; ++in)
        {
            const label face = incomingFaces[in];
            schedule.cols.push_back(mesh.owner[face]);
            schedule.coeffs.push_back(lower[face]);
        }
        for (label face=ownerStarts[cell]; face<ownerStarts[cell + 1]; ++face)
        {
            schedule.cols.push_back(neighbours[face]);
            schedule.coeffs.push_back(upper[face]);
        }
        schedule.diag.push_back(matrix.diag()[cell]);
        schedule.rowStarts.push_back(schedule.cols.size());
    }
    schedule.localCols.resize(schedule.cols.size());
    for (std::size_t p=0; p<schedule.cols.size(); ++p)
        schedule.localCols[p] = cellToWaveRow[schedule.cols[p]];
    if
    (
        schedule.rowStarts.size() != mesh.nCells + 1
     || schedule.cols.size() != 2*mesh.nInternalFaces()
     || schedule.coeffs.size() != schedule.cols.size()
     || schedule.localCols.size() != schedule.cols.size()
     || schedule.diag.size() != mesh.nCells
     || schedule.incomingCounts.size() != mesh.nCells
    )
    {
        throw std::runtime_error("invalid packed wavefront schedule size");
    }
    return schedule;
}

WavefrontSchedule makeLocalityReorderedSchedule
(
    const WavefrontSchedule& original
)
{
    WavefrontSchedule reordered;
    reordered.levelStarts = original.levelStarts;
    reordered.waveCells.reserve(original.waveCells.size());
    reordered.rowStarts.reserve(original.rowStarts.size());
    reordered.cols.reserve(original.cols.size());
    reordered.coeffs.reserve(original.coeffs.size());
    reordered.diag.reserve(original.diag.size());
    reordered.rowStarts.push_back(0);

    for (std::size_t level=0; level + 1<original.levelStarts.size(); ++level)
    {
        std::vector<label> rows;
        for (label row=original.levelStarts[level]; row<original.levelStarts[level + 1]; ++row)
        {
            rows.push_back(row);
        }
        const auto localityKey = [&](const label row)
        {
            long long sum = 0;
            const label begin = original.rowStarts[row];
            const label end = original.rowStarts[row + 1];
            for (label p=begin; p<end; ++p) sum += original.cols[p];
            return end == begin ? static_cast<long long>(original.waveCells[row])
                : sum/(end - begin);
        };
        std::stable_sort
        (
            rows.begin(), rows.end(), [&](const label a, const label b)
            {
                const long long keyA = localityKey(a);
                const long long keyB = localityKey(b);
                return keyA == keyB
                    ? original.waveCells[a] < original.waveCells[b]
                    : keyA < keyB;
            }
        );

        for (const label row : rows)
        {
            reordered.waveCells.push_back(original.waveCells[row]);
            reordered.diag.push_back(original.diag[row]);
            std::vector<std::pair<label, scalar>> contributions;
            for (label p=original.rowStarts[row]; p<original.rowStarts[row + 1]; ++p)
                contributions.emplace_back(original.cols[p], original.coeffs[p]);
            std::stable_sort
            (
                contributions.begin(), contributions.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; }
            );
            for (const auto& [column, coefficient] : contributions)
            {
                reordered.cols.push_back(column);
                reordered.coeffs.push_back(coefficient);
            }
            reordered.rowStarts.push_back(reordered.cols.size());
        }
    }
    return reordered;
}

struct ImplementationTiming
{
    label samples = 0;
    label sweepsPerSample = 0;
    scalar minimumSeconds = 0;
    scalar maximumSeconds = 0;
    scalar medianSeconds = 0;
    scalar meanSeconds = 0;
    scalar stddevSeconds = 0;
    scalar cvPercent = 0;
    scalar medianNsPerCellSweep = 0;
};

ImplementationTiming summarizeTimings
(
    const std::vector<scalar>& secondsPerSweep,
    const std::size_t nCells,
    const label sweepsPerSample
)
{
    std::vector<scalar> sorted = secondsPerSweep;
    std::sort(sorted.begin(), sorted.end());
    ImplementationTiming result;
    result.samples = static_cast<label>(secondsPerSweep.size());
    result.sweepsPerSample = sweepsPerSample;
    result.minimumSeconds = sorted.front();
    result.maximumSeconds = sorted.back();
    result.medianSeconds = sorted[sorted.size()/2];
    result.meanSeconds = std::accumulate
    (
        secondsPerSweep.begin(), secondsPerSweep.end(), scalar(0)
    )/secondsPerSweep.size();
    scalar squaredDeviation = 0;
    for (const scalar seconds : secondsPerSweep)
    {
        const scalar deviation = seconds - result.meanSeconds;
        squaredDeviation += deviation*deviation;
    }
    result.stddevSeconds = std::sqrt
    (
        squaredDeviation/secondsPerSweep.size()
    );
    result.cvPercent = 100*result.stddevSeconds/result.meanSeconds;
    result.medianNsPerCellSweep = 1e9*result.medianSeconds/nCells;
    return result;
}

template<class SweepFunction>
ImplementationTiming timeSweepImplementation
(
    const scalarField& initialPsi,
    const std::size_t nCells,
    const label samples,
    const label sweepsPerSample,
    const label warmupSweeps,
    SweepFunction&& executeSweeps
)
{
    scalarField warmupPsi = initialPsi;
    executeSweeps(warmupPsi, warmupSweeps);

    std::vector<scalar> secondsPerSweep;
    secondsPerSweep.reserve(samples);
    scalarField psi(initialPsi.size());
    for (label sample=0; sample<samples; ++sample)
    {
        // Sample initialization is deliberately outside the timed interval.
        psi = initialPsi;
        const auto begin = std::chrono::steady_clock::now();
        executeSweeps(psi, sweepsPerSample);
        const auto end = std::chrono::steady_clock::now();
        secondsPerSweep.push_back
        (
            std::chrono::duration<scalar>(end - begin).count()/sweepsPerSample
        );
    }

    return summarizeTimings(secondsPerSweep, nCells, sweepsPerSample);
}

struct RandomizedBenchmarkVariant
{
    const char* name;
    std::function<void(scalarField&, label)> executeSweeps;
};

std::vector<ImplementationTiming> timeRandomizedImplementations
(
    const scalarField& initialPsi,
    const std::size_t nCells,
    const label samples,
    const label sweepsPerSample,
    const label warmupSweeps,
    const std::uint32_t seed,
    const std::vector<RandomizedBenchmarkVariant>& variants
)
{
    std::vector<std::vector<scalar>> secondsPerSweep(variants.size());
    for (std::vector<scalar>& timings : secondsPerSweep) timings.reserve(samples);
    scalarField psi(initialPsi.size());
    for (const RandomizedBenchmarkVariant& variant : variants)
    {
        psi = initialPsi;
        variant.executeSweeps(psi, warmupSweeps);
    }

    std::vector<std::size_t> order(variants.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 random(seed);
    for (label sample=0; sample<samples; ++sample)
    {
        std::shuffle(order.begin(), order.end(), random);
        for (const std::size_t variantIndex : order)
        {
            psi = initialPsi;
            const auto begin = std::chrono::steady_clock::now();
            variants[variantIndex].executeSweeps(psi, sweepsPerSample);
            const auto end = std::chrono::steady_clock::now();
            secondsPerSweep[variantIndex].push_back
            (
                std::chrono::duration<scalar>(end - begin).count()
               /sweepsPerSample
            );
        }
    }

    std::vector<ImplementationTiming> result;
    result.reserve(variants.size());
    for (const std::vector<scalar>& timings : secondsPerSweep)
        result.push_back(summarizeTimings(timings, nCells, sweepsPerSample));
    return result;
}

void printTiming(const char* variant, const ImplementationTiming& timing)
{
    std::cout << "\nVariant: " << variant
        << "\n  samples: " << timing.samples
        << "\n  sweeps/sample: " << timing.sweepsPerSample
        << "\n  min: " << timing.minimumSeconds << " s/sweep"
        << "\n  max: " << timing.maximumSeconds << " s/sweep"
        << "\n  median: " << timing.medianSeconds << " s/sweep"
        << "\n  mean: " << timing.meanSeconds << " s/sweep"
        << "\n  stddev: " << timing.stddevSeconds << " s/sweep"
        << "\n  CV: " << timing.cvPercent << " %"
        << "\n  median sample duration: "
        << timing.medianSeconds*timing.sweepsPerSample << " s"
        << "\n  median ns/cell/sweep: "
        << timing.medianNsPerCellSweep << '\n';
}

scalar maxAbsDifference(const scalarField& a, const scalarField& b)
{
    scalar result = 0;
    for (label i=0; i<label(a.size()); ++i)
        result = std::max(result, std::abs(a[i] - b[i]));
    return result;
}

std::pair<scalar, scalar> l2Differences
(
    const scalarField& reference,
    const scalarField& candidate
)
{
    scalar squaredDifference = 0;
    scalar squaredReference = 0;
    for (label i=0; i<label(reference.size()); ++i)
    {
        const scalar difference = reference[i] - candidate[i];
        squaredDifference += difference*difference;
        squaredReference += reference[i]*reference[i];
    }
    const scalar l2 = std::sqrt(squaredDifference);
    return {l2, l2/std::max(std::sqrt(squaredReference), 1e-300)};
}

scalar fractionInLevelsAtLeast
(
    const std::vector<std::size_t>& widths,
    const std::size_t threshold,
    const std::size_t nCells
)
{
    std::size_t cells = 0;
    for (const std::size_t width : widths)
        if (width >= threshold) cells += width;
    return scalar(cells)/nCells;
}

struct RowLengthStatistics
{
    scalar mean = 0;
    scalar median = 0;
    label maximum = 0;
    scalar fractionAtMost8 = 0;
    scalar fractionAtMost16 = 0;
    std::size_t fullBatches = 0;
    std::size_t partialBatches = 0;
    scalar averageActiveLanes = 0;
};

RowLengthStatistics rowLengthStatistics(const WavefrontSchedule& schedule)
{
    std::vector<label> lengths(schedule.waveCells.size());
    std::size_t atMost8 = 0;
    std::size_t atMost16 = 0;
    std::size_t total = 0;
    for (std::size_t row=0; row<lengths.size(); ++row)
    {
        lengths[row] = schedule.rowStarts[row + 1] - schedule.rowStarts[row];
        total += lengths[row];
        if (lengths[row] <= 8) ++atMost8;
        if (lengths[row] <= 16) ++atMost16;
    }
    std::sort(lengths.begin(), lengths.end());
    const std::size_t middle = lengths.size()/2;
    RowLengthStatistics result;
    result.mean = scalar(total)/lengths.size();
    result.median = lengths.size() % 2
        ? scalar(lengths[middle])
        : 0.5*scalar(lengths[middle - 1] + lengths[middle]);
    result.maximum = lengths.back();
    result.fractionAtMost8 = scalar(atMost8)/lengths.size();
    result.fractionAtMost16 = scalar(atMost16)/lengths.size();
    std::size_t activeContributions = 0;
    std::size_t neighbourSteps = 0;
    for (std::size_t level=0; level + 1<schedule.levelStarts.size(); ++level)
    {
        const label levelEnd = schedule.levelStarts[level + 1];
        for (label first=schedule.levelStarts[level]; first<levelEnd; first += 8)
        {
            const label batchRows = std::min<label>(8, levelEnd - first);
            if (batchRows == 8) ++result.fullBatches;
            else ++result.partialBatches;
            label maximumLength = 0;
            for (label lane=0; lane<batchRows; ++lane)
            {
                const label length = schedule.rowStarts[first + lane + 1]
                    - schedule.rowStarts[first + lane];
                activeContributions += length;
                maximumLength = std::max(maximumLength, length);
            }
            neighbourSteps += maximumLength;
        }
    }
    result.averageActiveLanes = neighbourSteps
        ? scalar(activeContributions)/neighbourSteps
        : 0;
    return result;
}

std::vector<std::size_t> rowLengthHistogram(const WavefrontSchedule& schedule)
{
    label maximum = 0;
    for (std::size_t row=0; row<schedule.waveCells.size(); ++row)
        maximum = std::max(maximum, schedule.rowStarts[row + 1] - schedule.rowStarts[row]);
    std::vector<std::size_t> histogram(static_cast<std::size_t>(maximum) + 1, 0);
    for (std::size_t row=0; row<schedule.waveCells.size(); ++row)
        ++histogram[schedule.rowStarts[row + 1] - schedule.rowStarts[row]];
    return histogram;
}

void printLevelWidths
(
    const char* labelText,
    const std::vector<std::size_t>& widths,
    const std::size_t begin,
    const std::size_t end
)
{
    std::cout << labelText;
    for (std::size_t i=begin; i<end; ++i)
    {
        if (i != begin) std::cout << ' ';
        std::cout << i << ':' << widths[i];
    }
    std::cout << '\n';
}

}

int runTest(const std::string& meshDirectory, const label historySweeps)
{
    const PolyMeshTopology mesh = PolyMeshReader::read(meshDirectory);
    lduMatrix matrix = makeMatrix(mesh);

    scalarField exact(mesh.nCells);
    for (label cell=0; cell<label(exact.size()); ++cell)
    {
        const scalar position = scalar(cell)/scalar(exact.size() - 1);
        exact[cell] = 1.0 + 0.25*std::sin(2.0*M_PI*position)
            + 0.1*std::cos(10.0*M_PI*position);
    }
    const scalarField source = multiply(matrix, exact);

    // Serial placeholders for coupled/MPI interface data.
    FieldField<Field, scalar> interfaceCoeffs(0);
    lduInterfaceFieldPtrsList interfaces(0);

    // Level and incoming-CSR preprocessing is completed once, before any
    // timed sweep. The benchmark measures only consumption of this schedule.
    const auto preprocessingBegin = std::chrono::steady_clock::now();
    const WavefrontStatistics wavefronts = constructWavefronts(mesh);
    const WavefrontSchedule schedule =
        makeWavefrontSchedule(mesh, matrix, wavefronts);
    const WavefrontSchedule localitySchedule =
        makeLocalityReorderedSchedule(schedule);
    const auto preprocessingEnd = std::chrono::steady_clock::now();
    const scalar preprocessingSeconds =
        std::chrono::duration<scalar>
        (preprocessingEnd - preprocessingBegin).count();
    const RowLengthStatistics rowLengths = rowLengthStatistics(schedule);
    const std::vector<std::size_t> contributionHistogram =
        rowLengthHistogram(schedule);
    const scalarField initialPsi(exact.size(), 0.0);

    scalarField psiReference = initialPsi;
    GaussSeidelSmoother::smooth
    (
        "psi", psiReference, matrix, source,
        interfaceCoeffs, interfaces, 0, 1
    );
    scalarField psiSerialGather = initialPsi;
    serialGatherSmooth(psiSerialGather, source, schedule, 1);
    scalarField psiInterleavedRows = initialPsi;
    serialGatherInterleavedRowsSmooth
    (
        psiInterleavedRows, source, schedule, 1
    );
    scalarField psiDegree6 = initialPsi;
    serialGatherDegree6Smooth(psiDegree6, source, schedule, 1);
    scalarField psiDegree6Interleaved = initialPsi;
    serialGatherDegree6InterleavedRowsSmooth
    (
        psiDegree6Interleaved, source, schedule, 1
    );
    scalarField psiPrefetchFirst = initialPsi;
    serialGatherPrefetchSmooth
    (
        psiPrefetchFirst, source, schedule, 1, 8, 1
    );
    scalarField psiPrefetchTwo = initialPsi;
    serialGatherPrefetchSmooth
    (
        psiPrefetchTwo, source, schedule, 1, 8, 2
    );
    scalarField psiLocality = initialPsi;
    serialGatherSmooth(psiLocality, source, localitySchedule, 1);
    const bool avx512Available = avx512GatherAvailable();
    const bool avx2Available = avx2GatherAvailable();
    scalarField psiAvx2 = initialPsi;
    serialGatherAvx2Smooth(psiAvx2, source, schedule, 1);
    scalarField psiAvx2AcrossRows = initialPsi;
    serialGatherAvx2AcrossRowsSmooth(psiAvx2AcrossRows, source, schedule, 1);
    scalarField psiAvx512 = initialPsi;
    serialGatherAvx512Smooth(psiAvx512, source, schedule, 1);
    scalarField psiAvx512AcrossRows = initialPsi;
    serialGatherAvx512AcrossRowsSmooth
    (
        psiAvx512AcrossRows, source, schedule, 1
    );
    int perLevelThreads = 1;
    scalarField psiPerLevel = initialPsi;
    perLevelOpenMpSmooth
    (
        psiPerLevel, source, schedule, 1, perLevelThreads
    );
    int persistentThreads = 1;
    scalarField psiPersistent = initialPsi;
    persistentOpenMpSmooth
    (
        psiPersistent, source, schedule, 1, persistentThreads
    );

    const scalar serialGatherMaxAbs =
        maxAbsDifference(psiReference, psiSerialGather);
    const scalar interleavedRowsMaxAbs =
        maxAbsDifference(psiReference, psiInterleavedRows);
    const scalar interleavedRowsVsPackedMaxAbs =
        maxAbsDifference(psiSerialGather, psiInterleavedRows);
    const scalar degree6MaxAbs = maxAbsDifference(psiReference, psiDegree6);
    const scalar degree6VsPackedMaxAbs =
        maxAbsDifference(psiSerialGather, psiDegree6);
    const scalar degree6InterleavedMaxAbs =
        maxAbsDifference(psiReference, psiDegree6Interleaved);
    const scalar degree6InterleavedVsPackedMaxAbs =
        maxAbsDifference(psiSerialGather, psiDegree6Interleaved);
    const scalar localityMaxAbs = maxAbsDifference(psiReference, psiLocality);
    const scalar prefetchFirstMaxAbs =
        maxAbsDifference(psiReference, psiPrefetchFirst);
    const scalar prefetchTwoMaxAbs =
        maxAbsDifference(psiReference, psiPrefetchTwo);
    const scalar avx512MaxAbs = maxAbsDifference(psiReference, psiAvx512);
    const scalar avx2MaxAbs = maxAbsDifference(psiReference, psiAvx2);
    const scalar avx2AcrossRowsMaxAbs =
        maxAbsDifference(psiReference, psiAvx2AcrossRows);
    const scalar avx512AcrossRowsMaxAbs =
        maxAbsDifference(psiReference, psiAvx512AcrossRows);
    const scalar perLevelMaxAbs =
        maxAbsDifference(psiReference, psiPerLevel);
    const scalar persistentMaxAbs =
        maxAbsDifference(psiReference, psiPersistent);
    const auto [serialGatherL2, serialGatherRelativeL2] =
        l2Differences(psiReference, psiSerialGather);
    const auto [interleavedRowsL2, interleavedRowsRelativeL2] =
        l2Differences(psiReference, psiInterleavedRows);
    const auto [degree6L2, degree6RelativeL2] =
        l2Differences(psiReference, psiDegree6);
    const auto [degree6InterleavedL2, degree6InterleavedRelativeL2] =
        l2Differences(psiReference, psiDegree6Interleaved);
    const auto [localityL2, localityRelativeL2] =
        l2Differences(psiReference, psiLocality);
    const auto [prefetchFirstL2, prefetchFirstRelativeL2] =
        l2Differences(psiReference, psiPrefetchFirst);
    const auto [prefetchTwoL2, prefetchTwoRelativeL2] =
        l2Differences(psiReference, psiPrefetchTwo);
    const auto [avx512L2, avx512RelativeL2] =
        l2Differences(psiReference, psiAvx512);
    const auto [avx2L2, avx2RelativeL2] =
        l2Differences(psiReference, psiAvx2);
    const auto [avx2AcrossRowsL2, avx2AcrossRowsRelativeL2] =
        l2Differences(psiReference, psiAvx2AcrossRows);
    const auto [avx512AcrossRowsL2, avx512AcrossRowsRelativeL2] =
        l2Differences(psiReference, psiAvx512AcrossRows);
    const auto [perLevelL2, perLevelRelativeL2] =
        l2Differences(psiReference, psiPerLevel);
    const auto [persistentL2, persistentRelativeL2] =
        l2Differences(psiReference, psiPersistent);

    const scalar referenceOneSweepResidual =
        relativeResidual(matrix, psiReference, source);
    const scalar serialGatherResidual =
        relativeResidual(matrix, psiSerialGather, source);
    const scalar interleavedRowsResidual =
        relativeResidual(matrix, psiInterleavedRows, source);
    const scalar degree6Residual = relativeResidual(matrix, psiDegree6, source);
    const scalar degree6InterleavedResidual =
        relativeResidual(matrix, psiDegree6Interleaved, source);
    const scalar localityResidual = relativeResidual(matrix, psiLocality, source);
    const scalar prefetchFirstResidual =
        relativeResidual(matrix, psiPrefetchFirst, source);
    const scalar prefetchTwoResidual =
        relativeResidual(matrix, psiPrefetchTwo, source);
    const scalar avx512Residual = relativeResidual(matrix, psiAvx512, source);
    const scalar avx2Residual = relativeResidual(matrix, psiAvx2, source);
    const scalar avx2AcrossRowsResidual =
        relativeResidual(matrix, psiAvx2AcrossRows, source);
    const scalar avx512AcrossRowsResidual =
        relativeResidual(matrix, psiAvx512AcrossRows, source);
    const scalar perLevelResidual =
        relativeResidual(matrix, psiPerLevel, source);
    const scalar persistentResidual =
        relativeResidual(matrix, psiPersistent, source);
    const scalar avx512ResidualDifference =
        std::abs(referenceOneSweepResidual - avx512Residual);
    const scalar avx2ResidualDifference =
        std::abs(referenceOneSweepResidual - avx2Residual);
    const scalar avx2AcrossRowsResidualDifference =
        std::abs(referenceOneSweepResidual - avx2AcrossRowsResidual);
    const scalar avx512AcrossRowsResidualDifference =
        std::abs(referenceOneSweepResidual - avx512AcrossRowsResidual);
    const scalar localityResidualDifference =
        std::abs(referenceOneSweepResidual - localityResidual);
    const scalar prefetchFirstResidualDifference =
        std::abs(referenceOneSweepResidual - prefetchFirstResidual);
    const scalar prefetchTwoResidualDifference =
        std::abs(referenceOneSweepResidual - prefetchTwoResidual);
    const scalar interleavedRowsResidualDifference =
        std::abs(referenceOneSweepResidual - interleavedRowsResidual);
    const scalar degree6ResidualDifference =
        std::abs(referenceOneSweepResidual - degree6Residual);
    const scalar degree6InterleavedResidualDifference =
        std::abs(referenceOneSweepResidual - degree6InterleavedResidual);

    constexpr scalar equivalenceTolerance = 1e-12;
    const auto status = [&](const scalar difference)
    {
        return difference <= equivalenceTolerance ? "PASS" : "FAIL";
    };
    std::cout << "motorBike cells: " << mesh.nCells
        << "\ninternal faces: " << mesh.nInternalFaces()
        << "\n\nOne-sweep correctness:"
        << "\nReference GS residual:          " << referenceOneSweepResidual
        << "\nSerial gather residual:         " << serialGatherResidual
        << "\nInterleaved rows residual:      " << interleavedRowsResidual
        << "\nDegree-6 scalar residual:       " << degree6Residual
        << "\nDegree-6 interleaved residual:  " << degree6InterleavedResidual
        << "\nPrefetch (8, first) residual:   " << prefetchFirstResidual
        << "\nPrefetch (8, first 2) residual: " << prefetchTwoResidual
        << "\nLocality-reordered residual:    " << localityResidual
        << "\nSerial packed AVX2 residual:    " << avx2Residual
        << "\nAVX2 across-rows residual:      " << avx2AcrossRowsResidual
        << "\nSerial packed AVX-512 residual: " << avx512Residual
        << "\nAVX-512 across-rows residual:   " << avx512AcrossRowsResidual
        << "\nPer-level OpenMP residual:      " << perLevelResidual
        << "\nPersistent OpenMP residual:     " << persistentResidual
        << "\nSerial gather residual difference:     "
        << std::abs(referenceOneSweepResidual - serialGatherResidual)
        << "\nInterleaved rows residual difference: "
        << interleavedRowsResidualDifference
        << "\nDegree-6 residual difference:         "
        << degree6ResidualDifference
        << "\nDegree-6 interleaved residual difference: "
        << degree6InterleavedResidualDifference
        << "\nPrefetch first residual difference:    "
        << prefetchFirstResidualDifference
        << "\nPrefetch first 2 residual difference:  "
        << prefetchTwoResidualDifference
        << "\nLocality-reordered residual difference: "
        << localityResidualDifference
        << "\nAVX-512 residual difference:           "
        << avx512ResidualDifference
        << "\nAVX2 residual difference:              "
        << avx2ResidualDifference
        << "\nAVX2 across-rows residual difference:  "
        << avx2AcrossRowsResidualDifference
        << "\nAVX-512 across-rows residual difference: "
        << avx512AcrossRowsResidualDifference
        << "\nPer-level OMP residual difference:     "
        << std::abs(referenceOneSweepResidual - perLevelResidual)
        << "\nPersistent OMP residual difference:    "
        << std::abs(referenceOneSweepResidual - persistentResidual)
        << "\n\nmax |reference - serial gather|:  " << serialGatherMaxAbs
        << "\nmax |reference - interleaved|:    " << interleavedRowsMaxAbs
        << "\nmax |packed - interleaved|:       "
        << interleavedRowsVsPackedMaxAbs
        << "\nmax |reference - degree-6|:      " << degree6MaxAbs
        << "\nmax |packed - degree-6|:         " << degree6VsPackedMaxAbs
        << "\nmax |reference - degree-6 ILP|:  " << degree6InterleavedMaxAbs
        << "\nmax |packed - degree-6 ILP|:     "
        << degree6InterleavedVsPackedMaxAbs
        << "\nmax |reference - prefetch first|: " << prefetchFirstMaxAbs
        << "\nmax |reference - prefetch first 2|: " << prefetchTwoMaxAbs
        << "\nmax |reference - locality|:       " << localityMaxAbs
        << "\nmax |reference - AVX-512|:        " << avx512MaxAbs
        << "\nmax |reference - AVX2|:           " << avx2MaxAbs
        << "\nmax |reference - AVX2 rows|:      " << avx2AcrossRowsMaxAbs
        << "\nmax |reference - AVX-512 rows|:   " << avx512AcrossRowsMaxAbs
        << "\nmax |reference - per-level OMP|:  " << perLevelMaxAbs
        << "\nmax |reference - persistent OMP|: " << persistentMaxAbs
        << "\n\nserial gather L2 / relative L2:  "
        << serialGatherL2 << " / " << serialGatherRelativeL2
        << "\ninterleaved L2 / relative L2:   "
        << interleavedRowsL2 << " / " << interleavedRowsRelativeL2
        << "\ndegree-6 L2 / relative L2:     "
        << degree6L2 << " / " << degree6RelativeL2
        << "\ndegree-6 ILP L2 / relative L2: "
        << degree6InterleavedL2 << " / " << degree6InterleavedRelativeL2
        << "\nprefetch first L2 / relative L2: "
        << prefetchFirstL2 << " / " << prefetchFirstRelativeL2
        << "\nprefetch first 2 L2 / relative L2: "
        << prefetchTwoL2 << " / " << prefetchTwoRelativeL2
        << "\nlocality L2 / relative L2:       "
        << localityL2 << " / " << localityRelativeL2
        << "\nAVX-512 L2 / relative L2:        "
        << avx512L2 << " / " << avx512RelativeL2
        << "\nAVX2 L2 / relative L2:           "
        << avx2L2 << " / " << avx2RelativeL2
        << "\nAVX2 rows L2 / relative L2:     "
        << avx2AcrossRowsL2 << " / " << avx2AcrossRowsRelativeL2
        << "\nAVX-512 rows L2 / relative L2:   "
        << avx512AcrossRowsL2 << " / " << avx512AcrossRowsRelativeL2
        << "\nper-level OMP L2 / relative L2:  "
        << perLevelL2 << " / " << perLevelRelativeL2
        << "\npersistent OMP L2 / relative L2: "
        << persistentL2 << " / " << persistentRelativeL2
        << "\n\nserial gather exact equality:    "
        << (serialGatherMaxAbs == 0 ? "PASS" : "NO")
        << "\ninterleaved rows exact equality: "
        << (interleavedRowsMaxAbs == 0 ? "PASS" : "NO")
        << "\ndegree-6 exact equality:        "
        << (degree6MaxAbs == 0 ? "PASS" : "NO")
        << "\ndegree-6 ILP exact equality:    "
        << (degree6InterleavedMaxAbs == 0 ? "PASS" : "NO")
        << "\nprefetch first exact equality:   "
        << (prefetchFirstMaxAbs == 0 ? "PASS" : "NO")
        << "\nprefetch first 2 exact equality: "
        << (prefetchTwoMaxAbs == 0 ? "PASS" : "NO")
        << "\nAVX-512 rows exact equality:     "
        << (avx512AcrossRowsMaxAbs == 0 ? "PASS" : "NO")
        << "\nAVX2 rows exact equality:        "
        << (avx2AcrossRowsMaxAbs == 0 ? "PASS" : "NO")
        << "\nper-level OMP exact equality:    "
        << (perLevelMaxAbs == 0 ? "PASS" : "NO")
        << "\npersistent OMP exact equality:   "
        << (persistentMaxAbs == 0 ? "PASS" : "NO")
        << "\nserial gather equivalence:       " << status(serialGatherMaxAbs)
        << "\ninterleaved rows equivalence:    " << status(interleavedRowsMaxAbs)
        << "\ndegree-6 equivalence:           " << status(degree6MaxAbs)
        << "\ndegree-6 ILP equivalence:       "
        << status(degree6InterleavedMaxAbs)
        << "\nprefetch first equivalence:      " << status(prefetchFirstMaxAbs)
        << "\nprefetch first 2 equivalence:    " << status(prefetchTwoMaxAbs)
        << "\nlocality-reordered equivalence:  " << status(localityMaxAbs)
        << "\nAVX-512 equivalence:             " << status(avx512MaxAbs)
        << "\nAVX2 equivalence:                " << status(avx2MaxAbs)
        << "\nAVX2 across-rows equivalence:    " << status(avx2AcrossRowsMaxAbs)
        << "\nAVX-512 across-rows equivalence: "
        << status(avx512AcrossRowsMaxAbs)
        << "\nAVX-512 kernel available:        "
        << (avx512Available ? "YES" : "NO (scalar fallback)")
        << "\nAVX2 kernels available:          "
        << (avx2Available ? "YES" : "NO (scalar fallback)")
        << "\nper-level OMP equivalence:       " << status(perLevelMaxAbs)
        << "\npersistent OMP equivalence:      " << status(persistentMaxAbs)
        << "\nequivalence tolerance:           " << equivalenceTolerance
        << "\n\n";
    if
    (
        serialGatherMaxAbs > equivalenceTolerance
     || interleavedRowsMaxAbs > equivalenceTolerance
     || interleavedRowsVsPackedMaxAbs > equivalenceTolerance
     || interleavedRowsResidualDifference > equivalenceTolerance
     || degree6MaxAbs > equivalenceTolerance
     || degree6VsPackedMaxAbs > equivalenceTolerance
     || degree6ResidualDifference > equivalenceTolerance
     || degree6InterleavedMaxAbs > equivalenceTolerance
     || degree6InterleavedVsPackedMaxAbs > equivalenceTolerance
     || degree6InterleavedResidualDifference > equivalenceTolerance
     || prefetchFirstMaxAbs > equivalenceTolerance
     || prefetchFirstResidualDifference > equivalenceTolerance
     || prefetchTwoMaxAbs > equivalenceTolerance
     || prefetchTwoResidualDifference > equivalenceTolerance
     || localityMaxAbs > equivalenceTolerance
     || localityResidualDifference > equivalenceTolerance
     || avx2MaxAbs > equivalenceTolerance
     || avx2ResidualDifference > equivalenceTolerance
     || avx2AcrossRowsMaxAbs > equivalenceTolerance
     || avx2AcrossRowsResidualDifference > equivalenceTolerance
     || avx512MaxAbs > equivalenceTolerance
     || avx512ResidualDifference > equivalenceTolerance
     || avx512AcrossRowsMaxAbs > equivalenceTolerance
     || avx512AcrossRowsResidualDifference > equivalenceTolerance
     || perLevelMaxAbs > equivalenceTolerance
     || persistentMaxAbs > equivalenceTolerance
    )
    {
        throw std::runtime_error("one-sweep wavefront equivalence check failed");
    }

    constexpr label reorderedValidationSweeps = 20;
    scalarField psiReorderedReference = initialPsi;
    GaussSeidelSmoother::smooth
    (
        "psi", psiReorderedReference, matrix, source,
        interfaceCoeffs, interfaces, 0, reorderedValidationSweeps
    );
    scalarField psiReorderedOriginal = initialPsi;
    scalarField psiReorderedWorkspace(initialPsi.size());
    serialReorderedPsiSmooth
    (
        psiReorderedOriginal, psiReorderedWorkspace, source, schedule,
        reorderedValidationSweeps
    );
    const scalar reorderedPsiMaxAbs =
        maxAbsDifference(psiReorderedReference, psiReorderedOriginal);
    const scalar reorderedPsiResidualDifference = std::abs
    (
        relativeResidual(matrix, psiReorderedReference, source)
      - relativeResidual(matrix, psiReorderedOriginal, source)
    );
    std::cout << "Reordered-psi correctness (" << reorderedValidationSweeps
        << " sweeps):"
        << "\nmax |reference - reordered psi|: " << reorderedPsiMaxAbs
        << "\nresidual difference:              "
        << reorderedPsiResidualDifference
        << "\nequivalence:                     "
        << status(reorderedPsiMaxAbs) << "\n\n";
    if
    (
        reorderedPsiMaxAbs > equivalenceTolerance
     || reorderedPsiResidualDifference > equivalenceTolerance
    )
    {
        throw std::runtime_error("reordered-psi equivalence check failed");
    }

    constexpr label timingSamples = 7;
    constexpr label timingSweepsPerSample = 250;
    constexpr label warmupSweeps = 10;
    const ImplementationTiming sequentialTiming = timeSweepImplementation
    (
        initialPsi, mesh.nCells, timingSamples, timingSweepsPerSample,
        warmupSweeps, [&](scalarField& timedPsi, const label nSweeps)
        {
            GaussSeidelSmoother::smooth
            (
                "psi", timedPsi, matrix, source,
                interfaceCoeffs, interfaces, 0, nSweeps
            );
        }
    );
    const ImplementationTiming serialGatherTiming = timeSweepImplementation
    (
        initialPsi, mesh.nCells, timingSamples, timingSweepsPerSample,
        warmupSweeps, [&](scalarField& timedPsi, const label nSweeps)
        {
            serialGatherSmooth(timedPsi, source, schedule, nSweeps);
        }
    );
    constexpr std::array<label, 4> prefetchDistances{{4, 8, 16, 32}};
    std::array<ImplementationTiming, prefetchDistances.size()>
        prefetchFirstTimings;
    for (std::size_t i=0; i<prefetchDistances.size(); ++i)
    {
        const label distance = prefetchDistances[i];
        prefetchFirstTimings[i] = timeSweepImplementation
        (
            initialPsi, mesh.nCells, timingSamples, timingSweepsPerSample,
            warmupSweeps, [&](scalarField& timedPsi, const label nSweeps)
            {
                serialGatherPrefetchSmooth
                (
                    timedPsi, source, schedule, nSweeps, distance, 1
                );
            }
        );
    }
    const auto bestPrefetchIterator = std::min_element
    (
        prefetchFirstTimings.begin(), prefetchFirstTimings.end(),
        [](const ImplementationTiming& left, const ImplementationTiming& right)
        {
            return left.medianSeconds < right.medianSeconds;
        }
    );
    const std::size_t bestPrefetchIndex = static_cast<std::size_t>
    (
        std::distance(prefetchFirstTimings.begin(), bestPrefetchIterator)
    );
    const label bestPrefetchDistance = prefetchDistances[bestPrefetchIndex];
    const ImplementationTiming prefetchTwoTiming = timeSweepImplementation
    (
        initialPsi, mesh.nCells, timingSamples, timingSweepsPerSample,
        warmupSweeps, [&](scalarField& timedPsi, const label nSweeps)
        {
            serialGatherPrefetchSmooth
            (
                timedPsi, source, schedule, nSweeps, bestPrefetchDistance, 2
            );
        }
    );
    const ImplementationTiming localityTiming = timeSweepImplementation
    (
        initialPsi, mesh.nCells, timingSamples, timingSweepsPerSample,
        warmupSweeps, [&](scalarField& timedPsi, const label nSweeps)
        {
            serialGatherSmooth
            (
                timedPsi, source, localitySchedule, nSweeps
            );
        }
    );
    const ImplementationTiming avx2Timing = timeSweepImplementation
    (
        initialPsi, mesh.nCells, timingSamples, timingSweepsPerSample,
        warmupSweeps, [&](scalarField& timedPsi, const label nSweeps)
        {
            serialGatherAvx2Smooth(timedPsi, source, schedule, nSweeps);
        }
    );
    const ImplementationTiming avx2AcrossRowsTiming = timeSweepImplementation
    (
        initialPsi, mesh.nCells, timingSamples, timingSweepsPerSample,
        warmupSweeps, [&](scalarField& timedPsi, const label nSweeps)
        {
            serialGatherAvx2AcrossRowsSmooth
            (
                timedPsi, source, schedule, nSweeps
            );
        }
    );
    const ImplementationTiming avx512Timing = timeSweepImplementation
    (
        initialPsi, mesh.nCells, timingSamples, timingSweepsPerSample,
        warmupSweeps, [&](scalarField& timedPsi, const label nSweeps)
        {
            serialGatherAvx512Smooth(timedPsi, source, schedule, nSweeps);
        }
    );
    const ImplementationTiming avx512AcrossRowsTiming = timeSweepImplementation
    (
        initialPsi, mesh.nCells, timingSamples, timingSweepsPerSample,
        warmupSweeps, [&](scalarField& timedPsi, const label nSweeps)
        {
            serialGatherAvx512AcrossRowsSmooth
            (
                timedPsi, source, schedule, nSweeps
            );
        }
    );
    const ImplementationTiming perLevelTiming = timeSweepImplementation
    (
        initialPsi, mesh.nCells, timingSamples, timingSweepsPerSample,
        warmupSweeps, [&](scalarField& timedPsi, const label nSweeps)
        {
            perLevelOpenMpSmooth
            (
                timedPsi, source, schedule, nSweeps, perLevelThreads
            );
        }
    );
    const ImplementationTiming persistentTiming = timeSweepImplementation
    (
        initialPsi, mesh.nCells, timingSamples, timingSweepsPerSample,
        warmupSweeps, [&](scalarField& timedPsi, const label nSweeps)
        {
            persistentOpenMpSmooth
            (
                timedPsi, source, schedule, nSweeps, persistentThreads
            );
        }
    );

    constexpr label productionSweepsPerCall = 3;
    constexpr label productionCallsPerSample = 100;
    constexpr label productionSweepsPerSample =
        productionSweepsPerCall*productionCallsPerSample;
    constexpr label productionWarmupSweeps = 12;
    constexpr std::uint32_t productionBenchmarkSeed = 0x5EED1234u;
    const std::vector<RandomizedBenchmarkVariant> productionVariants
    {
        {
            "Reference GS",
            [&](scalarField& timedPsi, const label nSweeps)
            {
                for (label sweep=0; sweep<nSweeps; sweep += productionSweepsPerCall)
                {
                    GaussSeidelSmoother::smooth
                    (
                        "psi", timedPsi, matrix, source,
                        interfaceCoeffs, interfaces, 0,
                        productionSweepsPerCall
                    );
                }
            }
        },
        {
            "Packed scalar",
            [&](scalarField& timedPsi, const label nSweeps)
            {
                for (label sweep=0; sweep<nSweeps; sweep += productionSweepsPerCall)
                {
                    serialGatherSmooth
                    (
                        timedPsi, source, schedule, productionSweepsPerCall
                    );
                }
            }
        },
        {
            "Generic 4-way",
            [&](scalarField& timedPsi, const label nSweeps)
            {
                for (label sweep=0; sweep<nSweeps; sweep += productionSweepsPerCall)
                {
                    serialGatherInterleavedRowsSmooth
                    (
                        timedPsi, source, schedule, productionSweepsPerCall
                    );
                }
            }
        },
        {
            "Degree-6 scalar",
            [&](scalarField& timedPsi, const label nSweeps)
            {
                for (label sweep=0; sweep<nSweeps; sweep += productionSweepsPerCall)
                {
                    serialGatherDegree6Smooth
                    (
                        timedPsi, source, schedule, productionSweepsPerCall
                    );
                }
            }
        },
        {
            "Degree-6 4-way",
            [&](scalarField& timedPsi, const label nSweeps)
            {
                for (label sweep=0; sweep<nSweeps; sweep += productionSweepsPerCall)
                {
                    serialGatherDegree6InterleavedRowsSmooth
                    (
                        timedPsi, source, schedule, productionSweepsPerCall
                    );
                }
            }
        }
    };
    const std::vector<ImplementationTiming> productionTimings =
        timeRandomizedImplementations
        (
            initialPsi, mesh.nCells, timingSamples, productionSweepsPerSample,
            productionWarmupSweeps, productionBenchmarkSeed,
            productionVariants
        );
    const ImplementationTiming& productionReferenceTiming = productionTimings[0];
    const ImplementationTiming& productionPackedTiming = productionTimings[1];
    const ImplementationTiming& productionInterleavedTiming = productionTimings[2];
    const ImplementationTiming& productionDegree6Timing = productionTimings[3];
    const ImplementationTiming& productionDegree6InterleavedTiming =
        productionTimings[4];


    scalarField psi = initialPsi;
    const SweepMeasurements measurements = measureSweeps
    (
        psi, matrix, source, historySweeps, interfaceCoeffs, interfaces
    );
    const scalar initial = measurements.residuals.front();
    const scalar afterOne = measurements.residuals[1];
    if (!(afterOne < initial))
        throw std::runtime_error("one sweep did not reduce the residual");

    constexpr label totalCorrectnessSweeps = 200;
    if (historySweeps < totalCorrectnessSweeps)
    {
        GaussSeidelSmoother::smooth
        (
            "psi", psi, matrix, source, interfaceCoeffs, interfaces, 0,
            totalCorrectnessSweeps - historySweeps
        );
    }
    const scalar final = relativeResidual(matrix, psi, source);
    const label executedCorrectnessSweeps =
        std::max(historySweeps, totalCorrectnessSweeps);
    scalar maxError = 0;
    for (label i=0; i<label(psi.size()); ++i)
        maxError = std::max(maxError, std::abs(psi[i] - exact[i]));

    const std::size_t totalLevelCells =
        std::accumulate
        (
            wavefronts.widths.begin(), wavefronts.widths.end(), std::size_t(0)
        );
    if (totalLevelCells != mesh.nCells)
        throw std::runtime_error("wavefront levels do not cover every cell");

    const scalar totalSeconds =
        std::chrono::duration<scalar>(measurements.sweepTime).count();
    const scalar totalNanoseconds = scalar(measurements.sweepTime.count());
    const scalar averageSeconds = totalSeconds/historySweeps;
    const scalar nsPerCellSweep =
        totalNanoseconds/(scalar(mesh.nCells)*historySweeps);
    const scalar nsPerFaceSweep =
        totalNanoseconds/(scalar(mesh.nInternalFaces())*historySweeps);
    const scalar eta = -std::log
    (
        measurements.residuals.back()/measurements.residuals.front()
    )/totalSeconds;

    std::cout << "initial residual: " << initial
        << "\none-sweep residual: " << afterOne << "\n\n";

    std::cout << "Residual history (first " << historySweeps
        << " individually timed sweeps):\n"
        << "sweep   residual        ratio           effective-factor\n";
    const auto oldFlags = std::cout.flags();
    const auto oldPrecision = std::cout.precision();
    std::cout << std::scientific << std::setprecision(8);
    for (std::size_t sweep=0; sweep<measurements.residuals.size(); ++sweep)
    {
        std::cout << std::setw(5) << sweep << "   "
            << std::setw(14) << measurements.residuals[sweep] << "   ";
        if (sweep == 0)
        {
            std::cout << std::setw(14) << "-" << "   "
                << std::setw(16) << "-" << '\n';
        }
        else
        {
            const scalar ratio =
                measurements.residuals[sweep]/measurements.residuals[sweep - 1];
            const scalar effectiveFactor = std::pow
            (
                measurements.residuals[sweep]/measurements.residuals[0],
                1.0/scalar(sweep)
            );
            std::cout << std::setw(14) << ratio << "   "
                << std::setw(16) << effectiveFactor << '\n';
        }
    }
    std::cout.flags(oldFlags);
    std::cout.precision(oldPrecision);

    std::cout << "\nResidual-history reference timing "
        << "(diagnostic, not used for comparisons):\n"
        << "history/timed sweeps: " << historySweeps
        << "\ntotal sweep time: " << totalSeconds << " s"
        << "\naverage sweep time: " << averageSeconds << " s"
        << "\nns/cell/sweep: " << nsPerCellSweep
        << "\nns/internal-face/sweep: " << nsPerFaceSweep
        << "\nsmoothing efficiency eta: " << eta << " 1/s\n";

    std::cout << "\nWavefront dependency statistics:\n"
        << "preprocessing time: " << preprocessingSeconds << " s\n"
        << "levels: " << wavefronts.widths.size()
        << "\ncells: " << totalLevelCells
        << "\nmin width: " << wavefronts.minimumWidth
        << "\nmean width: " << wavefronts.meanWidth
        << "\nmedian width: " << wavefronts.medianWidth
        << "\np90 width: " << wavefronts.p90Width
        << "\np95 width: " << wavefronts.p95Width
        << "\nmax width: " << wavefronts.maximumWidth
        << "\nmean packed row length: " << rowLengths.mean
        << "\nmedian packed row length: " << rowLengths.median
        << "\nmax packed row length: " << rowLengths.maximum
        << "\nfraction of rows with <= 8 entries: " << rowLengths.fractionAtMost8
        << "\nfraction of rows with <= 16 entries: " << rowLengths.fractionAtMost16
        << "\nfull 8-row SIMD batches: " << rowLengths.fullBatches
        << "\npartial SIMD batches: " << rowLengths.partialBatches
        << "\naverage active SIMD lanes per neighbour step: "
        << rowLengths.averageActiveLanes
        << "\naverage available parallelism: " << wavefronts.meanWidth
        << "\nfraction in levels with width >= 2: "
        << fractionInLevelsAtLeast(wavefronts.widths, 2, mesh.nCells)
        << "\nfraction in levels with width >= 4: "
        << fractionInLevelsAtLeast(wavefronts.widths, 4, mesh.nCells)
        << "\nfraction in levels with width >= 8: "
        << fractionInLevelsAtLeast(wavefronts.widths, 8, mesh.nCells)
        << "\nfraction in levels with width >= 16: "
        << fractionInLevelsAtLeast(wavefronts.widths, 16, mesh.nCells) << '\n';
    std::cout << "Packed row contribution histogram:\n";
    for (std::size_t degree=0; degree<contributionHistogram.size(); ++degree)
    {
        if (contributionHistogram[degree] != 0)
            std::cout << "degree " << degree << ": "
                << contributionHistogram[degree] << '\n';
    }
    const std::size_t shown = std::min<std::size_t>(10, wavefronts.widths.size());
    printLevelWidths("first levels: ", wavefronts.widths, 0, shown);
    printLevelWidths
    (
        "last levels: ", wavefronts.widths,
        wavefronts.widths.size() - shown, wavefronts.widths.size()
    );
    std::cout << "Wavefront widths:\n";
    for (std::size_t level=0; level<wavefronts.widths.size(); ++level)
        std::cout << "level " << level << ": " << wavefronts.widths[level] << '\n';
    const scalar gatherSlowdown =
        serialGatherTiming.medianSeconds/sequentialTiming.medianSeconds;
    const scalar localitySpeedupScalar =
        serialGatherTiming.medianSeconds/localityTiming.medianSeconds;
    const scalar localitySpeedupReference =
        sequentialTiming.medianSeconds/localityTiming.medianSeconds;
    const scalar avx2SpeedupScalar =
        serialGatherTiming.medianSeconds/avx2Timing.medianSeconds;
    const scalar avx2SpeedupReference =
        sequentialTiming.medianSeconds/avx2Timing.medianSeconds;
    const scalar avx2AcrossRowsSpeedupScalar =
        serialGatherTiming.medianSeconds/avx2AcrossRowsTiming.medianSeconds;
    const scalar avx2AcrossRowsSpeedupReference =
        sequentialTiming.medianSeconds/avx2AcrossRowsTiming.medianSeconds;
    const scalar avx2AcrossRowsSpeedupIntra =
        avx2Timing.medianSeconds/avx2AcrossRowsTiming.medianSeconds;
    const scalar avx512SpeedupScalar =
        serialGatherTiming.medianSeconds/avx512Timing.medianSeconds;
    const scalar avx512SpeedupReference =
        sequentialTiming.medianSeconds/avx512Timing.medianSeconds;
    const scalar avx512AcrossRowsSpeedupScalar =
        serialGatherTiming.medianSeconds
       /avx512AcrossRowsTiming.medianSeconds;
    const scalar avx512AcrossRowsSpeedupReference =
        sequentialTiming.medianSeconds
       /avx512AcrossRowsTiming.medianSeconds;
    const scalar avx512AcrossRowsSpeedupIntra =
        avx512Timing.medianSeconds
       /avx512AcrossRowsTiming.medianSeconds;
    const scalar perLevelSpeedupGather =
        serialGatherTiming.medianSeconds/perLevelTiming.medianSeconds;
    const scalar perLevelSpeedupReference =
        sequentialTiming.medianSeconds/perLevelTiming.medianSeconds;
    const scalar persistentSpeedupGather =
        serialGatherTiming.medianSeconds/persistentTiming.medianSeconds;
    const scalar persistentSpeedupReference =
        sequentialTiming.medianSeconds/persistentTiming.medianSeconds;
    const scalar persistentVsPerLevel =
        perLevelTiming.medianSeconds/persistentTiming.medianSeconds;

    std::cout << "dependency validation: PASS\n"
        << "\nStable performance benchmark:"
        << "\n  warm-up sweeps per variant: " << warmupSweeps
        << "\n  OpenMP threads detected: " << persistentThreads
        << "\n  primary comparison statistic: median\n";
    printTiming("Reference sequential GS", sequentialTiming);
    printTiming("Serial packed scalar", serialGatherTiming);
    for (std::size_t i=0; i<prefetchDistances.size(); ++i)
    {
        const std::string name =
            "Serial packed scalar prefetch first, distance "
          + std::to_string(prefetchDistances[i]);
        printTiming(name.c_str(), prefetchFirstTimings[i]);
    }
    const std::string prefetchTwoName =
        "Serial packed scalar prefetch first 2, distance "
      + std::to_string(bestPrefetchDistance);
    printTiming(prefetchTwoName.c_str(), prefetchTwoTiming);
    printTiming("Serial packed locality-reordered", localityTiming);
    printTiming("Serial packed AVX2", avx2Timing);
    printTiming("Serial packed AVX2 across rows", avx2AcrossRowsTiming);
    printTiming("Serial packed AVX-512", avx512Timing);
    printTiming("Serial packed AVX-512 across rows", avx512AcrossRowsTiming);
    printTiming("Wavefront per-level OpenMP", perLevelTiming);
    printTiming("Wavefront persistent OpenMP", persistentTiming);

    const scalar productionInterleavedSpeedupPacked =
        productionPackedTiming.medianSeconds
       /productionInterleavedTiming.medianSeconds;
    const scalar productionInterleavedSpeedupReference =
        productionReferenceTiming.medianSeconds
       /productionInterleavedTiming.medianSeconds;
    const scalar productionDegree6SpeedupPacked =
        productionPackedTiming.medianSeconds/productionDegree6Timing.medianSeconds;
    const scalar productionDegree6SpeedupReference =
        productionReferenceTiming.medianSeconds/productionDegree6Timing.medianSeconds;
    const scalar productionDegree6InterleavedSpeedupPacked =
        productionPackedTiming.medianSeconds
       /productionDegree6InterleavedTiming.medianSeconds;
    const scalar productionDegree6InterleavedSpeedupReference =
        productionReferenceTiming.medianSeconds
       /productionDegree6InterleavedTiming.medianSeconds;
    std::cout << "\nProduction-like scalar ILP benchmark:"
        << "\n  calls/sample: " << productionCallsPerSample
        << "\n  sweeps/call: " << productionSweepsPerCall
        << "\n  randomized variant-order seed: " << productionBenchmarkSeed
        << "\n  OpenMP threads detected: " << persistentThreads << '\n';
    printTiming("3-sweep Reference GS", productionReferenceTiming);
    printTiming("3-sweep Packed scalar", productionPackedTiming);
    printTiming
    (
        "3-sweep Packed scalar interleaved rows",
        productionInterleavedTiming
    );
    printTiming("3-sweep Packed scalar degree-6", productionDegree6Timing);
    printTiming
    (
        "3-sweep Packed scalar degree-6 4-way interleaved",
        productionDegree6InterleavedTiming
    );
    std::cout << "\nInterleaved rows summary (median):"
        << "\nvariant          median(s)  ns/cell  CV(%)"
        << "\nreference        " << productionReferenceTiming.medianSeconds
        << "  " << productionReferenceTiming.medianNsPerCellSweep
        << "  " << productionReferenceTiming.cvPercent
        << "\npacked scalar    " << productionPackedTiming.medianSeconds
        << "  " << productionPackedTiming.medianNsPerCellSweep
        << "  " << productionPackedTiming.cvPercent
        << "\ninterleaved rows " << productionInterleavedTiming.medianSeconds
        << "  " << productionInterleavedTiming.medianNsPerCellSweep
        << "  " << productionInterleavedTiming.cvPercent
        << "\ndegree-6         " << productionDegree6Timing.medianSeconds
        << "  " << productionDegree6Timing.medianNsPerCellSweep
        << "  " << productionDegree6Timing.cvPercent
        << "\ndegree-6 ILP     "
        << productionDegree6InterleavedTiming.medianSeconds
        << "  " << productionDegree6InterleavedTiming.medianNsPerCellSweep
        << "  " << productionDegree6InterleavedTiming.cvPercent
        << "\n  generic interleaved speedup vs packed/reference: "
        << productionInterleavedSpeedupPacked << "x / "
        << productionInterleavedSpeedupReference << "x"
        << "\n  degree-6 speedup vs packed/reference: "
        << productionDegree6SpeedupPacked << "x / "
        << productionDegree6SpeedupReference << "x"
        << "\n  degree-6 ILP speedup vs packed/reference: "
        << productionDegree6InterleavedSpeedupPacked << "x / "
        << productionDegree6InterleavedSpeedupReference << "x"
        << "\n  speedup vs packed, reference: "
        << productionPackedTiming.medianSeconds
           /productionReferenceTiming.medianSeconds << "x"
        << "\n  speedup vs packed, packed scalar: 1x"
        << "\n  speedup vs packed, generic 4-way: "
        << productionInterleavedSpeedupPacked << "x"
        << "\n  speedup vs packed, degree-6 scalar: "
        << productionDegree6SpeedupPacked << "x"
        << "\n  speedup vs packed, degree-6 4-way: "
        << productionDegree6InterleavedSpeedupPacked << "x\n";

    std::cout << "\nPrefetch summary (median):"
        << "\nvariant          distance  neighbours  median(s)  ns/cell  "
           "speedup  CV(%)"
        << "\nno prefetch      -         0           "
        << serialGatherTiming.medianSeconds << "  "
        << serialGatherTiming.medianNsPerCellSweep << "  1  "
        << serialGatherTiming.cvPercent;
    for (std::size_t i=0; i<prefetchDistances.size(); ++i)
    {
        const ImplementationTiming& timing = prefetchFirstTimings[i];
        std::cout << "\nfirst neighbour  " << prefetchDistances[i]
            << "         1           " << timing.medianSeconds << "  "
            << timing.medianNsPerCellSweep << "  "
            << serialGatherTiming.medianSeconds/timing.medianSeconds << "  "
            << timing.cvPercent;
    }
    std::cout << "\nfirst 2          " << bestPrefetchDistance
        << "         2           " << prefetchTwoTiming.medianSeconds << "  "
        << prefetchTwoTiming.medianNsPerCellSweep << "  "
        << serialGatherTiming.medianSeconds/prefetchTwoTiming.medianSeconds
        << "  " << prefetchTwoTiming.cvPercent
        << "\n  best first-neighbour distance: " << bestPrefetchDistance << '\n';

    std::cout << "\nMedian-based comparisons:"
        << "\n  serial packed slowdown vs reference: " << gatherSlowdown << "x"
        << "\n  locality-reordered speedup vs scalar/reference: "
        << localitySpeedupScalar << "x / " << localitySpeedupReference << "x"
        << "\n  AVX2 speedup vs scalar/reference: "
        << avx2SpeedupScalar << "x / " << avx2SpeedupReference << "x"
        << "\n  AVX2 rows speedup vs scalar/reference/intra-row: "
        << avx2AcrossRowsSpeedupScalar << "x / "
        << avx2AcrossRowsSpeedupReference << "x / "
        << avx2AcrossRowsSpeedupIntra << "x"
        << "\n  AVX-512 speedup vs scalar/reference: "
        << avx512SpeedupScalar << "x / " << avx512SpeedupReference << "x"
        << "\n  AVX-512 rows speedup vs scalar/reference/intra-row: "
        << avx512AcrossRowsSpeedupScalar << "x / "
        << avx512AcrossRowsSpeedupReference << "x / "
        << avx512AcrossRowsSpeedupIntra << "x"
        << "\n  per-level OMP speedup vs scalar/reference: "
        << perLevelSpeedupGather << "x / " << perLevelSpeedupReference << "x"
        << "\n  persistent OMP speedup vs scalar/reference/per-level: "
        << persistentSpeedupGather << "x / "
        << persistentSpeedupReference << "x / " << persistentVsPerLevel << "x"
        << "\nRESULT variant=persistent threads=" << persistentThreads
        << " medianTime=" << persistentTiming.medianSeconds
        << " medianNsPerCell=" << persistentTiming.medianNsPerCellSweep
        << " speedupGather=" << persistentSpeedupGather
        << " speedupRef=" << persistentSpeedupReference << '\n'
        << "\ncorrectness sweeps: " << executedCorrectnessSweeps
        << "\nfinal residual: " << final
        << "\nmaximum error: " << maxError << '\n';
    if (final > 1e-10 || maxError > 1e-10)
        throw std::runtime_error("Gauss-Seidel did not converge");
    std::cout << "PASS\n";
    return 0;
}

int main(int argc, char** argv)
{
    const std::string usage =
        "Usage: Test-GaussSeidel <polyMesh-directory> "
        "[--history-sweeps N]\n";
    if (argc >= 2 && std::string(argv[1]) == "--help")
    {
        std::cout << usage;
        return 0;
    }
    if (argc != 2 && argc != 4)
    {
        std::cerr << usage;
        return 2;
    }
    try
    {
        label historySweeps = 20;
        if (argc == 4)
        {
            if (std::string(argv[2]) != "--history-sweeps")
            {
                std::cerr << usage;
                return 2;
            }
            std::size_t parsedCharacters = 0;
            historySweeps = std::stoi(argv[3], &parsedCharacters);
            if (parsedCharacters != std::string(argv[3]).size()
                || historySweeps < 1 || historySweeps > 10000)
            {
                throw std::runtime_error
                (
                    "history sweep count must be in the range [1, 10000]"
                );
            }
        }
        return runTest(argv[1], historySweeps);
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
