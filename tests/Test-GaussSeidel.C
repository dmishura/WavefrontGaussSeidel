#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include "GaussSeidelSmoother.H"
#include "PolyMeshReader.H"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <vector>

using namespace Foam;
using smootherTest::PolyMeshReader;
using smootherTest::PolyMeshTopology;

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

struct WavefrontSchedule
{
    WavefrontStatistics statistics;
    std::vector<std::vector<label>> cellsByLevel;
    std::vector<label> incomingStarts;
    std::vector<label> incomingFaces;
    std::vector<label> faceOwners;
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

WavefrontSchedule makeWavefrontSchedule(const PolyMeshTopology& mesh)
{
    WavefrontSchedule schedule;
    schedule.statistics = constructWavefronts(mesh);
    schedule.cellsByLevel.resize(schedule.statistics.widths.size());
    for (label cell=0; cell<label(mesh.nCells); ++cell)
        schedule.cellsByLevel[schedule.statistics.levels[cell]].push_back(cell);

    // Incoming-face CSR enables a race-free gather of contributions from
    // already updated lower-index cells. Filling in global face order preserves
    // the subtraction order used by the reference scatter implementation.
    schedule.incomingStarts.assign(mesh.nCells + 1, 0);
    for (const int neighbour : mesh.neighbour)
        ++schedule.incomingStarts[static_cast<std::size_t>(neighbour) + 1];
    for (std::size_t cell=0; cell<mesh.nCells; ++cell)
        schedule.incomingStarts[cell + 1] += schedule.incomingStarts[cell];

    schedule.incomingFaces.resize(mesh.nInternalFaces());
    schedule.faceOwners.assign
    (
        mesh.owner.begin(), mesh.owner.begin() + mesh.nInternalFaces()
    );
    std::vector<label> cursor = schedule.incomingStarts;
    for (label face=0; face<label(mesh.nInternalFaces()); ++face)
    {
        const label neighbour = mesh.neighbour[face];
        schedule.incomingFaces[cursor[neighbour]++] = face;
    }
    return schedule;
}

void wavefrontSmooth
(
    scalarField& psi,
    const lduMatrix& matrix,
    const scalarField& source,
    const WavefrontSchedule& schedule,
    const label nSweeps,
    const bool parallel
)
{
    const auto& ownerStarts = matrix.lduAddr().ownerStartAddr();
    const auto& neighbours = matrix.lduAddr().upperAddr();
    const scalarField& diag = matrix.diag();
    const scalarField& upper = matrix.upper();
    const scalarField& lower = matrix.lower();

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (const std::vector<label>& levelCells : schedule.cellsByLevel)
        {
            // Access classification inside the loop:
            // read-only: matrix arrays, source, addressing, schedule;
            // private: cell, psii, loop indices;
            // write: psi[cell] only;
            // neighbour access: psi[owner/neighbour] is read-only.
            #pragma omp parallel for if(parallel) schedule(static)
            for (std::size_t index=0; index<levelCells.size(); ++index)
            {
                const label cell = levelCells[index];
                scalar psii = source[cell];

                for
                (
                    label in=schedule.incomingStarts[cell];
                    in<schedule.incomingStarts[cell + 1];
                    ++in
                )
                {
                    const label face = schedule.incomingFaces[in];
                    // owner is the dependency for this incoming face.
                    const label owner = schedule.faceOwners[face];
                    psii -= lower[face]*psi[owner];
                }
                for
                (
                    label face=ownerStarts[cell];
                    face<ownerStarts[cell + 1];
                    ++face
                )
                {
                    psii -= upper[face]*psi[neighbours[face]];
                }
                psi[cell] = psii/diag[cell];
            }
        }
    }
}

struct ImplementationTiming
{
    scalar averageSeconds = 0;
    scalar nsPerCellSweep = 0;
};

template<class SweepFunction>
ImplementationTiming timeSweepImplementation
(
    const scalarField& initialPsi,
    const std::size_t nCells,
    const label repetitions,
    SweepFunction&& executeOneSweep
)
{
    scalar totalSeconds = 0;
    scalarField psi(initialPsi.size());
    for (label repetition=0; repetition<repetitions; ++repetition)
    {
        // Restoring input state is deliberately outside the timed interval.
        psi = initialPsi;
        const auto begin = std::chrono::steady_clock::now();
        executeOneSweep(psi);
        const auto end = std::chrono::steady_clock::now();
        totalSeconds += std::chrono::duration<scalar>(end - begin).count();
    }
    ImplementationTiming result;
    result.averageSeconds = totalSeconds/repetitions;
    result.nsPerCellSweep = 1e9*result.averageSeconds/nCells;
    return result;
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

int openMpThreadCount()
{
    int count = 1;
    #pragma omp parallel
    {
        #pragma omp single
        count = omp_get_num_threads();
    }
    return count;
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

    const WavefrontSchedule schedule = makeWavefrontSchedule(mesh);
    const scalarField initialPsi(exact.size(), 0.0);

    scalarField psiReference = initialPsi;
    GaussSeidelSmoother::smooth
    (
        "psi", psiReference, matrix, source,
        interfaceCoeffs, interfaces, 0, 1
    );
    scalarField psiWavefront = initialPsi;
    wavefrontSmooth(psiWavefront, matrix, source, schedule, 1, true);

    const scalar equivalenceMaxAbs =
        maxAbsDifference(psiReference, psiWavefront);
    const auto [equivalenceL2, equivalenceRelativeL2] =
        l2Differences(psiReference, psiWavefront);
    const scalar sequentialOneSweepResidual =
        relativeResidual(matrix, psiReference, source);
    const scalar wavefrontOneSweepResidual =
        relativeResidual(matrix, psiWavefront, source);
    const scalar oneSweepResidualDifference =
        std::abs(sequentialOneSweepResidual - wavefrontOneSweepResidual);

    constexpr scalar equivalenceTolerance = 1e-12;
    std::cout << "motorBike cells: " << mesh.nCells
        << "\ninternal faces: " << mesh.nInternalFaces()
        << "\n\nOne-sweep equivalence:"
        << "\nmax abs difference: " << equivalenceMaxAbs
        << "\nL2 difference: " << equivalenceL2
        << "\nrelative L2 difference: " << equivalenceRelativeL2
        << "\nSequential one-sweep residual: " << sequentialOneSweepResidual
        << "\nWavefront  one-sweep residual: " << wavefrontOneSweepResidual
        << "\nResidual difference: " << oneSweepResidualDifference
        << "\nequivalence tolerance: " << equivalenceTolerance << "\n\n";
    if (equivalenceMaxAbs > equivalenceTolerance)
        throw std::runtime_error("one-sweep wavefront equivalence check failed");

    constexpr label timingRepetitions = 20;
    const ImplementationTiming sequentialTiming = timeSweepImplementation
    (
        initialPsi, mesh.nCells, timingRepetitions,
        [&](scalarField& timedPsi)
        {
            GaussSeidelSmoother::smooth
            (
                "psi", timedPsi, matrix, source,
                interfaceCoeffs, interfaces, 0, 1
            );
        }
    );
    const ImplementationTiming wavefrontTiming = timeSweepImplementation
    (
        initialPsi, mesh.nCells, timingRepetitions,
        [&](scalarField& timedPsi)
        {
            wavefrontSmooth(timedPsi, matrix, source, schedule, 1, true);
        }
    );
    const int openMpThreads = openMpThreadCount();

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

    const WavefrontStatistics& wavefronts = schedule.statistics;
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

    std::cout << "\nSweep performance:\n"
        << "history/timed sweeps: " << historySweeps
        << "\ntotal sweep time: " << totalSeconds << " s"
        << "\naverage sweep time: " << averageSeconds << " s"
        << "\nns/cell/sweep: " << nsPerCellSweep
        << "\nns/internal-face/sweep: " << nsPerFaceSweep
        << "\nsmoothing efficiency eta: " << eta << " 1/s\n";

    std::cout << "\nWavefront dependency statistics:\n"
        << "levels: " << wavefronts.widths.size()
        << "\ncells: " << totalLevelCells
        << "\nmin width: " << wavefronts.minimumWidth
        << "\nmean width: " << wavefronts.meanWidth
        << "\nmedian width: " << wavefronts.medianWidth
        << "\np90 width: " << wavefronts.p90Width
        << "\np95 width: " << wavefronts.p95Width
        << "\nmax width: " << wavefronts.maximumWidth
        << "\naverage available parallelism: " << wavefronts.meanWidth
        << "\nfraction in levels with width >= 2: "
        << fractionInLevelsAtLeast(wavefronts.widths, 2, mesh.nCells)
        << "\nfraction in levels with width >= 4: "
        << fractionInLevelsAtLeast(wavefronts.widths, 4, mesh.nCells)
        << "\nfraction in levels with width >= 8: "
        << fractionInLevelsAtLeast(wavefronts.widths, 8, mesh.nCells)
        << "\nfraction in levels with width >= 16: "
        << fractionInLevelsAtLeast(wavefronts.widths, 16, mesh.nCells) << '\n';
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
    std::cout << "dependency validation: PASS\n"
        << "\nIsolated implementation timing (" << timingRepetitions
        << " repetitions):"
        << "\nSequential GS:"
        << "\n  average sweep time: " << sequentialTiming.averageSeconds << " s"
        << "\n  ns/cell/sweep: " << sequentialTiming.nsPerCellSweep
        << "\nWavefront GS:"
        << "\n  threads: " << openMpThreads
        << "\n  average sweep time: " << wavefrontTiming.averageSeconds << " s"
        << "\n  ns/cell/sweep: " << wavefrontTiming.nsPerCellSweep
        << "\n  speedup vs sequential: "
        << sequentialTiming.averageSeconds/wavefrontTiming.averageSeconds << '\n'
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
