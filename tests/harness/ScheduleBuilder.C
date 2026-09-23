#include "ScheduleBuilder.H"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

namespace smootherTest::harness
{
namespace
{

std::size_t percentile
(
    const std::vector<std::size_t>& sorted,
    const Foam::scalar fraction
)
{
    const std::size_t index = static_cast<std::size_t>
    (
        std::ceil(fraction*sorted.size()) - 1
    );
    return sorted[std::min(index, sorted.size() - 1)];
}

void finishStatistics
(
    WavefrontStatistics& statistics,
    const std::size_t nCells
)
{
    if (statistics.widths.empty())
        throw std::runtime_error("wavefront schedule has no levels");
    std::vector<std::size_t> sorted = statistics.widths;
    std::sort(sorted.begin(), sorted.end());
    statistics.minimumWidth = sorted.front();
    statistics.maximumWidth = sorted.back();
    statistics.meanWidth = Foam::scalar(nCells)/sorted.size();
    const std::size_t middle = sorted.size()/2;
    statistics.medianWidth = sorted.size()%2
        ? Foam::scalar(sorted[middle])
        : 0.5*Foam::scalar(sorted[middle - 1] + sorted[middle]);
    statistics.p90Width = percentile(sorted, 0.90);
    statistics.p95Width = percentile(sorted, 0.95);
}

} // namespace

Foam::lduMatrix makeMatrix(const PolyMeshTopology& mesh)
{
    Foam::labelField upperAddr(mesh.neighbour.begin(), mesh.neighbour.end());
    const std::vector<int> starts = mesh.ownerStartAddressing();
    Foam::labelField ownerStart(starts.begin(), starts.end());
    Foam::scalarField upper(mesh.nInternalFaces());
    Foam::scalarField lower(mesh.nInternalFaces());
    Foam::scalarField diag(mesh.nCells, 0.0);
    for (std::size_t face=0; face<mesh.nInternalFaces(); ++face)
    {
        const Foam::label owner = mesh.owner[face];
        const Foam::label neighbour = mesh.neighbour[face];
        const Foam::scalar coefficient =
            1.0 + 0.001*((17*owner + 13*neighbour) % 101);
        upper[face] = -coefficient;
        lower[face] = -coefficient;
        diag[owner] += coefficient;
        diag[neighbour] += coefficient;
    }
    const Foam::scalar meanDiag =
        std::accumulate(diag.begin(), diag.end(), 0.0)/mesh.nCells;
    for (Foam::scalar& value : diag) value += 0.25*meanDiag;
    return Foam::lduMatrix
    (
        Foam::lduAddressing(std::move(upperAddr), std::move(ownerStart)),
        std::move(diag), std::move(upper), std::move(lower)
    );
}

Foam::scalarField multiply
(
    const Foam::lduMatrix& matrix,
    const Foam::scalarField& psi
)
{
    Foam::scalarField result(psi.size(), 0.0);
    const auto& starts = matrix.lduAddr().ownerStartAddr();
    const auto& neighbours = matrix.lduAddr().upperAddr();
    for (Foam::label cell=0; cell<Foam::label(psi.size()); ++cell)
    {
        result[cell] += matrix.diag()[cell]*psi[cell];
        for (Foam::label face=starts[cell]; face<starts[cell + 1]; ++face)
        {
            const Foam::label neighbour = neighbours[face];
            result[cell] += matrix.upper()[face]*psi[neighbour];
            result[neighbour] += matrix.lower()[face]*psi[cell];
        }
    }
    return result;
}

Foam::scalar relativeResidual
(
    const Foam::lduMatrix& matrix,
    const Foam::scalarField& psi,
    const Foam::scalarField& source
)
{
    const Foam::scalarField product = multiply(matrix, psi);
    Foam::scalar numerator = 0;
    Foam::scalar denominator = 0;
    for (Foam::label cell=0; cell<Foam::label(psi.size()); ++cell)
    {
        numerator += std::abs(source[cell] - product[cell]);
        denominator += std::abs(source[cell]);
    }
    return numerator/std::max(denominator, Foam::scalar(1e-300));
}

Foam::scalar maxAbsDifference
(
    const Foam::scalarField& left,
    const Foam::scalarField& right
)
{
    if (left.size() != right.size())
        throw std::runtime_error("field comparison size mismatch");
    Foam::scalar maximum = 0;
    for (Foam::label cell=0; cell<Foam::label(left.size()); ++cell)
    {
        if (!std::isfinite(left[cell]) || !std::isfinite(right[cell]))
            throw std::runtime_error("field comparison contains a non-finite value");
        maximum = std::max(maximum, std::abs(left[cell] - right[cell]));
    }
    return maximum;
}

bool validateDependencies
(
    const PolyMeshTopology& mesh,
    const WavefrontStatistics& statistics
)
{
    if (statistics.levels.size() != mesh.nCells) return false;
    std::size_t assigned = 0;
    for (const std::size_t width : statistics.widths) assigned += width;
    if (assigned != mesh.nCells) return false;
    for (std::size_t face=0; face<mesh.nInternalFaces(); ++face)
    {
        if
        (
            statistics.levels[mesh.owner[face]]
         >= statistics.levels[mesh.neighbour[face]]
        ) return false;
    }
    return true;
}

WavefrontStatistics constructMinimalWavefronts(const PolyMeshTopology& mesh)
{
    WavefrontStatistics result;
    result.levels.assign(mesh.nCells, 0);
    for (std::size_t face=0; face<mesh.nInternalFaces(); ++face)
    {
        const Foam::label owner = mesh.owner[face];
        const Foam::label neighbour = mesh.neighbour[face];
        result.levels[neighbour] = std::max
        (
            result.levels[neighbour], result.levels[owner] + 1
        );
    }
    const Foam::label maximum =
        *std::max_element(result.levels.begin(), result.levels.end());
    result.widths.assign(static_cast<std::size_t>(maximum) + 1, 0);
    for (const Foam::label level : result.levels) ++result.widths[level];
    finishStatistics(result, mesh.nCells);
    if (!validateDependencies(mesh, result))
        throw std::runtime_error("minimal wavefront dependency validation failed");
    return result;
}

WavefrontStatistics constructGeometryWavefronts
(
    const PolyMeshTopology& mesh,
    const Foam::label xSlabs,
    const Foam::label targetWidth
)
{
    if (mesh.cellCentreX.size() != mesh.nCells || xSlabs < 1 || targetWidth < 1)
        throw std::runtime_error("invalid Geometry-X scheduler input");
    const auto bounds = std::minmax_element
    (
        mesh.cellCentreX.begin(), mesh.cellCentreX.end()
    );
    const double minimum = *bounds.first;
    const double range = std::max(*bounds.second - minimum, 1e-300);
    const auto slabFor = [&](const Foam::label cell)
    {
        return std::min
        (
            xSlabs - 1,
            static_cast<Foam::label>
            (
                (mesh.cellCentreX[cell] - minimum)/range*xSlabs
            )
        );
    };
    std::vector<Foam::label> predecessors(mesh.nCells, 0);
    std::vector<std::vector<Foam::label>> successors(mesh.nCells);
    for (std::size_t face=0; face<mesh.nInternalFaces(); ++face)
    {
        ++predecessors[mesh.neighbour[face]];
        successors[mesh.owner[face]].push_back(mesh.neighbour[face]);
    }
    std::vector<std::vector<Foam::label>> ready(xSlabs);
    std::vector<std::size_t> cursor(xSlabs, 0);
    for (Foam::label cell=0; cell<Foam::label(mesh.nCells); ++cell)
        if (predecessors[cell] == 0) ready[slabFor(cell)].push_back(cell);

    WavefrontStatistics result;
    result.levels.assign(mesh.nCells, -1);
    std::size_t assigned = 0;
    while (assigned < mesh.nCells)
    {
        std::vector<Foam::label> current;
        current.reserve(targetWidth);
        for (Foam::label slab=0; slab<xSlabs && Foam::label(current.size())<targetWidth; ++slab)
        {
            const Foam::label count = std::min
            (
                targetWidth - Foam::label(current.size()),
                Foam::label(ready[slab].size() - cursor[slab])
            );
            current.insert
            (
                current.end(), ready[slab].begin() + cursor[slab],
                ready[slab].begin() + cursor[slab] + count
            );
            cursor[slab] += count;
        }
        if (current.empty())
            throw std::runtime_error("Geometry-X scheduler stalled");
        const Foam::label level = result.widths.size();
        result.widths.push_back(current.size());
        assigned += current.size();
        for (const Foam::label cell : current) result.levels[cell] = level;
        for (const Foam::label owner : current)
            for (const Foam::label neighbour : successors[owner])
                if (--predecessors[neighbour] == 0)
                    ready[slabFor(neighbour)].push_back(neighbour);
    }
    finishStatistics(result, mesh.nCells);
    if (!validateDependencies(mesh, result))
        throw std::runtime_error("Geometry-X dependency validation failed");
    return result;
}

WavefrontStatistics constructIndexWavefronts
(
    const PolyMeshTopology& mesh,
    const Foam::lduMatrix& matrix,
    const Foam::label targetWidth,
    const bool window
)
{
    if (targetWidth < 1) throw std::runtime_error("invalid Index-Kahn width");
    const auto& starts = matrix.lduAddr().ownerStartAddr();
    const auto& neighbours = matrix.lduAddr().upperAddr();
    std::vector<Foam::label> predecessors(mesh.nCells, 0);
    for (const Foam::label cell : neighbours) ++predecessors[cell];
    std::set<Foam::label> ready;
    for (Foam::label cell=0; cell<Foam::label(mesh.nCells); ++cell)
        if (predecessors[cell] == 0) ready.insert(cell);
    WavefrontStatistics result;
    result.levels.assign(mesh.nCells, -1);
    Foam::label cursor = 0;
    std::size_t assigned = 0;
    while (!ready.empty())
    {
        std::vector<Foam::label> current;
        current.reserve(targetWidth);
        auto position = window ? ready.lower_bound(cursor) : ready.begin();
        while (!ready.empty() && current.size()<std::size_t(targetWidth))
        {
            if (position == ready.end()) position = ready.begin();
            current.push_back(*position);
            position = ready.erase(position);
        }
        cursor = current.back() + 1;
        const Foam::label level = result.widths.size();
        result.widths.push_back(current.size());
        assigned += current.size();
        for (const Foam::label cell : current) result.levels[cell] = level;
        for (const Foam::label cell : current)
            for (Foam::label face=starts[cell]; face<starts[cell + 1]; ++face)
                if (--predecessors[neighbours[face]] == 0)
                    ready.insert(neighbours[face]);
    }
    if (assigned != mesh.nCells)
        throw std::runtime_error("Index-Kahn scheduler did not cover all cells");
    finishStatistics(result, mesh.nCells);
    if (!validateDependencies(mesh, result))
        throw std::runtime_error("Index-Kahn dependency validation failed");
    return result;
}

WavefrontSchedule makeWavefrontSchedule
(
    const PolyMeshTopology& mesh,
    const Foam::lduMatrix& matrix,
    const WavefrontStatistics& statistics
)
{
    WavefrontSchedule schedule;
    schedule.levelStarts.resize(statistics.widths.size() + 1, 0);
    for (std::size_t level=0; level<statistics.widths.size(); ++level)
        schedule.levelStarts[level + 1] =
            schedule.levelStarts[level] + statistics.widths[level];
    schedule.waveCells.resize(mesh.nCells);
    std::vector<Foam::label> levelCursor = schedule.levelStarts;
    for (Foam::label cell=0; cell<Foam::label(mesh.nCells); ++cell)
        schedule.waveCells[levelCursor[statistics.levels[cell]]++] = cell;
    std::vector<Foam::label> cellToRow(mesh.nCells);
    for (Foam::label row=0; row<Foam::label(mesh.nCells); ++row)
        cellToRow[schedule.waveCells[row]] = row;

    std::vector<Foam::label> incomingStarts(mesh.nCells + 1, 0);
    for (const int neighbour : mesh.neighbour) ++incomingStarts[neighbour + 1];
    for (std::size_t cell=0; cell<mesh.nCells; ++cell)
        incomingStarts[cell + 1] += incomingStarts[cell];
    std::vector<Foam::label> incomingFaces(mesh.nInternalFaces());
    std::vector<Foam::label> incomingCursor = incomingStarts;
    for (Foam::label face=0; face<Foam::label(mesh.nInternalFaces()); ++face)
        incomingFaces[incomingCursor[mesh.neighbour[face]]++] = face;

    const auto& ownerStarts = matrix.lduAddr().ownerStartAddr();
    const auto& neighbours = matrix.lduAddr().upperAddr();
    schedule.rowStarts.reserve(mesh.nCells + 1);
    schedule.cols.reserve(2*mesh.nInternalFaces());
    schedule.coeffs.reserve(2*mesh.nInternalFaces());
    schedule.diag.reserve(mesh.nCells);
    schedule.incomingCounts.reserve(mesh.nCells);
    schedule.rowStarts.push_back(0);
    for (const Foam::label cell : schedule.waveCells)
    {
        schedule.incomingCounts.push_back
        (
            incomingStarts[cell + 1] - incomingStarts[cell]
        );
        for (Foam::label in=incomingStarts[cell]; in<incomingStarts[cell + 1]; ++in)
        {
            const Foam::label face = incomingFaces[in];
            schedule.cols.push_back(mesh.owner[face]);
            schedule.coeffs.push_back(matrix.lower()[face]);
        }
        for (Foam::label face=ownerStarts[cell]; face<ownerStarts[cell + 1]; ++face)
        {
            schedule.cols.push_back(neighbours[face]);
            schedule.coeffs.push_back(matrix.upper()[face]);
        }
        schedule.diag.push_back(matrix.diag()[cell]);
        schedule.rowStarts.push_back(schedule.cols.size());
    }
    schedule.localCols.resize(schedule.cols.size());
    for (std::size_t p=0; p<schedule.cols.size(); ++p)
        schedule.localCols[p] = cellToRow[schedule.cols[p]];
    if
    (
        schedule.rowStarts.size() != mesh.nCells + 1
     || schedule.cols.size() != 2*mesh.nInternalFaces()
    ) throw std::runtime_error("invalid packed wavefront schedule");
    return schedule;
}

namespace
{

template<class RowOrder>
WavefrontSchedule reorderRows
(
    const WavefrontSchedule& original,
    RowOrder&& rowOrder,
    const bool reorderContributions
)
{
    WavefrontSchedule result;
    result.levelStarts = original.levelStarts;
    result.waveCells.reserve(original.waveCells.size());
    result.rowStarts.reserve(original.rowStarts.size());
    result.cols.reserve(original.cols.size());
    result.coeffs.reserve(original.coeffs.size());
    result.diag.reserve(original.diag.size());
    result.rowStarts.push_back(0);
    for (std::size_t level=0; level + 1<original.levelStarts.size(); ++level)
    {
        std::vector<Foam::label> rows;
        for (Foam::label row=original.levelStarts[level]; row<original.levelStarts[level + 1]; ++row)
            rows.push_back(row);
        rowOrder(rows);
        for (const Foam::label row : rows)
        {
            result.waveCells.push_back(original.waveCells[row]);
            result.diag.push_back(original.diag[row]);
            std::vector<std::pair<Foam::label, Foam::scalar>> contributions;
            for (Foam::label p=original.rowStarts[row]; p<original.rowStarts[row + 1]; ++p)
                contributions.emplace_back(original.cols[p], original.coeffs[p]);
            if (reorderContributions)
                std::stable_sort
                (
                    contributions.begin(), contributions.end(),
                    [](const auto& left, const auto& right)
                    {
                        return left.first < right.first;
                    }
                );
            for (const auto& contribution : contributions)
            {
                result.cols.push_back(contribution.first);
                result.coeffs.push_back(contribution.second);
            }
            result.rowStarts.push_back(result.cols.size());
        }
    }
    return result;
}

} // namespace

WavefrontSchedule makeXSortedSchedule
(
    const WavefrontSchedule& original,
    const std::vector<double>& cellCentreX
)
{
    return reorderRows
    (
        original,
        [&](std::vector<Foam::label>& rows)
        {
            std::stable_sort
            (
                rows.begin(), rows.end(),
                [&](const Foam::label left, const Foam::label right)
                {
                    const Foam::label leftCell = original.waveCells[left];
                    const Foam::label rightCell = original.waveCells[right];
                    return cellCentreX[leftCell] == cellCentreX[rightCell]
                        ? leftCell < rightCell
                        : cellCentreX[leftCell] < cellCentreX[rightCell];
                }
            );
        },
        false
    );
}

WavefrontSchedule makeMedianRowSchedule(const WavefrontSchedule& original)
{
    return reorderRows
    (
        original,
        [&](std::vector<Foam::label>& rows)
        {
            const auto key = [&](const Foam::label row)
            {
                std::vector<Foam::label> columns
                (
                    original.cols.begin() + original.rowStarts[row],
                    original.cols.begin() + original.rowStarts[row + 1]
                );
                if (columns.empty()) return std::int64_t(2)*original.waveCells[row];
                std::sort(columns.begin(), columns.end());
                const std::size_t middle = columns.size()/2;
                return columns.size()%2
                    ? std::int64_t(2)*columns[middle]
                    : std::int64_t(columns[middle - 1]) + columns[middle];
            };
            std::stable_sort
            (
                rows.begin(), rows.end(),
                [&](const Foam::label left, const Foam::label right)
                {
                    return key(left) == key(right)
                        ? original.waveCells[left] < original.waveCells[right]
                        : key(left) < key(right);
                }
            );
        },
        false
    );
}

WavefrontSchedule makeLocalitySchedule(const WavefrontSchedule& original)
{
    return reorderRows
    (
        original,
        [&](std::vector<Foam::label>& rows)
        {
            const auto key = [&](const Foam::label row)
            {
                long long sum = 0;
                for (Foam::label p=original.rowStarts[row]; p<original.rowStarts[row + 1]; ++p)
                    sum += original.cols[p];
                const Foam::label degree =
                    original.rowStarts[row + 1] - original.rowStarts[row];
                return degree
                    ? sum/degree
                    : static_cast<long long>(original.waveCells[row]);
            };
            std::stable_sort
            (
                rows.begin(), rows.end(),
                [&](const Foam::label left, const Foam::label right)
                {
                    return key(left) == key(right)
                        ? original.waveCells[left] < original.waveCells[right]
                        : key(left) < key(right);
                }
            );
        },
        true
    );
}

WholeRowInt16Schedule makeWholeRowInt16Schedule(const WavefrontSchedule& original)
{
    WholeRowInt16Schedule result;
    result.levelStarts = original.levelStarts;
    result.waveCells = original.waveCells;
    result.rowStarts = original.rowStarts;
    result.coeffs = original.coeffs;
    result.diag = original.diag;
    result.compactRows.resize(original.waveCells.size(), 0);
    result.compactStarts.push_back(0);
    result.fallbackStarts.push_back(0);
    for (Foam::label row=0; row<Foam::label(original.waveCells.size()); ++row)
    {
        const Foam::label cell = original.waveCells[row];
        bool fits = true;
        for (Foam::label p=original.rowStarts[row]; p<original.rowStarts[row + 1]; ++p)
        {
            const std::int64_t offset =
                std::int64_t(original.cols[p]) - std::int64_t(cell);
            fits = fits
                && offset >= std::numeric_limits<std::int16_t>::min()
                && offset <= std::numeric_limits<std::int16_t>::max();
        }
        result.compactRows[row] = fits;
        for (Foam::label p=original.rowStarts[row]; p<original.rowStarts[row + 1]; ++p)
        {
            if (fits)
                result.offsets.push_back
                (
                    static_cast<std::int16_t>(original.cols[p] - cell)
                );
            else result.fallbackCols.push_back(original.cols[p]);
        }
        result.compactStarts.push_back(result.offsets.size());
        result.fallbackStarts.push_back(result.fallbackCols.size());
    }
    return result;
}

DeltaEscapeInt16Schedule makeDeltaEscapeInt16Schedule
(
    const WavefrontSchedule& original
)
{
    DeltaEscapeInt16Schedule result;
    result.levelStarts = original.levelStarts;
    result.waveCells = original.waveCells;
    result.rowStarts = original.rowStarts;
    result.coeffs = original.coeffs;
    result.diag = original.diag;
    result.firstCols.resize(original.waveCells.size(), 0);
    result.deltaStarts.push_back(0);
    result.escapeStarts.push_back(0);
    constexpr std::int64_t escape = std::numeric_limits<std::int16_t>::min();
    for (Foam::label row=0; row<Foam::label(original.waveCells.size()); ++row)
    {
        const Foam::label begin = original.rowStarts[row];
        const Foam::label end = original.rowStarts[row + 1];
        if (begin < end)
        {
            Foam::label previous = original.cols[begin];
            result.firstCols[row] = previous;
            for (Foam::label p=begin + 1; p<end; ++p)
            {
                const Foam::label column = original.cols[p];
                const std::int64_t delta =
                    std::int64_t(column) - std::int64_t(previous);
                if
                (
                    delta > escape
                 && delta <= std::numeric_limits<std::int16_t>::max()
                ) result.deltas.push_back(static_cast<std::int16_t>(delta));
                else
                {
                    result.deltas.push_back(static_cast<std::int16_t>(escape));
                    result.escapeCols.push_back(column);
                }
                previous = column;
            }
        }
        result.deltaStarts.push_back(result.deltas.size());
        result.escapeStarts.push_back(result.escapeCols.size());
    }
    return result;
}

PackedUint24Schedule makePackedUint24Schedule(const WavefrontSchedule& original)
{
    PackedUint24Schedule result;
    result.levelStarts = original.levelStarts;
    result.waveCells = original.waveCells;
    result.rowStarts = original.rowStarts;
    result.coeffs = original.coeffs;
    result.diag = original.diag;
    result.cols.reserve(3*original.cols.size());
    for (const Foam::label column : original.cols)
    {
        if (column < 0 || std::uint64_t(column) > 0xFFFFFFu)
            throw std::runtime_error("packed column does not fit uint24");
        const std::uint32_t value = column;
        result.cols.push_back(static_cast<std::uint8_t>(value));
        result.cols.push_back(static_cast<std::uint8_t>(value >> 8));
        result.cols.push_back(static_cast<std::uint8_t>(value >> 16));
    }
    return result;
}

RowDegreeSchedule makeRowDegreeSchedule(const WavefrontSchedule& original)
{
    RowDegreeSchedule result;
    result.packed = &original;
    std::size_t contributions = 0;
    for (Foam::label row=0; row<Foam::label(original.waveCells.size()); ++row)
    {
        const Foam::label degree =
            original.rowStarts[row + 1] - original.rowStarts[row];
        if (degree < 0 || degree > std::numeric_limits<std::uint8_t>::max())
            throw std::runtime_error("row degree does not fit uint8_t");
        result.degrees.push_back(static_cast<std::uint8_t>(degree));
        contributions += degree;
    }
    if (contributions != original.cols.size())
        throw std::runtime_error("row degrees do not cover contributions");
    return result;
}

BlockedRowDegreeSchedule makeBlockedRowDegreeSchedule
(
    const WavefrontSchedule& original,
    const Foam::label blockSize
)
{
    if (blockSize < 1) throw std::runtime_error("invalid row-degree block size");
    BlockedRowDegreeSchedule result;
    result.packed = &original;
    result.degrees = makeRowDegreeSchedule(original).degrees;
    result.levelBlockStarts.push_back(0);
    for (std::size_t level=0; level + 1<original.levelStarts.size(); ++level)
    {
        const Foam::label end = original.levelStarts[level + 1];
        for (Foam::label row=original.levelStarts[level]; row<end; row += blockSize)
        {
            result.blockRowStarts.push_back(row);
            result.blockRowEnds.push_back(std::min(row + blockSize, end));
            result.blockPStarts.push_back(original.rowStarts[row]);
        }
        result.levelBlockStarts.push_back(result.blockRowStarts.size());
    }
    return result;
}

HybridWavefrontSchedule makeHybridSchedule
(
    const PolyMeshTopology& mesh,
    const Foam::lduMatrix& matrix,
    const WavefrontSchedule& packed
)
{
    HybridWavefrontSchedule result;
    result.packed = &packed;
    result.lowerCoeffs = &matrix.lower();
    result.nInternalFaces = mesh.nInternalFaces();
    std::vector<Foam::label> incomingStarts(mesh.nCells + 1, 0);
    for (const int neighbour : mesh.neighbour) ++incomingStarts[neighbour + 1];
    for (std::size_t cell=0; cell<mesh.nCells; ++cell)
        incomingStarts[cell + 1] += incomingStarts[cell];
    std::vector<Foam::label> incomingFaces(mesh.nInternalFaces());
    std::vector<Foam::label> cursor = incomingStarts;
    for (Foam::label face=0; face<Foam::label(mesh.nInternalFaces()); ++face)
        incomingFaces[cursor[mesh.neighbour[face]]++] = face;
    const auto& ownerStarts = matrix.lduAddr().ownerStartAddr();
    result.incomingStarts.push_back(0);
    for (const Foam::label cell : packed.waveCells)
    {
        for (Foam::label p=incomingStarts[cell]; p<incomingStarts[cell + 1]; ++p)
            result.incomingFaces.push_back(incomingFaces[p]);
        result.incomingStarts.push_back(result.incomingFaces.size());
        result.outgoingFaceStarts.push_back(ownerStarts[cell]);
        result.outgoingFaceEnds.push_back(ownerStarts[cell + 1]);
    }
    if
    (
        result.incomingFaces.size() != mesh.nInternalFaces()
     || result.incomingStarts.size() != mesh.nCells + 1
    ) throw std::runtime_error("invalid hybrid schedule");
    return result;
}

DirectCoefficientSchedule makeDirectCoefficientSchedule
(
    const PolyMeshTopology& mesh,
    const Foam::lduMatrix& matrix,
    const WavefrontStatistics& statistics,
    const Foam::label blockSize
)
{
    if (blockSize < 1) throw std::runtime_error("invalid direct block size");
    DirectCoefficientSchedule result;
    result.levelStarts.resize(statistics.widths.size() + 1, 0);
    for (std::size_t level=0; level<statistics.widths.size(); ++level)
        result.levelStarts[level + 1] =
            result.levelStarts[level] + statistics.widths[level];
    result.waveCells.resize(mesh.nCells);
    std::vector<Foam::label> levelCursor = result.levelStarts;
    for (Foam::label cell=0; cell<Foam::label(mesh.nCells); ++cell)
        result.waveCells[levelCursor[statistics.levels[cell]]++] = cell;

    std::vector<Foam::label> incomingStarts(mesh.nCells + 1, 0);
    for (const int neighbour : mesh.neighbour) ++incomingStarts[neighbour + 1];
    for (std::size_t cell=0; cell<mesh.nCells; ++cell)
        incomingStarts[cell + 1] += incomingStarts[cell];
    std::vector<Foam::label> incomingFaces(mesh.nInternalFaces());
    std::vector<Foam::label> cursor = incomingStarts;
    for (Foam::label face=0; face<Foam::label(mesh.nInternalFaces()); ++face)
        incomingFaces[cursor[mesh.neighbour[face]]++] = face;
    const auto& ownerStarts = matrix.lduAddr().ownerStartAddr();
    const auto& neighbours = matrix.lduAddr().upperAddr();
    std::vector<Foam::label> rowPStarts;
    rowPStarts.reserve(mesh.nCells + 1);
    for (const Foam::label cell : result.waveCells)
    {
        const Foam::label begin = result.cols.size();
        rowPStarts.push_back(begin);
        const Foam::label incoming =
            incomingStarts[cell + 1] - incomingStarts[cell];
        for (Foam::label p=incomingStarts[cell]; p<incomingStarts[cell + 1]; ++p)
        {
            const Foam::label face = incomingFaces[p];
            result.cols.push_back(mesh.owner[face]);
            result.faceIds.push_back(face);
        }
        for (Foam::label face=ownerStarts[cell]; face<ownerStarts[cell + 1]; ++face)
        {
            result.cols.push_back(neighbours[face]);
            result.faceIds.push_back(face);
        }
        const Foam::label degree = result.cols.size() - begin;
        if (degree > std::numeric_limits<std::uint8_t>::max())
            throw std::runtime_error("direct row degree does not fit uint8_t");
        result.degrees.push_back(static_cast<std::uint8_t>(degree));
        result.incomingDegrees.push_back(static_cast<std::uint8_t>(incoming));
    }
    rowPStarts.push_back(result.cols.size());
    result.levelBlockStarts.push_back(0);
    for (std::size_t level=0; level + 1<result.levelStarts.size(); ++level)
    {
        const Foam::label end = result.levelStarts[level + 1];
        for (Foam::label row=result.levelStarts[level]; row<end; row += blockSize)
        {
            result.blockRowStarts.push_back(row);
            result.blockRowEnds.push_back(std::min(row + blockSize, end));
            result.blockPStarts.push_back(rowPStarts[row]);
        }
        result.levelBlockStarts.push_back(result.blockRowStarts.size());
    }
    if
    (
        result.cols.size() != 2*mesh.nInternalFaces()
     || result.faceIds.size() != result.cols.size()
     || result.degrees.size() != mesh.nCells
    ) throw std::runtime_error("invalid direct coefficient schedule");
    return result;
}

} // namespace smootherTest::harness
