#include "PolyMeshReader.H"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using label = int;

struct Similarity
{
    double jaccard = 0;
    double overlapMin = 0;
    label newLines = 0;
    label indexGap = 0;
};

struct WindowStats
{
    double mean = 0;
    double median = 0;
    double minimum = 0;
    double percentile = 0;
    label distanceToMinimum = 0;
};

struct CutRow
{
    std::size_t chunk = 0;
    std::size_t readySize = 0;
    std::size_t chosenWidth = 0;
    double jaccard = 0;
    double overlapMin = 0;
    label newLines = 0;
    label indexGap = 0;
    double strength = 0;
    WindowStats radius64, radius256, radius1024;
    bool strongHit = false;
};

template<class T>
T readLittle(std::istream& input)
{
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw std::runtime_error("truncated permutation file");
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    if constexpr (sizeof(T) == 4) value = __builtin_bswap32(value);
    if constexpr (sizeof(T) == 8) value = __builtin_bswap64(value);
#endif
    return value;
}

std::vector<label> readNewToOld(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    char magic[8];
    input.read(magic, sizeof(magic));
    if (!input || std::string(magic, 8) != std::string("IH1PERM", 8))
        throw std::runtime_error("invalid permutation magic");
    const auto version = readLittle<std::uint32_t>(input);
    const auto labelBytes = readLittle<std::uint32_t>(input);
    const auto nCells = readLittle<std::uint64_t>(input);
    (void)readLittle<std::uint32_t>(input); // level
    const auto nameBytes = readLittle<std::uint32_t>(input);
    std::uint64_t levelStarts = 0;
    if (version >= 2) levelStarts = readLittle<std::uint64_t>(input);
    if ((version != 1 && version != 2) || labelBytes != 4)
        throw std::runtime_error("unsupported permutation format");
    input.seekg(static_cast<std::streamoff>(nameBytes + 4*levelStarts), std::ios::cur);
    std::vector<label> result(static_cast<std::size_t>(nCells));
    input.read
    (
        reinterpret_cast<char*>(result.data()),
        static_cast<std::streamsize>(result.size()*sizeof(label))
    );
    if (!input) throw std::runtime_error("truncated newToOld payload");
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    for (auto& value : result) value = __builtin_bswap32(value);
#endif
    return result;
}

struct LineCSR
{
    std::vector<std::size_t> starts;
    std::vector<label> lines;
};

LineCSR buildLineCSR(const smootherTest::PolyMeshTopology& mesh)
{
    const std::size_t nCells = mesh.nCells;
    const std::size_t nFaces = mesh.neighbour.size();
    std::vector<std::size_t> starts(nCells + 1, 0);
    for (std::size_t face=0; face<nFaces; ++face)
    {
        ++starts[static_cast<std::size_t>(mesh.owner[face]) + 1];
        ++starts[static_cast<std::size_t>(mesh.neighbour[face]) + 1];
    }
    std::partial_sum(starts.begin(), starts.end(), starts.begin());
    std::vector<label> raw(starts.back());
    auto cursor = starts;
    for (std::size_t face=0; face<nFaces; ++face)
    {
        const label owner = mesh.owner[face];
        const label neighbour = mesh.neighbour[face];
        raw[cursor[static_cast<std::size_t>(owner)]++] = neighbour/8;
        raw[cursor[static_cast<std::size_t>(neighbour)]++] = owner/8;
    }
    std::vector<std::size_t> uniqueStarts(nCells + 1, 0);
    for (std::size_t cell=0; cell<nCells; ++cell)
    {
        auto begin = raw.begin() + static_cast<std::ptrdiff_t>(starts[cell]);
        auto end = raw.begin() + static_cast<std::ptrdiff_t>(starts[cell + 1]);
        std::sort(begin, end);
        uniqueStarts[cell + 1] = static_cast<std::size_t>(std::unique(begin, end) - begin);
    }
    std::partial_sum(uniqueStarts.begin(), uniqueStarts.end(), uniqueStarts.begin());
    std::vector<label> lines(uniqueStarts.back());
    for (std::size_t cell=0; cell<nCells; ++cell)
        std::copy_n
        (
            raw.begin() + static_cast<std::ptrdiff_t>(starts[cell]),
            uniqueStarts[cell + 1] - uniqueStarts[cell],
            lines.begin() + static_cast<std::ptrdiff_t>(uniqueStarts[cell])
        );
    return {std::move(uniqueStarts), std::move(lines)};
}

Similarity similarity(const LineCSR& csr, const label a, const label b)
{
    std::size_t ai = csr.starts[static_cast<std::size_t>(a)];
    std::size_t bi = csr.starts[static_cast<std::size_t>(b)];
    const std::size_t ae = csr.starts[static_cast<std::size_t>(a) + 1];
    const std::size_t be = csr.starts[static_cast<std::size_t>(b) + 1];
    std::size_t intersection = 0;
    while (ai < ae && bi < be)
    {
        if (csr.lines[ai] < csr.lines[bi]) ++ai;
        else if (csr.lines[bi] < csr.lines[ai]) ++bi;
        else { ++intersection; ++ai; ++bi; }
    }
    const std::size_t an = ae - csr.starts[static_cast<std::size_t>(a)];
    const std::size_t bn = be - csr.starts[static_cast<std::size_t>(b)];
    const std::size_t unionSize = an + bn - intersection;
    Similarity result;
    result.jaccard = unionSize ? double(intersection)/unionSize : 1.0;
    result.overlapMin = std::min(an, bn)
        ? double(intersection)/std::min(an, bn) : 1.0;
    result.newLines = static_cast<label>(bn - intersection);
    result.indexGap = b - a;
    return result;
}

double quantile(std::vector<double> values, const double fraction)
{
    if (values.empty()) return 0;
    const std::size_t index = static_cast<std::size_t>
    (
        fraction*double(values.size() - 1)
    );
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}

double quantileInt(std::vector<label> values, const double fraction)
{
    if (values.empty()) return 0;
    const std::size_t index = static_cast<std::size_t>
    (
        fraction*double(values.size() - 1)
    );
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}

WindowStats windowStats
(
    const std::vector<Similarity>& values,
    const std::size_t cut,
    const std::size_t radius
)
{
    const std::size_t begin = cut > radius ? cut - radius : 0;
    const std::size_t end = std::min(values.size(), cut + radius + 1);
    std::vector<double> local;
    local.reserve(end - begin);
    double total = 0;
    double minimum = std::numeric_limits<double>::infinity();
    for (std::size_t i=begin; i<end; ++i)
    {
        local.push_back(values[i].jaccard);
        total += values[i].jaccard;
        minimum = std::min(minimum, values[i].jaccard);
    }
    const double cutValue = values[cut].jaccard;
    std::size_t below = 0, equal = 0;
    label distance = std::numeric_limits<label>::max();
    for (std::size_t i=begin; i<end; ++i)
    {
        below += values[i].jaccard < cutValue;
        equal += values[i].jaccard == cutValue;
        if (values[i].jaccard == minimum)
            distance = std::min
            (
                distance,
                static_cast<label>(i > cut ? i - cut : cut - i)
            );
    }
    WindowStats result;
    result.mean = total/local.size();
    result.median = quantile(local, 0.5);
    result.minimum = minimum;
    result.percentile = (double(below) + 0.5*double(equal))/local.size();
    result.distanceToMinimum = distance;
    return result;
}

struct Correlation
{
    long double n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    void add(const double x, const double y)
    {
        ++n; sx += x; sy += y; sxx += x*x; syy += y*y; sxy += x*y;
    }
    double value() const
    {
        const long double numerator = n*sxy - sx*sy;
        const long double denominator = std::sqrt
        (
            (n*sxx - sx*sx)*(n*syy - sy*sy)
        );
        return denominator ? static_cast<double>(numerator/denominator) : 0;
    }
};

void writeWindow(std::ostream& output, const WindowStats& value)
{
    output << ',' << value.mean << ',' << value.median << ',' << value.minimum
        << ',' << value.percentile << ',' << value.distanceToMinimum;
}

void writeCutRows(const std::filesystem::path& path, const std::vector<CutRow>& rows)
{
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot create " + path.string());
    output << "chunkId,readySize,chosenWidth,cutJaccard,cutOverlapMin,"
        "cutNewLines,cellIndexGap,cutBoundaryStrength,"
        "local64Mean,local64Median,local64Min,cutPercentileLocal64,distanceToLocalMin64,"
        "local256Mean,local256Median,local256Min,cutPercentileLocal256,distanceToLocalMin256,"
        "local1024Mean,local1024Median,local1024Min,cutPercentileLocal1024,distanceToLocalMin1024,"
        "strongBoundaryHit\n";
    output << std::setprecision(17);
    for (const auto& row : rows)
    {
        output << row.chunk << ',' << row.readySize << ',' << row.chosenWidth
            << ',' << row.jaccard << ',' << row.overlapMin << ','
            << row.newLines << ',' << row.indexGap << ',' << row.strength;
        writeWindow(output, row.radius64);
        writeWindow(output, row.radius256);
        writeWindow(output, row.radius1024);
        output << ',' << row.strongHit << '\n';
    }
}

void writeSummary
(
    const std::filesystem::path& path,
    const std::string& mesh,
    const label width,
    const double runtime,
    const std::size_t totalChunks,
    const std::vector<CutRow>& rows,
    const std::vector<label>& strongSpacing,
    const Correlation& correlation,
    const bool exact
)
{
    std::vector<double> cut, percentile;
    std::vector<label> distance;
    cut.reserve(rows.size()); percentile.reserve(rows.size()); distance.reserve(rows.size());
    std::size_t bottom10 = 0, bottom25 = 0, strong = 0;
    for (const auto& row : rows)
    {
        cut.push_back(row.jaccard);
        percentile.push_back(row.radius1024.percentile);
        distance.push_back(row.radius256.distanceToMinimum);
        bottom10 += row.radius1024.percentile <= 0.10;
        bottom25 += row.radius1024.percentile <= 0.25;
        strong += row.strongHit;
    }
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot create " + path.string());
    output << "mesh,width,runtimeMedianMs,totalChunks,validCuts,"
        "medianCutJaccard,p90CutJaccard,medianCutPercentileLocal1024,"
        "cutsBottom10Fraction,cutsBottom25Fraction,medianDistanceToLocalMin256,"
        "strongBoundaryHitFraction,strongBoundarySpacingP50,strongBoundarySpacingP90,"
        "strongBoundarySpacingP99,indexGapBoundaryStrengthCorrelation,"
        "randomExpectedPercentile,replayExact\n";
    output << std::setprecision(17) << mesh << ',' << width << ',' << runtime
        << ',' << totalChunks << ',' << rows.size()
        << ',' << quantile(cut, 0.5) << ',' << quantile(cut, 0.9)
        << ',' << quantile(percentile, 0.5)
        << ',' << (rows.empty() ? 0.0 : double(bottom10)/rows.size())
        << ',' << (rows.empty() ? 0.0 : double(bottom25)/rows.size())
        << ',' << quantileInt(distance, 0.5)
        << ',' << (rows.empty() ? 0.0 : double(strong)/rows.size())
        << ',' << quantileInt(strongSpacing, 0.5)
        << ',' << quantileInt(strongSpacing, 0.9)
        << ',' << quantileInt(strongSpacing, 0.99)
        << ',' << correlation.value() << ",0.5," << exact << '\n';
}

struct Arguments
{
    std::filesystem::path mesh, permutation, output;
    std::string name;
    label width = 0;
    double runtime = 0;
};

Arguments arguments(const int argc, char** argv)
{
    Arguments result;
    for (int i=1; i<argc; ++i)
    {
        const std::string arg = argv[i];
        const auto value = [&](const char* prefix)
        {
            return arg.substr(std::char_traits<char>::length(prefix));
        };
        if (arg.rfind("--mesh=", 0) == 0) result.mesh = value("--mesh=");
        else if (arg.rfind("--name=", 0) == 0) result.name = value("--name=");
        else if (arg.rfind("--width=", 0) == 0) result.width = std::stoi(value("--width="));
        else if (arg.rfind("--permutation=", 0) == 0) result.permutation = value("--permutation=");
        else if (arg.rfind("--output=", 0) == 0) result.output = value("--output=");
        else if (arg.rfind("--runtime-ms=", 0) == 0) result.runtime = std::stod(value("--runtime-ms="));
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if (result.mesh.empty() || result.name.empty() || result.width < 1
        || result.permutation.empty() || result.output.empty())
        throw std::runtime_error("missing required argument");
    return result;
}

} // namespace

int main(int argc, char** argv)
try
{
    const Arguments args = arguments(argc, argv);
    const auto mesh = smootherTest::PolyMeshReader::read(args.mesh);
    const auto expected = readNewToOld(args.permutation);
    if (expected.size() != mesh.nCells)
        throw std::runtime_error("mesh/permutation cell count mismatch");
    const LineCSR lines = buildLineCSR(mesh);
    const auto ownerStarts = mesh.ownerStartAddressing();
    std::vector<label> predecessors(mesh.nCells, 0);
    for (const label cell : mesh.neighbour) ++predecessors[cell];
    std::set<label> ready;
    for (std::size_t cell=0; cell<mesh.nCells; ++cell)
        if (predecessors[cell] == 0) ready.insert(static_cast<label>(cell));

    constexpr std::size_t maxRadius = 1024;
    std::vector<CutRow> rows;
    std::vector<label> strongSpacing;
    Correlation correlation;
    std::size_t execution = 0, chunk = 0;
    bool exact = true;
    while (!ready.empty())
    {
        const std::size_t readySize = ready.size();
        std::vector<label> current;
        current.reserve(static_cast<std::size_t>(args.width));
        auto position = ready.begin();
        while (!ready.empty() && current.size() < static_cast<std::size_t>(args.width))
        {
            current.push_back(*position);
            position = ready.erase(position);
        }
        for (const label cell : current)
        {
            if (execution >= expected.size() || expected[execution] != cell) exact = false;
            ++execution;
        }

        if (!ready.empty())
        {
            const std::size_t leftCount = std::min(maxRadius + 1, current.size());
            std::vector<label> sequence
            (
                current.end() - static_cast<std::ptrdiff_t>(leftCount), current.end()
            );
            auto right = ready.begin();
            for (std::size_t count=0; count<maxRadius + 1 && right!=ready.end(); ++count, ++right)
                sequence.push_back(*right);
            std::vector<Similarity> landscape;
            landscape.reserve(sequence.size() - 1);
            for (std::size_t i=1; i<sequence.size(); ++i)
            {
                landscape.push_back(similarity(lines, sequence[i - 1], sequence[i]));
                correlation.add
                (
                    landscape.back().indexGap,
                    1.0 - landscape.back().jaccard
                );
            }
            const std::size_t cutIndex = leftCount - 1;
            const Similarity cut = landscape[cutIndex];
            CutRow row;
            row.chunk = chunk;
            row.readySize = readySize;
            row.chosenWidth = current.size();
            row.jaccard = cut.jaccard;
            row.overlapMin = cut.overlapMin;
            row.newLines = cut.newLines;
            row.indexGap = cut.indexGap;
            row.strength = 1.0 - cut.jaccard;
            row.radius64 = windowStats(landscape, cutIndex, 64);
            row.radius256 = windowStats(landscape, cutIndex, 256);
            row.radius1024 = windowStats(landscape, cutIndex, 1024);
            row.strongHit = row.radius1024.percentile <= 0.05;
            rows.push_back(row);

            const std::size_t begin = cutIndex > maxRadius ? cutIndex - maxRadius : 0;
            const std::size_t end = std::min
            (
                landscape.size(), cutIndex + maxRadius + 1
            );
            std::vector<double> local;
            local.reserve(end - begin);
            for (std::size_t i=begin; i<end; ++i) local.push_back(landscape[i].jaccard);
            const double strongThreshold = quantile(local, 0.05);
            label previousStrong = -1;
            for (std::size_t i=begin; i<end; ++i)
            {
                if (landscape[i].jaccard <= strongThreshold)
                {
                    if (previousStrong >= 0)
                        strongSpacing.push_back(static_cast<label>(i) - previousStrong);
                    previousStrong = static_cast<label>(i);
                }
            }
        }

        for (const label cell : current)
            for
            (
                label face=ownerStarts[static_cast<std::size_t>(cell)];
                face<ownerStarts[static_cast<std::size_t>(cell) + 1];
                ++face
            )
                if (--predecessors[mesh.neighbour[static_cast<std::size_t>(face)]] == 0)
                    ready.insert(mesh.neighbour[static_cast<std::size_t>(face)]);
        ++chunk;
    }
    exact = exact && execution == expected.size();
    if (!exact) throw std::runtime_error("offline replay differs from saved permutation");
    std::filesystem::create_directories(args.output);
    const std::string stem = args.name + "_W" + std::to_string(args.width);
    writeCutRows(args.output/(stem + "_cuts.csv"), rows);
    writeSummary
    (
        args.output/(stem + "_summary.csv"), args.name, args.width,
        args.runtime, chunk, rows, strongSpacing, correlation, exact
    );
    std::cout << "psi-ready mesh=" << args.name << " width=" << args.width
        << " chunks=" << chunk << " validCuts=" << rows.size()
        << " replayExact=PASS\n";
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << "psi_ready_boundary_analysis: " << error.what() << '\n';
    return 1;
}
