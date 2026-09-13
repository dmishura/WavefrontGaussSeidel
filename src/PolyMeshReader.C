#include "PolyMeshReader.H"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace smootherTest
{
namespace
{

struct LabelListFile
{
    std::vector<int> values;
    std::size_t headerCells = 0;
    std::size_t headerFaces = 0;
    std::size_t headerPoints = 0;
    std::size_t headerInternalFaces = 0;
};

struct GeometryFileHeader
{
    std::vector<char> bytes;
    std::string text;
    std::size_t position = 0;
    std::size_t count = 0;
    bool binary = false;
};

std::vector<char> readFile(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    const auto end = stream.tellg();
    if (end < 0) throw std::runtime_error("cannot size " + path.string());
    std::vector<char> bytes(static_cast<std::size_t>(end));
    stream.seekg(0);
    if (!bytes.empty())
        stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!stream) throw std::runtime_error("cannot read " + path.string());
    return bytes;
}

std::size_t parseHeaderNumber(const std::string& header, const char* name)
{
    const std::regex expression(std::string(name) + R"(:\s*([0-9]+))");
    std::smatch match;
    if (!std::regex_search(header, match, expression)) return 0;
    return static_cast<std::size_t>(std::stoull(match[1].str()));
}

void skipSpaceAndComments(const std::vector<char>& bytes, std::size_t& pos)
{
    while (pos < bytes.size())
    {
        if (std::isspace(static_cast<unsigned char>(bytes[pos]))) { ++pos; continue; }
        if (pos + 1 < bytes.size() && bytes[pos] == '/' && bytes[pos + 1] == '/')
        {
            pos += 2;
            while (pos < bytes.size() && bytes[pos] != '\n') ++pos;
            continue;
        }
        if (pos + 1 < bytes.size() && bytes[pos] == '/' && bytes[pos + 1] == '*')
        {
            pos += 2;
            while (pos + 1 < bytes.size()
                && !(bytes[pos] == '*' && bytes[pos + 1] == '/')) ++pos;
            if (pos + 1 >= bytes.size()) throw std::runtime_error("unterminated comment");
            pos += 2;
            continue;
        }
        break;
    }
}

GeometryFileHeader readGeometryHeader(const std::filesystem::path& path)
{
    GeometryFileHeader result;
    result.bytes = readFile(path);
    result.text.assign(result.bytes.begin(), result.bytes.end());
    const std::size_t headerBegin = result.text.find("FoamFile");
    const std::size_t headerEnd = result.text.find('}', headerBegin);
    if (headerBegin == std::string::npos || headerEnd == std::string::npos)
        throw std::runtime_error("invalid FoamFile header in " + path.string());
    const std::string header =
        result.text.substr(headerBegin, headerEnd - headerBegin + 1);
    result.binary = std::regex_search(header, std::regex(R"(format\s+binary\s*;)"));
    result.position = headerEnd + 1;
    skipSpaceAndComments(result.bytes, result.position);
    const std::size_t begin = result.position;
    while
    (
        result.position < result.bytes.size()
     && std::isdigit(static_cast<unsigned char>(result.bytes[result.position]))
    ) ++result.position;
    if (begin == result.position)
        throw std::runtime_error("missing list size in " + path.string());
    result.count = std::stoull
    (
        result.text.substr(begin, result.position - begin)
    );
    skipSpaceAndComments(result.bytes, result.position);
    if (result.position >= result.bytes.size() || result.bytes[result.position++] != '(')
        throw std::runtime_error("missing list opening in " + path.string());
    return result;
}

std::vector<std::array<double, 3>> readPoints(const std::filesystem::path& path)
{
    GeometryFileHeader file = readGeometryHeader(path);
    std::vector<std::array<double, 3>> points(file.count);
    if (!file.binary)
    {
        for (std::size_t point=0; point<file.count; ++point)
        {
            skipSpaceAndComments(file.bytes, file.position);
            if (file.position >= file.bytes.size() || file.bytes[file.position++] != '(')
                throw std::runtime_error("invalid ASCII point in " + path.string());
            char* end = nullptr;
            points[point][0] = std::strtod(file.text.c_str() + file.position, &end);
            points[point][1] = std::strtod(end, &end);
            points[point][2] = std::strtod(end, &end);
            file.position = static_cast<std::size_t>(end - file.text.c_str());
            skipSpaceAndComments(file.bytes, file.position);
            if (file.position >= file.bytes.size() || file.bytes[file.position++] != ')')
                throw std::runtime_error("invalid ASCII point closing");
        }
        return points;
    }
    const std::size_t byteCount = 3*file.count*sizeof(double);
    if (byteCount > file.bytes.size() - file.position)
        throw std::runtime_error("truncated binary points payload");
    for (std::size_t point=0; point<file.count; ++point)
    {
        for (std::size_t component=0; component<3; ++component)
        {
            double value;
            std::memcpy
            (
                &value,
                file.bytes.data() + file.position
                  + (3*point + component)*sizeof(double),
                sizeof(value)
            );
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            std::uint64_t raw;
            std::memcpy(&raw, &value, sizeof(raw));
            raw = __builtin_bswap64(raw);
            std::memcpy(&value, &raw, sizeof(value));
#endif
            points[point][component] = value;
        }
    }
    return points;
}

std::pair<std::vector<int>, std::vector<int>> readCompactFaces
(
    const std::filesystem::path& path
)
{
    GeometryFileHeader file = readGeometryHeader(path);
    if (!file.binary)
        throw std::runtime_error("ASCII faceCompactList is not supported yet");
    std::vector<int> starts(file.count);
    const std::size_t startsBytes = starts.size()*sizeof(std::int32_t);
    if (startsBytes > file.bytes.size() - file.position)
        throw std::runtime_error("truncated face starts payload");
    for (std::size_t i=0; i<starts.size(); ++i)
    {
        std::int32_t value;
        std::memcpy(&value, file.bytes.data() + file.position + 4*i, 4);
        starts[i] = value;
    }
    file.position += startsBytes;
    skipSpaceAndComments(file.bytes, file.position);
    if (file.position >= file.bytes.size() || file.bytes[file.position++] != ')')
        throw std::runtime_error("invalid face starts closing");
    skipSpaceAndComments(file.bytes, file.position);
    const std::size_t countBegin = file.position;
    while
    (
        file.position < file.bytes.size()
     && std::isdigit(static_cast<unsigned char>(file.bytes[file.position]))
    ) ++file.position;
    if (countBegin == file.position)
        throw std::runtime_error("missing compact face vertex count");
    const std::size_t count = std::stoull
    (
        file.text.substr(countBegin, file.position - countBegin)
    );
    skipSpaceAndComments(file.bytes, file.position);
    if (file.position >= file.bytes.size() || file.bytes[file.position++] != '(')
        throw std::runtime_error("missing compact face vertices opening");
    std::vector<int> vertices(count);
    if (count*sizeof(std::int32_t) > file.bytes.size() - file.position)
        throw std::runtime_error("truncated compact face vertices");
    for (std::size_t i=0; i<count; ++i)
    {
        std::int32_t value;
        std::memcpy(&value, file.bytes.data() + file.position + 4*i, 4);
        vertices[i] = value;
    }
    return {std::move(starts), std::move(vertices)};
}

LabelListFile readLabelList(const std::filesystem::path& path)
{
    const std::vector<char> bytes = readFile(path);
    const std::string all(bytes.begin(), bytes.end());
    const std::size_t headerBegin = all.find("FoamFile");
    const std::size_t headerEnd = all.find('}', headerBegin);
    if (headerBegin == std::string::npos || headerEnd == std::string::npos)
        throw std::runtime_error("invalid FoamFile header in " + path.string());

    const std::string header = all.substr(headerBegin, headerEnd - headerBegin + 1);
    if (header.find("labelList") == std::string::npos)
        throw std::runtime_error("expected labelList in " + path.string());
    const bool binary = header.find("format      binary") != std::string::npos
        || std::regex_search(header, std::regex(R"(format\s+binary\s*;)"));

    std::size_t pos = headerEnd + 1;
    skipSpaceAndComments(bytes, pos);
    const std::size_t countBegin = pos;
    while (pos < bytes.size() && std::isdigit(static_cast<unsigned char>(bytes[pos]))) ++pos;
    if (pos == countBegin) throw std::runtime_error("missing list size in " + path.string());
    const std::size_t count = std::stoull(all.substr(countBegin, pos - countBegin));
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("labelList is too large in " + path.string());
    skipSpaceAndComments(bytes, pos);
    if (pos >= bytes.size() || bytes[pos++] != '(')
        throw std::runtime_error("missing list opening in " + path.string());

    LabelListFile result;
    result.headerCells = parseHeaderNumber(header, "nCells");
    result.headerFaces = parseHeaderNumber(header, "nFaces");
    result.headerPoints = parseHeaderNumber(header, "nPoints");
    result.headerInternalFaces = parseHeaderNumber(header, "nInternalFaces");
    result.values.resize(count);

    if (binary)
    {
        const std::size_t byteCount = count*sizeof(std::int32_t);
        if (byteCount > bytes.size() - pos)
            throw std::runtime_error("truncated binary payload in " + path.string());
        for (std::size_t i=0; i<count; ++i)
        {
            std::uint32_t raw;
            std::memcpy(&raw, bytes.data() + pos + i*sizeof(raw), sizeof(raw));
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            raw = __builtin_bswap32(raw);
#endif
            result.values[i] = static_cast<std::int32_t>(raw);
        }
        pos += byteCount;
    }
    else
    {
        for (std::size_t i=0; i<count; ++i)
        {
            skipSpaceAndComments(bytes, pos);
            const std::size_t begin = pos;
            if (pos < bytes.size() && (bytes[pos] == '-' || bytes[pos] == '+')) ++pos;
            while (pos < bytes.size() && std::isdigit(static_cast<unsigned char>(bytes[pos]))) ++pos;
            if (pos == begin) throw std::runtime_error("invalid ASCII label in " + path.string());
            result.values[i] = std::stoi(all.substr(begin, pos - begin));
        }
    }
    skipSpaceAndComments(bytes, pos);
    if (pos >= bytes.size() || bytes[pos] != ')')
        throw std::runtime_error("missing list closing in " + path.string());
    return result;
}

}

void PolyMeshTopology::validate() const
{
    if (nCells == 0) throw std::runtime_error("polyMesh has no cells");
    if (owner.size() != nFaces) throw std::runtime_error("owner size differs from nFaces");
    if (neighbour.size() > owner.size()) throw std::runtime_error("too many neighbour entries");
    for (std::size_t face=0; face<owner.size(); ++face)
        if (owner[face] < 0 || static_cast<std::size_t>(owner[face]) >= nCells)
            throw std::runtime_error("owner label is outside cell range");
    for (std::size_t face=0; face<neighbour.size(); ++face)
    {
        if (neighbour[face] < 0 || static_cast<std::size_t>(neighbour[face]) >= nCells)
            throw std::runtime_error("neighbour label is outside cell range");
        if (owner[face] >= neighbour[face])
            throw std::runtime_error("internal face is not in upper-triangular order");
    }
    if (!std::is_sorted(owner.begin(), owner.begin() + neighbour.size()))
        throw std::runtime_error("internal-face owners are not grouped");

    std::vector<std::size_t> incidentFaces(nCells, 0);
    for (const int cell : owner) ++incidentFaces[static_cast<std::size_t>(cell)];
    for (const int cell : neighbour) ++incidentFaces[static_cast<std::size_t>(cell)];
    if (std::find(incidentFaces.begin(), incidentFaces.end(), 0) != incidentFaces.end())
        throw std::runtime_error("polyMesh contains a cell without faces");
}

std::vector<int> PolyMeshTopology::ownerStartAddressing() const
{
    validate();
    std::vector<int> starts(nCells + 1, 0);
    for (std::size_t face=0; face<neighbour.size(); ++face)
        ++starts[static_cast<std::size_t>(owner[face]) + 1];
    for (std::size_t cell=0; cell<nCells; ++cell) starts[cell + 1] += starts[cell];
    return starts;
}

PolyMeshTopology PolyMeshReader::read(const std::filesystem::path& dir)
{
    LabelListFile owner = readLabelList(dir/"owner");
    LabelListFile neighbour = readLabelList(dir/"neighbour");
    PolyMeshTopology mesh;
    mesh.owner = std::move(owner.values);
    mesh.neighbour = std::move(neighbour.values);
    mesh.nFaces = mesh.owner.size();
    mesh.nPoints = owner.headerPoints;
    mesh.nCells = owner.headerCells;
    if (mesh.nCells == 0 && !mesh.owner.empty())
        mesh.nCells = static_cast<std::size_t>(*std::max_element(mesh.owner.begin(), mesh.owner.end())) + 1;
    if (owner.headerFaces && owner.headerFaces != mesh.nFaces)
        throw std::runtime_error("owner header nFaces mismatch");
    if (owner.headerInternalFaces
        && owner.headerInternalFaces != mesh.nInternalFaces())
        throw std::runtime_error("owner header nInternalFaces mismatch");
    if (neighbour.headerCells && neighbour.headerCells != mesh.nCells)
        throw std::runtime_error("owner/neighbour nCells mismatch");
    if (neighbour.headerFaces && neighbour.headerFaces != mesh.nFaces)
        throw std::runtime_error("owner/neighbour nFaces mismatch");
    if (neighbour.headerInternalFaces
        && neighbour.headerInternalFaces != mesh.nInternalFaces())
        throw std::runtime_error("neighbour header nInternalFaces mismatch");
    if (neighbour.headerPoints && mesh.nPoints
        && neighbour.headerPoints != mesh.nPoints)
        throw std::runtime_error("owner/neighbour nPoints mismatch");
    if (std::filesystem::exists(dir/"points") && std::filesystem::exists(dir/"faces"))
    {
        const std::vector<std::array<double, 3>> points = readPoints(dir/"points");
        auto [faceStarts, faceVertices] = readCompactFaces(dir/"faces");
        if (faceStarts.size() != mesh.nFaces + 1)
            throw std::runtime_error("faceCompactList start size mismatch");
        if (points.size() != mesh.nPoints)
            throw std::runtime_error("points size differs from nPoints");
        mesh.cellCentreX.assign(mesh.nCells, 0.0);
        std::vector<double> cellVolumes(mesh.nCells, 0.0);
        for (std::size_t face=0; face<mesh.nFaces; ++face)
        {
            const int begin = faceStarts[face];
            const int end = faceStarts[face + 1];
            if (begin < 0 || end <= begin || std::size_t(end) > faceVertices.size())
                throw std::runtime_error("invalid compact face range");
            if (end - begin < 3)
                throw std::runtime_error("face contains fewer than three points");
            const int firstPoint = faceVertices[begin];
            if (firstPoint < 0 || std::size_t(firstPoint) >= points.size())
                throw std::runtime_error("face point outside point range");
            double faceVolume = 0.0;
            double faceMomentX = 0.0;
            const auto& a = points[firstPoint];
            for (int p=begin + 1; p<end - 1; ++p)
            {
                const int secondPoint = faceVertices[p];
                const int thirdPoint = faceVertices[p + 1];
                if
                (
                    secondPoint < 0 || thirdPoint < 0
                 || std::size_t(secondPoint) >= points.size()
                 || std::size_t(thirdPoint) >= points.size()
                )
                    throw std::runtime_error("face point outside point range");
                const auto& b = points[secondPoint];
                const auto& c = points[thirdPoint];
                const double volume =
                (
                    a[0]*(b[1]*c[2] - b[2]*c[1])
                  + a[1]*(b[2]*c[0] - b[0]*c[2])
                  + a[2]*(b[0]*c[1] - b[1]*c[0])
                )/6.0;
                faceVolume += volume;
                faceMomentX += volume*(a[0] + b[0] + c[0])/4.0;
            }
            const std::size_t ownerCell = mesh.owner[face];
            mesh.cellCentreX[ownerCell] += faceMomentX;
            cellVolumes[ownerCell] += faceVolume;
            if (face < mesh.nInternalFaces())
            {
                const std::size_t neighbourCell = mesh.neighbour[face];
                mesh.cellCentreX[neighbourCell] -= faceMomentX;
                cellVolumes[neighbourCell] -= faceVolume;
            }
        }
        for (std::size_t cell=0; cell<mesh.nCells; ++cell)
        {
            if (std::abs(cellVolumes[cell]) < 1e-300)
                throw std::runtime_error("cell has zero geometric volume");
            mesh.cellCentreX[cell] /= cellVolumes[cell];
        }
    }
    mesh.validate();
    return mesh;
}

}
