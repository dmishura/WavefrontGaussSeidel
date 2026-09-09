#include "PolyMeshReader.H"

#include <algorithm>
#include <cctype>
#include <cstdint>
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
    mesh.validate();
    return mesh;
}

}
