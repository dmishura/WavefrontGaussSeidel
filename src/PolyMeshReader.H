#ifndef PolyMeshReader_H
#define PolyMeshReader_H

#include <cstddef>
#include <filesystem>
#include <vector>

namespace smootherTest
{

struct PolyMeshTopology
{
    std::size_t nPoints = 0;
    std::size_t nCells = 0;
    std::size_t nFaces = 0;
    std::vector<int> owner;
    std::vector<int> neighbour;
    std::vector<double> cellCentreX;

    std::size_t nInternalFaces() const noexcept { return neighbour.size(); }

    // Build the owner-grouped addressing expected by GaussSeidelSmoother.
    std::vector<int> ownerStartAddressing() const;
    void validate() const;
};

class PolyMeshReader
{
public:
    // Read serial OpenFOAM polyMesh topology from owner and neighbour files.
    // Both ASCII and binary labelList files are supported; binary input is
    // currently restricted to little-endian 32-bit labels.
    static PolyMeshTopology read(const std::filesystem::path& polyMeshDir);
};

}

#endif
