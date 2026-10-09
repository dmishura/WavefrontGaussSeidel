#include "PolyMeshReader.H"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace
{

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

void writeCellCentres
(
    const smootherTest::PolyMeshTopology& mesh,
    const std::string& outputPath
)
{
    if
    (
        mesh.cellCentreX.size() != mesh.nCells
     || mesh.cellCentreY.size() != mesh.nCells
     || mesh.cellCentreZ.size() != mesh.nCells
    ) throw std::runtime_error
      (
          "polyMesh points/faces are required to export cell centres"
      );

    std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create " + outputPath);
    output.write("IHCENTR", 8);
    writeLittleEndian<std::uint32_t>(output, 1);
    writeLittleEndian<std::uint32_t>(output, 8);
    writeLittleEndian<std::uint64_t>(output, mesh.nCells);

    constexpr std::size_t chunkCells = 1u << 18;
    std::vector<double> buffer;
    buffer.reserve(3*std::min(chunkCells, mesh.nCells));
    for (std::size_t begin=0; begin<mesh.nCells; begin += chunkCells)
    {
        const std::size_t end = std::min(begin + chunkCells, mesh.nCells);
        buffer.clear();
        for (std::size_t cell=begin; cell<end; ++cell)
        {
            buffer.push_back(mesh.cellCentreX[cell]);
            buffer.push_back(mesh.cellCentreY[cell]);
            buffer.push_back(mesh.cellCentreZ[cell]);
        }
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        for (double& value : buffer)
        {
            std::uint64_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            bits = __builtin_bswap64(bits);
            std::memcpy(&value, &bits, sizeof(bits));
        }
#endif
        output.write
        (
            reinterpret_cast<const char*>(buffer.data()),
            static_cast<std::streamsize>(buffer.size()*sizeof(buffer.front()))
        );
    }
    if (!output) throw std::runtime_error("failed writing " + outputPath);
}

}

int main(int argc, char** argv)
{
    try
    {
        if (argc != 3)
        {
            std::cerr << "usage: mesh_centres_export POLYMESH OUTPUT.ihcentres\n";
            return 2;
        }
        const smootherTest::PolyMeshTopology mesh =
            smootherTest::PolyMeshReader::read(argv[1]);
        writeCellCentres(mesh, argv[2]);
        std::cout << "Cell centres exported cells=" << mesh.nCells
            << " path=" << argv[2] << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
