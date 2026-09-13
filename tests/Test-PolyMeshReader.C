#include "PolyMeshReader.H"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using smootherTest::PolyMeshReader;

namespace
{
void require(const bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void testMotorBike(const fs::path& meshDir)
{
    const auto mesh = PolyMeshReader::read(meshDir);
    require(mesh.nCells == 352253, "unexpected motorBike cell count");
    require(mesh.nPoints == 404931, "unexpected motorBike point count");
    require(mesh.nFaces == 1104083, "unexpected motorBike face count");
    require(mesh.nInternalFaces() == 1053964, "unexpected internal-face count");
    require(mesh.cellCentreX.size() == mesh.nCells, "cell centres were not read");
    require
    (
        std::all_of
        (
            mesh.cellCentreX.begin(), mesh.cellCentreX.end(),
            [](const double x) { return std::isfinite(x); }
        ),
        "cell centre contains a non-finite X coordinate"
    );
    const auto centreBounds = std::minmax_element
    (
        mesh.cellCentreX.begin(), mesh.cellCentreX.end()
    );
    require(*centreBounds.second > *centreBounds.first, "cell centre X range is empty");
    const auto starts = mesh.ownerStartAddressing();
    require(starts.size() == mesh.nCells + 1, "invalid ownerStart size");
    require(starts.front() == 0, "ownerStart must begin at zero");
    require
    (
        static_cast<std::size_t>(starts.back()) == mesh.nInternalFaces(),
        "ownerStart must end at nInternalFaces"
    );

    std::vector<std::size_t> incident(mesh.nCells, 0);
    for (const int cell : mesh.owner) ++incident[cell];
    for (const int cell : mesh.neighbour) ++incident[cell];
    require
    (
        std::none_of
        (
            incident.begin(), incident.end(),
            [](const std::size_t n) { return n == 0; }
        ),
        "at least one cell is missing from face addressing"
    );

    for (std::size_t cell=0; cell<mesh.nCells; ++cell)
    {
        require(starts[cell] <= starts[cell + 1], "ownerStart is not monotonic");
        for (int face=starts[cell]; face<starts[cell + 1]; ++face)
            require
            (
                mesh.owner[face] == static_cast<int>(cell),
                "ownerStart range mismatch"
            );
    }

    const auto bounds = std::minmax_element(incident.begin(), incident.end());
    const std::size_t sum =
        std::accumulate(incident.begin(), incident.end(), std::size_t(0));
    require(*bounds.first == 4, "unexpected minimum incident-face degree");
    require(*bounds.second == 21, "unexpected maximum incident-face degree");
    require(sum == 2158047, "unexpected total cell-face incidence count");
    std::cout << "motorBike incident-face degree: min=" << *bounds.first
        << ", max=" << *bounds.second << ", sum=" << sum << '\n';
    std::cout << "PASS: binary motorBike topology\n";
}

void writeAsciiList
(
    const fs::path& path,
    const std::string& object,
    const std::string& values,
    const int count
)
{
    std::ofstream out(path);
    out << "FoamFile\n{\n format ascii;\n class labelList;\n"
        << " note \"nCells:2  nFaces:1\";\n object " << object
        << ";\n}\n" << count << "\n(\n" << values << "\n)\n";
}

void testRejectsInvalidTopology()
{
    const fs::path dir = fs::temp_directory_path()/"smootherTest-invalid-polyMesh";
    fs::create_directories(dir);
    writeAsciiList(dir/"owner", "owner", "0", 1);
    writeAsciiList(dir/"neighbour", "neighbour", "3", 1);
    bool rejected = false;
    try { static_cast<void>(PolyMeshReader::read(dir)); }
    catch (const std::runtime_error&) { rejected = true; }
    fs::remove_all(dir);
    require(rejected, "reader accepted an out-of-range neighbour label");
    std::cout << "PASS: invalid topology is rejected\n";
}
}

int runTests(const fs::path& meshDir)
{
    testMotorBike(meshDir);
    testRejectsInvalidTopology();
    std::cout << "PolyMeshReader tests passed\n";
    return 0;
}

int main(int argc, char** argv)
{
    const std::string usage =
        "Usage: Test-PolyMeshReader <polyMesh-directory>\n";
    if (argc == 2 && std::string(argv[1]) == "--help")
    {
        std::cout << usage;
        return 0;
    }
    if (argc != 2)
    {
        std::cerr << usage;
        return 2;
    }
    try
    {
        return runTests(argv[1]);
    }
    catch (const std::exception& error)
    {
        std::cerr << "Test-PolyMeshReader: " << error.what() << '\n';
        return 1;
    }
}
