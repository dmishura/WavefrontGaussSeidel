#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include "GaussSeidelSmoother.H"
#include "PolyMeshReader.H"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

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

}

int runTest(const std::string& meshDirectory)
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
    scalarField psi(exact.size(), 0.0);

    // Serial placeholders for coupled/MPI interface data.
    FieldField<Field, scalar> interfaceCoeffs(0);
    lduInterfaceFieldPtrsList interfaces(0);

    const scalar initial = relativeResidual(matrix, psi, source);
    GaussSeidelSmoother::smooth
    (
        "psi", psi, matrix, source, interfaceCoeffs, interfaces, 0, 1
    );
    const scalar afterOne = relativeResidual(matrix, psi, source);
    if (!(afterOne < initial))
        throw std::runtime_error("one sweep did not reduce the residual");

    GaussSeidelSmoother::smooth
    (
        "psi", psi, matrix, source, interfaceCoeffs, interfaces, 0, 199
    );
    const scalar final = relativeResidual(matrix, psi, source);
    scalar maxError = 0;
    for (label i=0; i<label(psi.size()); ++i)
        maxError = std::max(maxError, std::abs(psi[i] - exact[i]));

    std::cout << "motorBike cells: " << mesh.nCells
        << "\ninternal faces: " << mesh.nInternalFaces()
        << "\ninitial residual: " << initial
        << "\none-sweep residual: " << afterOne
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
        "Usage: Test-GaussSeidel <polyMesh-directory>\n";
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
        return runTest(argv[1]);
    }
    catch (const std::exception& error)
    {
        std::cerr << "Test-GaussSeidel: " << error.what() << '\n';
        return 1;
    }
}
