#include "GAMGKernels.H"
#include "GAMGSolver.H"
#include "GaussSeidelSmoother.H"
#include "PolyMeshReader.H"
#include "ScheduleBuilder.H"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace gamg = smootherTest::gamg;
namespace harness = smootherTest::harness;

namespace
{

struct TestProblem
{
    smootherTest::PolyMeshTopology mesh;
    std::unique_ptr<Foam::lduMatrix> matrix;
    Foam::scalarField exact;
    Foam::scalarField source;
    std::unique_ptr<gamg::GAMGSolver> solver;
};

std::unique_ptr<TestProblem> problem;

std::string consumeMeshArgument(int& argc, char** argv)
{
#ifdef SMOOTHER_TEST_DEFAULT_MESH
    std::string mesh = SMOOTHER_TEST_DEFAULT_MESH;
#else
    std::string mesh;
#endif
    int destination = 1;
    for (int source=1; source<argc; ++source)
    {
        const std::string argument(argv[source]);
        if (argument.rfind("--mesh=", 0) == 0)
            mesh = argument.substr(7);
        else
            argv[destination++] = argv[source];
    }
    argc = destination;
    if (mesh.empty()) throw std::runtime_error("usage: gamg_tests --mesh=POLYMESH_DIR");
    return mesh;
}

Foam::scalar maximumDifference
(
    const Foam::scalarField& left,
    const Foam::scalarField& right
)
{
    Foam::scalar result = 0;
    for (Foam::label i=0; i<static_cast<Foam::label>(left.size()); ++i)
        result = std::max(result, std::abs(left[i] - right[i]));
    return result;
}

Foam::scalar matrixSum(const gamg::GAMGMatrix& matrix)
{
    return std::accumulate(matrix.diag().begin(), matrix.diag().end(), 0.0)
        + 2.0*std::accumulate(matrix.upper().begin(), matrix.upper().end(), 0.0);
}

TEST(GAMGHierarchy, AddressingAndCoefficientInvariants)
{
    const auto& agglomeration = problem->solver->agglomeration();
    const auto& matrices = problem->solver->matrices();
    ASSERT_EQ(matrices.size(), agglomeration.size() + 1);
    ASSERT_GT(matrices.size(), 1u);

    if (matrices[0].nCells() == 352253 && matrices[0].nFaces() == 1053964)
    {
        const std::vector<Foam::label> expectedCells
        {
            352253, 168901, 81066, 39298, 19004, 9224, 4484, 2180,
            1061, 517, 251, 123, 61, 30, 14
        };
        const std::vector<Foam::label> expectedFaces
        {
            1053964, 743717, 447843, 247749, 129699, 65644, 32571,
            15961, 7763, 3809, 1818, 887, 406, 170, 60
        };
        ASSERT_EQ(matrices.size(), expectedCells.size());
        for (std::size_t level=0; level<matrices.size(); ++level)
        {
            EXPECT_EQ(matrices[level].nCells(), expectedCells[level]);
            EXPECT_EQ(matrices[level].nFaces(), expectedFaces[level]);
        }
    }

    const Foam::scalar finestSum = matrixSum(matrices[0]);
    std::cout << "GAMG hierarchy checksums:\n";
    for (std::size_t level=0; level<matrices.size(); ++level)
    {
        const gamg::GAMGMatrix& matrix = matrices[level];
        EXPECT_GT(matrix.nCells(), 0);
        EXPECT_EQ(matrix.lowerAddr().size(), matrix.upperAddr().size());
        EXPECT_EQ(matrix.upperAddr().size(), matrix.upper().size());
        EXPECT_NEAR(matrixSum(matrix), finestSum, 1e-10*std::max(1.0, std::abs(finestSum)));
        if (level > 0)
        {
            EXPECT_LT(matrix.nCells(), matrices[level - 1].nCells());
        }

        Foam::scalar diagonalSum = std::accumulate
        (
            matrix.diag().begin(), matrix.diag().end(), 0.0
        );
        Foam::scalar coefficientL1 = 0;
        for (const Foam::scalar coefficient : matrix.upper())
            coefficientL1 += std::abs(coefficient);
        std::cout
            << "  level=" << level
            << " cells=" << matrix.nCells()
            << " faces=" << matrix.nFaces()
            << " diagSum=" << diagonalSum
            << " coeffL1=" << coefficientL1 << '\n';

        if (level == 0) continue;
        const auto& map = agglomeration[level - 1];
        EXPECT_EQ(map.fineToCoarse.size(), static_cast<std::size_t>(map.nFineCells));
        EXPECT_EQ(map.faceToCoarse.size(), static_cast<std::size_t>(matrices[level - 1].nFaces()));
        for (const Foam::label coarseCell : map.fineToCoarse)
        {
            EXPECT_GE(coarseCell, 0);
            EXPECT_LT(coarseCell, map.nCoarseCells);
        }
    }
}

TEST(GAMGHierarchy, CoarseOperatorEqualsRestrictionFineOperatorProlongation)
{
    const auto& agglomeration = problem->solver->agglomeration();
    const auto& matrices = problem->solver->matrices();
    for (std::size_t level=0; level<agglomeration.size(); ++level)
    {
        const gamg::GAMGMatrix& coarseMatrix = matrices[level + 1];
        Foam::scalarField coarse(coarseMatrix.nCells());
        for (Foam::label cell=0; cell<coarseMatrix.nCells(); ++cell)
            coarse[cell] = 0.5 + std::sin(0.37*cell) + 0.1*std::cos(0.11*cell);

        Foam::scalarField fine;
        gamg::prolongField(fine, coarse, agglomeration[level]);
        Foam::scalarField fineProduct;
        gamg::multiply(fineProduct, matrices[level], fine);
        Foam::scalarField restricted;
        gamg::restrictField(restricted, fineProduct, agglomeration[level]);
        Foam::scalarField coarseProduct;
        gamg::multiply(coarseProduct, coarseMatrix, coarse);
        EXPECT_LE
        (
            maximumDifference(restricted, coarseProduct),
            2e-10*std::max(1.0, gamg::l1Norm(coarseProduct))
        ) << "level=" << level;
    }
}

TEST(GAMGKernels, FinestGaussSeidelMatchesReference)
{
    Foam::scalarField expected(problem->mesh.nCells, 0.0);
    Foam::scalarField actual(expected);
    Foam::FieldField<Foam::Field, Foam::scalar> interfaceCoefficients;
    Foam::lduInterfaceFieldPtrsList interfaces;
    Foam::GaussSeidelSmoother::smooth
    (
        "psi", expected, *problem->matrix, problem->source,
        interfaceCoefficients, interfaces, 0, 3
    );
    gamg::gaussSeidelSmooth
    (
        actual, problem->solver->matrices()[0], problem->source, 3
    );
    EXPECT_EQ(maximumDifference(expected, actual), 0.0);
}

TEST(GAMGSolve, ConvergesAndReportsResidualHistory)
{
    Foam::scalarField solution(problem->mesh.nCells, 0.0);
    const gamg::GAMGSolverPerformance performance =
        problem->solver->solve(solution, problem->source);
    ASSERT_FALSE(performance.residualHistory.empty());
    EXPECT_EQ
    (
        performance.residualHistory.size(),
        static_cast<std::size_t>(performance.iterations) + 1
    );
    EXPECT_TRUE(performance.converged);
    EXPECT_LT(performance.finalResidual, performance.initialResidual);
    EXPECT_LE(performance.finalResidual, 1e-8);

    const Foam::scalar residual = harness::relativeResidual
    (
        *problem->matrix, solution, problem->source
    );
    std::cout
        << "GAMG solve: levels=" << problem->solver->matrices().size()
        << " iterations=" << performance.iterations
        << " initialResidual=" << performance.initialResidual
        << " finalResidual=" << performance.finalResidual
        << " standaloneRelativeResidual=" << residual
        << " maxSolutionError=" << maximumDifference(solution, problem->exact)
        << "\nResidual history:";
    for (const Foam::scalar value : performance.residualHistory)
        std::cout << ' ' << value;
    std::cout << '\n';
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const std::string meshDirectory = consumeMeshArgument(argc, argv);
        ::testing::InitGoogleTest(&argc, argv);
        problem = std::make_unique<TestProblem>();
        problem->mesh = smootherTest::PolyMeshReader::read(meshDirectory);
        problem->matrix = std::make_unique<Foam::lduMatrix>
        (
            harness::makeMatrix(problem->mesh)
        );
        problem->exact.resize(problem->mesh.nCells);
        const Foam::scalar pi = std::acos(-1.0);
        for (Foam::label cell=0; cell<static_cast<Foam::label>(problem->mesh.nCells); ++cell)
        {
            const Foam::scalar position = Foam::scalar(cell)
                /Foam::scalar(problem->mesh.nCells - 1);
            problem->exact[cell] = 1.0 + 0.25*std::sin(2.0*pi*position)
                + 0.1*std::cos(10.0*pi*position);
        }
        problem->source = harness::multiply(*problem->matrix, problem->exact);
        gamg::GAMGControls controls;
        controls.relativeTolerance = 0;
        controls.tolerance = 1e-8;
        controls.maxIterations = 100;
        controls.diagnostics = true;
        problem->solver = std::make_unique<gamg::GAMGSolver>
        (
            *problem->matrix, controls
        );
        return RUN_ALL_TESTS();
    }
    catch (const std::exception& error)
    {
        std::cerr << "GAMG test setup failed: " << error.what() << '\n';
        return 2;
    }
}
