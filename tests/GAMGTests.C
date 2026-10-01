#include "GAMGKernels.H"
#include "GAMGSolver.H"
#include "fwGAMG.H"
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
namespace fwgamg = smootherTest::fwgamg;
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
    std::unique_ptr<fwgamg::Solver> fwSolver;
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

Foam::scalar maximumRelativeDifference
(
    const Foam::scalarField& left,
    const Foam::scalarField& right
)
{
    Foam::scalar result = 0;
    for (Foam::label i=0; i<static_cast<Foam::label>(left.size()); ++i)
        result = std::max
        (
            result,
            std::abs(left[i] - right[i])
           /std::max(Foam::scalar(1e-300), std::abs(right[i]))
        );
    return result;
}

Foam::scalar relativeL2Difference
(
    const Foam::scalarField& left,
    const Foam::scalarField& right
)
{
    Foam::scalar difference = 0;
    Foam::scalar reference = 0;
    for (Foam::label i=0; i<static_cast<Foam::label>(left.size()); ++i)
    {
        const Foam::scalar delta = left[i] - right[i];
        difference += delta*delta;
        reference += right[i]*right[i];
    }
    return std::sqrt(difference/std::max(Foam::scalar(1e-300), reference));
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

TEST(GAMGTiming, InstrumentationPreservesSolveAndAccountsForWork)
{
    gamg::GAMGControls controls;
    controls.relativeTolerance = 0;
    controls.tolerance = 1e-8;
    controls.maxIterations = 100;

    Foam::scalarField referenceSolution(problem->mesh.nCells, 0.0);
    const gamg::GAMGSolverPerformance reference =
        problem->solver->solve(referenceSolution, problem->source);

    gamg::GAMGTimingStats timing;
    gamg::GAMGSolver timedSolver(*problem->matrix, controls, &timing);
    Foam::scalarField timedSolution(problem->mesh.nCells, 0.0);
    const gamg::GAMGSolverPerformance timed =
        timedSolver.solve(timedSolution, problem->source);

    EXPECT_EQ(timed.iterations, reference.iterations);
    EXPECT_EQ(timed.initialResidual, reference.initialResidual);
    EXPECT_EQ(timed.finalResidual, reference.finalResidual);
    EXPECT_EQ(timed.residualHistory, reference.residualHistory);
    EXPECT_EQ(maximumDifference(timedSolution, referenceSolution), 0.0);
    EXPECT_GT(timing.hierarchyBuildSeconds, 0.0);
    EXPECT_GT(timing.coarseMatrixBuildSeconds, 0.0);
    EXPECT_EQ(timing.levels.size(), timedSolver.matrices().size());
    EXPECT_EQ(timing.transitions.size(), timedSolver.agglomeration().size());
    EXPECT_EQ(timing.solveDurations.size(), 1u);
    EXPECT_EQ
    (
        timing.vcycleDurations.size(),
        static_cast<std::size_t>(timed.iterations)
    );
    EXPECT_EQ
    (
        timing.levels.front().smoothingSweeps,
        static_cast<std::size_t>(timed.iterations*controls.nFinestSweeps)
    );
    EXPECT_EQ
    (
        timing.levels.back().coarsestSolve.calls,
        static_cast<std::size_t>(timed.iterations)
    );
}

TEST(GAMGSolve, NativeGsDirectFlatSpmvMatchesNativeLduExactly)
{
    gamg::GAMGControls controls;
    controls.relativeTolerance = 0;
    controls.tolerance = 1e-8;
    controls.maxIterations = 100;

    gamg::GAMGSolver nativeSolver
    (
        *problem->matrix,
        controls,
        nullptr,
        gamg::GAMGSpmvTraversal::NativeLdu
    );
    gamg::GAMGSolver directSolver
    (
        *problem->matrix,
        controls,
        nullptr,
        gamg::GAMGSpmvTraversal::DirectFlat
    );
    Foam::scalarField nativeSolution(problem->mesh.nCells, 0.0);
    Foam::scalarField directSolution(problem->mesh.nCells, 0.0);
    const gamg::GAMGSolverPerformance native =
        nativeSolver.solve(nativeSolution, problem->source);
    const gamg::GAMGSolverPerformance direct =
        directSolver.solve(directSolution, problem->source);

    EXPECT_EQ(direct.iterations, native.iterations);
    EXPECT_EQ(direct.initialResidual, native.initialResidual);
    EXPECT_EQ(direct.finalResidual, native.finalResidual);
    EXPECT_EQ(direct.residualHistory, native.residualHistory);
    EXPECT_EQ(maximumDifference(directSolution, nativeSolution), 0.0);
}

TEST(fwGAMGHierarchy, PackedOperatorsMatchBaselineAtEveryLevel)
{
    const auto& baseline = problem->solver->matrices();
    const auto& fw = problem->fwSolver->hierarchy();
    ASSERT_EQ(fw.size(), baseline.size());
    for (std::size_t level=0; level<fw.size(); ++level)
    {
        ASSERT_EQ(fw[level].nCells(), baseline[level].nCells());
        ASSERT_EQ(fw[level].nFaces(), baseline[level].nFaces());
        Foam::scalarField original(baseline[level].nCells());
        for (Foam::label cell=0; cell<baseline[level].nCells(); ++cell)
            original[cell] = 0.25 + std::sin(0.17*cell) + 0.2*std::cos(0.031*cell);
        Foam::scalarField expected;
        gamg::multiply(expected, baseline[level], original);
        const Foam::scalarField reordered = problem->fwSolver->toFw(original, level);
        Foam::scalarField packedProduct;
        fw[level].multiply(packedProduct, reordered);
        const Foam::scalarField actual =
            problem->fwSolver->toOriginal(packedProduct, level);
        EXPECT_LE(maximumDifference(expected, actual), 1e-12)
            << "level=" << level;
    }
}

TEST(fwGAMGHierarchy, NaturalPackedKernelsMatchBaseline)
{
    const auto& matrices = problem->solver->matrices();
    for (std::size_t level=0; level<matrices.size(); ++level)
    {
        const gamg::GAMGMatrix& ldu = matrices[level];
        fwgamg::NativeMatrixData native;
        native.nCells = ldu.nCells();
        native.lowerAddr = ldu.lowerAddr();
        native.upperAddr = ldu.upperAddr();
        native.diag = ldu.diag();
        native.upper = ldu.upper();
        const fwgamg::Matrix packed = fwgamg::Matrix::buildNatural
        (
            native, nullptr, true, true, true
        );
        Foam::scalarField source(ldu.nCells());
        Foam::scalarField field(ldu.nCells());
        for (Foam::label cell=0; cell<ldu.nCells(); ++cell)
        {
            source[cell] = 0.7 + std::sin(0.03*cell);
            field[cell] = 0.2 + std::cos(0.07*cell);
        }

        Foam::scalarField expected;
        Foam::scalarField actual;
        gamg::multiply(expected, ldu, field);
        packed.multiply(actual, field);
        EXPECT_LE(maximumDifference(expected, actual), 1e-12)
            << "SpMV level=" << level;
        Foam::scalarField edgeActual;
        packed.multiplyFlatEdgeWise(edgeActual, field);
        EXPECT_LE(maximumDifference(expected, edgeActual), 1e-12)
            << "edge versus LDU SpMV level=" << level;
        EXPECT_LE(maximumDifference(actual, edgeActual), 1e-12)
            << "edge versus row CSR SpMV level=" << level;
        Foam::scalarField directEdgeActual;
        packed.multiplyDirectFlatEdgeWise(directEdgeActual, field);
        EXPECT_LE(maximumDifference(expected, directEdgeActual), 1e-12)
            << "direct edge versus LDU SpMV level=" << level;
        EXPECT_LE(maximumDifference(actual, directEdgeActual), 1e-12)
            << "direct edge versus row CSR SpMV level=" << level;
        Foam::scalarField lduOwnerAccumulated;
        gamg::multiplyOwnerAccumulated(lduOwnerAccumulated, ldu, field);
        EXPECT_LE(maximumDifference(expected, lduOwnerAccumulated), 1e-12)
            << "owner-accumulated LDU versus LDU SpMV level=" << level;
        Foam::scalarField directOwnerAccumulated;
        packed.multiplyDirectFlatOwnerAccumulated
        (
            directOwnerAccumulated, field
        );
        EXPECT_LE(maximumDifference(expected, directOwnerAccumulated), 1e-12)
            << "owner-accumulated direct edge versus LDU SpMV level="
            << level;
        EXPECT_EQ
        (
            maximumDifference
            (
                lduOwnerAccumulated, directOwnerAccumulated
            ),
            0.0
        ) << "owner-accumulated LDU/direct parity level=" << level;
        Foam::scalarField directOwnerRunAccumulated;
        packed.multiplyDirectFlatOwnerRunAccumulated
        (
            directOwnerRunAccumulated, field
        );
        EXPECT_EQ
        (
            maximumDifference
            (
                directEdgeActual, directOwnerRunAccumulated
            ),
            0.0
        ) << "owner-run direct edge exact parity level=" << level;

        gamg::residual(expected, ldu, field, source);
        packed.residual(actual, field, source);
        EXPECT_LE(maximumDifference(expected, actual), 1e-12)
            << "residual level=" << level;

        Foam::scalarField expectedPsi(ldu.nCells(), 0.0);
        Foam::scalarField actualPsi(expectedPsi);
        gamg::gaussSeidelSmooth(expectedPsi, ldu, source, 1);
        packed.smooth(actualPsi, source, 1, false);
        EXPECT_EQ(maximumDifference(expectedPsi, actualPsi), 0.0)
            << "GS level=" << level;

        Foam::scalarField expectedScale(field), expectedWork;
        Foam::scalarField actualScale(field), actualWork;
        gamg::scaleCorrection(expectedScale, ldu, source, expectedWork);
        packed.scaleCorrection(actualScale, source, actualWork);
        EXPECT_LE(maximumDifference(expectedScale, actualScale), 1e-12)
            << "scaleCorrection level=" << level;
    }
}

TEST(fwGAMGHierarchy, DirectRestrictionAndProlongationMatchBaseline)
{
    const auto& baselineAgglomeration = problem->solver->agglomeration();
    const auto& fw = problem->fwSolver->hierarchy();
    for (std::size_t level=0; level<baselineAgglomeration.size(); ++level)
    {
        Foam::scalarField fine(fw[level].nCells());
        for (Foam::label cell=0; cell<fw[level].nCells(); ++cell)
            fine[cell] = std::sin(0.13*cell) + 0.01*cell;
        const Foam::scalarField fineFw = problem->fwSolver->toFw(fine, level);

        Foam::scalarField expectedCoarse;
        gamg::restrictField
        (
            expectedCoarse, fine, baselineAgglomeration[level]
        );
        Foam::scalarField actualCoarseFw;
        fw.restrictField(actualCoarseFw, fineFw, level);
        const Foam::scalarField actualCoarse =
            problem->fwSolver->toOriginal(actualCoarseFw, level + 1);
        EXPECT_LE(maximumDifference(expectedCoarse, actualCoarse), 2e-10)
            << "restriction level=" << level;

        Foam::scalarField expectedFine;
        gamg::prolongField
        (
            expectedFine, actualCoarse, baselineAgglomeration[level]
        );
        Foam::scalarField actualFineFw;
        fw.prolongField(actualFineFw, actualCoarseFw, level);
        const Foam::scalarField actualFine =
            problem->fwSolver->toOriginal(actualFineFw, level);
        EXPECT_EQ(maximumDifference(expectedFine, actualFine), 0.0)
            << "prolongation level=" << level;
    }
}

TEST(fwGAMGKernels, FinestSmootherMatchesIndexKahn1024)
{
    const auto statistics = harness::constructIndexWavefronts
    (
        problem->mesh, *problem->matrix, 1024, false
    );
    const smootherTest::WavefrontSchedule schedule = harness::makeWavefrontSchedule
    (
        problem->mesh, *problem->matrix, statistics
    );
    Foam::scalarField expected(problem->mesh.nCells, 0.0);
    smootherTest::serialGatherSmooth(expected, problem->source, schedule, 3);

    Foam::scalarField actual = problem->fwSolver->toFw
    (
        Foam::scalarField(problem->mesh.nCells, 0.0)
    );
    const Foam::scalarField source = problem->fwSolver->toFw(problem->source);
    problem->fwSolver->hierarchy()[0].smooth(actual, source, 3, false);
    actual = problem->fwSolver->toOriginal(actual);
    EXPECT_EQ(maximumDifference(expected, actual), 0.0);
}

TEST(fwGAMGKernels, FusedFinalSweepProducesPostSmoothMatrixProduct)
{
    fwgamg::Controls controls;
    controls.spmvTraversal = fwgamg::SpmvTraversal::DirectFlatEdgeWise;
    fwgamg::Solver directFlatSolver(*problem->matrix, controls);
    const auto& matrices = directFlatSolver.hierarchy();
    for (std::size_t level=0; level<matrices.size(); ++level)
    {
        const fwgamg::Matrix& matrix = matrices[level];
        Foam::scalarField source(matrix.nCells());
        Foam::scalarField baseline(matrix.nCells());
        for (Foam::label row=0; row<matrix.nCells(); ++row)
        {
            source[row] = 0.7 + std::sin(0.031*row);
            baseline[row] = 0.2 + 0.1*std::cos(0.017*row);
        }
        Foam::scalarField fused(baseline);
        matrix.smooth(baseline, source, 2, false);
        Foam::scalarField expectedProduct;
        matrix.multiplyDirectFlatEdgeWise(expectedProduct, baseline);
        Foam::scalarField fusedProduct;
        matrix.smoothAndMultiply(fused, source, fusedProduct, 2, false);
        Foam::scalarField deltaFused = baseline;
        // baseline already contains the post-smooth field; restore the same
        // deterministic initial state for the delta formulation.
        for (Foam::label row=0; row<matrix.nCells(); ++row)
            deltaFused[row] = 0.2 + 0.1*std::cos(0.017*row);
        Foam::scalarField deltaProduct(matrix.nCells());
        matrix.smoothAndMultiplyDelta
        (
            deltaFused, source, deltaProduct, 2, false
        );

        EXPECT_EQ(maximumDifference(fused, baseline), 0.0)
            << "smoothed field level=" << level;
        EXPECT_LE(maximumDifference(fusedProduct, expectedProduct), 1e-12)
            << "A*psi level=" << level;
        EXPECT_EQ(maximumDifference(deltaFused, baseline), 0.0)
            << "delta smoothed field level=" << level;
        EXPECT_LE(maximumDifference(deltaProduct, expectedProduct), 1e-12)
            << "delta A*psi level=" << level;

        for (const int variant : {0, 1, 2})
        {
            Foam::scalarField variantPsi(matrix.nCells());
            for (Foam::label row=0; row<matrix.nCells(); ++row)
                variantPsi[row] = 0.2 + 0.1*std::cos(0.017*row);
            Foam::scalarField variantProduct(matrix.nCells());
            if (variant == 0)
                matrix.smoothAndMultiplyDeltaCurrent
                (
                    variantPsi, source, variantProduct, 2, false
                );
            else if (variant == 1)
                matrix.smoothAndMultiplyDeltaEdgeRows
                (
                    variantPsi, source, variantProduct, 2, false
                );
            else
                matrix.smoothAndMultiplyDeltaPackedBaseline
                (
                    variantPsi, source, variantProduct, 2, false
                );
            EXPECT_EQ(maximumDifference(variantPsi, baseline), 0.0)
                << "delta variant field level=" << level
                << " variant=" << variant;
            EXPECT_LE(maximumDifference(variantProduct, expectedProduct), 1e-12)
                << "delta variant A*psi level=" << level
                << " variant=" << variant;
        }
    }
}

TEST(fwGAMGSolve, FusedFinestIH1SpmvPreservesConvergence)
{
    gamg::GAMGControls gamgControls;
    gamgControls.relativeTolerance = 0;
    gamgControls.tolerance = 1e-8;
    gamgControls.maxIterations = 100;

    fwgamg::Controls baselineControls;
    baselineControls.gamg = gamgControls;
    baselineControls.width = 1024;
    fwgamg::Controls fusedControls = baselineControls;
    fusedControls.finestSpmvFusion = fwgamg::FinestSpmvFusion::IncomingHeld;

    fwgamg::Solver baselineSolver(*problem->matrix, baselineControls);
    Foam::scalarField baselineSolution(problem->mesh.nCells, 0.0);
    const fwgamg::Performance baseline = baselineSolver.solve
    (
        baselineSolution, problem->source
    );
    for (const fwgamg::FinestSpmvFusion fusion :
        {fwgamg::FinestSpmvFusion::IncomingHeld,
         fwgamg::FinestSpmvFusion::DeltaCorrection,
         fwgamg::FinestSpmvFusion::DeltaCorrectionCurrent,
         fwgamg::FinestSpmvFusion::DeltaCorrectionEdgeRows,
         fwgamg::FinestSpmvFusion::DeltaCorrectionPackedBaseline})
    {
        fusedControls.finestSpmvFusion = fusion;
        fwgamg::Solver fusedSolver(*problem->matrix, fusedControls);
        Foam::scalarField fusedSolution(problem->mesh.nCells, 0.0);
        const fwgamg::Performance fused = fusedSolver.solve
        (
            fusedSolution, problem->source
        );

        EXPECT_EQ(fused.iterations, baseline.iterations);
        EXPECT_EQ(fused.initialResidual, baseline.initialResidual);
        EXPECT_NEAR(fused.finalResidual, baseline.finalResidual, 1e-12);
        ASSERT_EQ(fused.residualHistory.size(), baseline.residualHistory.size());
        for (std::size_t cycle=0; cycle<fused.residualHistory.size(); ++cycle)
            EXPECT_NEAR
            (
                fused.residualHistory[cycle],
                baseline.residualHistory[cycle],
                1e-12
            ) << "cycle=" << cycle << " fusion=" << static_cast<int>(fusion);
        EXPECT_LE(maximumDifference(fusedSolution, baselineSolution), 1e-12);
    }
}

TEST(fwGAMGSolve, ConvergenceIsEquivalentToBaseline)
{
    Foam::scalarField baselineSolution(problem->mesh.nCells, 0.0);
    Foam::scalarField fwSolution(problem->mesh.nCells, 0.0);
    const gamg::GAMGSolverPerformance baseline =
        problem->solver->solve(baselineSolution, problem->source);
    const fwgamg::Performance fw =
        problem->fwSolver->solve(fwSolution, problem->source);
    ASSERT_TRUE(fw.converged);
    EXPECT_EQ(fw.iterations, baseline.iterations);
    EXPECT_NEAR(fw.initialResidual, baseline.initialResidual, 1e-14);
    EXPECT_NEAR(fw.finalResidual, baseline.finalResidual, 1e-10);
    ASSERT_EQ(fw.residualHistory.size(), baseline.residualHistory.size());
    for (std::size_t i=0; i<fw.residualHistory.size(); ++i)
        EXPECT_NEAR(fw.residualHistory[i], baseline.residualHistory[i], 1e-9)
            << "cycle=" << i;
    EXPECT_LE(maximumDifference(fwSolution, baselineSolution), 1e-7);
    std::cout << "fwGAMG convergence: iterations=" << fw.iterations
        << " initial=" << fw.initialResidual
        << " final=" << fw.finalResidual
        << " maxSolutionDifference="
        << maximumDifference(fwSolution, baselineSolution)
        << " maxRelativeDifference="
        << maximumRelativeDifference(fwSolution, baselineSolution)
        << " relativeL2Difference="
        << relativeL2Difference(fwSolution, baselineSolution)
        << "\nfwGAMG residual history:";
    for (const Foam::scalar value : fw.residualHistory) std::cout << ' ' << value;
    std::cout << '\n';
}

TEST(fwGAMGSolve, LocalityRenumberedHierarchiesConverge)
{
    gamg::GAMGControls gamgControls;
    gamgControls.relativeTolerance = 0;
    gamgControls.tolerance = 1e-8;
    gamgControls.maxIterations = 100;
    Foam::scalarField baselineSolution(problem->mesh.nCells, 0.0);
    const gamg::GAMGSolverPerformance baselinePerformance =
        problem->solver->solve(baselineSolution, problem->source);
    const auto& baselineMatrices = problem->solver->matrices();
    for (const fwgamg::RowOrdering ordering :
        {fwgamg::RowOrdering::Natural, fwgamg::RowOrdering::IndexKahn})
    {
        fwgamg::Controls controls;
        controls.gamg = gamgControls;
        controls.width = 1024;
        controls.rowOrdering = ordering;
        controls.coarseRenumbering =
            fwgamg::CoarseRenumbering::MinimumFineIndex;
        fwgamg::TimingStats timing;
        fwgamg::Solver solver(*problem->matrix, controls, &timing);

        ASSERT_EQ(solver.hierarchy().size(), baselineMatrices.size());
        for (std::size_t level=0; level<baselineMatrices.size(); ++level)
        {
            const gamg::GAMGMatrix& baselineMatrix = baselineMatrices[level];
            const fwgamg::Matrix& renumberedMatrix = solver.hierarchy()[level];
            ASSERT_EQ(renumberedMatrix.nCells(), baselineMatrix.nCells());
            ASSERT_EQ(renumberedMatrix.nFaces(), baselineMatrix.nFaces());
            Foam::scalarField original(baselineMatrix.nCells());
            for (Foam::label cell=0; cell<baselineMatrix.nCells(); ++cell)
                original[cell] = std::sin(0.13*cell) + 0.01*cell;
            Foam::scalarField expected;
            gamg::multiply(expected, baselineMatrix, original);
            Foam::scalarField native(original.size());
            const Foam::labelField& nativeToAgg =
                solver.hierarchy().nativeToAggCells(level);
            for (Foam::label cell=0; cell<baselineMatrix.nCells(); ++cell)
                native[cell] = original[nativeToAgg[cell]];
            const Foam::scalarField reordered = solver.toFw(native, level);
            Foam::scalarField packedProduct;
            renumberedMatrix.multiply(packedProduct, reordered);
            const Foam::scalarField nativeProduct = solver.toOriginal
            (
                packedProduct, level
            );
            Foam::scalarField actual(original.size());
            const Foam::labelField& aggToNative =
                solver.hierarchy().aggToNativeCells(level);
            for (Foam::label cell=0; cell<baselineMatrix.nCells(); ++cell)
                actual[cell] = nativeProduct[aggToNative[cell]];
            EXPECT_LE(maximumDifference(expected, actual), 2e-10)
                << "level=" << level;
        }

        Foam::scalarField solution(problem->mesh.nCells, 0.0);
        const fwgamg::Performance performance =
            solver.solve(solution, problem->source);
        EXPECT_TRUE(performance.converged);
        EXPECT_EQ(performance.iterations, baselinePerformance.iterations);
        EXPECT_LE(performance.finalResidual, 1e-8);
        EXPECT_LE(maximumDifference(solution, baselineSolution), 1e-7);
        EXPECT_LE
        (
            harness::relativeResidual
            (
                *problem->matrix, solution, problem->source
            ),
            1e-8
        );
        double renumberSeconds = 0;
        for (const fwgamg::LevelSetupTiming& level : timing.levelSetup)
            renumberSeconds += level.localityRenumberSeconds;
        EXPECT_GT(renumberSeconds, 0.0);
        std::cout << "fwGAMG locality-renumbered ordering="
            << (ordering == fwgamg::RowOrdering::Natural
                ? "natural" : "indexKahn")
            << " iterations=" << performance.iterations
            << " finalResidual=" << performance.finalResidual
            << " renumberMs=" << 1e3*renumberSeconds << '\n';
    }
}

TEST(fwGAMGSolve, FlatEdgeWiseSpmvConvergesEquivalently)
{
    gamg::GAMGControls gamgControls;
    gamgControls.relativeTolerance = 0;
    gamgControls.tolerance = 1e-8;
    gamgControls.maxIterations = 100;
    fwgamg::Controls indirectControls;
    indirectControls.gamg = gamgControls;
    indirectControls.width = 1024;
    indirectControls.spmvTraversal = fwgamg::SpmvTraversal::FlatEdgeWise;
    fwgamg::Solver indirectSolver(*problem->matrix, indirectControls);
    fwgamg::Controls rowControls = indirectControls;
    rowControls.spmvTraversal = fwgamg::SpmvTraversal::RowWise;
    fwgamg::Solver rowSolver(*problem->matrix, rowControls);

    Foam::scalarField rowSolution(problem->mesh.nCells, 0.0);
    Foam::scalarField indirectSolution(problem->mesh.nCells, 0.0);
    Foam::scalarField directSolution(problem->mesh.nCells, 0.0);
    const fwgamg::Performance row =
        rowSolver.solve(rowSolution, problem->source);
    const fwgamg::Performance indirect =
        indirectSolver.solve(indirectSolution, problem->source);
    const fwgamg::Performance direct =
        problem->fwSolver->solve(directSolution, problem->source);
    ASSERT_TRUE(indirect.converged);
    ASSERT_TRUE(direct.converged);
    EXPECT_EQ(indirect.iterations, row.iterations);
    EXPECT_EQ(direct.iterations, row.iterations);
    EXPECT_NEAR(indirect.finalResidual, row.finalResidual, 1e-10);
    EXPECT_NEAR(direct.finalResidual, row.finalResidual, 1e-10);
    EXPECT_LE(maximumDifference(indirectSolution, rowSolution), 1e-7);
    EXPECT_LE(maximumDifference(directSolution, rowSolution), 1e-7);
}

TEST(fwGAMGSolve, NativeFaceOrderDirectFlatConvergesEquivalently)
{
    gamg::GAMGControls gamgControls;
    gamgControls.relativeTolerance = 0;
    gamgControls.tolerance = 1e-8;
    gamgControls.maxIterations = 100;

    fwgamg::Controls currentControls;
    currentControls.gamg = gamgControls;
    currentControls.width = 1024;
    currentControls.spmvTraversal = fwgamg::SpmvTraversal::DirectFlatEdgeWise;
    fwgamg::Controls nativeFaceControls = currentControls;
    nativeFaceControls.spmvTraversal =
        fwgamg::SpmvTraversal::NativeFaceOrderDirectFlatEdgeWise;

    fwgamg::Solver current(*problem->matrix, currentControls);
    fwgamg::Solver nativeFace(*problem->matrix, nativeFaceControls);
    for (const fwgamg::Matrix& matrix : nativeFace.hierarchy().matrices())
    {
        EXPECT_TRUE(matrix.directEdgeOwners().empty());
        EXPECT_TRUE(matrix.directEdgeColumns().empty());
        EXPECT_TRUE(matrix.directEdgeCoeffs().empty());
        EXPECT_EQ
        (
            matrix.lduOrderDirectEdgeOwners().size(),
            static_cast<std::size_t>(matrix.nFaces())
        );
    }
    Foam::scalarField currentSolution(problem->mesh.nCells, 0.0);
    Foam::scalarField nativeFaceSolution(problem->mesh.nCells, 0.0);
    const fwgamg::Performance currentPerformance =
        current.solve(currentSolution, problem->source);
    const fwgamg::Performance nativeFacePerformance =
        nativeFace.solve(nativeFaceSolution, problem->source);

    ASSERT_TRUE(currentPerformance.converged);
    ASSERT_TRUE(nativeFacePerformance.converged);
    EXPECT_EQ(nativeFacePerformance.iterations, currentPerformance.iterations);
    EXPECT_NEAR
    (
        nativeFacePerformance.finalResidual,
        currentPerformance.finalResidual,
        1e-10
    );
    EXPECT_LE(maximumDifference(nativeFaceSolution, currentSolution), 1e-7);
}

TEST(GAMGSolve, L1EvictionIsNumericallyInvisible)
{
    gamg::GAMGControls controls;
    controls.relativeTolerance = 0;
    controls.tolerance = 1e-8;
    controls.maxIterations = 100;
    gamg::GAMGSolver gsDirect
    (
        *problem->matrix, controls, nullptr,
        gamg::GAMGSpmvTraversal::DirectFlat, false
    );
    gamg::GAMGSolver gsDirectEvicted
    (
        *problem->matrix, controls, nullptr,
        gamg::GAMGSpmvTraversal::DirectFlat, true
    );
    Foam::scalarField gsSolution(problem->mesh.nCells, 0.0);
    Foam::scalarField gsEvictedSolution(problem->mesh.nCells, 0.0);
    const gamg::GAMGSolverPerformance gsPerformance =
        gsDirect.solve(gsSolution, problem->source);
    const gamg::GAMGSolverPerformance gsEvictedPerformance =
        gsDirectEvicted.solve(gsEvictedSolution, problem->source);
    EXPECT_EQ(gsSolution, gsEvictedSolution);
    EXPECT_EQ
    (
        gsPerformance.residualHistory,
        gsEvictedPerformance.residualHistory
    );

    fwgamg::Controls ih1Controls;
    ih1Controls.gamg = controls;
    ih1Controls.width = 1024;
    fwgamg::Controls ih1EvictedControls = ih1Controls;
    ih1EvictedControls.evictL1BeforeDirectFlat = true;
    fwgamg::Solver ih1(*problem->matrix, ih1Controls);
    fwgamg::Solver ih1Evicted(*problem->matrix, ih1EvictedControls);
    Foam::scalarField ih1Solution(problem->mesh.nCells, 0.0);
    Foam::scalarField ih1EvictedSolution(problem->mesh.nCells, 0.0);
    const fwgamg::Performance ih1Performance =
        ih1.solve(ih1Solution, problem->source);
    const fwgamg::Performance ih1EvictedPerformance =
        ih1Evicted.solve(ih1EvictedSolution, problem->source);
    EXPECT_EQ(ih1Solution, ih1EvictedSolution);
    EXPECT_EQ
    (
        ih1Performance.residualHistory,
        ih1EvictedPerformance.residualHistory
    );
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
        fwgamg::Controls fwControls;
        fwControls.gamg = controls;
        fwControls.width = 1024;
        problem->fwSolver = std::make_unique<fwgamg::Solver>
        (
            *problem->matrix, fwControls
        );
        return RUN_ALL_TESTS();
    }
    catch (const std::exception& error)
    {
        std::cerr << "GAMG test setup failed: " << error.what() << '\n';
        return 2;
    }
}
