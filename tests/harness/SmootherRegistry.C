#include "SmootherRegistry.H"

#include "GaussSeidelSmoother.H"
#include "symGaussSeidelSmoother.H"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

namespace smootherTest::harness
{

void SmootherRegistry::add(SmootherVariant variant)
{
    if (variant.meta.id.empty() || !variant.run)
        throw std::runtime_error("invalid solver registration");
    for (const SmootherVariant& existing : variants_)
        if (existing.meta.id == variant.meta.id)
            throw std::runtime_error
            (
                "duplicate solver id: " + variant.meta.id
            );
    variants_.push_back(std::move(variant));
}

const std::vector<SmootherVariant>& SmootherRegistry::variants() const noexcept
{
    return variants_;
}

const SmootherVariant& SmootherRegistry::at(const std::string& id) const
{
    for (const SmootherVariant& variant : variants_)
        if (variant.meta.id == id) return variant;
    throw std::runtime_error("unknown solver variant: " + id);
}

std::vector<VariantMetadata> SmootherRegistry::metadata() const
{
    std::vector<VariantMetadata> result;
    result.reserve(variants_.size());
    for (const SmootherVariant& variant : variants_)
        result.push_back(variant.meta);
    return result;
}

SolverFunction makeBatchedSolver
(
    SolverFunction solver,
    const Foam::label sweepsPerCall
)
{
    if (sweepsPerCall < 1)
        throw std::runtime_error("sweepsPerCall must be positive");
    return [solver=std::move(solver), sweepsPerCall]
    (
        Foam::scalarField& psi,
        const Foam::label totalSweeps
    )
    {
        for (Foam::label sweep=0; sweep<totalSweeps; sweep += sweepsPerCall)
        {
            const Foam::label count = std::min
            (
                sweepsPerCall, totalSweeps - sweep
            );
            solver(psi, count);
        }
    };
}

SmootherRegistry createSmootherRegistry(Workload& workload)
{
    SmootherRegistry registry;
    const Foam::lduMatrix& matrix = *workload.matrix;
    const Foam::scalarField& source = workload.source;

    registry.add
    ({
        "Reference GS", "ReferenceGS",
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            Foam::GaussSeidelSmoother::smooth
            (
                "psi", psi, matrix, source,
                workload.interfaceCoeffs, workload.interfaces, 0, nSweeps
            );
        },
        CorrectnessPolicy::Reference, 0, true, true, 3, false,
        VariantTier::Core, {"reference"}
    });
    registry.add
    ({
        "Reference symmetric GS", "SymmetricGS",
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            Foam::symGaussSeidelSmoother::smooth
            (
                "psi", psi, matrix, source,
                workload.interfaceCoeffs, workload.interfaces, 0, nSweeps
            );
        },
        CorrectnessPolicy::ResidualReduction, 1e-12,
        true, true, 3, false,
        VariantTier::Core, {"reference", "symmetric"}
    });

    const auto addPacked =
    [&]
    (
        const std::string& name,
        const std::string& id,
        const WavefrontSchedule& schedule,
        const CorrectnessPolicy policy,
        SolverFunction function,
        const VariantTier tier,
        std::vector<std::string> tags,
        const bool benchmark = true,
        const bool parallel = false
    )
    {
        static_cast<void>(schedule);
        tags.push_back("packed");
        registry.add
        ({
            name, id, std::move(function), policy, 1e-12,
            true, benchmark, 3, parallel, tier, std::move(tags)
        });
    };

    addPacked
    (
        "Packed scalar", "PackedScalar", workload.packed,
        CorrectnessPolicy::ExactReference,
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            serialGatherSmooth(psi, source, workload.packed, nSweeps);
        },
        VariantTier::Core, {}
    );
    addPacked
    (
        "Packed scalar generic 4-way", "PackedGeneric4", workload.packed,
        CorrectnessPolicy::ExactReference,
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            serialGatherInterleavedRowsSmooth
            (
                psi, source, workload.packed, nSweeps
            );
        },
        VariantTier::Legacy, {"interleaved"}
    );
    addPacked
    (
        "Packed scalar degree-6", "PackedDegree6", workload.packed,
        CorrectnessPolicy::ExactReference,
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            serialGatherDegree6Smooth(psi, source, workload.packed, nSweeps);
        },
        VariantTier::Legacy, {"degree-6"}
    );
    addPacked
    (
        "Packed degree-6 4-way", "PackedDegree6Interleaved", workload.packed,
        CorrectnessPolicy::ExactReference,
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            serialGatherDegree6InterleavedRowsSmooth
            (
                psi, source, workload.packed, nSweeps
            );
        },
        VariantTier::Legacy, {"degree-6", "interleaved"}
    );

    for (const Foam::label distance : {4, 8, 16, 32})
    {
        const std::string suffix = std::to_string(distance);
        addPacked
        (
            "Packed prefetch first D=" + suffix,
            "PackedPrefetchFirstD" + suffix,
            workload.packed, CorrectnessPolicy::ExactReference,
            [&, distance](Foam::scalarField& psi, const Foam::label nSweeps)
            {
                serialGatherPrefetchSmooth
                (
                    psi, source, workload.packed, nSweeps, distance, 1
                );
            },
            VariantTier::Legacy, {"prefetch"}
        );
        addPacked
        (
            "Packed lightweight prefetch D=" + suffix,
            "PackedLightPrefetchD" + suffix,
            workload.packed, CorrectnessPolicy::ExactReference,
            [&, distance](Foam::scalarField& psi, const Foam::label nSweeps)
            {
                serialGatherFirstPsiPrefetchSmooth
                (
                    psi, source, workload.packed, nSweeps, distance
                );
            },
            VariantTier::Legacy, {"prefetch"}
        );
    }
    addPacked
    (
        "Packed prefetch first two D=8", "PackedPrefetchFirst2D8",
        workload.packed, CorrectnessPolicy::ExactReference,
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            serialGatherPrefetchSmooth
            (
                psi, source, workload.packed, nSweeps, 8, 2
            );
        },
        VariantTier::Legacy, {"prefetch"}
    );
    addPacked
    (
        "Locality-reordered packed scalar", "PackedLocalityReordered",
        workload.locality, CorrectnessPolicy::ToleranceReference,
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            serialGatherSmooth(psi, source, workload.locality, nSweeps);
        },
        VariantTier::Legacy, {"reordered", "locality"}
    );
    addPacked
    (
        "Median-row packed scalar", "PackedMedianRow", workload.medianRow,
        CorrectnessPolicy::ExactReference,
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            serialGatherSmooth(psi, source, workload.medianRow, nSweeps);
        },
        VariantTier::Legacy, {"reordered", "median-row"}
    );
    addPacked
    (
        "X-sorted packed scalar", "PackedXSorted", workload.xSorted,
        CorrectnessPolicy::ExactReference,
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            serialGatherSmooth(psi, source, workload.xSorted, nSweeps);
        },
        VariantTier::Legacy, {"reordered", "geometry-x"}
    );
    const auto reorderedWorkspace = std::make_shared<Foam::scalarField>
    (
        workload.initialPsi.size()
    );
    registry.add
    ({
        "Packed reordered-psi", "PackedReorderedPsi",
        [&, reorderedWorkspace]
        (
            Foam::scalarField& psi,
            const Foam::label nSweeps
        )
        {
            serialReorderedPsiSmooth
            (
                psi, *reorderedWorkspace, source, workload.packed, nSweeps
            );
        },
        CorrectnessPolicy::ExactReference, 0, true, false, 3, false,
        VariantTier::Legacy, {"packed", "reordered-psi"}
    });

    registry.add
    ({
        "Whole-row int16/fallback", "WholeRowInt16",
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            serialWholeRowInt16Smooth
            (
                psi, source, workload.wholeRowInt16, nSweeps
            );
        },
        CorrectnessPolicy::ExactReference, 0, true, true, 3, false,
        VariantTier::Legacy, {"compression", "int16"}
    });
    registry.add
    ({
        "Chained int16 delta/escape", "DeltaEscapeInt16",
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            serialDeltaEscapeInt16Smooth
            (
                psi, source, workload.deltaEscapeInt16, nSweeps
            );
        },
        CorrectnessPolicy::ExactReference, 0, true, true, 3, false,
        VariantTier::Legacy, {"compression", "int16", "delta"}
    });
    registry.add
    ({
        "Absolute uint24 columns", "AbsoluteUint24",
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            serialPackedUint24Smooth
            (
                psi, source, workload.packedUint24, nSweeps
            );
        },
        CorrectnessPolicy::ExactReference, 0, true, true, 3, false,
        VariantTier::Legacy, {"compression", "uint24"}
    });

    addPacked
    (
        "Packed AVX2 intra-row", "PackedAVX2", workload.packed,
        CorrectnessPolicy::ToleranceReference,
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            serialGatherAvx2Smooth(psi, source, workload.packed, nSweeps);
        },
        VariantTier::Legacy, {"simd", "avx2", "intra-row"}
    );
    addPacked
    (
        "Packed AVX2 across rows", "PackedAVX2Rows", workload.packed,
        CorrectnessPolicy::ExactReference,
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            serialGatherAvx2AcrossRowsSmooth
            (
                psi, source, workload.packed, nSweeps
            );
        },
        VariantTier::Legacy, {"simd", "avx2", "across-rows"}
    );
    addPacked
    (
        "Packed AVX-512 intra-row", "PackedAVX512", workload.packed,
        CorrectnessPolicy::ToleranceReference,
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            serialGatherAvx512Smooth(psi, source, workload.packed, nSweeps);
        },
        VariantTier::Legacy, {"simd", "avx512", "intra-row"}
    );
    addPacked
    (
        "Packed AVX-512 across rows", "PackedAVX512Rows", workload.packed,
        CorrectnessPolicy::ExactReference,
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            serialGatherAvx512AcrossRowsSmooth
            (
                psi, source, workload.packed, nSweeps
            );
        },
        VariantTier::Legacy, {"simd", "avx512", "across-rows"}
    );

    addPacked
    (
        "Wavefront per-level OpenMP", "PackedPerLevelOMP", workload.packed,
        CorrectnessPolicy::ExactReference,
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            int threads = 1;
            perLevelOpenMpSmooth
            (
                psi, source, workload.packed, nSweeps, threads
            );
        },
        VariantTier::Legacy, {"openmp", "per-level"}, true, true
    );
    addPacked
    (
        "Wavefront persistent OpenMP", "PackedPersistentOMP", workload.packed,
        CorrectnessPolicy::ExactReference,
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            int threads = 1;
            persistentOpenMpSmooth
            (
                psi, source, workload.packed, nSweeps, threads
            );
        },
        VariantTier::Core, {"openmp", "persistent"}, true, true
    );

    registry.add
    ({
        "Packed row-degree scalar", "PackedRowDegree",
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            serialRowDegreeSmooth(psi, source, workload.rowDegree, nSweeps);
        },
        CorrectnessPolicy::ExactReference, 0, true, true, 3, false,
        VariantTier::Experimental, {"packed", "row-degree"}
    });
    registry.add
    ({
        "Packed row-degree unrolled-8", "PackedRowDegreeUnrolled8",
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            serialRowDegreeUnrolled8Smooth
            (
                psi, source, workload.rowDegree, nSweeps
            );
        },
        CorrectnessPolicy::ExactReference, 0, true, true, 3, false,
        VariantTier::Experimental,
        {"packed", "row-degree", "unrolled-8"}
    });
    registry.add
    ({
        "Packed row-degree AVX-512 divide", "PackedRowDegreeAVX512Divide",
        [&](Foam::scalarField& psi, const Foam::label nSweeps)
        {
            serialRowDegreeAvx512DivideSmooth
            (
                psi, source, workload.rowDegree, nSweeps
            );
        },
        CorrectnessPolicy::ToleranceReference, 1e-12,
        true, true, 3, false, VariantTier::Experimental,
        {"packed", "row-degree", "simd", "avx512"}
    });

    const auto packedEdgeValues = std::make_shared<Foam::scalarField>
    (
        workload.mesh.nInternalFaces(), 0.0
    );
    registry.add
    ({
        "Hybrid packed scalar", "HybridPackedScalar",
        [&, packedEdgeValues]
        (
            Foam::scalarField& psi,
            const Foam::label nSweeps
        )
        {
            serialHybridSmooth
            (
                psi, *packedEdgeValues, source,
                workload.hybridPacked, nSweeps
            );
        },
        CorrectnessPolicy::ExactReference, 0, true, true, 3, false,
        VariantTier::Experimental, {"packed", "hybrid"}
    });

    for (const auto& entry : workload.geometrySchedules)
    {
        const Foam::label width = entry.first;
        const WavefrontSchedule* const schedule = &entry.second;
        const RowDegreeSchedule* const rowDegree =
            &workload.geometryRowDegrees.at(width);
        const BlockedRowDegreeSchedule* const blocked =
            &workload.geometryBlocked64.at(width);
        const std::string suffix = std::to_string(width);
        const VariantTier scalarTier = width == 8192
            ? VariantTier::Core : VariantTier::Experimental;
        const VariantTier rowDegreeTier = width == 2048
            ? VariantTier::Core : VariantTier::Experimental;
        const VariantTier blockedTier = width == 2048 || width == 8192
            ? VariantTier::Core : VariantTier::Experimental;
        addPacked
        (
            "Geometry-X " + suffix + " scalar", "GeometryX" + suffix,
            *schedule, CorrectnessPolicy::ExactReference,
            [&, schedule](Foam::scalarField& psi, const Foam::label nSweeps)
            {
                serialGatherSmooth(psi, source, *schedule, nSweeps);
            },
            scalarTier, {"geometry-x"}
        );
        registry.add
        ({
            "Geometry-X " + suffix + " row-degree",
            "GeometryX" + suffix + "RowDegree",
            [&, rowDegree](Foam::scalarField& psi, const Foam::label nSweeps)
            {
                serialRowDegreeSmooth(psi, source, *rowDegree, nSweeps);
            },
            CorrectnessPolicy::ExactReference, 0,
            true, true, 3, false, rowDegreeTier,
            {"geometry-x", "row-degree"}
        });
        if (width == 2048)
        {
            registry.add
            ({
                "Geometry-X 2048 row-degree unrolled-8",
                "GeometryX2048RowDegreeUnrolled8",
                [&, rowDegree]
                (
                    Foam::scalarField& psi,
                    const Foam::label nSweeps
                )
                {
                    serialRowDegreeUnrolled8Smooth
                    (
                        psi, source, *rowDegree, nSweeps
                    );
                },
                CorrectnessPolicy::ExactReference, 0,
                true, true, 3, false, VariantTier::Experimental,
                {"geometry-x", "row-degree", "unrolled-8"}
            });
            registry.add
            ({
                "Geometry-X 2048 row-degree AVX-512 divide",
                "GeometryX2048RowDegreeAVX512Divide",
                [&, rowDegree]
                (
                    Foam::scalarField& psi,
                    const Foam::label nSweeps
                )
                {
                    serialRowDegreeAvx512DivideSmooth
                    (
                        psi, source, *rowDegree, nSweeps
                    );
                },
                CorrectnessPolicy::ToleranceReference, 1e-12,
                true, true, 3, false, VariantTier::Experimental,
                {"geometry-x", "row-degree", "simd", "avx512"}
            });
        }
        registry.add
        ({
            "Geometry-X " + suffix + " blocked-64 OpenMP",
            "GeometryX" + suffix + "Blocked64OMP",
            [&, blocked](Foam::scalarField& psi, const Foam::label nSweeps)
            {
                int threads = 1;
                persistentOpenMpBlockedRowDegreeSmooth
                (
                    psi, source, *blocked, nSweeps, threads
                );
            },
            CorrectnessPolicy::ExactReference, 0,
            true, true, 3, true, blockedTier,
            {"geometry-x", "row-degree", "openmp", "persistent"}
        });
        if (width == 2048 || width == 4096 || width == 8192)
        {
            addPacked
            (
                "Geometry-X " + suffix + " generic 4-way",
                "GeometryX" + suffix + "Generic4", *schedule,
                CorrectnessPolicy::ExactReference,
                [&, schedule]
                (
                    Foam::scalarField& psi,
                    const Foam::label nSweeps
                )
                {
                    serialGatherInterleavedRowsSmooth
                    (
                        psi, source, *schedule, nSweeps
                    );
                },
                VariantTier::Legacy, {"geometry-x", "interleaved"}
            );
            addPacked
            (
                "Geometry-X " + suffix + " AVX-512 rows",
                "GeometryX" + suffix + "AVX512Rows", *schedule,
                CorrectnessPolicy::ExactReference,
                [&, schedule]
                (
                    Foam::scalarField& psi,
                    const Foam::label nSweeps
                )
                {
                    serialGatherAvx512AcrossRowsSmooth
                    (
                        psi, source, *schedule, nSweeps
                    );
                },
                VariantTier::Experimental,
                {"geometry-x", "simd", "avx512", "across-rows"}
            );
        }
        if (width == 8192)
        {
            for (const Foam::label distance : {4, 8, 16, 32})
            {
                const std::string distanceText = std::to_string(distance);
                addPacked
                (
                    "Geometry-X 8192 lightweight prefetch D=" + distanceText,
                    "GeometryX8192LightPrefetchD" + distanceText,
                    *schedule, CorrectnessPolicy::ExactReference,
                    [&, schedule, distance]
                    (
                        Foam::scalarField& psi,
                        const Foam::label nSweeps
                    )
                    {
                        serialGatherFirstPsiPrefetchSmooth
                        (
                            psi, source, *schedule, nSweeps, distance
                        );
                    },
                    VariantTier::Legacy, {"geometry-x", "prefetch"}
                );
            }
        }
    }

    for (const auto& entry : workload.geometryBlockedBySize.at(2048))
    {
        const Foam::label blockSize = entry.first;
        if (blockSize == 64) continue;
        const BlockedRowDegreeSchedule* const blocked = &entry.second;
        const std::string suffix = std::to_string(blockSize);
        registry.add
        ({
            "Geometry-X 2048 blocked-" + suffix + " OpenMP",
            "GeometryX2048Blocked" + suffix + "OMP",
            [&, blocked](Foam::scalarField& psi, const Foam::label nSweeps)
            {
                int threads = 1;
                persistentOpenMpBlockedRowDegreeSmooth
                (
                    psi, source, *blocked, nSweeps, threads
                );
            },
            CorrectnessPolicy::ExactReference, 0,
            true, true, 3, true, VariantTier::Experimental,
            {"geometry-x", "row-degree", "openmp", "persistent"}
        });
    }

    const auto geometryEdgeValues = std::make_shared<Foam::scalarField>
    (
        workload.mesh.nInternalFaces(), 0.0
    );
    registry.add
    ({
        "Hybrid Geometry-X 8192 scalar", "HybridGeometryX8192",
        [&, geometryEdgeValues]
        (
            Foam::scalarField& psi,
            const Foam::label nSweeps
        )
        {
            serialHybridSmooth
            (
                psi, *geometryEdgeValues, source,
                workload.geometryHybrid.at(8192), nSweeps
            );
        },
        CorrectnessPolicy::ExactReference, 0, true, true, 3, false,
        VariantTier::Experimental, {"geometry-x", "hybrid"}
    });
    const auto geometryParallelEdgeValues = std::make_shared<Foam::scalarField>
    (
        workload.mesh.nInternalFaces(), 0.0
    );
    registry.add
    ({
        "Hybrid Geometry-X 8192 persistent OpenMP",
        "HybridGeometryX8192PersistentOMP",
        [&, geometryParallelEdgeValues]
        (
            Foam::scalarField& psi,
            const Foam::label nSweeps
        )
        {
            int threads = 1;
            persistentOpenMpHybridSmooth
            (
                psi, *geometryParallelEdgeValues, source,
                workload.geometryHybrid.at(8192), nSweeps, threads
            );
        },
        CorrectnessPolicy::ExactReference, 0, true, true, 3, true,
        VariantTier::Experimental,
        {"geometry-x", "hybrid", "openmp", "persistent"}
    });

    for (const auto& entry : workload.indexRowDegrees)
    {
        const Foam::label width = entry.first;
        const RowDegreeSchedule* const rowDegree = &entry.second;
        const BlockedRowDegreeSchedule* const blocked =
            &workload.indexBlocked64.at(width);
        const std::string suffix = std::to_string(width);
        const VariantTier serialTier = width == 1024 || width == 2048
            ? VariantTier::Core : VariantTier::Experimental;
        const VariantTier mainWidthTier = width == 1024
            ? VariantTier::Core : VariantTier::Experimental;
        registry.add
        ({
            "Index-Kahn " + suffix + " row-degree",
            "IndexKahn" + suffix,
            [&, rowDegree](Foam::scalarField& psi, const Foam::label nSweeps)
            {
                serialRowDegreeSmooth(psi, source, *rowDegree, nSweeps);
            },
            CorrectnessPolicy::ExactReference, 0,
            true, true, 3, false, serialTier,
            {"index-kahn", "row-degree"}
        });
        registry.add
        ({
            "Index-Kahn " + suffix + " blocked-64 OpenMP",
            "IndexKahn" + suffix + "Blocked64OMP",
            [&, blocked](Foam::scalarField& psi, const Foam::label nSweeps)
            {
                int threads = 1;
                persistentOpenMpBlockedRowDegreeSmooth
                (
                    psi, source, *blocked, nSweeps, threads
                );
            },
            CorrectnessPolicy::ExactReference, 0,
            true, true, 3, true, mainWidthTier,
            {"index-kahn", "row-degree", "openmp", "persistent"}
        });
        const DirectCoefficientSchedule* const direct =
            &workload.indexDirect.at(width);
        registry.add
        ({
            "Index-Kahn " + suffix + " direct coefficients",
            "IndexKahn" + suffix + "DirectCoefficients",
            [&, direct](Foam::scalarField& psi, const Foam::label nSweeps)
            {
                serialDirectCoefficientSmooth
                (
                    psi, source, matrix, *direct, nSweeps
                );
            },
            CorrectnessPolicy::ExactReference, 0,
            true, true, 3, false, mainWidthTier,
            {"index-kahn", "direct-coefficients"}
        });
        registry.add
        ({
            "Index-Kahn " + suffix + " direct coefficients OpenMP",
            "IndexKahn" + suffix + "DirectCoefficientsOMP",
            [&, direct](Foam::scalarField& psi, const Foam::label nSweeps)
            {
                int threads = 1;
                persistentOpenMpDirectCoefficientSmooth
                (
                    psi, source, matrix, *direct, nSweeps, threads
                );
            },
            CorrectnessPolicy::ExactReference, 0,
            true, true, 3, true, mainWidthTier,
            {"index-kahn", "direct-coefficients", "openmp", "persistent"}
        });
    }

    for (const auto& entry : workload.indexWindowRowDegrees)
    {
        const Foam::label width = entry.first;
        const RowDegreeSchedule* const rowDegree = &entry.second;
        const std::string suffix = std::to_string(width);
        const VariantTier tier = width == 1024
            ? VariantTier::Core : VariantTier::Experimental;
        registry.add
        ({
            "Index-window Kahn " + suffix + " row-degree",
            "IndexWindowKahn" + suffix,
            [&, rowDegree](Foam::scalarField& psi, const Foam::label nSweeps)
            {
                serialRowDegreeSmooth(psi, source, *rowDegree, nSweeps);
            },
            CorrectnessPolicy::ExactReference, 0,
            true, true, 3, false, tier,
            {"index-window-kahn", "row-degree"}
        });
    }

    return registry;
}

} // namespace smootherTest::harness
