#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include "Workload.H"

#include <algorithm>
#include <array>
#include <cmath>
#include <omp.h>
#include <stdexcept>
#include <utility>

namespace smootherTest::harness
{

void WorkloadRegistry::add(WorkloadDefinition definition)
{
    for (const WorkloadDefinition& existing : definitions_)
        if (existing.id == definition.id)
            throw std::runtime_error("duplicate workload id: " + definition.id);
    definitions_.push_back(std::move(definition));
}

const std::vector<WorkloadDefinition>& WorkloadRegistry::definitions() const noexcept
{
    return definitions_;
}

std::unique_ptr<Workload> WorkloadRegistry::create(const std::string& id) const
{
    for (const WorkloadDefinition& definition : definitions_)
        if (definition.id == id) return definition.create();
    throw std::runtime_error("unknown workload: " + id);
}

std::unique_ptr<Workload> createMeshWorkload
(
    const std::string& name,
    const std::string& id,
    const std::string& meshDirectory,
    std::vector<std::string> labels
)
{
    auto workload = std::make_unique<Workload>();
    workload->name = name;
    workload->id = id;
    workload->meshDirectory = meshDirectory;
    workload->labels = std::move(labels);
    workload->mesh = PolyMeshReader::read(meshDirectory);
    workload->matrix = std::make_unique<Foam::lduMatrix>
    (
        makeMatrix(workload->mesh)
    );
    workload->exact.resize(workload->mesh.nCells);
    for (Foam::label cell=0; cell<Foam::label(workload->exact.size()); ++cell)
    {
        const Foam::scalar position = Foam::scalar(cell)
            /Foam::scalar(workload->exact.size() - 1);
        workload->exact[cell] = 1.0 + 0.25*std::sin(2.0*M_PI*position)
            + 0.1*std::cos(10.0*M_PI*position);
    }
    workload->source = multiply(*workload->matrix, workload->exact);
    workload->initialPsi = Foam::scalarField(workload->mesh.nCells, 0.0);

    const auto begin = std::chrono::steady_clock::now();
    const auto lexicographicPackedBegin = std::chrono::steady_clock::now();
    workload->lexicographicPacked = makeLexicographicPackedSchedule
    (
        *workload->matrix
    );
    workload->lexicographicPackedPreprocessingSeconds =
        std::chrono::duration<Foam::scalar>
        (
            std::chrono::steady_clock::now() - lexicographicPackedBegin
        ).count();
    workload->minimalStatistics = constructMinimalWavefronts(workload->mesh);
    workload->packed = makeWavefrontSchedule
    (
        workload->mesh, *workload->matrix, workload->minimalStatistics
    );
    workload->locality = makeLocalitySchedule(workload->packed);
    workload->medianRow = makeMedianRowSchedule(workload->packed);
    workload->xSorted = makeXSortedSchedule
    (
        workload->packed, workload->mesh.cellCentreX
    );
    workload->wholeRowInt16 = makeWholeRowInt16Schedule(workload->packed);
    workload->deltaEscapeInt16 = makeDeltaEscapeInt16Schedule(workload->packed);
    workload->packedUint24 = makePackedUint24Schedule(workload->packed);
    workload->rowDegree = makeRowDegreeSchedule(workload->packed);
    workload->blockedRowDegree64 = makeBlockedRowDegreeSchedule
    (
        workload->packed, 64
    );
    workload->hybridPacked = makeHybridSchedule
    (
        workload->mesh, *workload->matrix, workload->packed
    );

    constexpr Foam::label xSlabs = 256;
    constexpr std::array<Foam::label, 7> geometryWidths
    {{256, 512, 1024, 2048, 4096, 8192, 16384}};
    for (const Foam::label width : geometryWidths)
    {
        workload->geometryStatistics.emplace
        (
            width,
            constructGeometryWavefronts(workload->mesh, xSlabs, width)
        );
        workload->geometrySchedules.emplace
        (
            width,
            makeWavefrontSchedule
            (
                workload->mesh,
                *workload->matrix,
                workload->geometryStatistics.at(width)
            )
        );
        workload->geometryRowDegrees.emplace
        (
            width,
            makeRowDegreeSchedule(workload->geometrySchedules.at(width))
        );
        workload->geometryBlocked64.emplace
        (
            width,
            makeBlockedRowDegreeSchedule
            (
                workload->geometrySchedules.at(width), 64
            )
        );
    }
    for (const Foam::label blockSize : {32, 64, 128, 256})
        workload->geometryBlockedBySize[2048].emplace
        (
            blockSize,
            makeBlockedRowDegreeSchedule
            (
                workload->geometrySchedules.at(2048), blockSize
            )
        );
    workload->geometryHybrid.emplace
    (
        8192,
        makeHybridSchedule
        (
            workload->mesh, *workload->matrix,
            workload->geometrySchedules.at(8192)
        )
    );

    constexpr std::array<Foam::label, 5> indexWidths
    {{512, 1024, 2048, 4096, 8192}};
    for (const Foam::label width : indexWidths)
    {
        workload->indexStatistics.emplace
        (
            width,
            constructIndexWavefronts
            (
                workload->mesh, *workload->matrix, width, false
            )
        );
        workload->indexSchedules.emplace
        (
            width,
            makeWavefrontSchedule
            (
                workload->mesh,
                *workload->matrix,
                workload->indexStatistics.at(width)
            )
        );
        workload->indexRowDegrees.emplace
        (
            width,
            makeRowDegreeSchedule(workload->indexSchedules.at(width))
        );
        workload->indexBlocked64.emplace
        (
            width,
            makeBlockedRowDegreeSchedule(workload->indexSchedules.at(width), 64)
        );
        workload->indexDirect.emplace
        (
            width,
            makeDirectCoefficientSchedule
            (
                workload->mesh, *workload->matrix,
                workload->indexStatistics.at(width), 64
            )
        );
    }

    const auto adaptiveBegin = std::chrono::steady_clock::now();
    const int configuredThreads = std::max(1, omp_get_max_threads());
    const BlockedRowDegreeSchedule& ih1Blocked =
        workload->indexBlocked64.at(1024);
    for (const Foam::label threshold : {0, 16, 32, 64, 128, 256, 512, 1024, 2048})
    {
        workload->index1024AdaptiveThresholds.emplace
        (
            threshold,
            makeAdaptiveBlockedRowDegreeSchedule
            (
                ih1Blocked, configuredThreads, threshold, threshold
            )
        );
    }
    for (const auto& thresholds : {
            std::pair<Foam::label, Foam::label>{32, 128},
            {64, 256},
            {128, 512}
        })
    {
        workload->index1024AdaptiveClasses.emplace
        (
            thresholds,
            makeAdaptiveBlockedRowDegreeSchedule
            (
                ih1Blocked, configuredThreads,
                thresholds.first, thresholds.second
            )
        );
    }
    workload->adaptiveIh1PreprocessingSeconds =
        std::chrono::duration<Foam::scalar>
        (
            std::chrono::steady_clock::now() - adaptiveBegin
        ).count();
    for (const auto& entry : workload->index1024AdaptiveThresholds)
        workload->adaptiveIh1MetadataBytes +=
            entry.second.activeThreads.size()*sizeof(std::uint8_t);
    for (const auto& entry : workload->index1024AdaptiveClasses)
        workload->adaptiveIh1MetadataBytes +=
            entry.second.activeThreads.size()*sizeof(std::uint8_t);

    for (const Foam::label width : {1024, 2048, 8192})
    {
        workload->indexWindowStatistics.emplace
        (
            width,
            constructIndexWavefronts
            (
                workload->mesh, *workload->matrix, width, true
            )
        );
        workload->indexWindowSchedules.emplace
        (
            width,
            makeWavefrontSchedule
            (
                workload->mesh, *workload->matrix,
                workload->indexWindowStatistics.at(width)
            )
        );
        workload->indexWindowRowDegrees.emplace
        (
            width,
            makeRowDegreeSchedule(workload->indexWindowSchedules.at(width))
        );
    }

    workload->dependenciesValid = validateDependencies
    (
        workload->mesh, workload->minimalStatistics
    );
    for (const auto& entry : workload->geometryStatistics)
        workload->dependenciesValid = workload->dependenciesValid
            && validateDependencies(workload->mesh, entry.second);
    for (const auto& entry : workload->indexStatistics)
        workload->dependenciesValid = workload->dependenciesValid
            && validateDependencies(workload->mesh, entry.second);
    for (const auto& entry : workload->indexWindowStatistics)
        workload->dependenciesValid = workload->dependenciesValid
            && validateDependencies(workload->mesh, entry.second);
    workload->preprocessingSeconds = std::chrono::duration<Foam::scalar>
    (
        std::chrono::steady_clock::now() - begin
    ).count();
    return workload;
}

WorkloadRegistry createWorkloadRegistry
(
    const std::string& motorBikeMeshDirectory
)
{
    WorkloadRegistry registry;
    registry.add
    ({
        "motorBike",
        "MotorBike",
        {"motorBike", "correctness", "benchmark"},
        [motorBikeMeshDirectory]
        {
            return createMeshWorkload
            (
                "motorBike", "MotorBike", motorBikeMeshDirectory,
                {"motorBike", "correctness", "benchmark"}
            );
        }
    });
    return registry;
}

} // namespace smootherTest::harness
