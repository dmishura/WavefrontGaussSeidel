#ifndef SmootherTest_Workload_H
#define SmootherTest_Workload_H

#include "ScheduleBuilder.H"

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace smootherTest::harness
{

struct Workload
{
    std::string name;
    std::string id;
    std::string meshDirectory;
    std::vector<std::string> labels;

    PolyMeshTopology mesh;
    std::unique_ptr<Foam::lduMatrix> matrix;
    Foam::scalarField exact;
    Foam::scalarField source;
    Foam::scalarField initialPsi;
    Foam::FieldField<Foam::Field, Foam::scalar> interfaceCoeffs;
    Foam::lduInterfaceFieldPtrsList interfaces;

    WavefrontStatistics minimalStatistics;
    WavefrontSchedule packed;
    WavefrontSchedule locality;
    WavefrontSchedule medianRow;
    WavefrontSchedule xSorted;
    WholeRowInt16Schedule wholeRowInt16;
    DeltaEscapeInt16Schedule deltaEscapeInt16;
    PackedUint24Schedule packedUint24;
    RowDegreeSchedule rowDegree;
    BlockedRowDegreeSchedule blockedRowDegree64;
    HybridWavefrontSchedule hybridPacked;

    std::map<Foam::label, WavefrontStatistics> geometryStatistics;
    std::map<Foam::label, WavefrontSchedule> geometrySchedules;
    std::map<Foam::label, RowDegreeSchedule> geometryRowDegrees;
    std::map<Foam::label, BlockedRowDegreeSchedule> geometryBlocked64;
    std::map<Foam::label, std::map<Foam::label, BlockedRowDegreeSchedule>>
        geometryBlockedBySize;
    std::map<Foam::label, HybridWavefrontSchedule> geometryHybrid;

    std::map<Foam::label, WavefrontStatistics> indexStatistics;
    std::map<Foam::label, WavefrontSchedule> indexSchedules;
    std::map<Foam::label, RowDegreeSchedule> indexRowDegrees;
    std::map<Foam::label, BlockedRowDegreeSchedule> indexBlocked64;
    std::map<Foam::label, DirectCoefficientSchedule> indexDirect;

    std::map<Foam::label, WavefrontStatistics> indexWindowStatistics;
    std::map<Foam::label, WavefrontSchedule> indexWindowSchedules;
    std::map<Foam::label, RowDegreeSchedule> indexWindowRowDegrees;

    Foam::scalar preprocessingSeconds = 0;
    bool dependenciesValid = false;

    Workload() = default;
    Workload(const Workload&) = delete;
    Workload& operator=(const Workload&) = delete;
    Workload(Workload&&) = delete;
    Workload& operator=(Workload&&) = delete;
};

using WorkloadFactory = std::function<std::unique_ptr<Workload>()>;

struct WorkloadDefinition
{
    std::string name;
    std::string id;
    std::vector<std::string> labels;
    WorkloadFactory create;
};

class WorkloadRegistry
{
    std::vector<WorkloadDefinition> definitions_;

public:
    void add(WorkloadDefinition definition);
    const std::vector<WorkloadDefinition>& definitions() const noexcept;
    std::unique_ptr<Workload> create(const std::string& id) const;
};

std::unique_ptr<Workload> createMeshWorkload
(
    const std::string& name,
    const std::string& id,
    const std::string& meshDirectory,
    std::vector<std::string> labels
);

WorkloadRegistry createWorkloadRegistry
(
    const std::string& motorBikeMeshDirectory
);

} // namespace smootherTest::harness

#endif
