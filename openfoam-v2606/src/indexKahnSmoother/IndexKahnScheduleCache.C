#include "IndexKahnScheduleCache.H"

#include "IOstreams.H"
#include "MeshObject.H"
#include "fvMesh.H"
#include "lduAddressing.H"
#include "lduPrimitiveMesh.H"
#include "error.H"

#include <algorithm>
#include <limits>
#include <set>

namespace Foam
{
    defineTypeNameAndDebug(IndexKahnScheduleCache, 0);
namespace
{

std::unique_ptr<Foam::IndexKahnSchedule> buildIndexKahnSchedule
(
    const Foam::lduAddressing& addr,
    const Foam::label width
)
{
    using Foam::label;

    const label nCells = addr.size();
    const Foam::labelUList& owners = addr.lowerAddr();
    const Foam::labelUList& neighbours = addr.upperAddr();
    const label nFaces = owners.size();
    if (neighbours.size() != nFaces || nFaces > std::numeric_limits<label>::max()/2)
    {
        FatalErrorInFunction
            << "invalid or oversized LDU addressing" << exit(FatalError);
    }

    const Foam::labelUList& ownerStarts = addr.ownerStartAddr();
    if
    (
        ownerStarts.size() != nCells + 1
     || ownerStarts[0] != 0
     || ownerStarts[nCells] != nFaces
    )
    {
        FatalErrorInFunction
            << "invalid LDU owner-start addressing" << exit(FatalError);
    }

    // The standalone Index-Kahn scheduler uses owner -> neighbour edges.
    // Normal OpenFOAM LDU addressing orders every edge lower cell first.
    std::vector<label> predecessors(nCells, 0);
    for (label face=0; face<nFaces; ++face)
    {
        const label owner = owners[face];
        const label neighbour = neighbours[face];
        if (owner < 0 || owner >= neighbour || neighbour >= nCells)
        {
            FatalErrorInFunction
                << "Index-Kahn requires lowerAddr[face] < upperAddr[face]; "
                << "invalid face " << face << exit(FatalError);
        }
        ++predecessors[neighbour];
    }

    std::set<label> ready;
    for (label cell=0; cell<nCells; ++cell)
    {
        if (predecessors[cell] == 0) ready.insert(cell);
    }

    auto result = std::make_unique<Foam::IndexKahnSchedule>();
    result->width = width;
    result->levelStarts.push_back(0);
    result->waveCells.reserve(nCells);
    std::vector<label> levels(nCells, -1);
    std::vector<label> current;
    current.reserve(static_cast<std::size_t>(std::min(width, nCells)));

    while (!ready.empty())
    {
        current.clear();
        // Snapshot the ready set: newly unlocked cells enter the next level.
        while (!ready.empty() && current.size() < std::size_t(width))
        {
            const auto it = ready.begin();
            current.push_back(*it);
            ready.erase(it);
        }

        const label level = static_cast<label>(result->levelStarts.size() - 1);
        for (const label cell : current)
        {
            if (levels[cell] != -1 || predecessors[cell] != 0)
            {
                FatalErrorInFunction
                    << "invalid Index-Kahn ready set" << exit(FatalError);
            }
            levels[cell] = level;
            result->waveCells.push_back(cell);
        }
        result->levelStarts.push_back
        (
            static_cast<label>(result->waveCells.size())
        );

        for (const label cell : current)
        {
            for (label face=ownerStarts[cell]; face<ownerStarts[cell + 1]; ++face)
            {
                const label neighbour = neighbours[face];
                if (--predecessors[neighbour] == 0) ready.insert(neighbour);
            }
        }
    }

    if (result->waveCells.size() != std::size_t(nCells))
    {
        FatalErrorInFunction
            << "Index-Kahn schedule does not cover all cells" << exit(FatalError);
    }
    for (label face=0; face<nFaces; ++face)
    {
        if (levels[owners[face]] >= levels[neighbours[face]])
        {
            FatalErrorInFunction
                << "Index-Kahn dependency validation failed at face "
                << face << exit(FatalError);
        }
    }

    // losort retains the original face order within each upper cell. This
    // matches the reference GS lower-side scatter arithmetic order.
    const Foam::labelUList& incomingFaces = addr.losortAddr();
    const Foam::labelUList& incomingStarts = addr.losortStartAddr();
    if
    (
        incomingFaces.size() != nFaces
     || incomingStarts.size() != nCells + 1
     || incomingStarts[0] != 0
     || incomingStarts[nCells] != nFaces
    )
    {
        FatalErrorInFunction
            << "invalid LDU losort addressing" << exit(FatalError);
    }

    result->cols.reserve(2*std::size_t(nFaces));
    result->faceIds.reserve(2*std::size_t(nFaces));
    result->degrees.reserve(nCells);
    result->incomingDegrees.reserve(nCells);
    for (const label cell : result->waveCells)
    {
        const label incoming = incomingStarts[cell + 1] - incomingStarts[cell];
        const label outgoing = ownerStarts[cell + 1] - ownerStarts[cell];
        const label degree = incoming + outgoing;
        if (incoming < 0 || outgoing < 0
         || degree > std::numeric_limits<std::uint8_t>::max())
        {
            FatalErrorInFunction
                << "cell " << cell << " has degree " << degree
                << ", which does not fit uint8_t" << exit(FatalError);
        }
        result->degrees.push_back(static_cast<std::uint8_t>(degree));
        result->incomingDegrees.push_back(static_cast<std::uint8_t>(incoming));

        for (label in=incomingStarts[cell]; in<incomingStarts[cell + 1]; ++in)
        {
            const label face = incomingFaces[in];
            if (face < 0 || face >= nFaces || neighbours[face] != cell)
            {
                FatalErrorInFunction
                    << "invalid LDU incoming face for cell " << cell
                    << exit(FatalError);
            }
            result->cols.push_back(owners[face]);
            result->faceIds.push_back(face);
        }
        for (label face=ownerStarts[cell]; face<ownerStarts[cell + 1]; ++face)
        {
            result->cols.push_back(neighbours[face]);
            result->faceIds.push_back(face);
        }
    }
    if (result->cols.size() != 2*std::size_t(nFaces))
    {
        FatalErrorInFunction
            << "incomplete Index-Kahn contribution mapping" << exit(FatalError);
    }
    return result;
}

} // namespace
} // namespace Foam


Foam::IndexKahnScheduleCache::IndexKahnScheduleCache
(
    const lduAddressing& addressing
)
:
    addressing_(addressing)
{}


namespace Foam
{

// Retain the original topology-aware fvMesh ownership for finest-level
// matrices and ordinary smoothSolver fields.
class IndexKahnFvMeshCache
:
    public MeshObject<fvMesh, TopologicalMeshObject, IndexKahnFvMeshCache>
{
    IndexKahnScheduleCache cache_;

public:

    TypeName("IndexKahnFvMeshCache");

    explicit IndexKahnFvMeshCache(const fvMesh& mesh)
    :
        MeshObject<fvMesh, TopologicalMeshObject, IndexKahnFvMeshCache>(mesh),
        cache_(mesh.lduAddr())
    {}

    const IndexKahnScheduleCache& cache() const noexcept { return cache_; }
};

defineTypeNameAndDebug(IndexKahnFvMeshCache, 0);

} // namespace Foam


const Foam::IndexKahnScheduleCache& Foam::IndexKahnScheduleCache::New
(
    const lduMesh& mesh
)
{
    if (const fvMesh* fvmesh = isA<fvMesh>(mesh))
    {
        return IndexKahnFvMeshCache::New(*fvmesh).cache();
    }
    if (const lduPrimitiveMesh* coarse = isA<lduPrimitiveMesh>(mesh))
    {
        const auto& stored = coarse->indexKahnScheduleCache();
        if (stored)
        {
            return *std::static_pointer_cast<IndexKahnScheduleCache>(stored);
        }
        auto cache = std::make_shared<IndexKahnScheduleCache>(mesh.lduAddr());
        coarse->setIndexKahnScheduleCache(cache);
        return *cache;
    }
    FatalErrorInFunction
        << "indexKahn requires fvMesh or lduPrimitiveMesh addressing"
        << exit(FatalError);
    return IndexKahnFvMeshCache::New(refCast<const fvMesh>(mesh)).cache();
}


const Foam::IndexKahnSchedule& Foam::IndexKahnScheduleCache::getOrCreate
(
    const label width,
    bool* const scheduleBuilt
) const
{
    if (width < 1)
    {
        FatalErrorInFunction
            << "Index-Kahn width must be positive, got " << width
            << exit(FatalError);
    }

    const ScheduleKey key{width};
    const auto found = schedules_.find(key);
    if (found != schedules_.end())
    {
        if (scheduleBuilt) *scheduleBuilt = false;
        if (debug)
        {
            Pout
                << "IndexKahn: reusing schedule width=" << width
                << " cells=" << found->second->waveCells.size() << endl;
        }
        return *found->second;
    }

    if (debug)
    {
        Pout
            << "IndexKahn: building schedule width=" << width
            << " cells=" << addressing_.size() << endl;
    }
    auto schedule = buildIndexKahnSchedule(addressing_, width);
    if (scheduleBuilt) *scheduleBuilt = true;
    const IndexKahnSchedule& result = *schedule;
    schedules_.emplace(key, std::move(schedule));
    return result;
}
