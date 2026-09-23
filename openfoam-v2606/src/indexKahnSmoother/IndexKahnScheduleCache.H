#ifndef Foam_IndexKahnScheduleCache_H
#define Foam_IndexKahnScheduleCache_H

#include "lduAddressing.H"
#include "typeInfo.H"

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace Foam
{

class lduMesh;

// Contains topology and addressing only. No matrix coefficient or field data.
struct IndexKahnSchedule
{
    label width = 0;
    std::vector<label> levelStarts;
    std::vector<label> waveCells;
    std::vector<label> cols;
    std::vector<label> faceIds;
    std::vector<std::uint8_t> degrees;
    std::vector<std::uint8_t> incomingDegrees;
};


class IndexKahnScheduleCache
{
    const lduAddressing& addressing_;

    // Extend this key when another parameter changes schedule generation.
    struct ScheduleKey
    {
        label width;

        bool operator<(const ScheduleKey& other) const noexcept
        {
            return width < other.width;
        }
    };

    mutable std::map<ScheduleKey, std::unique_ptr<IndexKahnSchedule>> schedules_;

public:

    TypeName("IndexKahnScheduleCache");

    explicit IndexKahnScheduleCache(const lduAddressing& addressing);

    // Fine fvMesh and coarse lduPrimitiveMesh own their respective caches.
    static const IndexKahnScheduleCache& New(const lduMesh& mesh);

    const IndexKahnSchedule& getOrCreate
    (
        label width,
        bool* scheduleBuilt = nullptr
    ) const;

    virtual ~IndexKahnScheduleCache() = default;
};

} // namespace Foam

#endif
