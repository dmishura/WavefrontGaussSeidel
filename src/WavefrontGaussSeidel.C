#include "WavefrontGaussSeidel.H"

#include <algorithm>
#include <deque>
#include <limits>
#include <omp.h>
#include <set>
#include <stdexcept>

namespace smootherTest
{

using Foam::label;
using Foam::scalar;

IndexKahnOrdering makeIndexKahnOrdering
(
    const label nCells,
    const Foam::labelField& ownerStarts,
    const Foam::labelField& upperAddr,
    const label targetWidth
)
{
    if (targetWidth < 1) throw std::runtime_error("invalid Index-Kahn width");
    if
    (
        ownerStarts.size() != static_cast<std::size_t>(nCells + 1)
     || ownerStarts.back() != static_cast<label>(upperAddr.size())
    ) throw std::runtime_error("invalid Index-Kahn LDU addressing");

    std::vector<label> predecessors(nCells, 0);
    for (const label cell : upperAddr) ++predecessors[cell];
    std::set<label> ready;
    for (label cell=0; cell<nCells; ++cell)
        if (predecessors[cell] == 0) ready.insert(cell);

    IndexKahnOrdering result;
    result.levelStarts.push_back(0);
    result.newToOld.reserve(nCells);
    while (!ready.empty())
    {
        std::vector<label> current;
        current.reserve(targetWidth);
        auto position = ready.begin();
        while (!ready.empty() && current.size()<static_cast<std::size_t>(targetWidth))
        {
            current.push_back(*position);
            position = ready.erase(position);
        }
        result.newToOld.insert
        (
            result.newToOld.end(), current.begin(), current.end()
        );
        result.levelStarts.push_back(result.newToOld.size());
        for (const label cell : current)
            for (label face=ownerStarts[cell]; face<ownerStarts[cell + 1]; ++face)
                if (--predecessors[upperAddr[face]] == 0)
                    ready.insert(upperAddr[face]);
    }
    if (result.newToOld.size() != static_cast<std::size_t>(nCells))
        throw std::runtime_error("Index-Kahn scheduler did not cover all cells");
    result.oldToNew.resize(nCells);
    for (label row=0; row<nCells; ++row)
        result.oldToNew[result.newToOld[row]] = row;
    return result;
}

IndexKahnOrdering makeIndexKahnOrderingReleaseDiagnostic
(
    const label nCells,
    const Foam::labelField& ownerStarts,
    const Foam::labelField& upperAddr,
    const label targetWidth,
    std::vector<ReleaseImpactChunkDiagnostic>* const diagnostics
)
{
    if (targetWidth < 1) throw std::runtime_error("invalid Index-Kahn width");
    if
    (
        ownerStarts.size() != static_cast<std::size_t>(nCells + 1)
     || ownerStarts.back() != static_cast<label>(upperAddr.size())
    ) throw std::runtime_error("invalid Index-Kahn LDU addressing");

    if (diagnostics) diagnostics->clear();
    std::vector<label> predecessors(nCells, 0);
    for (const label cell : upperAddr) ++predecessors[cell];
    std::set<label> ready;
    for (label cell=0; cell<nCells; ++cell)
        if (predecessors[cell] == 0) ready.insert(cell);

    IndexKahnOrdering result;
    result.levelStarts.push_back(0);
    result.newToOld.reserve(nCells);
    while (!ready.empty())
    {
        std::vector<label> current;
        current.reserve(targetWidth);
        auto position = ready.begin();
        while
        (
            !ready.empty()
         && current.size() < static_cast<std::size_t>(targetWidth)
        )
        {
            current.push_back(*position);
            position = ready.erase(position);
        }
        result.newToOld.insert
        (
            result.newToOld.end(), current.begin(), current.end()
        );
        result.levelStarts.push_back(result.newToOld.size());

        ReleaseImpactChunkDiagnostic diagnostic;
        diagnostic.chosenChunkSize = static_cast<label>(current.size());
        diagnostic.cut = current.back();
        const bool haveHeadBefore = !ready.empty();
        if (haveHeadBefore) diagnostic.headBefore = *ready.begin();
        for (const label cell : current)
        {
            for
            (
                label face=ownerStarts[cell];
                face<ownerStarts[cell + 1];
                ++face
            )
            {
                const label successor = upperAddr[face];
                if (--predecessors[successor] == 0)
                {
                    ++diagnostic.releasedCount;
                    diagnostic.releasedBelowCut += successor < diagnostic.cut;
                    diagnostic.releasedBeforeHead +=
                        haveHeadBefore && successor < diagnostic.headBefore;
                    ready.insert(successor);
                }
            }
        }
        if (!ready.empty()) diagnostic.headAfter = *ready.begin();
        if
        (
            haveHeadBefore
         && diagnostic.headAfter >= 0
         && diagnostic.headAfter < diagnostic.headBefore
        ) diagnostic.headBackstep = diagnostic.headBefore - diagnostic.headAfter;
        diagnostic.headChanged = diagnostic.releasedBeforeHead > 0;
        if (diagnostics) diagnostics->push_back(diagnostic);
    }
    if (result.newToOld.size() != static_cast<std::size_t>(nCells))
        throw std::runtime_error("Index-Kahn scheduler did not cover all cells");
    result.oldToNew.resize(nCells);
    for (label row=0; row<nCells; ++row)
        result.oldToNew[result.newToOld[row]] = row;
    return result;
}

IndexKahnOrdering makeIndexKahnOrderingOriented
(
    const label nCells,
    const Foam::labelField& ownerStarts,
    const Foam::labelField& upperAddr,
    const label targetWidth
)
{
    if (targetWidth < 1)
        throw std::runtime_error("invalid oriented Index-Kahn width");
    if
    (
        ownerStarts.size() != static_cast<std::size_t>(nCells + 1)
     || ownerStarts.back() != static_cast<label>(upperAddr.size())
    ) throw std::runtime_error("invalid oriented Index-Kahn LDU addressing");

    std::vector<label> predecessors(nCells, 0);
    for (const label cell : upperAddr) ++predecessors[cell];
    std::set<label> ready;
    for (label cell=0; cell<nCells; ++cell)
        if (predecessors[cell] == 0) ready.insert(cell);

    IndexKahnOrdering result;
    result.levelStarts.push_back(0);
    result.newToOld.reserve(nCells);
    bool havePrevious = false;
    label previous = 0;
    while (!ready.empty())
    {
        // This is deliberately identical to the production IH1W membership
        // selection: take the first targetWidth entries from the same sorted
        // ready set before releasing any successors.
        std::vector<label> current;
        current.reserve(targetWidth);
        auto position = ready.begin();
        while
        (
            !ready.empty()
         && current.size() < static_cast<std::size_t>(targetWidth)
        )
        {
            current.push_back(*position);
            position = ready.erase(position);
        }

        bool reverse = false;
        if (havePrevious)
        {
            const auto distance = [](const label a, const label b)
            {
                return a > b ? a - b : b - a;
            };
            reverse = distance(previous, current.back())
                    < distance(previous, current.front());
        }
        if (reverse)
            result.newToOld.insert
            (
                result.newToOld.end(), current.rbegin(), current.rend()
            );
        else
            result.newToOld.insert
            (
                result.newToOld.end(), current.begin(), current.end()
            );
        previous = result.newToOld.back();
        havePrevious = true;
        result.levelStarts.push_back(result.newToOld.size());

        // Release only after the complete, immutable chunk has been appended.
        // Use the original ascending membership order; orientation cannot
        // influence dependency processing or the following ready set.
        for (const label cell : current)
            for (label face=ownerStarts[cell]; face<ownerStarts[cell + 1]; ++face)
                if (--predecessors[upperAddr[face]] == 0)
                    ready.insert(upperAddr[face]);
    }
    if (result.newToOld.size() != static_cast<std::size_t>(nCells))
        throw std::runtime_error
        (
            "oriented Index-Kahn scheduler did not cover all cells"
        );
    result.oldToNew.resize(nCells);
    for (label row=0; row<nCells; ++row)
        result.oldToNew[result.newToOld[row]] = row;
    return result;
}

IndexKahnOrdering makeIndexKahnOrderingSlidingWindow
(
    const label nCells,
    const Foam::labelField& ownerStarts,
    const Foam::labelField& upperAddr,
    const label activeWidth,
    const label advanceStep
)
{
    if (activeWidth < 1 || advanceStep < 1 || advanceStep > activeWidth)
        throw std::runtime_error("invalid sliding-window Index-Kahn controls");
    if
    (
        ownerStarts.size() != static_cast<std::size_t>(nCells + 1)
     || ownerStarts.back() != static_cast<label>(upperAddr.size())
    ) throw std::runtime_error
      (
          "invalid sliding-window Index-Kahn LDU addressing"
      );

    std::vector<label> predecessors(nCells, 0);
    for (const label cell : upperAddr) ++predecessors[cell];
    std::set<label> ready;
    for (label cell=0; cell<nCells; ++cell)
        if (predecessors[cell] == 0) ready.insert(cell);

    std::deque<label> active;
    const auto refill = [&]
    {
        while
        (
            active.size() < static_cast<std::size_t>(activeWidth)
         && !ready.empty()
        )
        {
            const auto first = ready.begin();
            active.push_back(*first);
            ready.erase(first);
        }
    };

    IndexKahnOrdering result;
    result.levelStarts.push_back(0);
    result.newToOld.reserve(nCells);
    refill();
    while (!active.empty())
    {
        const std::size_t count = std::min
        (
            active.size(), static_cast<std::size_t>(advanceStep)
        );
        std::vector<label> current;
        current.reserve(count);
        for (std::size_t i=0; i<count; ++i)
        {
            current.push_back(active.front());
            active.pop_front();
        }
        result.newToOld.insert
        (
            result.newToOld.end(), current.begin(), current.end()
        );
        result.levelStarts.push_back(result.newToOld.size());

        // Cells released by this step become eligible only for a later step.
        for (const label cell : current)
            for (label face=ownerStarts[cell]; face<ownerStarts[cell + 1]; ++face)
                if (--predecessors[upperAddr[face]] == 0)
                    ready.insert(upperAddr[face]);
        refill();
    }
    if (!ready.empty() || result.newToOld.size() != static_cast<std::size_t>(nCells))
        throw std::runtime_error
        (
            "sliding-window Index-Kahn scheduler did not cover all cells"
        );
    result.oldToNew.resize(nCells);
    for (label row=0; row<nCells; ++row)
        result.oldToNew[result.newToOld[row]] = row;
    return result;
}

IndexKahnOrdering makeIndexKahnOrderingAdaptiveGap
(
    const label nCells,
    const Foam::labelField& ownerStarts,
    const Foam::labelField& upperAddr,
    const label minWidth,
    const label targetWidth,
    const label maxWidth
)
{
    if
    (
        minWidth < 1
     || targetWidth < minWidth
     || targetWidth > maxWidth
    ) throw std::runtime_error("invalid adaptive-gap Index-Kahn controls");
    if
    (
        ownerStarts.size() != static_cast<std::size_t>(nCells + 1)
     || ownerStarts.back() != static_cast<label>(upperAddr.size())
    ) throw std::runtime_error
      (
          "invalid adaptive-gap Index-Kahn LDU addressing"
      );

    std::vector<label> predecessors(nCells, 0);
    for (const label cell : upperAddr) ++predecessors[cell];
    std::set<label> ready;
    for (label cell=0; cell<nCells; ++cell)
        if (predecessors[cell] == 0) ready.insert(cell);

    IndexKahnOrdering result;
    result.levelStarts.push_back(0);
    result.newToOld.reserve(nCells);
    while (!ready.empty())
    {
        const std::size_t readySize = ready.size();
        std::size_t chunkSize = std::min
        (
            readySize, static_cast<std::size_t>(maxWidth)
        );
        if (readySize > static_cast<std::size_t>(minWidth))
        {
            const std::size_t largestCandidateCut = std::min
            (
                static_cast<std::size_t>(maxWidth), readySize - 1
            );
            std::vector<label> candidates;
            candidates.reserve(largestCandidateCut + 1);
            auto position = ready.begin();
            for
            (
                std::size_t index=0;
                index<=largestCandidateCut;
                ++index, ++position
            ) candidates.push_back(*position);

            chunkSize = static_cast<std::size_t>(minWidth);
            label largestGap = candidates[chunkSize] - candidates[chunkSize - 1];
            for
            (
                std::size_t cut=chunkSize + 1;
                cut<=largestCandidateCut;
                ++cut
            )
            {
                const label gap = candidates[cut] - candidates[cut - 1];
                // Earliest maximum is deterministic. targetWidth is an
                // intentionally non-operative diagnostic reference for this
                // first experiment.
                if (gap > largestGap)
                {
                    largestGap = gap;
                    chunkSize = cut;
                }
            }
        }

        std::vector<label> current;
        current.reserve(chunkSize);
        auto position = ready.begin();
        for (std::size_t index=0; index<chunkSize; ++index)
        {
            current.push_back(*position);
            position = ready.erase(position);
        }
        result.newToOld.insert
        (
            result.newToOld.end(), current.begin(), current.end()
        );
        result.levelStarts.push_back(result.newToOld.size());

        for (const label cell : current)
            for (label face=ownerStarts[cell]; face<ownerStarts[cell + 1]; ++face)
                if (--predecessors[upperAddr[face]] == 0)
                    ready.insert(upperAddr[face]);
    }
    if (result.newToOld.size() != static_cast<std::size_t>(nCells))
        throw std::runtime_error
        (
            "adaptive-gap Index-Kahn scheduler did not cover all cells"
        );
    result.oldToNew.resize(nCells);
    for (label row=0; row<nCells; ++row)
        result.oldToNew[result.newToOld[row]] = row;
    return result;
}

IndexKahnOrdering makeIndexKahnOrderingReadyFractionAdaptiveGap
(
    const label nCells,
    const Foam::labelField& ownerStarts,
    const Foam::labelField& upperAddr,
    const label readyDivisor,
    std::vector<AdaptiveReadyChunkDiagnostic>* const diagnostics
)
{
    if (readyDivisor < 1)
        throw std::runtime_error("invalid ready-fraction Index-Kahn divisor");
    if
    (
        ownerStarts.size() != static_cast<std::size_t>(nCells + 1)
     || ownerStarts.back() != static_cast<label>(upperAddr.size())
    ) throw std::runtime_error
      (
          "invalid ready-fraction Index-Kahn LDU addressing"
      );

    if (diagnostics) diagnostics->clear();
    std::vector<label> predecessors(nCells, 0);
    for (const label cell : upperAddr) ++predecessors[cell];
    std::set<label> ready;
    for (label cell=0; cell<nCells; ++cell)
        if (predecessors[cell] == 0) ready.insert(cell);

    IndexKahnOrdering result;
    result.levelStarts.push_back(0);
    result.newToOld.reserve(nCells);
    while (!ready.empty())
    {
        const std::size_t readySize = ready.size();
        const std::size_t target = readySize < 256
            ? readySize
            : std::clamp<std::size_t>(readySize/readyDivisor, 256, 8192);
        const std::size_t minWidth = std::max<std::size_t>(1, target/2);
        const std::size_t maxWidth = std::min(readySize, 2*target);
        std::size_t chunkSize = readySize;
        label largestGap = 0;

        // Below the safety floor, consume the complete available frontier.
        // The clamp must not manufacture a wider target or split a naturally
        // small ready set.
        if (readySize >= 256 && readySize > minWidth)
        {
            const std::size_t largestCandidateCut = std::min
            (
                maxWidth, readySize - 1
            );
            auto position = ready.begin();
            label previous = *position;
            ++position;
            bool haveGap = false;
            for
            (
                std::size_t cut=1;
                cut<=largestCandidateCut;
                ++cut, ++position
            )
            {
                const label current = *position;
                if (cut >= minWidth)
                {
                    const label gap = current - previous;
                    if (!haveGap || gap > largestGap)
                    {
                        haveGap = true;
                        largestGap = gap;
                        chunkSize = cut;
                    }
                }
                previous = current;
            }
        }

        if (diagnostics)
            diagnostics->push_back
            (
                AdaptiveReadyChunkDiagnostic
                {
                    static_cast<label>(readySize),
                    static_cast<label>(target),
                    static_cast<label>(minWidth),
                    static_cast<label>(maxWidth),
                    static_cast<label>(chunkSize),
                    largestGap
                }
            );

        std::vector<label> current;
        current.reserve(chunkSize);
        auto position = ready.begin();
        for (std::size_t index=0; index<chunkSize; ++index)
        {
            current.push_back(*position);
            position = ready.erase(position);
        }
        result.newToOld.insert
        (
            result.newToOld.end(), current.begin(), current.end()
        );
        result.levelStarts.push_back(result.newToOld.size());

        for (const label cell : current)
            for (label face=ownerStarts[cell]; face<ownerStarts[cell + 1]; ++face)
                if (--predecessors[upperAddr[face]] == 0)
                    ready.insert(upperAddr[face]);
    }
    if (result.newToOld.size() != static_cast<std::size_t>(nCells))
        throw std::runtime_error
        (
            "ready-fraction Index-Kahn scheduler did not cover all cells"
        );
    result.oldToNew.resize(nCells);
    for (label row=0; row<nCells; ++row)
        result.oldToNew[result.newToOld[row]] = row;
    return result;
}

IndexKahnOrdering makeIndexKahnOrderingUnbounded
(
    const label nCells,
    const Foam::labelField& ownerStarts,
    const Foam::labelField& upperAddr
)
{
    if
    (
        ownerStarts.size() != static_cast<std::size_t>(nCells + 1)
     || ownerStarts.back() != static_cast<label>(upperAddr.size())
    ) throw std::runtime_error("invalid unbounded Index-Kahn LDU addressing");

    std::vector<label> predecessors(nCells, 0);
    for (const label cell : upperAddr) ++predecessors[cell];
    std::set<label> ready;
    for (label cell=0; cell<nCells; ++cell)
        if (predecessors[cell] == 0) ready.insert(cell);

    IndexKahnOrdering result;
    result.levelStarts.push_back(0);
    result.newToOld.reserve(nCells);
    while (!ready.empty())
    {
        std::vector<label> current(ready.begin(), ready.end());
        ready.clear();
        result.newToOld.insert
        (
            result.newToOld.end(), current.begin(), current.end()
        );
        result.levelStarts.push_back(result.newToOld.size());
        for (const label cell : current)
            for (label face=ownerStarts[cell]; face<ownerStarts[cell + 1]; ++face)
                if (--predecessors[upperAddr[face]] == 0)
                    ready.insert(upperAddr[face]);
    }
    if (result.newToOld.size() != static_cast<std::size_t>(nCells))
        throw std::runtime_error
        (
            "unbounded Index-Kahn scheduler did not cover all cells"
        );
    result.oldToNew.resize(nCells);
    for (label row=0; row<nCells; ++row)
        result.oldToNew[result.newToOld[row]] = row;
    return result;
}

IndexKahnOrdering makeIndexKahnOrderingSliced
(
    const label nCells,
    const Foam::labelField& ownerStarts,
    const Foam::labelField& upperAddr,
    const label targetWidth
)
{
    if (targetWidth < 1)
        throw std::runtime_error("invalid sliced Index-Kahn width");
    if
    (
        ownerStarts.size() != static_cast<std::size_t>(nCells + 1)
     || ownerStarts.back() != static_cast<label>(upperAddr.size())
    ) throw std::runtime_error("invalid sliced Index-Kahn LDU addressing");

    std::vector<label> predecessors(nCells, 0);
    for (const label cell : upperAddr) ++predecessors[cell];
    std::set<label> ready;
    for (label cell=0; cell<nCells; ++cell)
        if (predecessors[cell] == 0) ready.insert(cell);

    IndexKahnOrdering result;
    result.levelStarts.push_back(0);
    result.newToOld.reserve(nCells);
    while (!ready.empty())
    {
        // Freeze the complete natural frontier before releasing successors.
        // std::set iteration preserves the same index order as IH1U.
        std::vector<label> currentNatural(ready.begin(), ready.end());
        ready.clear();
        result.newToOld.insert
        (
            result.newToOld.end(),
            currentNatural.begin(),
            currentNatural.end()
        );
        const std::size_t naturalBegin =
            result.newToOld.size() - currentNatural.size();
        for
        (
            std::size_t offset=targetWidth;
            offset<currentNatural.size();
            offset += targetWidth
        )
        {
            result.levelStarts.push_back(naturalBegin + offset);
        }
        result.levelStarts.push_back(result.newToOld.size());

        // Successors populate only the next natural Kahn frontier.
        for (const label cell : currentNatural)
            for (label face=ownerStarts[cell]; face<ownerStarts[cell + 1]; ++face)
                if (--predecessors[upperAddr[face]] == 0)
                    ready.insert(upperAddr[face]);
    }
    if (result.newToOld.size() != static_cast<std::size_t>(nCells))
        throw std::runtime_error
        (
            "sliced Index-Kahn scheduler did not cover all cells"
        );
    result.oldToNew.resize(nCells);
    for (label row=0; row<nCells; ++row)
        result.oldToNew[result.newToOld[row]] = row;
    return result;
}

IndexKahnOrdering makeIndexKahnOrderingContinuing
(
    const label nCells,
    const Foam::labelField& ownerStarts,
    const Foam::labelField& upperAddr,
    const label targetWidth
)
{
    if (targetWidth < 1)
        throw std::runtime_error("invalid continuing Index-Kahn width");
    if
    (
        ownerStarts.size() != static_cast<std::size_t>(nCells + 1)
     || ownerStarts.back() != static_cast<label>(upperAddr.size())
    ) throw std::runtime_error("invalid continuing Index-Kahn LDU addressing");

    std::vector<label> predecessors(nCells, 0);
    for (const label cell : upperAddr) ++predecessors[cell];
    std::set<label> ready;
    for (label cell=0; cell<nCells; ++cell)
        if (predecessors[cell] == 0) ready.insert(cell);

    IndexKahnOrdering result;
    result.levelStarts.push_back(0);
    result.newToOld.reserve(nCells);
    label anchor = 0;
    while (!ready.empty())
    {
        std::vector<label> current;
        current.reserve(targetWidth);
        auto position = ready.lower_bound(anchor);
        if (position == ready.end()) position = ready.begin();
        label lastSelected = *position;
        while
        (
            !ready.empty()
         && current.size() < static_cast<std::size_t>(targetWidth)
        )
        {
            lastSelected = *position;
            current.push_back(lastSelected);
            position = ready.erase(position);
            if (position == ready.end() && !ready.empty())
                position = ready.begin();
        }
        anchor = lastSelected == std::numeric_limits<label>::max()
            ? 0 : lastSelected + 1;
        result.newToOld.insert
        (
            result.newToOld.end(), current.begin(), current.end()
        );
        result.levelStarts.push_back(result.newToOld.size());
        for (const label cell : current)
            for (label face=ownerStarts[cell]; face<ownerStarts[cell + 1]; ++face)
                if (--predecessors[upperAddr[face]] == 0)
                    ready.insert(upperAddr[face]);
    }
    if (result.newToOld.size() != static_cast<std::size_t>(nCells))
        throw std::runtime_error
        (
            "continuing Index-Kahn scheduler did not cover all cells"
        );
    result.oldToNew.resize(nCells);
    for (label row=0; row<nCells; ++row)
        result.oldToNew[result.newToOld[row]] = row;
    return result;
}

IndexKahnOrdering makeIndexKahnOrderingFrontierContinuing
(
    const label nCells,
    const Foam::labelField& ownerStarts,
    const Foam::labelField& upperAddr,
    const label targetWidth
)
{
    if (targetWidth < 1)
        throw std::runtime_error("invalid frontier-continuing Index-Kahn width");
    if
    (
        ownerStarts.size() != static_cast<std::size_t>(nCells + 1)
     || ownerStarts.back() != static_cast<label>(upperAddr.size())
    ) throw std::runtime_error
      (
          "invalid frontier-continuing Index-Kahn LDU addressing"
      );

    std::vector<label> predecessors(nCells, 0);
    for (const label cell : upperAddr) ++predecessors[cell];
    std::set<label> ready;
    for (label cell=0; cell<nCells; ++cell)
        if (predecessors[cell] == 0) ready.insert(cell);

    IndexKahnOrdering result;
    result.levelStarts.push_back(0);
    result.newToOld.reserve(nCells);
    while (!ready.empty())
    {
        // Freeze membership without copying the complete frontier.  Newly
        // released successors are inserted into ready and cannot participate
        // until frontierReady has been exhausted.
        std::set<label> frontierReady;
        frontierReady.swap(ready);
        label anchor = *frontierReady.begin();

        while (!frontierReady.empty())
        {
            std::vector<label> current;
            current.reserve(targetWidth);
            auto position = frontierReady.lower_bound(anchor);
            if (position == frontierReady.end())
                position = frontierReady.begin();
            label lastSelected = *position;
            while
            (
                !frontierReady.empty()
             && current.size() < static_cast<std::size_t>(targetWidth)
            )
            {
                lastSelected = *position;
                current.push_back(lastSelected);
                position = frontierReady.erase(position);
                if (position == frontierReady.end() && !frontierReady.empty())
                    position = frontierReady.begin();
            }
            anchor = lastSelected == std::numeric_limits<label>::max()
                ? 0 : lastSelected + 1;
            result.newToOld.insert
            (
                result.newToOld.end(), current.begin(), current.end()
            );
            result.levelStarts.push_back(result.newToOld.size());

            // Preserve IH1W incremental release, but place every newly-ready
            // successor in the next-frontier set.
            for (const label cell : current)
            {
                for
                (
                    label face=ownerStarts[cell];
                    face<ownerStarts[cell + 1];
                    ++face
                )
                {
                    if (--predecessors[upperAddr[face]] == 0)
                        ready.insert(upperAddr[face]);
                }
            }
        }
    }
    if (result.newToOld.size() != static_cast<std::size_t>(nCells))
        throw std::runtime_error
        (
            "frontier-continuing Index-Kahn scheduler did not cover all cells"
        );
    result.oldToNew.resize(nCells);
    for (label row=0; row<nCells; ++row)
        result.oldToNew[result.newToOld[row]] = row;
    return result;
}

LexicographicPackedSchedule makeLexicographicPackedSchedule
(
    const Foam::lduMatrix& matrix
)
{
    LexicographicPackedSchedule schedule;
    const auto& ownerStarts = matrix.lduAddr().ownerStartAddr();
    const auto& neighbours = matrix.lduAddr().upperAddr();
    const label nCells = static_cast<label>(matrix.diag().size());
    const label nFaces = static_cast<label>(matrix.upper().size());

    std::vector<label> incomingCounts(nCells, 0);
    for (label face=0; face<nFaces; ++face) ++incomingCounts[neighbours[face]];

    schedule.rowStarts.resize(nCells + 1, 0);
    for (label cell=0; cell<nCells; ++cell)
    {
        schedule.rowStarts[cell + 1] = schedule.rowStarts[cell]
            + incomingCounts[cell]
            + ownerStarts[cell + 1] - ownerStarts[cell];
    }
    schedule.cols.resize(2*static_cast<std::size_t>(nFaces));
    schedule.coeffs.resize(schedule.cols.size());
    schedule.diag.assign(matrix.diag().begin(), matrix.diag().end());

    std::vector<label> incomingCursor(schedule.rowStarts.begin(), schedule.rowStarts.end());
    for (label owner=0; owner<nCells; ++owner)
    {
        label outgoing = schedule.rowStarts[owner] + incomingCounts[owner];
        for (label face=ownerStarts[owner]; face<ownerStarts[owner + 1]; ++face)
        {
            const label neighbour = neighbours[face];
            const label incoming = incomingCursor[neighbour]++;
            schedule.cols[incoming] = owner;
            schedule.coeffs[incoming] = matrix.lower()[face];
            schedule.cols[outgoing] = neighbour;
            schedule.coeffs[outgoing] = matrix.upper()[face];
            ++outgoing;
        }
    }

    if
    (
        schedule.rowStarts.back() != 2*nFaces
     || schedule.cols.size() != 2*static_cast<std::size_t>(nFaces)
     || schedule.coeffs.size() != schedule.cols.size()
     || schedule.diag.size() != static_cast<std::size_t>(nCells)
    ) throw std::runtime_error("invalid lexicographic packed schedule");

    return schedule;
}

void serialLexicographicPackedSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const LexicographicPackedSchedule& schedule,
    const label nSweeps
)
{
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const cols = schedule.cols.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const label nCells = static_cast<label>(schedule.diag.size());

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (label cell=0; cell<nCells; ++cell)
        {
            scalar psii = source[cell];
            for (label p=rowStarts[cell]; p<rowStarts[cell + 1]; ++p)
                psii -= coeffs[p]*psi[cols[p]];
            psi[cell] = psii/diag[cell];
        }
    }
}

void serialGatherSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps
)
{
    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const cols = schedule.cols.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
            {
                const label cell = waveCells[row];
                scalar psii = source[cell];
                for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                    psii -= coeffs[p]*psi[cols[p]];
                psi[cell] = psii/diag[row];
            }
        }
    }
}

void serialSymmetricGatherSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps
)
{
    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const cols = schedule.cols.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
            {
                const label cell = waveCells[row];
                scalar psii = source[cell];
                for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                    psii -= coeffs[p]*psi[cols[p]];
                psi[cell] = psii/diag[row];
            }
        }

        for (std::size_t level=nLevels; level-- > 0;)
        {
            for (label row=levelStarts[level + 1]; row-- > levelStarts[level];)
            {
                const label cell = waveCells[row];
                scalar psii = source[cell];
                for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                    psii -= coeffs[p]*psi[cols[p]];
                psi[cell] = psii/diag[row];
            }
        }
    }
}

void serialRowDegreeSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const RowDegreeSchedule& schedule,
    const label nSweeps
)
{
    const WavefrontSchedule& packed = *schedule.packed;
    const label* const waveCells = packed.waveCells.data();
    const std::uint8_t* const degrees = schedule.degrees.data();
    const label* const cols = packed.cols.data();
    const scalar* const coeffs = packed.coeffs.data();
    const scalar* const diag = packed.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const label nRows = static_cast<label>(packed.waveCells.size());

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        label p = 0;
        for (label row=0; row<nRows; ++row)
        {
            const label cell = waveCells[row];
            scalar psii = source[cell];
            const std::uint8_t degree = degrees[row];
            const label end = p + degree;
            for (; p<end; ++p)
                psii -= coeffs[p]*psi[cols[p]];
            psi[cell] = psii/diag[row];
        }
    }
}

void serialDirectCoefficientSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const Foam::lduMatrix& matrix,
    const DirectCoefficientSchedule& schedule,
    const label nSweeps
)
{
    const label* const waveCells = schedule.waveCells.data();
    const label* const cols = schedule.cols.data();
    const label* const faceIds = schedule.faceIds.data();
    const std::uint8_t* const degrees = schedule.degrees.data();
    const std::uint8_t* const incomingDegrees = schedule.incomingDegrees.data();
    const scalar* const lower = matrix.lower().data();
    const scalar* const upper = matrix.upper().data();
    const scalar* const diag = matrix.diag().data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const label nRows = static_cast<label>(schedule.waveCells.size());

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        label p = 0;
        for (label row=0; row<nRows; ++row)
        {
            const label cell = waveCells[row];
            scalar psii = source[cell];
            const label split = p + incomingDegrees[row];
            const label end = p + degrees[row];
            for (; p<split; ++p) psii -= lower[faceIds[p]]*psi[cols[p]];
            for (; p<end; ++p) psii -= upper[faceIds[p]]*psi[cols[p]];
            psi[cell] = psii/diag[cell];
        }
    }
}

void persistentOpenMpDirectCoefficientSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const Foam::lduMatrix& matrix,
    const DirectCoefficientSchedule& schedule,
    const label nSweeps,
    int& detectedThreads
)
{
    const label* const waveCells = schedule.waveCells.data();
    const label* const cols = schedule.cols.data();
    const label* const faceIds = schedule.faceIds.data();
    const std::uint8_t* const degrees = schedule.degrees.data();
    const std::uint8_t* const incomingDegrees = schedule.incomingDegrees.data();
    const label* const levelBlockStarts = schedule.levelBlockStarts.data();
    const label* const blockRowStarts = schedule.blockRowStarts.data();
    const label* const blockRowEnds = schedule.blockRowEnds.data();
    const label* const blockPStarts = schedule.blockPStarts.data();
    const scalar* const lower = matrix.lower().data();
    const scalar* const upper = matrix.upper().data();
    const scalar* const diag = matrix.diag().data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    #pragma omp parallel
    {
        #pragma omp single
        detectedThreads = omp_get_num_threads();

        for (label sweep=0; sweep<nSweeps; ++sweep)
        {
            for (std::size_t level=0; level<nLevels; ++level)
            {
                #pragma omp for schedule(static)
                for (label block=levelBlockStarts[level]; block<levelBlockStarts[level + 1]; ++block)
                {
                    label p = blockPStarts[block];
                    for (label row=blockRowStarts[block]; row<blockRowEnds[block]; ++row)
                    {
                        const label cell = waveCells[row];
                        scalar psii = source[cell];
                        const label split = p + incomingDegrees[row];
                        const label end = p + degrees[row];
                        for (; p<split; ++p) psii -= lower[faceIds[p]]*psi[cols[p]];
                        for (; p<end; ++p) psii -= upper[faceIds[p]]*psi[cols[p]];
                        psi[cell] = psii/diag[cell];
                    }
                }
            }
        }
    }
}

void serialRowDegreeUnrolled8Smooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const RowDegreeSchedule& schedule,
    const label nSweeps
)
{
    const WavefrontSchedule& packed = *schedule.packed;
    const label* const levelStarts = packed.levelStarts.data();
    const label* const waveCells = packed.waveCells.data();
    const std::uint8_t* const degrees = schedule.degrees.data();
    const label* const cols = packed.cols.data();
    const scalar* const coeffs = packed.coeffs.data();
    const scalar* const diag = packed.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = packed.levelStarts.size() - 1;

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        label p = 0;
        for (std::size_t level=0; level<nLevels; ++level)
        {
            label row = levelStarts[level];
            const label levelEnd = levelStarts[level + 1];
            for (; row + 8<=levelEnd; row += 8)
            {
                const label p0 = p;
                const label p1 = p0 + degrees[row];
                const label p2 = p1 + degrees[row + 1];
                const label p3 = p2 + degrees[row + 2];
                const label p4 = p3 + degrees[row + 3];
                const label p5 = p4 + degrees[row + 4];
                const label p6 = p5 + degrees[row + 5];
                const label p7 = p6 + degrees[row + 6];
                const label p8 = p7 + degrees[row + 7];
                const label cell0 = waveCells[row];
                const label cell1 = waveCells[row + 1];
                const label cell2 = waveCells[row + 2];
                const label cell3 = waveCells[row + 3];
                const label cell4 = waveCells[row + 4];
                const label cell5 = waveCells[row + 5];
                const label cell6 = waveCells[row + 6];
                const label cell7 = waveCells[row + 7];
                scalar s0 = source[cell0];
                scalar s1 = source[cell1];
                scalar s2 = source[cell2];
                scalar s3 = source[cell3];
                scalar s4 = source[cell4];
                scalar s5 = source[cell5];
                scalar s6 = source[cell6];
                scalar s7 = source[cell7];
                for (label q=p0; q<p1; ++q) s0 -= coeffs[q]*psi[cols[q]];
                for (label q=p1; q<p2; ++q) s1 -= coeffs[q]*psi[cols[q]];
                for (label q=p2; q<p3; ++q) s2 -= coeffs[q]*psi[cols[q]];
                for (label q=p3; q<p4; ++q) s3 -= coeffs[q]*psi[cols[q]];
                for (label q=p4; q<p5; ++q) s4 -= coeffs[q]*psi[cols[q]];
                for (label q=p5; q<p6; ++q) s5 -= coeffs[q]*psi[cols[q]];
                for (label q=p6; q<p7; ++q) s6 -= coeffs[q]*psi[cols[q]];
                for (label q=p7; q<p8; ++q) s7 -= coeffs[q]*psi[cols[q]];
                p = p8;
                s0 /= diag[row];
                s1 /= diag[row + 1];
                s2 /= diag[row + 2];
                s3 /= diag[row + 3];
                s4 /= diag[row + 4];
                s5 /= diag[row + 5];
                s6 /= diag[row + 6];
                s7 /= diag[row + 7];
                psi[cell0] = s0;
                psi[cell1] = s1;
                psi[cell2] = s2;
                psi[cell3] = s3;
                psi[cell4] = s4;
                psi[cell5] = s5;
                psi[cell6] = s6;
                psi[cell7] = s7;
            }
            for (; row<levelEnd; ++row)
            {
                const label cell = waveCells[row];
                scalar psii = source[cell];
                const label end = p + degrees[row];
                for (; p<end; ++p) psii -= coeffs[p]*psi[cols[p]];
                psi[cell] = psii/diag[row];
            }
        }
    }
}

void persistentOpenMpBlockedRowDegreeSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const BlockedRowDegreeSchedule& schedule,
    const label nSweeps,
    int& detectedThreads
)
{
    const WavefrontSchedule& packed = *schedule.packed;
    const label* const waveCells = packed.waveCells.data();
    const std::uint8_t* const degrees = schedule.degrees.data();
    const label* const levelBlockStarts = schedule.levelBlockStarts.data();
    const label* const blockRowStarts = schedule.blockRowStarts.data();
    const label* const blockRowEnds = schedule.blockRowEnds.data();
    const label* const blockPStarts = schedule.blockPStarts.data();
    const label* const cols = packed.cols.data();
    const scalar* const coeffs = packed.coeffs.data();
    const scalar* const diag = packed.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = packed.levelStarts.size() - 1;

    #pragma omp parallel
    {
        #pragma omp single
        detectedThreads = omp_get_num_threads();

        for (label sweep=0; sweep<nSweeps; ++sweep)
        {
            for (std::size_t level=0; level<nLevels; ++level)
            {
                #pragma omp for schedule(static)
                for (label block=levelBlockStarts[level]; block<levelBlockStarts[level + 1]; ++block)
                {
                    label p = blockPStarts[block];
                    for (label row=blockRowStarts[block]; row<blockRowEnds[block]; ++row)
                    {
                        const label cell = waveCells[row];
                        scalar psii = source[cell];
                        const label end = p + degrees[row];
                        for (; p<end; ++p)
                            psii -= coeffs[p]*psi[cols[p]];
                        psi[cell] = psii/diag[row];
                    }
                }
            }
        }
    }
}

void persistentOpenMpAdaptiveBlockedRowDegreeSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const AdaptiveBlockedRowDegreeSchedule& adaptive,
    const label nSweeps,
    int& detectedThreads
)
{
    const BlockedRowDegreeSchedule& schedule = *adaptive.blocked;
    const WavefrontSchedule& packed = *schedule.packed;
    const label* const waveCells = packed.waveCells.data();
    const std::uint8_t* const degrees = schedule.degrees.data();
    const label* const levelBlockStarts = schedule.levelBlockStarts.data();
    const label* const blockRowStarts = schedule.blockRowStarts.data();
    const label* const blockRowEnds = schedule.blockRowEnds.data();
    const label* const blockPStarts = schedule.blockPStarts.data();
    const std::uint8_t* const scheduledThreads = adaptive.activeThreads.data();
    const label* const cols = packed.cols.data();
    const scalar* const coeffs = packed.coeffs.data();
    const scalar* const diag = packed.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = packed.levelStarts.size() - 1;

    #pragma omp parallel
    {
        const int thread = omp_get_thread_num();
        const int teamThreads = omp_get_num_threads();
        #pragma omp single nowait
        detectedThreads = teamThreads;

        for (label sweep=0; sweep<nSweeps; ++sweep)
        {
            for (std::size_t level=0; level<nLevels; ++level)
            {
                const int activeThreads = std::min
                (
                    teamThreads, static_cast<int>(scheduledThreads[level])
                );
                const label firstBlock = levelBlockStarts[level];
                const label nBlocks = levelBlockStarts[level + 1] - firstBlock;
                if (thread < activeThreads)
                {
                    const label begin = firstBlock
                        + nBlocks*thread/activeThreads;
                    const label end = firstBlock
                        + nBlocks*(thread + 1)/activeThreads;
                    for (label block=begin; block<end; ++block)
                    {
                        label p = blockPStarts[block];
                        for (label row=blockRowStarts[block]; row<blockRowEnds[block]; ++row)
                        {
                            const label cell = waveCells[row];
                            scalar psii = source[cell];
                            const label contributionEnd = p + degrees[row];
                            for (; p<contributionEnd; ++p)
                                psii -= coeffs[p]*psi[cols[p]];
                            psi[cell] = psii/diag[row];
                        }
                    }
                }
                #pragma omp barrier
            }
        }
    }
}

void persistentOpenMpThresholdBlockedRowDegreeSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const AdaptiveBlockedRowDegreeSchedule& adaptive,
    const label nSweeps,
    int& detectedThreads
)
{
    const BlockedRowDegreeSchedule& schedule = *adaptive.blocked;
    const WavefrontSchedule& packed = *schedule.packed;
    const label* const waveCells = packed.waveCells.data();
    const std::uint8_t* const degrees = schedule.degrees.data();
    const label* const levelBlockStarts = schedule.levelBlockStarts.data();
    const label* const blockRowStarts = schedule.blockRowStarts.data();
    const label* const blockRowEnds = schedule.blockRowEnds.data();
    const label* const blockPStarts = schedule.blockPStarts.data();
    const std::uint8_t* const scheduledThreads = adaptive.activeThreads.data();
    const label* const cols = packed.cols.data();
    const scalar* const coeffs = packed.coeffs.data();
    const scalar* const diag = packed.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = packed.levelStarts.size() - 1;

    #pragma omp parallel
    {
        const int thread = omp_get_thread_num();
        const int teamThreads = omp_get_num_threads();
        #pragma omp single nowait
        detectedThreads = teamThreads;

        for (label sweep=0; sweep<nSweeps; ++sweep)
        {
            for (std::size_t level=0; level<nLevels; ++level)
            {
                const label firstBlock = levelBlockStarts[level];
                const label lastBlock = levelBlockStarts[level + 1];
                if (scheduledThreads[level] == 1 && teamThreads > 1)
                {
                    if (thread == 0)
                    {
                        for (label block=firstBlock; block<lastBlock; ++block)
                        {
                            label p = blockPStarts[block];
                            for (label row=blockRowStarts[block]; row<blockRowEnds[block]; ++row)
                            {
                                const label cell = waveCells[row];
                                scalar psii = source[cell];
                                const label contributionEnd = p + degrees[row];
                                for (; p<contributionEnd; ++p)
                                    psii -= coeffs[p]*psi[cols[p]];
                                psi[cell] = psii/diag[row];
                            }
                        }
                    }
                    #pragma omp barrier
                }
                else
                {
                    #pragma omp for schedule(static)
                    for (label block=firstBlock; block<lastBlock; ++block)
                    {
                        label p = blockPStarts[block];
                        for (label row=blockRowStarts[block]; row<blockRowEnds[block]; ++row)
                        {
                            const label cell = waveCells[row];
                            scalar psii = source[cell];
                            const label contributionEnd = p + degrees[row];
                            for (; p<contributionEnd; ++p)
                                psii -= coeffs[p]*psi[cols[p]];
                            psi[cell] = psii/diag[row];
                        }
                    }
                }
            }
        }
    }
}

void serialHybridSmooth
(
    Foam::scalarField& psiField,
    Foam::scalarField& edgeField,
    const Foam::scalarField& sourceField,
    const HybridWavefrontSchedule& schedule,
    const label nSweeps
)
{
    const WavefrontSchedule& packed = *schedule.packed;
    const label* const levelStarts = packed.levelStarts.data();
    const label* const waveCells = packed.waveCells.data();
    const label* const rowStarts = packed.rowStarts.data();
    const label* const incomingCounts = packed.incomingCounts.data();
    const label* const incomingStarts = schedule.incomingStarts.data();
    const label* const incomingFaces = schedule.incomingFaces.data();
    const label* const cols = packed.cols.data();
    const scalar* const coeffs = packed.coeffs.data();
    const scalar* const diag = packed.diag.data();
    const label* const outgoingFaceStarts = schedule.outgoingFaceStarts.data();
    const label* const outgoingFaceEnds = schedule.outgoingFaceEnds.data();
    const scalar* const lower = schedule.lowerCoeffs->data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    scalar* const edgeValues = edgeField.data();
    const std::size_t nLevels = packed.levelStarts.size() - 1;

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
            {
                const label cell = waveCells[row];
                scalar psii = source[cell];
                const label incomingEnd = rowStarts[row] + incomingCounts[row];
                for (label p=incomingStarts[row]; p<incomingStarts[row + 1]; ++p)
                    psii -= edgeValues[incomingFaces[p]];
                for (label p=incomingEnd; p<rowStarts[row + 1]; ++p)
                    psii -= coeffs[p]*psi[cols[p]];
                const scalar value = psii/diag[row];
                psi[cell] = value;
                for (label face=outgoingFaceStarts[row]; face<outgoingFaceEnds[row]; ++face)
                    edgeValues[face] = lower[face]*value;
            }
        }
    }
}

void persistentOpenMpHybridSmooth
(
    Foam::scalarField& psiField,
    Foam::scalarField& edgeField,
    const Foam::scalarField& sourceField,
    const HybridWavefrontSchedule& schedule,
    const label nSweeps,
    int& detectedThreads
)
{
    const WavefrontSchedule& packed = *schedule.packed;
    const label* const levelStarts = packed.levelStarts.data();
    const label* const waveCells = packed.waveCells.data();
    const label* const rowStarts = packed.rowStarts.data();
    const label* const incomingCounts = packed.incomingCounts.data();
    const label* const incomingStarts = schedule.incomingStarts.data();
    const label* const incomingFaces = schedule.incomingFaces.data();
    const label* const cols = packed.cols.data();
    const scalar* const coeffs = packed.coeffs.data();
    const scalar* const diag = packed.diag.data();
    const label* const outgoingFaceStarts = schedule.outgoingFaceStarts.data();
    const label* const outgoingFaceEnds = schedule.outgoingFaceEnds.data();
    const scalar* const lower = schedule.lowerCoeffs->data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    scalar* const edgeValues = edgeField.data();
    const std::size_t nLevels = packed.levelStarts.size() - 1;

    #pragma omp parallel
    {
        #pragma omp single
        detectedThreads = omp_get_num_threads();

        for (label sweep=0; sweep<nSweeps; ++sweep)
        {
            for (std::size_t level=0; level<nLevels; ++level)
            {
                #pragma omp for schedule(static)
                for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
                {
                    const label cell = waveCells[row];
                    scalar psii = source[cell];
                    const label incomingEnd = rowStarts[row] + incomingCounts[row];
                    for (label p=incomingStarts[row]; p<incomingStarts[row + 1]; ++p)
                        psii -= edgeValues[incomingFaces[p]];
                    for (label p=incomingEnd; p<rowStarts[row + 1]; ++p)
                        psii -= coeffs[p]*psi[cols[p]];
                    const scalar value = psii/diag[row];
                    psi[cell] = value;
                    for (label face=outgoingFaceStarts[row]; face<outgoingFaceEnds[row]; ++face)
                        edgeValues[face] = lower[face]*value;
                }
            }
        }
    }
}

void serialWholeRowInt16Smooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WholeRowInt16Schedule& schedule,
    const label nSweeps
)
{
    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const compactStarts = schedule.compactStarts.data();
    const label* const fallbackStarts = schedule.fallbackStarts.data();
    const label* const fallbackCols = schedule.fallbackCols.data();
    const std::int16_t* const offsets = schedule.offsets.data();
    const unsigned char* const compactRows = schedule.compactRows.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
            {
                const label cell = waveCells[row];
                scalar psii = source[cell];
                const label coefficientBegin = rowStarts[row];
                if (compactRows[row])
                {
                    const label begin = compactStarts[row];
                    const label end = compactStarts[row + 1];
                    for (label q=begin; q<end; ++q)
                        psii -= coeffs[coefficientBegin + q - begin]*psi[cell + offsets[q]];
                }
                else
                {
                    const label begin = fallbackStarts[row];
                    const label end = fallbackStarts[row + 1];
                    for (label q=begin; q<end; ++q)
                        psii -= coeffs[coefficientBegin + q - begin]*psi[fallbackCols[q]];
                }
                psi[cell] = psii/diag[row];
            }
        }
    }
}

void serialDeltaEscapeInt16Smooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const DeltaEscapeInt16Schedule& schedule,
    const label nSweeps
)
{
    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const firstCols = schedule.firstCols.data();
    const label* const deltaStarts = schedule.deltaStarts.data();
    const label* const escapeStarts = schedule.escapeStarts.data();
    const label* const escapeCols = schedule.escapeCols.data();
    const std::int16_t* const deltas = schedule.deltas.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::int16_t escape = std::numeric_limits<std::int16_t>::min();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
            {
                const label cell = waveCells[row];
                scalar psii = source[cell];
                const label begin = rowStarts[row];
                const label end = rowStarts[row + 1];
                if (begin < end)
                {
                    label col = firstCols[row];
                    psii -= coeffs[begin]*psi[col];
                    label deltaIndex = deltaStarts[row];
                    label escapeIndex = escapeStarts[row];
                    for (label p=begin + 1; p<end; ++p)
                    {
                        const std::int16_t delta = deltas[deltaIndex++];
                        col = delta == escape ? escapeCols[escapeIndex++] : col + delta;
                        psii -= coeffs[p]*psi[col];
                    }
                }
                psi[cell] = psii/diag[row];
            }
        }
    }
}

void serialPackedUint24Smooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const PackedUint24Schedule& schedule,
    const label nSweeps
)
{
    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const std::uint8_t* const cols = schedule.cols.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
            {
                const label cell = waveCells[row];
                scalar psii = source[cell];
                for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                {
                    const std::size_t byte = 3*std::size_t(p);
                    const std::uint32_t col =
                        std::uint32_t(cols[byte])
                      | (std::uint32_t(cols[byte + 1]) << 8)
                      | (std::uint32_t(cols[byte + 2]) << 16);
                    psii -= coeffs[p]*psi[col];
                }
                psi[cell] = psii/diag[row];
            }
        }
    }
}

void serialGatherPrefetchSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps,
    const label rowDistance,
    const label neighboursToPrefetch
)
{
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const cols = schedule.cols.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const label nRows = static_cast<label>(schedule.waveCells.size());
    const label prefetchEnd = std::max<label>(0, nRows - rowDistance);

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (label row=0; row<prefetchEnd; ++row)
        {
            const label futureRow = row + rowDistance;
            const label futureBegin = rowStarts[futureRow];
            const label futureCount = std::min
            (
                neighboursToPrefetch,
                rowStarts[futureRow + 1] - futureBegin
            );
            for (label neighbour=0; neighbour<futureCount; ++neighbour)
            {
                __builtin_prefetch
                (
                    psi + cols[futureBegin + neighbour], 0, 3
                );
            }

            const label cell = waveCells[row];
            scalar psii = source[cell];
            for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                psii -= coeffs[p]*psi[cols[p]];
            psi[cell] = psii/diag[row];
        }
        for (label row=prefetchEnd; row<nRows; ++row)
        {
            const label cell = waveCells[row];
            scalar psii = source[cell];
            for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                psii -= coeffs[p]*psi[cols[p]];
            psi[cell] = psii/diag[row];
        }
    }
}

void serialGatherFirstPsiPrefetchSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps,
    const label rowDistance
)
{
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const cols = schedule.cols.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const label nRows = static_cast<label>(schedule.waveCells.size());
    const label prefetchEnd = std::max<label>(0, nRows - rowDistance);

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (label row=0; row<prefetchEnd; ++row)
        {
            const label futureRow = row + rowDistance;
            const label futureBegin = rowStarts[futureRow];
            if (futureBegin < rowStarts[futureRow + 1])
                __builtin_prefetch(psi + cols[futureBegin], 0, 1);
            const label cell = waveCells[row];
            scalar psii = source[cell];
            for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                psii -= coeffs[p]*psi[cols[p]];
            psi[cell] = psii/diag[row];
        }
        for (label row=prefetchEnd; row<nRows; ++row)
        {
            const label cell = waveCells[row];
            scalar psii = source[cell];
            for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                psii -= coeffs[p]*psi[cols[p]];
            psi[cell] = psii/diag[row];
        }
    }
}

void serialGatherInterleavedRowsSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps
)
{
    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const cols = schedule.cols.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            const label levelBegin = levelStarts[level];
            const label levelEnd = levelStarts[level + 1];
            const label groupEnd = levelEnd - (levelEnd - levelBegin)%4;
            label row = levelBegin;
            for (; row<groupEnd; row += 4)
            {
                const label cell0 = waveCells[row];
                const label cell1 = waveCells[row + 1];
                const label cell2 = waveCells[row + 2];
                const label cell3 = waveCells[row + 3];
                scalar s0 = source[cell0];
                scalar s1 = source[cell1];
                scalar s2 = source[cell2];
                scalar s3 = source[cell3];
                label p0 = rowStarts[row];
                label p1 = rowStarts[row + 1];
                label p2 = rowStarts[row + 2];
                label p3 = rowStarts[row + 3];
                const label end0 = rowStarts[row + 1];
                const label end1 = rowStarts[row + 2];
                const label end2 = rowStarts[row + 3];
                const label end3 = rowStarts[row + 4];

                while (p0<end0 || p1<end1 || p2<end2 || p3<end3)
                {
                    if (p0<end0) { s0 -= coeffs[p0]*psi[cols[p0]]; ++p0; }
                    if (p1<end1) { s1 -= coeffs[p1]*psi[cols[p1]]; ++p1; }
                    if (p2<end2) { s2 -= coeffs[p2]*psi[cols[p2]]; ++p2; }
                    if (p3<end3) { s3 -= coeffs[p3]*psi[cols[p3]]; ++p3; }
                }

                psi[cell0] = s0/diag[row];
                psi[cell1] = s1/diag[row + 1];
                psi[cell2] = s2/diag[row + 2];
                psi[cell3] = s3/diag[row + 3];
            }
            for (; row<levelEnd; ++row)
            {
                const label cell = waveCells[row];
                scalar psii = source[cell];
                for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                    psii -= coeffs[p]*psi[cols[p]];
                psi[cell] = psii/diag[row];
            }
        }
    }
}

void serialGatherDegree6Smooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps
)
{
    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const cols = schedule.cols.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
            {
                const label cell = waveCells[row];
                scalar psii = source[cell];
                const label begin = rowStarts[row];
                if (rowStarts[row + 1] - begin == 6)
                {
                    psii -= coeffs[begin]*psi[cols[begin]];
                    psii -= coeffs[begin + 1]*psi[cols[begin + 1]];
                    psii -= coeffs[begin + 2]*psi[cols[begin + 2]];
                    psii -= coeffs[begin + 3]*psi[cols[begin + 3]];
                    psii -= coeffs[begin + 4]*psi[cols[begin + 4]];
                    psii -= coeffs[begin + 5]*psi[cols[begin + 5]];
                }
                else
                {
                    for (label p=begin; p<rowStarts[row + 1]; ++p)
                        psii -= coeffs[p]*psi[cols[p]];
                }
                psi[cell] = psii/diag[row];
            }
        }
    }
}

void serialGatherDegree6InterleavedRowsSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps
)
{
    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const cols = schedule.cols.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            const label levelEnd = levelStarts[level + 1];
            label row = levelStarts[level];
            while (row < levelEnd)
            {
                const bool fourDegree6 = row + 3 < levelEnd
                    && rowStarts[row + 1] - rowStarts[row] == 6
                    && rowStarts[row + 2] - rowStarts[row + 1] == 6
                    && rowStarts[row + 3] - rowStarts[row + 2] == 6
                    && rowStarts[row + 4] - rowStarts[row + 3] == 6;
                if (fourDegree6)
                {
                    const label cell0 = waveCells[row];
                    const label cell1 = waveCells[row + 1];
                    const label cell2 = waveCells[row + 2];
                    const label cell3 = waveCells[row + 3];
                    const label p0 = rowStarts[row];
                    const label p1 = rowStarts[row + 1];
                    const label p2 = rowStarts[row + 2];
                    const label p3 = rowStarts[row + 3];
                    scalar s0 = source[cell0];
                    scalar s1 = source[cell1];
                    scalar s2 = source[cell2];
                    scalar s3 = source[cell3];

                    s0 -= coeffs[p0]*psi[cols[p0]];
                    s1 -= coeffs[p1]*psi[cols[p1]];
                    s2 -= coeffs[p2]*psi[cols[p2]];
                    s3 -= coeffs[p3]*psi[cols[p3]];
                    s0 -= coeffs[p0 + 1]*psi[cols[p0 + 1]];
                    s1 -= coeffs[p1 + 1]*psi[cols[p1 + 1]];
                    s2 -= coeffs[p2 + 1]*psi[cols[p2 + 1]];
                    s3 -= coeffs[p3 + 1]*psi[cols[p3 + 1]];
                    s0 -= coeffs[p0 + 2]*psi[cols[p0 + 2]];
                    s1 -= coeffs[p1 + 2]*psi[cols[p1 + 2]];
                    s2 -= coeffs[p2 + 2]*psi[cols[p2 + 2]];
                    s3 -= coeffs[p3 + 2]*psi[cols[p3 + 2]];
                    s0 -= coeffs[p0 + 3]*psi[cols[p0 + 3]];
                    s1 -= coeffs[p1 + 3]*psi[cols[p1 + 3]];
                    s2 -= coeffs[p2 + 3]*psi[cols[p2 + 3]];
                    s3 -= coeffs[p3 + 3]*psi[cols[p3 + 3]];
                    s0 -= coeffs[p0 + 4]*psi[cols[p0 + 4]];
                    s1 -= coeffs[p1 + 4]*psi[cols[p1 + 4]];
                    s2 -= coeffs[p2 + 4]*psi[cols[p2 + 4]];
                    s3 -= coeffs[p3 + 4]*psi[cols[p3 + 4]];
                    s0 -= coeffs[p0 + 5]*psi[cols[p0 + 5]];
                    s1 -= coeffs[p1 + 5]*psi[cols[p1 + 5]];
                    s2 -= coeffs[p2 + 5]*psi[cols[p2 + 5]];
                    s3 -= coeffs[p3 + 5]*psi[cols[p3 + 5]];

                    psi[cell0] = s0/diag[row];
                    psi[cell1] = s1/diag[row + 1];
                    psi[cell2] = s2/diag[row + 2];
                    psi[cell3] = s3/diag[row + 3];
                    row += 4;
                }
                else
                {
                    const label cell = waveCells[row];
                    scalar psii = source[cell];
                    for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                        psii -= coeffs[p]*psi[cols[p]];
                    psi[cell] = psii/diag[row];
                    ++row;
                }
            }
        }
    }
}

void serialReorderedPsiSmooth
(
    Foam::scalarField& originalPsiField,
    Foam::scalarField& reorderedPsiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps
)
{
    if (nSweeps <= 1)
    {
        serialGatherSmooth
        (
            originalPsiField, sourceField, schedule, nSweeps
        );
        return;
    }

    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const cols = schedule.cols.data();
    const label* const localCols = schedule.localCols.data();
    const label* const incomingCounts = schedule.incomingCounts.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const originalPsi = originalPsiField.data();
    scalar* const reorderedPsi = reorderedPsiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    // First sweep: updated dependencies are already in reorderedPsi, while
    // not-yet-updated upper neighbours still live in the original field.
    for (std::size_t level=0; level<nLevels; ++level)
    {
        for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
        {
            const label cell = waveCells[row];
            scalar psii = source[cell];
            const label incomingEnd = rowStarts[row] + incomingCounts[row];
            for (label p=rowStarts[row]; p<incomingEnd; ++p)
                psii -= coeffs[p]*reorderedPsi[localCols[p]];
            for (label p=incomingEnd; p<rowStarts[row + 1]; ++p)
                psii -= coeffs[p]*originalPsi[cols[p]];
            reorderedPsi[row] = psii/diag[row];
        }
    }

    // Middle sweeps remain entirely in wavefront-row numbering.
    for (label sweep=1; sweep<nSweeps - 1; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
            {
                scalar psii = source[waveCells[row]];
                for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                    psii -= coeffs[p]*reorderedPsi[localCols[p]];
                reorderedPsi[row] = psii/diag[row];
            }
        }
    }

    // Last sweep: updated dependencies are written in original numbering;
    // upper neighbours retain their previous-sweep reordered values.
    for (std::size_t level=0; level<nLevels; ++level)
    {
        for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
        {
            const label cell = waveCells[row];
            scalar psii = source[cell];
            const label incomingEnd = rowStarts[row] + incomingCounts[row];
            for (label p=rowStarts[row]; p<incomingEnd; ++p)
                psii -= coeffs[p]*originalPsi[cols[p]];
            for (label p=incomingEnd; p<rowStarts[row + 1]; ++p)
                psii -= coeffs[p]*reorderedPsi[localCols[p]];
            originalPsi[cell] = psii/diag[row];
        }
    }
}

void perLevelOpenMpSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps,
    int& detectedThreads
)
{
    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const cols = schedule.cols.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        for (std::size_t level=0; level<nLevels; ++level)
        {
            #pragma omp parallel
            {
                #pragma omp single
                {
                    if (sweep == 0 && level == 0)
                        detectedThreads = omp_get_num_threads();
                }
                #pragma omp for schedule(static)
                for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
                {
                    const label cell = waveCells[row];
                    scalar psii = source[cell];
                    for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                        psii -= coeffs[p]*psi[cols[p]];
                    psi[cell] = psii/diag[row];
                }
            }
        }
    }
}

void persistentOpenMpSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps,
    int& detectedThreads
)
{
    const label* const levelStarts = schedule.levelStarts.data();
    const label* const waveCells = schedule.waveCells.data();
    const label* const rowStarts = schedule.rowStarts.data();
    const label* const cols = schedule.cols.data();
    const scalar* const coeffs = schedule.coeffs.data();
    const scalar* const diag = schedule.diag.data();
    const scalar* const source = sourceField.data();
    scalar* const psi = psiField.data();
    const std::size_t nLevels = schedule.levelStarts.size() - 1;

    #pragma omp parallel
    {
        #pragma omp single
        detectedThreads = omp_get_num_threads();

        for (label sweep=0; sweep<nSweeps; ++sweep)
        {
            for (std::size_t level=0; level<nLevels; ++level)
            {
                #pragma omp for schedule(static)
                for (label row=levelStarts[level]; row<levelStarts[level + 1]; ++row)
                {
                    const label cell = waveCells[row];
                    scalar psii = source[cell];
                    for (label p=rowStarts[row]; p<rowStarts[row + 1]; ++p)
                        psii -= coeffs[p]*psi[cols[p]];
                    psi[cell] = psii/diag[row];
                }
                // No nowait: the implicit barrier orders dependency levels.
            }
        }
    }
}

}
