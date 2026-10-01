#ifndef SmootherTest_fwHierarchy_H
#define SmootherTest_fwHierarchy_H

#include "GAMGAgglomeration.H"
#include "GAMGControls.H"
#include "fwMatrix.H"

#include <vector>

namespace smootherTest::fwgamg
{

enum class RowOrdering
{
    Natural,
    IndexKahn
};

enum class CoarseRenumbering
{
    None,
    MinimumFineIndex
};

struct HierarchyOptions
{
    Foam::label width = 1024;
    RowOrdering rowOrdering = RowOrdering::IndexKahn;
    CoarseRenumbering coarseRenumbering = CoarseRenumbering::None;
    bool buildFlatEdges = false;
    bool buildDirectFlatEdges = false;
    bool buildNativeFaceOrderDirectEdges = false;
};

class Hierarchy
{
    gamg::GAMGAgglomeration agglomeration_;
    std::vector<Matrix> matrices_;
    std::vector<Foam::labelField> restrictionMaps_;
    std::vector<Foam::labelField> nativeToAggCells_;
    std::vector<Foam::labelField> aggToNativeCells_;

public:
    Hierarchy
    (
        const Foam::lduMatrix& matrix,
        const gamg::GAMGControls& controls,
        HierarchyOptions options,
        TimingStats* timing = nullptr
    );

    Hierarchy
    (
        const Foam::lduMatrix& matrix,
        const gamg::GAMGControls& controls,
        Foam::label width,
        TimingStats* timing = nullptr
    )
    :
        Hierarchy
        (
            matrix,
            controls,
            HierarchyOptions
            {
                width, RowOrdering::IndexKahn,
                CoarseRenumbering::None, false, false, false
            },
            timing
        )
    {}

    std::size_t size() const noexcept { return matrices_.size(); }
    const Matrix& operator[](const std::size_t level) const
    {
        return matrices_.at(level);
    }
    const std::vector<Matrix>& matrices() const noexcept { return matrices_; }
    const gamg::GAMGAgglomeration& agglomeration() const noexcept
    {
        return agglomeration_;
    }
    const Foam::labelField& restrictionMap(const std::size_t fineLevel) const
    {
        return restrictionMaps_.at(fineLevel);
    }
    const Foam::labelField& nativeToAggCells(const std::size_t level) const
    {
        return nativeToAggCells_.at(level);
    }
    const Foam::labelField& aggToNativeCells(const std::size_t level) const
    {
        return aggToNativeCells_.at(level);
    }

    void restrictField
    (
        Foam::scalarField& coarse,
        const Foam::scalarField& fine,
        std::size_t fineLevel,
        TimingStats* timing = nullptr
    ) const;

    void prolongField
    (
        Foam::scalarField& fine,
        const Foam::scalarField& coarse,
        std::size_t fineLevel,
        TimingStats* timing = nullptr
    ) const;
};

} // namespace smootherTest::fwgamg

#endif
