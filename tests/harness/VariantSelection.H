#ifndef SmootherTest_VariantSelection_H
#define SmootherTest_VariantSelection_H

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace smootherTest::harness
{

enum class VariantTier
{
    Core,
    Experimental,
    Legacy
};

struct VariantMetadata
{
    std::string name;
    std::string id;
    VariantTier tier = VariantTier::Experimental;
    std::vector<std::string> tags;
};

struct VariantSelection
{
    VariantTier tier = VariantTier::Core;
    bool allTiers = false;
    std::vector<std::string> tags;
    bool listVariants = false;
};

const char* variantTierName(VariantTier tier) noexcept;
VariantTier parseVariantTier(const std::string& value);

// Multiple tags use AND semantics: every requested tag must be present.
bool matchesSelection
(
    const VariantMetadata& variant,
    const VariantSelection& selection
);

std::size_t matchingVariantCount
(
    const std::vector<VariantMetadata>& variants,
    const VariantSelection& selection
);

VariantSelection consumeVariantSelectionArguments(int& argc, char** argv);

void printVariantListing
(
    std::ostream& output,
    const std::vector<VariantMetadata>& variants
);

} // namespace smootherTest::harness

#endif
