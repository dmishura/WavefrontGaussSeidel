#include "VariantSelection.H"

#include <algorithm>
#include <array>
#include <iomanip>
#include <ostream>
#include <stdexcept>

namespace smootherTest::harness
{

const char* variantTierName(const VariantTier tier) noexcept
{
    switch (tier)
    {
        case VariantTier::Core: return "core";
        case VariantTier::Experimental: return "experimental";
        case VariantTier::Legacy: return "legacy";
    }
    return "unknown";
}

VariantTier parseVariantTier(const std::string& value)
{
    if (value == "core") return VariantTier::Core;
    if (value == "experimental") return VariantTier::Experimental;
    if (value == "legacy") return VariantTier::Legacy;
    throw std::runtime_error
    (
        "invalid tier '" + value
      + "' (expected core, experimental, legacy, or all)"
    );
}

bool matchesSelection
(
    const VariantMetadata& variant,
    const VariantSelection& selection
)
{
    if (!selection.allTiers && variant.tier != selection.tier) return false;
    for (const std::string& requested : selection.tags)
    {
        if
        (
            std::find
            (
                variant.tags.begin(), variant.tags.end(), requested
            ) == variant.tags.end()
        ) return false;
    }
    return true;
}

std::size_t matchingVariantCount
(
    const std::vector<VariantMetadata>& variants,
    const VariantSelection& selection
)
{
    return static_cast<std::size_t>
    (
        std::count_if
        (
            variants.begin(), variants.end(),
            [&selection](const VariantMetadata& variant)
            {
                return matchesSelection(variant, selection);
            }
        )
    );
}

VariantSelection consumeVariantSelectionArguments(int& argc, char** argv)
{
    VariantSelection selection;
    bool tierSpecified = false;
    int destination = 1;
    for (int source=1; source<argc; ++source)
    {
        const std::string argument = argv[source];
        std::string tierValue;
        std::string tagValue;
        if (argument == "--list-variants")
        {
            selection.listVariants = true;
        }
        else if (argument == "--tier")
        {
            if (++source >= argc)
                throw std::runtime_error("--tier requires a value");
            tierValue = argv[source];
        }
        else if (argument.rfind("--tier=", 0) == 0)
        {
            tierValue = argument.substr(7);
            if (tierValue.empty())
                throw std::runtime_error("--tier requires a value");
        }
        else if (argument == "--tag")
        {
            if (++source >= argc)
                throw std::runtime_error("--tag requires a value");
            tagValue = argv[source];
        }
        else if (argument.rfind("--tag=", 0) == 0)
        {
            tagValue = argument.substr(6);
            if (tagValue.empty())
                throw std::runtime_error("--tag requires a value");
        }
        else
        {
            argv[destination++] = argv[source];
        }

        if (!tierValue.empty())
        {
            tierSpecified = true;
            if (tierValue == "all") selection.allTiers = true;
            else
            {
                selection.tier = parseVariantTier(tierValue);
                selection.allTiers = false;
            }
        }
        if (!tagValue.empty()) selection.tags.push_back(tagValue);
    }
    if (!tierSpecified && !selection.tags.empty()) selection.allTiers = true;
    argc = destination;
    return selection;
}

void printVariantListing
(
    std::ostream& output,
    const std::vector<VariantMetadata>& variants
)
{
    std::array<std::size_t, 3> counts{{0, 0, 0}};
    output << std::left << std::setw(42) << "ID"
        << std::setw(15) << "Tier" << "Tags\n";
    for (const VariantMetadata& variant : variants)
    {
        ++counts[static_cast<std::size_t>(variant.tier)];
        output << std::left << std::setw(42) << variant.id
            << std::setw(15) << variantTierName(variant.tier);
        for (std::size_t tag=0; tag<variant.tags.size(); ++tag)
        {
            if (tag) output << ',';
            output << variant.tags[tag];
        }
        output << '\n';
    }
    output << "\nTotal: " << variants.size()
        << "\nCore: " << counts[static_cast<std::size_t>(VariantTier::Core)]
        << "\nExperimental: "
        << counts[static_cast<std::size_t>(VariantTier::Experimental)]
        << "\nLegacy: "
        << counts[static_cast<std::size_t>(VariantTier::Legacy)] << '\n';
}

} // namespace smootherTest::harness
