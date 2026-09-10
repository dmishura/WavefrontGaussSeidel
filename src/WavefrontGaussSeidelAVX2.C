#include "WavefrontGaussSeidel.H"

#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace smootherTest
{

using Foam::label;
using Foam::scalar;

bool avx2GatherAvailable()
{
#if defined(__AVX2__) && defined(__x86_64__)
    return sizeof(label) == sizeof(int) && __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

void serialGatherAvx2Smooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps
)
{
#if defined(__AVX2__)
    if (sizeof(label) != sizeof(int) || !avx2GatherAvailable())
    {
        serialGatherSmooth(psiField, sourceField, schedule, nSweeps);
        return;
    }

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
                __m256d accumulated = _mm256_setzero_pd();
                const label rowEnd = rowStarts[row + 1];
                for (label p=rowStarts[row]; p<rowEnd; p += 4)
                {
                    const label remaining = std::min<label>(4, rowEnd - p);
                    const __m256i mask64 = _mm256_set_epi64x
                    (
                        remaining > 3 ? -1LL : 0LL,
                        remaining > 2 ? -1LL : 0LL,
                        remaining > 1 ? -1LL : 0LL,
                        remaining > 0 ? -1LL : 0LL
                    );
                    const __m128i mask32 = _mm_set_epi32
                    (
                        remaining > 3 ? -1 : 0,
                        remaining > 2 ? -1 : 0,
                        remaining > 1 ? -1 : 0,
                        remaining > 0 ? -1 : 0
                    );
                    const __m256d coefficient =
                        _mm256_maskload_pd(coeffs + p, mask64);
                    const __m128i columns =
                        _mm_maskload_epi32(cols + p, mask32);
                    const __m256d neighbour = _mm256_mask_i32gather_pd
                    (
                        _mm256_setzero_pd(), psi, columns,
                        _mm256_castsi256_pd(mask64), sizeof(scalar)
                    );
                    accumulated = _mm256_add_pd
                    (
                        accumulated, _mm256_mul_pd(coefficient, neighbour)
                    );
                }
                alignas(32) scalar lanes[4];
                _mm256_store_pd(lanes, accumulated);
                const scalar sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
                const label cell = waveCells[row];
                psi[cell] = (source[cell] - sum)/diag[row];
            }
        }
    }
#else
    serialGatherSmooth(psiField, sourceField, schedule, nSweeps);
#endif
}

void serialGatherAvx2AcrossRowsSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps
)
{
#if defined(__AVX2__)
    if (sizeof(label) != sizeof(int) || !avx2GatherAvailable())
    {
        serialGatherSmooth(psiField, sourceField, schedule, nSweeps);
        return;
    }

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
            for (label firstRow=levelStarts[level]; firstRow<levelEnd; firstRow += 4)
            {
                const label rows = std::min<label>(4, levelEnd - firstRow);
                alignas(32) label cells[4]{};
                alignas(32) label starts[4]{};
                alignas(32) label lengths[4]{};
                alignas(32) scalar sourceLanes[4]{};
                alignas(32) scalar diagonalLanes[4]{1, 1, 1, 1};
                alignas(32) scalar coefficientLanes[4]{};
                alignas(32) scalar neighbourLanes[4]{};
                alignas(32) scalar resultLanes[4]{};
                label maxLength = 0;
                for (label lane=0; lane<rows; ++lane)
                {
                    const label row = firstRow + lane;
                    cells[lane] = waveCells[row];
                    starts[lane] = rowStarts[row];
                    lengths[lane] = rowStarts[row + 1] - starts[lane];
                    sourceLanes[lane] = source[cells[lane]];
                    diagonalLanes[lane] = diag[row];
                    maxLength = std::max(maxLength, lengths[lane]);
                }
                __m256d psii = _mm256_load_pd(sourceLanes);
                for (label k=0; k<maxLength; ++k)
                {
                    unsigned activeBits = 0;
                    for (label lane=0; lane<rows; ++lane)
                    {
                        if (k >= lengths[lane]) continue;
                        const label p = starts[lane] + k;
                        coefficientLanes[lane] = coeffs[p];
                        neighbourLanes[lane] = psi[cols[p]];
                        activeBits |= 1u << lane;
                    }
                    const __m256d mask = _mm256_castsi256_pd
                    (
                        _mm256_set_epi64x
                        (
                            activeBits & 8u ? -1LL : 0LL,
                            activeBits & 4u ? -1LL : 0LL,
                            activeBits & 2u ? -1LL : 0LL,
                            activeBits & 1u ? -1LL : 0LL
                        )
                    );
                    const __m256d product = _mm256_mul_pd
                    (
                        _mm256_load_pd(coefficientLanes),
                        _mm256_load_pd(neighbourLanes)
                    );
                    psii = _mm256_blendv_pd(psii, _mm256_sub_pd(psii, product), mask);
                }
                const __m256d result = _mm256_div_pd
                (
                    psii, _mm256_load_pd(diagonalLanes)
                );
                _mm256_store_pd(resultLanes, result);
                for (label lane=0; lane<rows; ++lane)
                    psi[cells[lane]] = resultLanes[lane];
            }
        }
    }
#else
    serialGatherSmooth(psiField, sourceField, schedule, nSweeps);
#endif
}

}
