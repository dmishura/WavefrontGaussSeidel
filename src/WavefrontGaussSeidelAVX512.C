#include "WavefrontGaussSeidel.H"

#if defined(__AVX512F__) && defined(__AVX512VL__)
#include <immintrin.h>
#endif

namespace smootherTest
{

using Foam::label;
using Foam::scalar;

bool avx512GatherAvailable()
{
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__x86_64__)
    return sizeof(label) == sizeof(int) && __builtin_cpu_supports("avx512f")
        && __builtin_cpu_supports("avx512vl");
#else
    return false;
#endif
}

void serialGatherAvx512Smooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const WavefrontSchedule& schedule,
    const label nSweeps
)
{
#if defined(__AVX512F__) && defined(__AVX512VL__)
    if (sizeof(label) != sizeof(int) || !avx512GatherAvailable())
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
                __m512d accumulated = _mm512_setzero_pd();
                const label rowEnd = rowStarts[row + 1];
                for (label p=rowStarts[row]; p<rowEnd; p += 8)
                {
                    const unsigned remaining =
                        static_cast<unsigned>(rowEnd - p);
                    const __mmask8 mask = remaining >= 8
                        ? __mmask8(0xff)
                        : __mmask8((1u << remaining) - 1u);
                    const __m512d coefficient =
                        _mm512_maskz_loadu_pd(mask, coeffs + p);
                    const __m256i column =
                        _mm256_maskz_loadu_epi32(mask, cols + p);
                    const __m512d value = _mm512_mask_i32gather_pd
                    (
                        _mm512_setzero_pd(), mask, column, psi, sizeof(scalar)
                    );
                    accumulated = _mm512_add_pd
                    (
                        accumulated, _mm512_mul_pd(coefficient, value)
                    );
                }
                const label cell = waveCells[row];
                const scalar psii =
                    source[cell] - _mm512_reduce_add_pd(accumulated);
                psi[cell] = psii/diag[row];
            }
        }
    }
#else
    serialGatherSmooth(psiField, sourceField, schedule, nSweeps);
#endif
}

}
