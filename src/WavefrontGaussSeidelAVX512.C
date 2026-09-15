#include "WavefrontGaussSeidel.H"

#if defined(__AVX512F__)
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

bool avx512FAvailable()
{
#if defined(__AVX512F__) && defined(__x86_64__)
    return __builtin_cpu_supports("avx512f");
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

void serialGatherAvx512AcrossRowsSmooth
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
            const label levelEnd = levelStarts[level + 1];
            for (label firstRow=levelStarts[level]; firstRow<levelEnd; firstRow += 8)
            {
                const label rows = std::min<label>(8, levelEnd - firstRow);
                const __mmask8 rowMask = rows == 8
                    ? __mmask8(0xff)
                    : __mmask8((1u << rows) - 1u);
                alignas(64) label cells[8]{};
                alignas(64) label starts[8]{};
                alignas(64) label lengths[8]{};
                alignas(64) scalar sourceLanes[8]{};
                alignas(64) scalar coefficientLanes[8]{};
                alignas(64) scalar neighbourLanes[8]{};
                alignas(64) scalar resultLanes[8]{};
                label maxLength = 0;
                // Irregular sparse accesses are deliberately ordinary scalar
                // loads/stores. AVX-512 is used only for the lane accumulators.
                for (label lane=0; lane<rows; ++lane)
                {
                    const label row = firstRow + lane;
                    cells[lane] = waveCells[row];
                    starts[lane] = rowStarts[row];
                    lengths[lane] = rowStarts[row + 1] - starts[lane];
                    sourceLanes[lane] = source[cells[lane]];
                    maxLength = std::max(maxLength, lengths[lane]);
                }
                __m512d psii = _mm512_load_pd(sourceLanes);
                const __m512d diagonal =
                    _mm512_maskz_loadu_pd(rowMask, diag + firstRow);

                for (label k=0; k<maxLength; ++k)
                {
                    __mmask8 active = 0;
                    for (label lane=0; lane<rows; ++lane)
                    {
                        if (k >= lengths[lane]) continue;
                        const label p = starts[lane] + k;
                        coefficientLanes[lane] = coeffs[p];
                        neighbourLanes[lane] = psi[cols[p]];
                        active |= __mmask8(1u << lane);
                    }
                    const __m512d coefficient =
                        _mm512_maskz_load_pd(active, coefficientLanes);
                    const __m512d neighbour =
                        _mm512_maskz_load_pd(active, neighbourLanes);
                    psii = _mm512_mask_sub_pd
                    (
                        psii, active, psii,
                        _mm512_mul_pd(coefficient, neighbour)
                    );
                }
                const __m512d result = _mm512_maskz_div_pd
                (
                    rowMask, psii, diagonal
                );
                _mm512_store_pd(resultLanes, result);
                for (label lane=0; lane<rows; ++lane)
                    psi[cells[lane]] = resultLanes[lane];
            }
        }
    }
#else
    serialGatherSmooth(psiField, sourceField, schedule, nSweeps);
#endif
}

void serialRowDegreeAvx512DivideSmooth
(
    Foam::scalarField& psiField,
    const Foam::scalarField& sourceField,
    const RowDegreeSchedule& schedule,
    const label nSweeps
)
{
#if defined(__AVX512F__)
    if (!avx512FAvailable())
    {
        serialRowDegreeSmooth(psiField, sourceField, schedule, nSweeps);
        return;
    }

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
                const __m128d pair01 = _mm_unpacklo_pd
                (
                    _mm_set_sd(s0), _mm_set_sd(s1)
                );
                const __m128d pair23 = _mm_unpacklo_pd
                (
                    _mm_set_sd(s2), _mm_set_sd(s3)
                );
                const __m128d pair45 = _mm_unpacklo_pd
                (
                    _mm_set_sd(s4), _mm_set_sd(s5)
                );
                const __m128d pair67 = _mm_unpacklo_pd
                (
                    _mm_set_sd(s6), _mm_set_sd(s7)
                );
                const __m256d lowerHalf = _mm256_insertf128_pd
                (
                    _mm256_castpd128_pd256(pair01), pair23, 1
                );
                const __m256d upperHalf = _mm256_insertf128_pd
                (
                    _mm256_castpd128_pd256(pair45), pair67, 1
                );
                const __m512d numerators = _mm512_shuffle_f64x2
                (
                    _mm512_castpd256_pd512(lowerHalf),
                    _mm512_castpd256_pd512(upperHalf),
                    0x44
                );
                const __m512d results = _mm512_div_pd
                (
                    numerators, _mm512_loadu_pd(diag + row)
                );
                alignas(64) scalar divided[8];
                _mm512_store_pd(divided, results);
                psi[cell0] = divided[0];
                psi[cell1] = divided[1];
                psi[cell2] = divided[2];
                psi[cell3] = divided[3];
                psi[cell4] = divided[4];
                psi[cell5] = divided[5];
                psi[cell6] = divided[6];
                psi[cell7] = divided[7];
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
#else
    serialRowDegreeSmooth(psiField, sourceField, schedule, nSweeps);
#endif
}

}
