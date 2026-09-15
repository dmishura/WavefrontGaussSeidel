# 1T performance recovery — 2026-09-15

## Context

Goal: accelerate the Wavefront implementation; original GaussSeidel is the reference. Unroll-8 previously showed no useful improvement. These measurements restore the performance baseline after a session restart.

## Method

- Intel Core i5-11400, exposed through a Microsoft hypervisor; 6 cores / 12 logical CPUs.
- Mesh: `build/mesh-data/MTBHPC_small_renumbered/polyMesh`, 8,613,999 cells, 25,952,973 internal faces.
- Existing build: `make -j2` reported up to date; default flags `-O2 -g`, no source changes for this run.
- `OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE taskset -c 2`.
- Each series: 6 warm-up sweeps, 5 samples × 81 sweeps, calls of 3 sweeps. Setup/reset excluded from timing.
- Variant order shuffled per sample with seed 1592594996. Series executed sequentially.
- One invocation per series. CV is within-series sample variation, not a confidence interval. Compare optimizations against their baseline in the same series.
- Existing working tree changes (requirement 63 / unroll-8 and AVX-512 division) retained. HEAD: d0d3014.

## Results

### hpc-series

Raw log: [hpc-series.log](build/recovery-1t-20260915/hpc-series.log)

| Variant | Median ms/sweep | ns/cell/sweep | CV % |
|---|---:|---:|---:|
| Reference GS | 95.194 | 11.051 | 0.27 |
| Packed scalar | 136.129 | 15.803 | 0.45 |
| Geometry width 256 scalar | 90.701 | 10.530 | 1.59 |
| Geometry width 512 scalar | 90.199 | 10.471 | 4.77 |
| Geometry width 1024 scalar | 91.284 | 10.597 | 0.86 |
| Geometry width 2048 scalar | 97.125 | 11.275 | 1.96 |
| Geometry width 4096 scalar | 100.597 | 11.678 | 1.11 |
| Geometry width 8192 scalar | 104.378 | 12.117 | 2.60 |
| Geometry width 16384 scalar | 111.872 | 12.987 | 0.45 |

### hpc-geometry-kernels

Raw log: [hpc-geometry-kernels.log](build/recovery-1t-20260915/hpc-geometry-kernels.log)

| Variant | Median ms/sweep | ns/cell/sweep | CV % |
|---|---:|---:|---:|
| Geometry 2048 scalar | 96.562 | 11.210 | 0.84 |
| Geometry 2048 generic 4-way | 97.135 | 11.276 | 0.54 |
| Geometry 2048 AVX-512 rows | 178.001 | 20.664 | 0.38 |
| Geometry 4096 scalar | 99.281 | 11.525 | 0.39 |
| Geometry 4096 generic 4-way | 100.495 | 11.666 | 0.77 |
| Geometry 4096 AVX-512 rows | 181.985 | 21.127 | 0.58 |
| Geometry 8192 scalar | 104.015 | 12.075 | 0.29 |
| Geometry 8192 generic 4-way | 103.918 | 12.064 | 0.78 |
| Geometry 8192 AVX-512 rows | 188.403 | 21.872 | 1.23 |

### hpc-row-degree

Raw log: [hpc-row-degree.log](build/recovery-1t-20260915/hpc-row-degree.log)

| Variant | Median ms/sweep | ns/cell/sweep | CV % |
|---|---:|---:|---:|
| Geometry 2048 scalar | 96.249 | 11.174 | 0.34 |
| Geometry 2048 row-degree | 94.639 | 10.987 | 0.40 |
| Geometry 2048 row-degree scalar unrolled-8 | 96.548 | 11.208 | 0.53 |
| Geometry 2048 row-degree AVX-512 divide | 96.212 | 11.169 | 0.93 |

### hpc-prefetch-series

Raw log: [hpc-prefetch-series.log](build/recovery-1t-20260915/hpc-prefetch-series.log)

| Variant | Median ms/sweep | ns/cell/sweep | CV % |
|---|---:|---:|---:|
| Packed scalar | 134.348 | 15.597 | 0.72 |
| Packed prefetch D=4 | 143.715 | 16.684 | 0.60 |
| Packed prefetch D=8 | 144.408 | 16.764 | 0.39 |
| Packed prefetch D=16 | 143.186 | 16.622 | 0.84 |
| Packed prefetch D=32 | 142.731 | 16.570 | 0.61 |
| Geometry width 8192 scalar | 101.671 | 11.803 | 0.33 |
| Geometry width 8192 prefetch D=4 | 109.311 | 12.690 | 0.46 |
| Geometry width 8192 prefetch D=8 | 109.922 | 12.761 | 0.70 |
| Geometry width 8192 prefetch D=16 | 108.836 | 12.635 | 0.41 |
| Geometry width 8192 prefetch D=32 | 109.107 | 12.666 | 0.69 |

### hpc-hybrid-series

Raw log: [hpc-hybrid-series.log](build/recovery-1t-20260915/hpc-hybrid-series.log)

| Variant | Median ms/sweep | ns/cell/sweep | CV % |
|---|---:|---:|---:|
| Packed scalar | 135.156 | 15.690 | 0.61 |
| Hybrid Packed scalar | 321.306 | 37.300 | 0.30 |
| Geometry width 8192 scalar | 101.911 | 11.831 | 1.22 |
| Hybrid Geometry width 8192 scalar | 254.442 | 29.538 | 0.40 |

## Conclusions

- Geometry widths 256–1024 take about 90–91 ms/sweep versus 95.19 ms for Reference GS and 136.13 ms for minimal-level Packed. The ranking within 256–1024 is not established given the variability.
- Generic 4-way row interleaving gives essentially no improvement. AVX-512 across rows is about 1.8× slower than scalar.
- Geometry-2048 row-degree: 94.64 ms versus 96.25 ms for its scalar baseline (1.017×).
- Scalar unroll-8: 96.55 ms; AVX-512 divide: 96.21 ms. Neither improves on scalar row-degree.
- First-neighbour prefetch increases time by about 6–8% for all tested distances.
- Hybrid edge values are about 2.38× slower for Packed and 2.50× slower for Geometry-8192.
- All five series passed their existing correctness checks. Row-degree and unroll-8 were exactly equal to scalar; AVX-512 divide max difference was 3.33067e-16, within 1e-12.
- These are single-thread results; they do not establish the best width or kernel for multiple threads.

## Reproduction

```sh
make -j2
for mode in hpc-series hpc-geometry-kernels hpc-row-degree hpc-prefetch-series hpc-hybrid-series; do
  OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE taskset -c 2 ./build/Test-GaussSeidel build/mesh-data/MTBHPC_small_renumbered/polyMesh --"$mode"
done
```
