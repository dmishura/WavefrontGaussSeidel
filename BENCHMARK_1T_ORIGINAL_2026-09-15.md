# 1T ordinary MTBHPC vs renumbered — 2026-09-15

## Method

- Ordinary mesh: `build/mesh-data/MTBHPC_small/polyMesh`, 8,613,999 cells and 25,952,973 internal faces.
- Same Intel i5-11400 environment and CPU 2 pinning as the earlier renumbered run.
- `OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE taskset -c 2`; sequential series, 6 warm-up sweeps, 5 randomized samples of 81 sweeps, calls grouped in 3 sweeps; seed 1592594996. Setup and resets excluded.
- Separate driver copied from `tests/Test-GaussSeidel.C` to `build/recovery-1t-original-20260915/Test-GaussSeidel.C`. Only the two width arrays for hpc-series and hpc-geometry-kernels changed to 1024, 2048, 8192. All other series unchanged.
- Driver compiled with the same -O2 -g, C++17, OpenMP, include and ITT flags; linked to the existing unchanged kernel objects. Main source and executable unchanged.
- One invocation per series. Earlier renumbered values are from separate runs, not an interleaved cross-mesh experiment. Renumbering also changes dependency order; this comparison is not an isolated measurement of locality.
- All five series passed their existing checks. Row-degree/unroll-8 differences were zero; AVX-512 division max difference 3.33067e-16 passed 1e-12 tolerance.

## Results

Times are medians in ms/sweep. Ratio >1 means the ordinary mesh takes longer. CV is within-series sample variation, not a confidence interval.

### hpc-series

[Raw ordinary log](build/recovery-1t-original-20260915/hpc-series.log) · [Raw renumbered log](build/recovery-1t-20260915/hpc-series.log)

| Variant | Ordinary ms | CV % | ns/cell | Renumbered ms | CV % | Ordinary / renumbered |
|---|---:|---:|---:|---:|---:|---:|
| Reference GS | 109.485 | 1.31 | 12.710 | 95.194 | 0.27 | 1.150× |
| Packed scalar | 162.933 | 1.13 | 18.915 | 136.129 | 0.45 | 1.197× |
| Geometry width 1024 scalar | 132.580 | 0.63 | 15.391 | 91.284 | 0.86 | 1.452× |
| Geometry width 2048 scalar | 133.098 | 1.13 | 15.451 | 97.125 | 1.96 | 1.370× |
| Geometry width 8192 scalar | 131.013 | 1.72 | 15.209 | 104.378 | 2.60 | 1.255× |

### hpc-geometry-kernels

[Raw ordinary log](build/recovery-1t-original-20260915/hpc-geometry-kernels.log) · [Raw renumbered log](build/recovery-1t-20260915/hpc-geometry-kernels.log)

| Variant | Ordinary ms | CV % | ns/cell | Renumbered ms | CV % | Ordinary / renumbered |
|---|---:|---:|---:|---:|---:|---:|
| Geometry 1024 scalar | 131.611 | 4.79 | 15.279 | — | — | — |
| Geometry 1024 generic 4-way | 133.020 | 4.52 | 15.442 | — | — | — |
| Geometry 1024 AVX-512 rows | 228.685 | 2.51 | 26.548 | — | — | — |
| Geometry 2048 scalar | 134.471 | 0.75 | 15.611 | 96.562 | 0.84 | 1.393× |
| Geometry 2048 generic 4-way | 134.785 | 1.40 | 15.647 | 97.135 | 0.54 | 1.388× |
| Geometry 2048 AVX-512 rows | 226.969 | 1.67 | 26.349 | 178.001 | 0.38 | 1.275× |
| Geometry 8192 scalar | 132.582 | 5.85 | 15.391 | 104.015 | 0.29 | 1.275× |
| Geometry 8192 generic 4-way | 131.401 | 0.76 | 15.254 | 103.918 | 0.78 | 1.264× |
| Geometry 8192 AVX-512 rows | 224.963 | 1.93 | 26.116 | 188.403 | 1.23 | 1.194× |

### hpc-row-degree

[Raw ordinary log](build/recovery-1t-original-20260915/hpc-row-degree.log) · [Raw renumbered log](build/recovery-1t-20260915/hpc-row-degree.log)

| Variant | Ordinary ms | CV % | ns/cell | Renumbered ms | CV % | Ordinary / renumbered |
|---|---:|---:|---:|---:|---:|---:|
| Geometry 2048 scalar | 131.354 | 0.58 | 15.249 | 96.249 | 0.34 | 1.365× |
| Geometry 2048 row-degree | 127.688 | 0.48 | 14.823 | 94.639 | 0.40 | 1.349× |
| Geometry 2048 row-degree scalar unrolled-8 | 131.810 | 0.75 | 15.302 | 96.548 | 0.53 | 1.365× |
| Geometry 2048 row-degree AVX-512 divide | 131.907 | 0.66 | 15.313 | 96.212 | 0.93 | 1.371× |

### hpc-prefetch-series

[Raw ordinary log](build/recovery-1t-original-20260915/hpc-prefetch-series.log) · [Raw renumbered log](build/recovery-1t-20260915/hpc-prefetch-series.log)

| Variant | Ordinary ms | CV % | ns/cell | Renumbered ms | CV % | Ordinary / renumbered |
|---|---:|---:|---:|---:|---:|---:|
| Packed scalar | 159.596 | 2.07 | 18.528 | 134.348 | 0.72 | 1.188× |
| Packed prefetch D=4 | 169.313 | 3.35 | 19.656 | 143.715 | 0.60 | 1.178× |
| Packed prefetch D=8 | 164.826 | 1.96 | 19.135 | 144.408 | 0.39 | 1.141× |
| Packed prefetch D=16 | 163.725 | 2.26 | 19.007 | 143.186 | 0.84 | 1.143× |
| Packed prefetch D=32 | 164.969 | 2.75 | 19.151 | 142.731 | 0.61 | 1.156× |
| Geometry width 8192 scalar | 129.915 | 4.45 | 15.082 | 101.671 | 0.33 | 1.278× |
| Geometry width 8192 prefetch D=4 | 138.395 | 3.56 | 16.066 | 109.311 | 0.46 | 1.266× |
| Geometry width 8192 prefetch D=8 | 135.169 | 4.68 | 15.692 | 109.922 | 0.70 | 1.230× |
| Geometry width 8192 prefetch D=16 | 135.724 | 2.91 | 15.756 | 108.836 | 0.41 | 1.247× |
| Geometry width 8192 prefetch D=32 | 137.672 | 3.39 | 15.982 | 109.107 | 0.69 | 1.262× |

### hpc-hybrid-series

[Raw ordinary log](build/recovery-1t-original-20260915/hpc-hybrid-series.log) · [Raw renumbered log](build/recovery-1t-20260915/hpc-hybrid-series.log)

| Variant | Ordinary ms | CV % | ns/cell | Renumbered ms | CV % | Ordinary / renumbered |
|---|---:|---:|---:|---:|---:|---:|
| Packed scalar | 158.278 | 1.73 | 18.375 | 135.156 | 0.61 | 1.171× |
| Hybrid Packed scalar | 302.951 | 1.80 | 35.170 | 321.306 | 0.30 | 0.943× |
| Geometry width 8192 scalar | 125.872 | 2.90 | 14.612 | 101.911 | 1.22 | 1.235× |
| Hybrid Geometry width 8192 scalar | 319.961 | 3.23 | 37.144 | 254.442 | 0.40 | 1.258× |

## Conclusions

- Ordinary Reference GS: 109.485 ms; Packed: 162.933 ms. Geometry 1024/2048/8192: 132.580/133.098/131.013 ms. Geometry improves Packed by about 1.23–1.24×, but takes about 20–22% longer than GS. Width ranking is weak relative to variability.
- Geometry 2048 row-degree takes 127.688 ms vs 131.354 ms for its same-series scalar baseline: 1.02871×. Unroll-8 (131.810 ms) and AVX-512 divide (131.907 ms) do not improve row-degree.
- Generic 4-way remains approximately neutral (±1%). AVX-512 across rows is about 1.69–1.74× slower than scalar.
- Prefetch medians are 2.6–6.5% slower than their baselines. CV is elevated (about 2–4.7%); these measurements show no useful gain and do not establish a fine ranking among distances.
- Hybrid takes about 1.91× longer than Packed and 2.54× longer than Geometry 8192.
- Compared with the earlier renumbered run, ordinary GS is about 15% slower, Packed 20%, and the selected scalar Geometry widths 26–45%. Cross-run machine variability remains a limitation.

## Reproduce with retained executable

```sh
for mode in hpc-series hpc-geometry-kernels hpc-row-degree hpc-prefetch-series hpc-hybrid-series; do
  OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE taskset -c 2 stdbuf -oL ./build/recovery-1t-original-20260915/Test-GaussSeidel build/mesh-data/MTBHPC_small/polyMesh --"$mode"
done
```
