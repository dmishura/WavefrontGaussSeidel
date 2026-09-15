# Index-prioritized Kahn, ordinary MTBHPC, 1T

Command:

```sh
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE taskset -c 2 ./build/Test-GaussSeidel build/mesh-data/MTBHPC_small/polyMesh --hpc-index-series
```

8,613,999 cells; existing GCC -O2 flags and unchanged scalar row-degree kernel.
Five randomized samples of 81 sweeps, calls of three, six warm-up sweeps,
seed 1592594996. All eight implementations measured in one process.
Setup, diagnostics, correctness and vector resets excluded from timing.
[Raw log](build/index-kahn-original-1t.log).

Index-Kahn selects smallest ready IDs afresh on every level. Index-window
starts at a persistent forward cursor, extends its ID range to fill the
target, and wraps when no higher ready IDs remain. Newly ready cells enter
only the next level. The existing packer sorts selected cells within levels.
Geometry baseline is the existing row-degree Geometry-X width 2048.

| Variant | ms/sweep | CV % | ns/cell | Speedup Ref | Speedup Geometry |
|---|---:|---:|---:|---:|---:|
| Reference | 101.508 | 0.623 | 11.784 | 1.000 | 1.181 |
| Geometry 2048 | 119.844 | 1.428 | 13.913 | 0.847 | 1.000 |
| Index-Kahn 1024 | 76.365 | 1.004 | 8.865 | 1.329 | 1.569 |
| Index-Kahn 2048 | 79.030 | 0.702 | 9.175 | 1.284 | 1.516 |
| Index-Kahn 8192 | 89.464 | 0.977 | 10.386 | 1.135 | 1.340 |
| Index-window 1024 | 132.088 | 0.442 | 15.334 | 0.768 | 0.907 |
| Index-window 2048 | 133.469 | 0.595 | 15.495 | 0.761 | 0.898 |
| Index-window 8192 | 136.721 | 2.083 | 15.872 | 0.742 | 0.877 |

Cell-ID gaps are absolute differences between successive packed rows within
levels, excluding transitions between levels. They do not measure psi cache
reuse or cross-level locality.

| Schedule | Levels | Mean width | Median width | Gap median | Gap p90 | Gap p95 |
|---|---:|---:|---:|---:|---:|---:|
| Geometry 2048 | 4253 | 2025.39 | 2048 | 13 | 1470 | 4419 |
| Index-Kahn 1024 | 8415 | 1023.65 | 1024 | 2 | 9 | 19 |
| Index-Kahn 2048 | 4219 | 2041.72 | 2048 | 2 | 10 | 20 |
| Index-Kahn 8192 | 1073 | 8027.96 | 8192 | 2 | 11 | 21 |
| Index-window 1024 | 8430 | 1021.83 | 1024 | 3 | 18 | 31 |
| Index-window 2048 | 4237 | 2033.04 | 2048 | 4 | 19 | 32 |
| Index-window 8192 | 1094 | 7873.86 | 8192 | 4 | 20 | 35 |

All schedules: exact one-sweep max difference versus Reference = 0; dependency
validation PASS. Index construction also validates complete unique cell
coverage and ready-set predecessor counts. Reader tests PASS.

Best measured candidate: Index-Kahn 1024, 1.329x Reference (24.8% less time)
and 1.569x Geometry (36.3% less time). Higher widths hurt 1T performance here.
The cyclic window loses despite small within-level ID gaps; these diagnostics
alone cannot identify the cause. No parallel timings or further optimizations
were performed.
