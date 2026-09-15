# Focused Index-Kahn experiment, renumbered MTBHPC, 1T

```sh
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE taskset -c 2 ./build/Test-GaussSeidel build/mesh-data/MTBHPC_small_renumbered/polyMesh --hpc-index-small-series
```

8,613,999 cells. Existing GCC -O2 flags; unchanged kernels, packing and
scheduler algorithms. Geometry baseline is Geometry-X row-degree width 2048.
All five variants in one randomized process: five samples, 81 sweeps/sample,
three sweeps/call, six warm-up sweeps, seed 1592594996.
Setup, correctness, diagnostics and vector resets excluded from timing.
[Raw output](build/index-kahn-renumbered-small-1t.log).

| Variant | Median ms/sweep | CV % | ns/cell | Speedup Ref | Speedup Geometry |
|---|---:|---:|---:|---:|---:|
| Reference GS | 88.6531 | 0.648 | 10.2917 | 1.0000 | 0.9653 |
| Geometry-X row-degree 2048 | 85.5751 | 0.740 | 9.9344 | 1.0360 | 1.0000 |
| Index-Kahn 512 | 55.6590 | 0.377 | 6.4615 | 1.5928 | 1.5375 |
| Index-Kahn 1024 | 55.5948 | 0.733 | 6.4540 | 1.5946 | 1.5393 |
| Index-Kahn 2048 | 57.5168 | 1.226 | 6.6771 | 1.5413 | 1.4878 |

| Schedule | Levels | Mean width | Median width | Cell-ID gap median/p90/p95 |
|---|---:|---:|---:|---:|
| Geometry-X 2048 | 4322 | 1993.06 | 2048 | 8 / 519 / 4629 |
| Index-Kahn 512 | 16885 | 510.157 | 512 | 4 / 12 / 20 |
| Index-Kahn 1024 | 8499 | 1013.53 | 1024 | 4 / 13 / 22 |
| Index-Kahn 2048 | 4316 | 1995.83 | 2048 | 4 / 14 / 24 |

All four schedules passed exact one-sweep comparison to Reference (max
difference zero), and all dependency edges passed strict level ordering.
Index construction checks full unique cell coverage and ready membership.
Gaps exclude transitions across dependency levels.

Widths 512 and 1024 are effectively tied (0.12% median difference), around
1.59x Reference and 1.54x Geometry. Width 2048 takes about 3.46% longer
than width 1024 in this run. Width 1024 uses roughly half as many levels as
512, but no parallel performance was measured.

These absolute timings come from a fresh run; comparisons with prior
ordinary-mesh runs are not a controlled measurement of renumbering alone.
