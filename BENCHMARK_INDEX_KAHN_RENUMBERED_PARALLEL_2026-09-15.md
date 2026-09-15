# Index-Kahn renumbered MTBHPC: 2T and 4T

Same five variants as the focused 1T series: Reference GS, Geometry-X
row-degree 2048, Index-Kahn 512/1024/2048. Reference always runs on one
thread. All wavefront variants use the existing persistent OpenMP blocked
row-degree kernel at multiple threads with blockSize=64. No kernel changes.
The previous 1T series uses the unblocked linear row-degree kernel.

Existing GCC -O2 flags, 8,613,999 cells, five randomized samples of 81 sweeps,
three sweeps/call, six warm-up sweeps, seed 1592594996.
Preprocessing, correctness and field resets excluded from timing.
Runs performed sequentially with OMP_DYNAMIC=FALSE.

```sh
OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE taskset -c 2-3 ./build/Test-GaussSeidel build/mesh-data/MTBHPC_small_renumbered/polyMesh --hpc-index-small-series
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE taskset -c 2-5 ./build/Test-GaussSeidel build/mesh-data/MTBHPC_small_renumbered/polyMesh --hpc-index-small-series
```

[2T log](build/index-kahn-renumbered-small-2t.log),
[4T log](build/index-kahn-renumbered-small-4t.log).

| Run | Variant | Actual threads | Median ms/sweep | CV % | ns/cell | Speedup Ref | Speedup Geometry |
|---|---|---:|---:|---:|---:|---:|---:|
| 2T | Reference GS | 1 | 89.5554 | 0.520 | 10.3965 | 1.000 | 0.595 |
| 2T | Geometry 2048 | 2 | 53.2444 | 1.894 | 6.1812 | 1.682 | 1.000 |
| 2T | Index-Kahn 512 | 2 | 40.8231 | 1.521 | 4.7392 | 2.194 | 1.304 |
| 2T | Index-Kahn 1024 | 2 | 38.3880 | 0.912 | 4.4565 | 2.333 | 1.387 |
| 2T | Index-Kahn 2048 | 2 | 37.4366 | 1.435 | 4.3460 | 2.392 | 1.422 |
| 4T | Reference GS | 1 | 89.3477 | 1.240 | 10.3724 | 1.000 | 0.455 |
| 4T | Geometry 2048 | 4 | 40.6869 | 1.302 | 4.7233 | 2.196 | 1.000 |
| 4T | Index-Kahn 512 | 4 | 34.3918 | 0.845 | 3.9926 | 2.598 | 1.183 |
| 4T | Index-Kahn 1024 | 4 | 31.9888 | 0.847 | 3.7136 | 2.793 | 1.272 |
| 4T | Index-Kahn 2048 | 4 | 30.5625 | 1.001 | 3.5480 | 2.923 | 1.331 |

All schedules in both runs: exact one-sweep max difference from Reference
zero; dependency validation PASS; detected team sizes match requests.
Blocks cannot cross levels. Reader tests PASS.

| Schedule | Levels | Blocks |
|---|---:|---:|
| Geometry 2048 | 4322 | 134712 |
| Index-Kahn 512 | 16885 | 134645 |
| Index-Kahn 1024 | 8499 | 134663 |
| Index-Kahn 2048 | 4316 | 134671 |

Best among measured widths at both thread counts is 2048. It improves from
57.5168 ms at 1T in the previous focused run to 37.4366 ms at 2T (1.536x)
and 30.5625 ms at 4T (1.882x). The 2T-to-4T improvement is 1.225x.
These scaling ratios compare separate runs and include the change from the
serial linear kernel to the parallel blocked kernel.

Wider fronts help here, consistent with reduced barrier overhead and more
work per level; this experiment does not isolate the separate costs.
