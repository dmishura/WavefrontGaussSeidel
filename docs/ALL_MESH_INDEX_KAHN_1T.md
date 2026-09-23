# Reference GS vs Index-Kahn across all benchmark meshes

Run date: 2026-09-22

## Method

- Build: GCC, `-O2 -g`
- Threads: 1 (`OMP_NUM_THREADS=1`)
- Pinning: `taskset -c 2`
- Samples: 5
- Sweeps per sample: 81
- Sweeps per smoother call: 3
- Warm-up sweeps: 6
- Randomized implementation order, seed `1592594996`
- Index-Kahn kernel: packed row-degree scalar

Schedule construction, mesh reading, correctness checks, and field resets are
outside the timed regions.

## Correctness

Every Index-Kahn schedule passed dependency validation. For every mesh and
width, the candidate was bitwise identical to Reference GS after:

- one sweep from zero initial state;
- one sweep from non-zero initial state;
- three sweeps from non-zero initial state;
- two calls of three sweeps from non-zero initial state.

## Performance

Times are median milliseconds per sweep. Parentheses contain speedup relative
to Reference GS. CV is shown separately below.

| Mesh | Reference GS | Index-Kahn 1024 | Index-Kahn 2048 | Index-Kahn 4096 | Best |
|---|---:|---:|---:|---:|---|
| MTB_example | 3.271 | 2.249 (1.45x) | 2.354 (1.39x) | 2.615 (1.25x) | 1024 |
| AHBody original | 23.061 | 20.763 (1.11x) | 19.990 (1.15x) | 21.844 (1.06x) | 2048 |
| AHBody renumbered | 21.108 | 17.130 (1.23x) | 17.749 (1.19x) | 20.565 (1.03x) | 1024 |
| MTBHPC_small original | 113.662 | 87.430 (1.30x) | 91.650 (1.24x) | 99.031 (1.15x) | 1024 |
| MTBHPC_small renumbered | 90.681 | 60.919 (1.49x) | 60.961 (1.49x) | 67.478 (1.34x) | 1024 |
| WD_DamBreak original | 107.256 | 77.422 (1.39x) | 106.154 (1.01x) | 112.666 (0.95x) | 1024 |
| WD_DamBreak renumbered | 53.785 | 44.079 (1.22x) | 42.958 (1.25x) | 43.186 (1.25x) | 2048 |

| Mesh | Reference CV | IK-1024 CV | IK-2048 CV | IK-4096 CV |
|---|---:|---:|---:|---:|
| MTB_example | 0.68% | 0.40% | 0.97% | 1.31% |
| AHBody original | 2.46% | 5.91% | 6.09% | 9.58% |
| AHBody renumbered | 2.09% | 0.54% | 1.85% | 1.22% |
| MTBHPC_small original | 2.49% | 3.45% | 4.76% | 3.31% |
| MTBHPC_small renumbered | 1.05% | 1.27% | 0.82% | 0.80% |
| WD_DamBreak original | 0.83% | 1.33% | 1.25% | 1.00% |
| WD_DamBreak renumbered | 1.76% | 0.88% | 0.97% | 1.13% |

## Observations

- Index-Kahn 1024 wins on five of seven archives.
- Width 2048 wins narrowly on AHBody original and WD_DamBreak renumbered.
- Width 4096 never wins this 1-thread series.
- The only regression relative to Reference GS is WD_DamBreak original at
  width 4096 (about 4.8% slower).
- Renumbering substantially improves both Reference GS and Index-Kahn on the
  two largest mesh families. On WD_DamBreak, the median within-level cell-ID
  delta changes from 228 to 1.
- AHBody original has high timing noise for the Index-Kahn variants; its small
  difference between widths 1024 and 2048 should be treated cautiously.

Raw logs are generated under `build/all-mesh-index-results/` and are not
tracked.
