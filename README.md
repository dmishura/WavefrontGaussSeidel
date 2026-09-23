# Wavefront Gauss-Seidel Lab

This public repository explores Gauss-Seidel smoothers for OpenFOAM-style
LDU matrices. It contains a standalone compatibility layer, realistic mesh
workloads, correctness tests, performance benchmarks, and experimental
wavefront/Index-Kahn implementations for serial, OpenMP, and SIMD execution.

The main goal is to understand how dependency scheduling, data layout, cache
locality, and parallel execution affect smoothing quality and performance,
while keeping the original OpenFOAM Gauss-Seidel implementation as the
reference baseline.

The OpenMP implementation is included as a straightforward parallelization
example. No OpenMP-specific tuning or optimization has been applied: it runs
essentially "as is", and its current performance results are unlikely to be
optimal.

The repository also contains experimental OpenFOAM v2606 integration patches.
OpenFOAM source trees themselves are treated as external, immutable references.
The author is currently working on integration with OpenFOAM/GAMG, but
this integration is still under development and is not ready for production
use.

## Test meshes

The benchmark mesh archives will be published as soon as a suitable way to
host these large files is available.

## Usage and attribution

The original work in this repository may be used, modified, and redistributed
freely, provided that credit is given to **Dmitry Mishura** and this repository
is identified as the source.

Files copied or derived from OpenFOAM remain subject to their original GNU
General Public License and copyright notices. Third-party components retain
their respective licenses.
