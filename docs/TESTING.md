# Registry test and benchmark commands

The original Makefile and `Test-GaussSeidel` CLI remain available during the
incremental migration. The registry-based executables use CMake:

```bash
cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-cmake -j
```

System GoogleTest and Google Benchmark packages are used by default. An
explicit network fallback is available with
`-DSMOOTHER_TEST_FETCH_DEPENDENCIES=ON`.

## Correctness

```bash
OMP_NUM_THREADS=1 ctest --test-dir build-cmake -L correctness --output-on-failure
./build-cmake/smoother_tests --gtest_filter='*IndexKahn1024*'
./build-cmake/smoother_tests --mesh=/path/to/polyMesh --gtest_filter='*Packed*'
```

CTest labels currently include `correctness`, `benchmark`, and `motorBike`:

```bash
ctest --test-dir build-cmake -R motorBike
```

The default correctness suite validates one-sweep behavior for every
Core solver, three-sweep reference equivalence where applicable,
200-sweep convergence for the reference smoothers, and every dependency
schedule.

## Variant selection

The registry contains all 89 variants, but both executables select the Core
tier by default. The current counts are 14 Core, 44 Experimental, and 31
Legacy variants.

```bash
./build-cmake/smoother_tests --tier core
./build-cmake/smoother_tests --tier experimental
./build-cmake/smoother_tests --tier legacy
./build-cmake/smoother_tests --tier all

./build-cmake/smoother_bench --tier all --tag index-kahn
./build-cmake/smoother_bench --tier experimental --tag simd --tag avx512
```

Tags are arbitrary strings with no central whitelist. Repeated `--tag`
arguments use AND semantics. A tag without an explicit tier searches all
tiers; use `--tier core --tag TAG` to restrict it to Core. Tier/tag selection
happens before, and does not replace, native `--gtest_filter` or
`--benchmark_filter` processing.

List all registered metadata without running tests or benchmarks:

```bash
./build-cmake/smoother_tests --list-variants
```

The default CTest configuration registers only Core tests. This keeps plain
`ctest` and `ctest -L correctness` small. To expose opt-in `experimental` and
`legacy` CTest labels, configure a separate build with:

```bash
cmake -S . -B build-cmake-extended \
    -DSMOOTHER_TEST_REGISTER_EXTENDED_TIERS=ON
ctest --test-dir build-cmake-extended -L experimental
```

## Performance

Pin the executable itself, not `make`:

```bash
OMP_NUM_THREADS=1 taskset -c 2 ./build-cmake/smoother_bench \
    --benchmark_filter='MotorBike/(ReferenceGS|IndexKahn1024)/.*' \
    --smoother_sweeps=250 \
    --smoother_warmup=10 \
    --benchmark_repetitions=7 \
    --benchmark_report_aggregates_only=true
```

Machine-readable output is provided by Google Benchmark:

```bash
OMP_NUM_THREADS=1 taskset -c 2 ./build-cmake/smoother_bench \
    --benchmark_filter=IndexKahn \
    --benchmark_out=results.json \
    --benchmark_out_format=json
```

Each Google Benchmark iteration is one fixed-size sample. Field reset occurs
while timing is paused. The measured manual time is divided by the requested
sweep count.
The primary `Time` column is therefore time per sweep. With this batched
manual-time harness, Google Benchmark's `CPU` column still describes the
whole callback batch and should not be used for per-sweep comparisons.
`sweepsPerCall=3` remains solver metadata and is applied by the common batching
wrapper.
Preprocessing occurs before benchmark registration. Warm-up runs in each
repetition before timing begins.
`--smoother_samples=7` is retained as a convenience default/alias when the
standard `--benchmark_repetitions` option is absent.

The installed Google Benchmark 1.6.1 does not provide
`--benchmark_enable_random_interleaving`. The legacy production benchmark is
therefore retained as the exact randomized-order parity oracle. With a newer
Google Benchmark, its random-interleaving option can be used without changing
the registry.

CTest contains only a one-sweep `benchmark` smoke test. It verifies dynamic
registration and execution but is not a performance measurement. Production
timing should always be launched manually on an idle, pinned CPU.
