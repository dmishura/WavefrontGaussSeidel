#!/bin/sh
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 POLYMESH_DIR [RESULT_ROOT] [CPU]" >&2
    exit 2
fi

mesh_dir=$1
result_root=${2:-build/vtune-memory-access}
profile_cpu=${3:-2}
vtune_bin=${VTUNE_BIN:-/opt/intel/oneapi/vtune/latest/bin64/vtune}
test_bin=${TEST_GAUSS_SEIDEL:-./build/Test-GaussSeidel}

mkdir -p "$result_root"
for variant in reference packed geometry2048; do
    result_dir="$result_root/$variant"
    OMP_NUM_THREADS=1 taskset -c "$profile_cpu" "$vtune_bin" \
        -collect memory-access \
        -result-dir "$result_dir" \
        -- "$test_bin" "$mesh_dir" --profile-variant "$variant"
    "$vtune_bin" -report summary -result-dir "$result_dir" \
        -format csv -report-output "$result_dir/summary.csv"
done

echo "VTune results: $result_root"
