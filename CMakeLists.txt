cmake_minimum_required(VERSION 3.18)
project(SmootherTest LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(
    SMOOTHER_TEST_FETCH_DEPENDENCIES
    "Fetch GoogleTest/Google Benchmark when system packages are unavailable"
    OFF
)
option(
    SMOOTHER_TEST_REGISTER_EXTENDED_TIERS
    "Register opt-in experimental/legacy CTest entries"
    OFF
)

find_package(OpenMP REQUIRED)
find_package(GTest QUIET)
find_package(benchmark QUIET)

if((NOT GTest_FOUND OR NOT benchmark_FOUND) AND SMOOTHER_TEST_FETCH_DEPENDENCIES)
    include(FetchContent)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz
    )
    FetchContent_Declare(
        googlebenchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.7.1.tar.gz
    )
    FetchContent_MakeAvailable(googletest googlebenchmark)
endif()

if(NOT TARGET GTest::gtest)
    message(FATAL_ERROR "GoogleTest not found; install it or enable SMOOTHER_TEST_FETCH_DEPENDENCIES")
endif()
if(NOT TARGET benchmark::benchmark)
    message(FATAL_ERROR "Google Benchmark not found; install it or enable SMOOTHER_TEST_FETCH_DEPENDENCIES")
endif()

set(SMOOTHER_COMMON_OPTIONS -Wall -Wextra -Wpedantic)

add_library(
    smoother_kernels
    src/GaussSeidelSmoother.C
    src/symGaussSeidelSmoother.C
    src/PolyMeshReader.C
    src/WavefrontGaussSeidel.C
    src/WavefrontGaussSeidelAVX2.C
    src/WavefrontGaussSeidelAVX512.C
)
target_include_directories(smoother_kernels PUBLIC compat src)
target_compile_options(smoother_kernels PRIVATE ${SMOOTHER_COMMON_OPTIONS})
target_link_libraries(smoother_kernels PUBLIC OpenMP::OpenMP_CXX)
set_source_files_properties(
    src/WavefrontGaussSeidelAVX2.C
    PROPERTIES COMPILE_OPTIONS "-mavx2"
)
set_source_files_properties(
    src/WavefrontGaussSeidelAVX512.C
    PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512vl"
)

find_path(ITTNOTIFY_INCLUDE_DIR ittnotify.h HINTS /opt/intel/oneapi/vtune/latest/include)
find_library(ITTNOTIFY_LIBRARY ittnotify HINTS /opt/intel/oneapi/vtune/latest/lib64)
if(ITTNOTIFY_INCLUDE_DIR AND ITTNOTIFY_LIBRARY)
    target_compile_definitions(smoother_kernels PUBLIC HAVE_ITTNOTIFY)
    target_include_directories(smoother_kernels PUBLIC ${ITTNOTIFY_INCLUDE_DIR})
    target_link_libraries(smoother_kernels PUBLIC ${ITTNOTIFY_LIBRARY} ${CMAKE_DL_LIBS})
endif()

add_library(
    smoother_harness
    tests/harness/VariantSelection.C
    tests/harness/ScheduleBuilder.C
    tests/harness/Workload.C
    tests/harness/SmootherRegistry.C
    tests/harness/CorrectnessRunner.C
)
target_include_directories(smoother_harness PUBLIC tests/harness)
target_compile_options(smoother_harness PRIVATE ${SMOOTHER_COMMON_OPTIONS})
target_link_libraries(smoother_harness PUBLIC smoother_kernels)

set(MOTORBIKE_ARCHIVE ${CMAKE_CURRENT_SOURCE_DIR}/meshes/MTB_example_polyMesh.tgz)
set(MOTORBIKE_ROOT ${CMAKE_CURRENT_BINARY_DIR}/mesh-data)
set(MOTORBIKE_MESH ${MOTORBIKE_ROOT}/MTB_example_polyMesh)
if(NOT EXISTS ${MOTORBIKE_MESH}/owner)
    file(MAKE_DIRECTORY ${MOTORBIKE_ROOT})
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${MOTORBIKE_ARCHIVE}
        WORKING_DIRECTORY ${MOTORBIKE_ROOT}
        COMMAND_ERROR_IS_FATAL ANY
    )
endif()

add_executable(smoother_tests tests/CorrectnessTests.C)
target_compile_options(smoother_tests PRIVATE ${SMOOTHER_COMMON_OPTIONS})
target_compile_definitions(
    smoother_tests PRIVATE SMOOTHER_TEST_DEFAULT_MESH="${MOTORBIKE_MESH}"
)
target_link_libraries(smoother_tests PRIVATE smoother_harness GTest::gtest)

add_executable(smoother_bench tests/Benchmarks.C)
target_compile_options(smoother_bench PRIVATE ${SMOOTHER_COMMON_OPTIONS})
target_compile_definitions(
    smoother_bench PRIVATE SMOOTHER_TEST_DEFAULT_MESH="${MOTORBIKE_MESH}"
)
target_link_libraries(smoother_bench PRIVATE smoother_harness benchmark::benchmark)

add_executable(Test-PolyMeshReader tests/Test-PolyMeshReader.C)
target_compile_options(Test-PolyMeshReader PRIVATE ${SMOOTHER_COMMON_OPTIONS})
target_link_libraries(Test-PolyMeshReader PRIVATE smoother_kernels)

add_executable(Test-GaussSeidel tests/Test-GaussSeidel.C)
target_compile_options(Test-GaussSeidel PRIVATE ${SMOOTHER_COMMON_OPTIONS})
target_link_libraries(Test-GaussSeidel PRIVATE smoother_kernels)

if(BUILD_TESTING)
    add_test(
        NAME motorBike_reader_correctness
        COMMAND Test-PolyMeshReader ${MOTORBIKE_MESH}
    )
    set_tests_properties(
        motorBike_reader_correctness
        PROPERTIES LABELS "correctness;core;motorBike"
    )

    add_test(
        NAME motorBike_smoother_correctness
        COMMAND smoother_tests --mesh=${MOTORBIKE_MESH}
    )
    set_tests_properties(
        motorBike_smoother_correctness
        PROPERTIES LABELS "correctness;core;motorBike"
    )

    # A one-sweep smoke test verifies benchmark registration without placing
    # the long production benchmark in the default CTest suite.
    add_test(
        NAME motorBike_benchmark_smoke
        COMMAND smoother_bench
            --mesh=${MOTORBIKE_MESH}
            --smoother_sweeps=1
            --smoother_samples=1
            --smoother_warmup=1
            --benchmark_filter=^MotorBike/ReferenceGS/.*
            --benchmark_report_aggregates_only=true
    )
    set_tests_properties(
        motorBike_benchmark_smoke
        PROPERTIES LABELS "benchmark;core;motorBike"
    )

    if(SMOOTHER_TEST_REGISTER_EXTENDED_TIERS)
        foreach(tier experimental legacy)
            add_test(
                NAME motorBike_smoother_${tier}
                COMMAND smoother_tests
                    --mesh=${MOTORBIKE_MESH}
                    --tier=${tier}
            )
            set_tests_properties(
                motorBike_smoother_${tier}
                PROPERTIES LABELS "correctness;${tier};motorBike"
            )

            add_test(
                NAME motorBike_benchmark_${tier}_smoke
                COMMAND smoother_bench
                    --mesh=${MOTORBIKE_MESH}
                    --tier=${tier}
                    --smoother_sweeps=1
                    --smoother_samples=1
                    --smoother_warmup=1
                    --benchmark_report_aggregates_only=true
            )
            set_tests_properties(
                motorBike_benchmark_${tier}_smoke
                PROPERTIES LABELS "benchmark;${tier};motorBike"
            )
        endforeach()
    endif()
endif()
