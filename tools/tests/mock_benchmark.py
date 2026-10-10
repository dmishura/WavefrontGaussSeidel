#!/usr/bin/env python3
"""Controllable Google Benchmark stand-in used by runner regression tests."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def option(prefix: str) -> str:
    for argument in sys.argv[1:]:
        if argument.startswith(prefix):
            return argument[len(prefix):]
    raise RuntimeError(f"missing option {prefix}")


def optional_option(prefix: str, default: str) -> str:
    for argument in sys.argv[1:]:
        if argument.startswith(prefix):
            return argument[len(prefix):]
    return default


counter_path = Path(os.environ["MOCK_BENCH_COUNTER"])
try:
    count = int(counter_path.read_text(encoding="utf-8")) + 1
except (OSError, ValueError):
    count = 1
counter_path.write_text(str(count), encoding="utf-8")
if "MOCK_BENCH_SELF_PID" in os.environ:
    Path(os.environ["MOCK_BENCH_SELF_PID"]).write_text(
        str(os.getpid()), encoding="utf-8"
    )

mode = os.environ.get("MOCK_BENCH_MODE", "success")
if mode.endswith("_once") and count > 1:
    mode = "success"
elif mode == "spawn_timeout_once":
    mode = "spawn_timeout"

if mode in {"spawn_timeout", "interrupt_once"}:
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    Path(os.environ["MOCK_BENCH_CHILD_PID"]).write_text(
        str(child.pid), encoding="utf-8"
    )
    print(f"mock child={child.pid}", flush=True)
    time.sleep(60)
elif mode == "timeout_once":
    time.sleep(60)
elif mode == "kill9_once":
    time.sleep(60)
elif mode in {"crash_once", "crash"}:
    print("controlled crash", flush=True)
    raise SystemExit(7)

result_path = Path(option("--benchmark_out="))
result_path.parent.mkdir(parents=True, exist_ok=True)
benchmark_names = os.environ.get(
    "MOCK_BENCH_NAMES", os.environ.get("MOCK_BENCH_NAME", "Mesh/Solver")
).split(",")
repetitions = int(optional_option("--benchmark_repetitions=", "1"))
benchmarks = []
for benchmark_name in benchmark_names:
    for repetition in range(repetitions):
        milliseconds = float(os.environ.get("MOCK_BENCH_MS", "2")) + repetition
        benchmarks.append(
            {
                "name": benchmark_name,
                "run_type": "iteration",
                "repetition_index": repetition,
                "repetitions": repetitions,
                "real_time": milliseconds,
                "cpu_time": milliseconds,
                "time_unit": "ms",
                "cells": 10,
            }
        )
if mode == "multiple":
    benchmarks.extend(
        {
            **benchmarks[0],
            "name": f"Mesh/SolverVariant{variant}/iterations:1/manual_time",
        }
        for variant in range(1, 16)
    )

result_path.write_text(
    json.dumps(
        {
            "benchmarks": benchmarks
        }
    ),
    encoding="utf-8",
)
print("controlled success", flush=True)
