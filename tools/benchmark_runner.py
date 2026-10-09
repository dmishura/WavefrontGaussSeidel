#!/usr/bin/env python3
"""Process-isolated, resumable runner for Google Benchmark executables.

Each mesh/solver/repetition tuple is a separate child process.  Child output is
always written to disk; this program only emits compact progress diagnostics.
The native Google Benchmark JSON is retained alongside a restart checkpoint,
and summary.csv is rebuilt atomically after every successful child.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


VERBOSITY = {"quiet": 0, "normal": 1, "debug": 2}


@dataclass(frozen=True)
class Task:
    mesh_name: str
    mesh_path: str
    solver_name: str
    benchmark_name: str
    benchmark_filter: str
    repetition: int
    run_fingerprint: str

    @property
    def key(self) -> str:
        return "/".join(
            (
                self.run_fingerprint,
                safe_name(self.mesh_name),
                safe_name(self.solver_name),
                f"rep-{self.repetition:04d}",
            )
        )


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "unnamed"


def literal_from_exact_filter(value: str) -> str:
    if len(value) < 2 or not value.startswith("^") or not value.endswith("$"):
        raise ValueError(
            "legacy solver filter must be an anchored exact match; "
            "prefer the 'benchmark' field"
        )
    body = value[1:-1]
    result: list[str] = []
    index = 0
    metacharacters = set(".^$*+?{}[]|()")
    while index < len(body):
        character = body[index]
        if character == "\\":
            index += 1
            if index >= len(body):
                raise ValueError("invalid trailing escape in solver filter")
            result.append(body[index])
        elif character in metacharacters:
            raise ValueError(
                f"solver filter is not exact: {value!r}; use 'benchmark'"
            )
        else:
            result.append(character)
        index += 1
    if not result:
        raise ValueError("expected benchmark name must not be empty")
    return "".join(result)


def solver_benchmark_name(solver: dict[str, Any]) -> str:
    if "benchmark" in solver:
        name = str(solver["benchmark"])
        if not name:
            raise ValueError("solver benchmark must not be empty")
        return name
    if "filter" not in solver:
        raise ValueError("solver requires a 'benchmark' name")
    return literal_from_exact_filter(str(solver["filter"]))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def output_lock(output_root: Path) -> Iterable[None]:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".runner.lock"
    lock = lock_path.open("a+")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another benchmark runner owns {lock_path}"
            ) from error
        yield
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path_text: str) -> dict[str, str]:
    path = Path(path_text).resolve(strict=True)
    digest = hashlib.sha256()
    digest.update(str(path).encode())
    if path.is_file():
        digest.update(b"\0file\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    elif path.is_dir():
        digest.update(b"\0directory\0")
        files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
        for candidate in files:
            digest.update(str(candidate.relative_to(path)).encode())
            digest.update(b"\0")
            digest.update(bytes.fromhex(sha256_file(candidate)))
    else:
        raise ValueError(f"cannot fingerprint non-file path: {path}")
    return {"path": str(path), "sha256": digest.hexdigest()}


def measurement_identity(plan: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    identity_plan = {
        key: value for key, value in plan.items()
        if key not in {"output", "timeout_seconds", "random_seed"}
    }
    executable = sha256_path(str(plan["executable"]))
    meshes = [
        {
            "name": str(mesh["name"]),
            **(
                {"path": str(Path(mesh["path"]).resolve()), "sha256": str(mesh["fingerprint"])}
                if "fingerprint" in mesh
                else sha256_path(str(mesh["path"]))
            ),
        }
        for mesh in plan["meshes"]
    ]
    explicit_environment = {
        str(key): str(value) for key, value in plan.get("env", {}).items()
    }
    inherited_omp = {
        key: value for key, value in os.environ.items()
        if key.startswith("OMP_") and key not in explicit_environment
    }
    material = {
        "schema": 2,
        "plan": identity_plan,
        "executable": executable,
        "meshes": meshes,
        "inherited_omp_environment": inherited_omp,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20], material


def load_plan(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        plan = json.load(source)
    required = ("executable", "meshes", "solvers")
    missing = [name for name in required if name not in plan]
    if missing:
        raise ValueError("plan is missing: " + ", ".join(missing))
    if not plan["meshes"] or not plan["solvers"]:
        raise ValueError("plan meshes and solvers must not be empty")
    return plan


def make_tasks(plan: dict[str, Any], run_fingerprint: str = "unfingerprinted") -> list[Task]:
    repetitions = int(plan.get("repetitions", 1))
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    mesh_names = [str(mesh["name"]) for mesh in plan["meshes"]]
    solver_names = [str(solver["name"]) for solver in plan["solvers"]]
    if len(set(mesh_names)) != len(mesh_names):
        raise ValueError("mesh names must be unique")
    if len(set(solver_names)) != len(solver_names):
        raise ValueError("solver names must be unique")
    if len({safe_name(name) for name in mesh_names}) != len(mesh_names):
        raise ValueError("mesh names collide after filesystem normalization")
    if len({safe_name(name) for name in solver_names}) != len(solver_names):
        raise ValueError("solver names collide after filesystem normalization")
    tasks = [
        Task(
            str(mesh["name"]),
            str(mesh["path"]),
            str(solver["name"]),
            solver_benchmark_name(solver),
            "^" + re.escape(solver_benchmark_name(solver)) + "$",
            repetition,
            run_fingerprint,
        )
        for mesh in plan["meshes"]
        for solver in plan["solvers"]
        for repetition in range(1, repetitions + 1)
    ]
    seed = int(plan.get("random_seed", 0x1A2B3C4D))
    random.Random(seed).shuffle(tasks)
    return tasks


def result_is_valid(result_path: Path, benchmark_name: str | None = None) -> bool:
    try:
        with result_path.open(encoding="utf-8") as source:
            result = json.load(source)
        entries = list(iteration_entries(result))
        if len(entries) != 1:
            return False
        return benchmark_name is None or str(entries[0].get("name", "")) == benchmark_name
    except (OSError, ValueError, TypeError):
        return False


def completed(task_dir: Path, expected_fingerprint: str | None = None) -> bool:
    status_path = task_dir / "status.json"
    try:
        with status_path.open(encoding="utf-8") as source:
            status = json.load(source)
        result_path = task_dir / "benchmark.json"
        return (
            status.get("state") == "complete"
            and status.get("returncode") == 0
            and (
                expected_fingerprint is None
                or status.get("run_fingerprint") == expected_fingerprint
            )
            and result_is_valid(
                result_path, status.get("task", {}).get("benchmark_name")
            )
            and status.get("result_sha256") == sha256_file(result_path)
        )
    except (OSError, ValueError, TypeError):
        return False


def benchmark_seconds(entry: dict[str, Any], field: str) -> float:
    scale = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}
    unit = entry.get("time_unit", "ns")
    if unit not in scale:
        raise ValueError(f"unsupported Google Benchmark time unit: {unit}")
    return float(entry[field]) * scale[unit]


def iteration_entries(result: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for entry in result.get("benchmarks", []):
        if entry.get("run_type", "iteration") == "iteration":
            yield entry


def collect_rows(output_root: Path, run_fingerprint: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_root = output_root / "runs" / run_fingerprint
    for status_path in sorted(run_root.glob("**/status.json")):
        task_dir = status_path.parent
        if not completed(task_dir, run_fingerprint):
            continue
        with status_path.open(encoding="utf-8") as source:
            status = json.load(source)
        with (task_dir / "benchmark.json").open(encoding="utf-8") as source:
            result = json.load(source)
        for entry in iteration_entries(result):
            row = {
                "mesh": status["task"]["mesh_name"],
                "solver": status["task"]["solver_name"],
                "repetition": int(status["task"]["repetition"]),
                "benchmark": entry["name"],
                "run_fingerprint": run_fingerprint,
                "real_seconds": benchmark_seconds(entry, "real_time"),
                "cpu_seconds": benchmark_seconds(entry, "cpu_time"),
                "log": status["log"],
                "result_json": str((task_dir / "benchmark.json").resolve()),
            }
            for name, value in entry.items():
                if name not in {
                    "name", "family_index", "per_family_instance_index",
                    "run_name", "run_type", "repetitions", "repetition_index",
                    "threads", "iterations", "real_time", "cpu_time",
                    "time_unit", "aggregate_name", "aggregate_unit",
                } and isinstance(value, (int, float)):
                    row[f"counter_{name}"] = value
            rows.append(row)
    return rows


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        output.flush()
        os.fsync(output.fileno())
    if path.is_file() and sha256_file(temporary) == sha256_file(path):
        temporary.unlink()
        fsync_directory(path.parent)
        return
    os.replace(temporary, path)
    fsync_directory(path.parent)


def rebuild_csv(output_root: Path, run_fingerprint: str) -> None:
    rows = collect_rows(output_root, run_fingerprint)
    raw_fields = [
        "run_fingerprint", "mesh", "solver", "repetition", "benchmark", "real_seconds",
        "cpu_seconds", "log", "result_json",
    ]
    counter_fields = sorted(
        {name for row in rows for name in row if name.startswith("counter_")}
    )
    csv_roots = [output_root, output_root / "summaries" / run_fingerprint]
    for csv_root in csv_roots:
        write_csv_atomic(csv_root / "results.csv", raw_fields + counter_fields, rows)

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["mesh"], row["solver"], row["benchmark"])
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (mesh, solver, benchmark), values in sorted(groups.items()):
        real = [float(value["real_seconds"]) for value in values]
        cpu = [float(value["cpu_seconds"]) for value in values]
        real_mean = statistics.fmean(real)
        real_stddev = statistics.stdev(real) if len(real) > 1 else 0.0
        summaries.append(
            {
                "mesh": mesh,
                "run_fingerprint": run_fingerprint,
                "solver": solver,
                "benchmark": benchmark,
                "completed_repetitions": len(values),
                "real_median_seconds": statistics.median(real),
                "real_mean_seconds": real_mean,
                "real_stddev_seconds": real_stddev,
                "real_cv_percent": 100.0 * real_stddev / real_mean
                    if real_mean else math.nan,
                "cpu_median_seconds": statistics.median(cpu),
            }
        )
    summary_fields = [
        "run_fingerprint", "mesh", "solver", "benchmark", "completed_repetitions",
        "real_median_seconds", "real_mean_seconds", "real_stddev_seconds",
        "real_cv_percent", "cpu_median_seconds",
    ]
    for csv_root in csv_roots:
        write_csv_atomic(csv_root / "summary.csv", summary_fields, summaries)


def task_command(plan: dict[str, Any], task: Task, result_path: Path) -> list[str]:
    prefix = [str(value) for value in plan.get("prefix", [])]
    common = [str(value) for value in plan.get("common_args", [])]
    return prefix + [
        str(plan["executable"]),
        f"--mesh={task.mesh_path}",
        *common,
        f"--benchmark_filter={task.benchmark_filter}",
        "--benchmark_repetitions=1",
        "--benchmark_report_aggregates_only=false",
        f"--benchmark_out={result_path}",
        "--benchmark_out_format=json",
    ]


class RunnerInterrupted(Exception):
    def __init__(self, signum: int):
        super().__init__(f"runner interrupted by signal {signum}")
        self.signum = signum


def terminate_process_group(process: subprocess.Popen[bytes], grace_seconds: float = 2.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    # The group can outlive its leader (eg, a grandchild ignores SIGTERM).
    # Always probe and kill the remaining group before returning.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait()


def wait_for_child(
    command: list[str], log: Any, environment: dict[str, str], timeout: float
) -> int:
    parent_pid = os.getpid()

    def set_parent_death_signal() -> None:
        # Linux guarantees that even an uncatchable runner death terminates
        # the direct benchmark child. Normal timeout/interrupt cleanup still
        # targets the entire process group below.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL) != 0:  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGKILL)

    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
        preexec_fn=set_parent_death_signal if sys.platform.startswith("linux") else None,
    )
    previous_handlers: dict[int, Any] = {}

    def interrupted(signum: int, _frame: Any) -> None:
        raise RunnerInterrupted(signum)

    handled_signals = (signal.SIGTERM, signal.SIGHUP)
    for signum in handled_signals:
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupted)
    try:
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_process_group(process)
            raise
        except KeyboardInterrupt:
            terminate_process_group(process)
            raise RunnerInterrupted(signal.SIGINT)
        except RunnerInterrupted:
            terminate_process_group(process)
            raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def run_task(
    plan: dict[str, Any], task: Task, output_root: Path, verbosity: int
) -> bool:
    task_dir = output_root / "runs" / task.key
    task_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir = task_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    attempt_id = f"{time.time_ns()}-{os.getpid()}"
    result_path = task_dir / "benchmark.json"
    temporary_result = attempts_dir / f"{attempt_id}.benchmark.json.tmp"
    log_path = attempts_dir / f"{attempt_id}.log"
    command = task_command(plan, task, temporary_result.resolve())
    environment = os.environ.copy()
    environment.update({str(k): str(v) for k, v in plan.get("env", {}).items()})
    timeout = float(plan.get("timeout_seconds", 1800))
    started = time.time()
    running = {
        "state": "running",
        "task": task.__dict__,
        "run_fingerprint": task.run_fingerprint,
        "attempt_id": attempt_id,
        "command": command,
        "started_unix": started,
        "timeout_seconds": timeout,
    }
    attempt_status = attempts_dir / f"{attempt_id}.status.json"
    atomic_json(attempt_status, running)
    if verbosity >= VERBOSITY["debug"]:
        print("RUN", " ".join(command), flush=True)
    try:
        with log_path.open("wb") as log:
            returncode = wait_for_child(command, log, environment, timeout)
        valid_result = returncode == 0 and result_is_valid(
            temporary_result, task.benchmark_name
        )
        state = "complete" if valid_result else "failed"
        error = None if state == "complete" else "non-zero exit or invalid benchmark JSON"
    except subprocess.TimeoutExpired:
        returncode = None
        state = "timeout"
        error = f"exceeded {timeout:g} seconds"
    except OSError as exception:
        returncode = None
        state = "failed"
        error = str(exception)
    except RunnerInterrupted as exception:
        returncode = None
        state = "interrupted"
        error = str(exception)
        final = {
            **running,
            "state": state,
            "returncode": returncode,
            "finished_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "error": error,
            "log": str(log_path.resolve()),
            "result_json": None,
            "result_sha256": None,
        }
        atomic_json(attempt_status, final)
        atomic_json(task_dir / "status.json", final)
        rebuild_csv(output_root, task.run_fingerprint)
        raise
    result_sha256 = None
    if state == "complete":
        with temporary_result.open("rb") as result_file:
            os.fsync(result_file.fileno())
        os.replace(temporary_result, result_path)
        fsync_directory(task_dir)
        result_sha256 = sha256_file(result_path)
    final = {
        **running,
        "state": state,
        "returncode": returncode,
        "finished_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "error": error,
        "log": str(log_path.resolve()),
        "result_json": str(result_path.resolve()) if state == "complete" else None,
        "result_sha256": result_sha256,
    }
    atomic_json(attempt_status, final)
    atomic_json(task_dir / "status.json", final)
    if verbosity >= VERBOSITY["normal"]:
        print(
            f"{state.upper():8s} {task.key} {final['elapsed_seconds']:.3f}s",
            flush=True,
        )
    return state == "complete"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--verbosity", choices=tuple(VERBOSITY), default="quiet",
        help="runner diagnostics only; child output always goes to log files",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    plan = load_plan(arguments.plan)
    output_root = arguments.output or Path(plan.get("output", "benchmark-runs"))
    run_fingerprint, identity = measurement_identity(plan)
    with output_lock(output_root):
        atomic_json(output_root / "identities" / f"{run_fingerprint}.json", identity)
        tasks = make_tasks(plan, run_fingerprint)
        verbosity = VERBOSITY[arguments.verbosity]
        complete_count = 0
        skipped_count = 0
        failed_count = 0
        for task in tasks:
            task_dir = output_root / "runs" / task.key
            if not arguments.no_resume and completed(task_dir, run_fingerprint):
                skipped_count += 1
                continue
            if arguments.dry_run:
                if verbosity >= VERBOSITY["normal"]:
                    print("DRY-RUN", task.key)
                continue
            if run_task(plan, task, output_root, verbosity):
                complete_count += 1
            else:
                failed_count += 1
            rebuild_csv(output_root, run_fingerprint)
        if not arguments.dry_run:
            rebuild_csv(output_root, run_fingerprint)
    print(
        f"benchmark_runner complete={complete_count} resumed={skipped_count} "
        f"failed={failed_count} fingerprint={run_fingerprint} "
        f"output={output_root.resolve()}"
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerInterrupted as error:
        print(f"benchmark_runner: {error}", file=sys.stderr)
        raise SystemExit(128 + error.signum)
    except (ValueError, KeyError, OSError) as error:
        print(f"benchmark_runner: {error}", file=sys.stderr)
        raise SystemExit(2)
