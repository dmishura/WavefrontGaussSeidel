#!/usr/bin/env python3

import csv
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1]
RUNNER_PATH = TOOLS / "benchmark_runner.py"
MOCK_PATH = Path(__file__).with_name("mock_benchmark.py")
SPEC = importlib.util.spec_from_file_location("benchmark_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class BenchmarkRunnerTests(unittest.TestCase):
    def make_plan(
        self, root: Path, mode: str = "success", timeout: float = 2.0
    ) -> dict:
        mesh = root / "mesh"
        mesh.mkdir(exist_ok=True)
        (mesh / "owner").write_text("mesh-v1", encoding="utf-8")
        return {
            "executable": str(MOCK_PATH),
            "prefix": [sys.executable],
            "output": str(root / "output"),
            "env": {
                "OMP_NUM_THREADS": "1",
                "MOCK_BENCH_MODE": mode,
                "MOCK_BENCH_COUNTER": str(root / "counter"),
                "MOCK_BENCH_CHILD_PID": str(root / "child.pid"),
                "MOCK_BENCH_SELF_PID": str(root / "self.pid"),
            },
            "common_args": ["--mock-argument=one"],
            "meshes": [{"name": "mesh", "path": str(mesh)}],
            "solvers": [{"name": "solver", "benchmark": "Mesh/Solver"}],
            "repetitions": 1,
            "timeout_seconds": timeout,
        }

    def run_one(self, plan: dict) -> tuple[str, runner.Task, Path, bool]:
        fingerprint, _ = runner.measurement_identity(plan)
        task = runner.make_tasks(plan, fingerprint)[0]
        output = Path(plan["output"])
        success = runner.run_task(plan, task, output, runner.VERBOSITY["quiet"])
        return fingerprint, task, output, success

    def test_tasks_are_process_granular_and_deterministic(self) -> None:
        plan = {
            "executable": "bench",
            "meshes": [{"name": "m1", "path": "p1"}, {"name": "m2", "path": "p2"}],
            "solvers": [
                {"name": "s1", "benchmark": "Mesh/One"},
                {"name": "s2", "benchmark": "Mesh/Two"},
            ],
            "repetitions": 3,
            "random_seed": 17,
        }
        first = runner.make_tasks(plan, "fingerprint")
        second = runner.make_tasks(plan, "fingerprint")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(len({task.key for task in first}), 12)

    def test_series_is_one_process_per_mesh_with_exact_selection(self) -> None:
        plan = {
            "executable": "bench",
            "meshes": [{"name": "m1", "path": "p1"}, {"name": "m2", "path": "p2"}],
            "solvers": [
                {"name": "s1", "benchmark": "Mesh/One"},
                {"name": "s2", "benchmark": "Mesh/Two+Exact"},
            ],
            "repetitions": 3,
            "random_seed": 17,
            "isolation": "series",
        }
        tasks = runner.make_series_tasks(plan, "fingerprint")
        self.assertEqual(len(tasks), 2)
        self.assertEqual(len({task.key for task in tasks}), 2)
        command = runner.series_command(plan, tasks[0], Path("result.json"))
        benchmark_filter = next(
            value.removeprefix("--benchmark_filter=")
            for value in command if value.startswith("--benchmark_filter=")
        )
        self.assertTrue(re.fullmatch(benchmark_filter, "Mesh/One"))
        self.assertTrue(re.fullmatch(benchmark_filter, "Mesh/Two+Exact"))
        self.assertFalse(re.fullmatch(benchmark_filter, "Mesh/OneVariant"))
        self.assertIn("--benchmark_repetitions=3", command)
        self.assertIn("--benchmark_enable_random_interleaving=true", command)

    def test_series_run_collects_all_samples_and_resumes_as_one_unit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.make_plan(root)
            plan["isolation"] = "series"
            plan["repetitions"] = 3
            plan["solvers"] = [
                {"name": "one", "benchmark": "Mesh/One"},
                {"name": "two", "benchmark": "Mesh/Two"},
            ]
            plan["env"]["MOCK_BENCH_NAMES"] = "Mesh/One,Mesh/Two"
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            command = [sys.executable, str(RUNNER_PATH), "--plan", str(plan_path)]
            first = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=5, check=False,
            )
            self.assertEqual(first.returncode, 0, first.stdout)
            self.assertIn("complete=1 resumed=0 failed=0", first.stdout)
            self.assertEqual((root / "counter").read_text(), "1")
            with (Path(plan["output"]) / "results.csv").open(
                newline="", encoding="utf-8"
            ) as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["solver"] for row in rows}, {"one", "two"})
            self.assertEqual(
                {int(row["repetition"]) for row in rows}, {1, 2, 3}
            )

            tracked = [
                Path(plan["output"]) / "results.csv",
                Path(plan["output"]) / "summary.csv",
            ]
            before = {
                path: (runner.sha256_file(path), path.stat().st_mtime_ns)
                for path in tracked
            }
            second = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=5, check=False,
            )
            self.assertEqual(second.returncode, 0, second.stdout)
            self.assertIn("complete=0 resumed=1 failed=0", second.stdout)
            self.assertEqual((root / "counter").read_text(), "1")
            self.assertEqual(
                before,
                {
                    path: (runner.sha256_file(path), path.stat().st_mtime_ns)
                    for path in tracked
                },
            )

    def test_series_rejects_missing_extra_or_incomplete_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"

            def write(names: list[str]) -> None:
                result.write_text(json.dumps({"benchmarks": [
                    {"name": name, "run_type": "iteration"} for name in names
                ]}), encoding="utf-8")

            write(["Mesh/One", "Mesh/One", "Mesh/Two", "Mesh/Two"])
            self.assertTrue(runner.series_result_is_valid(
                result, ("Mesh/One", "Mesh/Two"), 2
            ))
            write(["Mesh/One", "Mesh/One", "Mesh/Two"])
            self.assertFalse(runner.series_result_is_valid(
                result, ("Mesh/One", "Mesh/Two"), 2
            ))
            write(["Mesh/One", "Mesh/One", "Mesh/Two", "Mesh/Two", "Mesh/Extra"])
            self.assertFalse(runner.series_result_is_valid(
                result, ("Mesh/One", "Mesh/Two"), 2
            ))

    def test_output_lock_rejects_concurrent_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with runner.output_lock(root):
                with self.assertRaises(RuntimeError):
                    with runner.output_lock(root):
                        pass

    def test_identity_changes_for_every_result_affecting_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.make_plan(root)
            baseline, _ = runner.measurement_identity(plan)
            changes = []
            (root / "mesh" / "owner").write_text("mesh-v2", encoding="utf-8")
            changes.append(runner.measurement_identity(plan)[0])
            (root / "mesh" / "owner").write_text("mesh-v1", encoding="utf-8")
            changed_args = json.loads(json.dumps(plan))
            changed_args["common_args"].append("--new")
            changes.append(runner.measurement_identity(changed_args)[0])
            changed_omp = json.loads(json.dumps(plan))
            changed_omp["env"]["OMP_NUM_THREADS"] = "2"
            changes.append(runner.measurement_identity(changed_omp)[0])
            changed_solver = json.loads(json.dumps(plan))
            changed_solver["solvers"][0]["benchmark"] = "Mesh/Other"
            changes.append(runner.measurement_identity(changed_solver)[0])
            binary_copy = root / "mock-copy.py"
            binary_copy.write_bytes(MOCK_PATH.read_bytes() + b"\n# changed\n")
            changed_binary = json.loads(json.dumps(plan))
            changed_binary["executable"] = str(binary_copy)
            changes.append(runner.measurement_identity(changed_binary)[0])
            self.assertEqual(len(set(changes + [baseline])), 6)

    def test_broad_legacy_filter_is_rejected_and_multiple_json_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            broad = self.make_plan(root)
            broad["solvers"] = [
                {"name": "IndexKahn1024", "filter": "^Mesh/Solver.*"}
            ]
            with self.assertRaises(ValueError):
                runner.make_tasks(broad, "fingerprint")

            plan = self.make_plan(root, "multiple")
            fingerprint, task, output, success = self.run_one(plan)
            self.assertFalse(success)
            task_dir = output / "runs" / task.key
            self.assertFalse(runner.completed(task_dir, fingerprint))
            self.assertFalse((task_dir / "benchmark.json").exists())
            self.assertEqual(
                runner.task_command(plan, task, Path("result.json"))[-5],
                "--benchmark_filter=^Mesh/Solver$",
            )

    def test_success_resume_and_result_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan = self.make_plan(Path(directory))
            fingerprint, task, output, success = self.run_one(plan)
            self.assertTrue(success)
            task_dir = output / "runs" / task.key
            self.assertTrue(runner.completed(task_dir, fingerprint))
            self.assertEqual(Path(directory, "counter").read_text(), "1")
            result = task_dir / "benchmark.json"
            result.write_text(result.read_text() + " ", encoding="utf-8")
            self.assertFalse(runner.completed(task_dir, fingerprint))
            self.assertEqual(runner.collect_rows(output, fingerprint), [])

    def test_clean_cli_resume_preserves_csv_hash_and_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.make_plan(root)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            command = [sys.executable, str(RUNNER_PATH), "--plan", str(plan_path)]
            first = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=5, check=False,
            )
            self.assertEqual(first.returncode, 0, first.stdout)
            fingerprint, _ = runner.measurement_identity(plan)
            tracked = [
                Path(plan["output"]) / "results.csv",
                Path(plan["output"]) / "summary.csv",
                Path(plan["output"]) / "summaries" / fingerprint / "results.csv",
                Path(plan["output"]) / "summaries" / fingerprint / "summary.csv",
            ]
            before = {
                path: (runner.sha256_file(path), path.stat().st_mtime_ns)
                for path in tracked
            }
            second = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=5, check=False,
            )
            self.assertEqual(second.returncode, 0, second.stdout)
            self.assertIn("complete=0 resumed=1 failed=0", second.stdout)
            self.assertEqual(
                before,
                {
                    path: (runner.sha256_file(path), path.stat().st_mtime_ns)
                    for path in tracked
                },
            )
            self.assertEqual((root / "counter").read_text(), "1")

    def test_crash_is_retried_and_old_json_cannot_mask_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.make_plan(root, "crash_once")
            fingerprint, task, output, success = self.run_one(plan)
            self.assertFalse(success)
            task_dir = output / "runs" / task.key
            self.assertFalse(runner.completed(task_dir, fingerprint))
            self.assertTrue(runner.run_task(plan, task, output, 0))
            self.assertTrue(runner.completed(task_dir, fingerprint))
            self.assertEqual((root / "counter").read_text(), "2")
            self.assertEqual(len(list((task_dir / "attempts").glob("*.log"))), 2)

    def test_timeout_kills_entire_child_process_group_and_resume_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.make_plan(root, "spawn_timeout_once", timeout=0.15)
            fingerprint, task, output, success = self.run_one(plan)
            self.assertFalse(success)
            child_pid = int((root / "child.pid").read_text())
            time.sleep(0.2)
            self.assertFalse(pid_exists(child_pid), f"orphan process {child_pid}")
            status = json.loads((output / "runs" / task.key / "status.json").read_text())
            self.assertEqual(status["state"], "timeout")

            # The same plan resumes: the mock succeeds on its second attempt.
            self.assertTrue(runner.run_task(plan, task, output, 0))
            self.assertTrue(runner.completed(output / "runs" / task.key, fingerprint))

    def test_runner_interrupt_kills_children_and_resume_succeeds(self) -> None:
        for interrupt_signal in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=interrupt_signal):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    plan = self.make_plan(root, "interrupt_once", timeout=10)
                    plan_path = root / "plan.json"
                    plan_path.write_text(json.dumps(plan), encoding="utf-8")
                    process = subprocess.Popen(
                        [sys.executable, str(RUNNER_PATH), "--plan", str(plan_path)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    time.sleep(0.5)
                    self.assertTrue((root / "child.pid").is_file())
                    child_pid = int((root / "child.pid").read_text())
                    process.send_signal(interrupt_signal)
                    process.communicate(timeout=5)
                    self.assertEqual(process.returncode, 128 + interrupt_signal)
                    time.sleep(0.2)
                    self.assertFalse(
                        pid_exists(child_pid), f"orphan process {child_pid}"
                    )

                    resumed = subprocess.run(
                        [sys.executable, str(RUNNER_PATH), "--plan", str(plan_path)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=5,
                        check=False,
                    )
                    self.assertEqual(resumed.returncode, 0, resumed.stdout)
                    self.assertEqual((root / "counter").read_text(), "2")

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux pdeathsig")
    def test_sigkill_runner_does_not_leave_direct_child_and_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.make_plan(root, "kill9_once", timeout=10)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, str(RUNNER_PATH), "--plan", str(plan_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            time.sleep(0.5)
            child_pid = int((root / "self.pid").read_text())
            process.kill()
            process.communicate(timeout=5)
            self.assertEqual(process.returncode, -signal.SIGKILL)
            time.sleep(0.2)
            self.assertFalse(pid_exists(child_pid), f"orphan process {child_pid}")

            resumed = subprocess.run(
                [sys.executable, str(RUNNER_PATH), "--plan", str(plan_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout)
            self.assertEqual((root / "counter").read_text(), "2")

    def test_csv_median_sample_cv_and_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fingerprint = "f" * 20
            for repetition, milliseconds in ((1, 2.0), (2, 4.0)):
                task = runner.Task(
                    "mesh", "path", "solver", "Mesh/Solver",
                    "^Mesh/Solver$", repetition, fingerprint,
                )
                task_dir = root / "runs" / task.key
                task_dir.mkdir(parents=True)
                result_path = task_dir / "benchmark.json"
                runner.atomic_json(
                    result_path,
                    {"benchmarks": [{
                        "name": "Mesh/Solver", "run_type": "iteration",
                        "real_time": milliseconds, "cpu_time": milliseconds,
                        "time_unit": "ms", "cells": 10,
                    }]},
                )
                status = {
                    "state": "complete", "returncode": 0,
                    "run_fingerprint": fingerprint, "task": task.__dict__,
                    "result_sha256": runner.sha256_file(result_path),
                    "log": str(task_dir / "log"),
                }
                runner.atomic_json(task_dir / "status.json", status)
            runner.rebuild_csv(root, fingerprint)
            tracked = [
                root / "results.csv",
                root / "summary.csv",
                root / "summaries" / fingerprint / "results.csv",
                root / "summaries" / fingerprint / "summary.csv",
            ]
            original = {
                path: (runner.sha256_file(path), path.stat().st_mtime_ns)
                for path in tracked
            }
            runner.rebuild_csv(root, fingerprint)
            self.assertEqual(
                original,
                {
                    path: (runner.sha256_file(path), path.stat().st_mtime_ns)
                    for path in tracked
                },
            )
            with (root / "summary.csv").open(newline="", encoding="utf-8") as source:
                summary = next(csv.DictReader(source))
            self.assertAlmostEqual(float(summary["real_median_seconds"]), 0.003)
            self.assertAlmostEqual(
                float(summary["real_cv_percent"]), 100 * math.sqrt(2) / 3
            )
            self.assertEqual(summary["completed_repetitions"], "2")
            self.assertFalse((root / "summary.csv.tmp").exists())
            self.assertFalse((root / "results.csv.tmp").exists())
            archived = root / "summaries" / fingerprint
            self.assertTrue((archived / "summary.csv").is_file())
            self.assertTrue((archived / "results.csv").is_file())

            expected_results = (root / "results.csv").read_bytes()
            expected_summary = (root / "summary.csv").read_bytes()
            (root / "results.csv").write_text("corrupt", encoding="utf-8")
            (root / "summary.csv").unlink()
            runner.rebuild_csv(root, fingerprint)
            self.assertEqual((root / "results.csv").read_bytes(), expected_results)
            self.assertEqual((root / "summary.csv").read_bytes(), expected_summary)


if __name__ == "__main__":
    unittest.main()
