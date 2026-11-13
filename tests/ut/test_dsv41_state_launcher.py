# SPDX-License-Identifier: Apache-2.0
"""Standalone CLI checks; no model, server or accelerator is started."""

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"examples/dsv41_state/{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load_example("launch")
benchmark = load_example("bench")


class TestStateLauncher(unittest.TestCase):
    def test_launch_forwards_arguments_and_preserves_environment(self):
        environment = {"PATH": "/bin", "PYTHONPATH": "/installed", "VLLM_USE_V2_MODEL_RUNNER": "1"}
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("sys.argv", ["launch.py", "--", "/weights", "--tensor-parallel-size", "4"]),
            patch.object(launcher.subprocess, "call", return_value=7) as call,
        ):
            with self.assertRaises(SystemExit) as result:
                launcher.main()
            self.assertEqual(result.exception.code, 7)
            self.assertEqual(call.call_args.args[0], ["vllm", "serve", "/weights", "--tensor-parallel-size", "4"])
            self.assertEqual(call.call_args.kwargs["env"], {**environment, "VLLM_USE_V2_MODEL_RUNNER": "0"})
            self.assertEqual(dict(os.environ), environment)

    def test_missing_server_arguments_does_not_launch(self):
        with patch("sys.argv", ["launch.py"]), patch.object(launcher.subprocess, "call") as call:
            with self.assertRaises(SystemExit) as result:
                launcher.main()
            self.assertEqual(result.exception.code, 2)
            call.assert_not_called()

    def test_benchmark_plan_uses_explicit_paths_without_starting_server(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            config = directory / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "port": 8000,
                        "model": "/weights",
                        "tokenizer": "/tokenizer",
                        "served_model_name": "example-model",
                        "repeats": 1,
                        "pressure_repeats": 2,
                        "deadline": "2030-01-01T00:00:00+08:00",
                    }
                )
            )
            output = directory / "nested/results"
            printed = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "bench.py",
                        "--config",
                        str(config),
                        "--results-dir",
                        str(output),
                        "--suite",
                        "diagnostic",
                        "--plan-only",
                    ],
                ),
                patch.object(benchmark, "run_one") as run,
                patch.object(benchmark.subprocess, "check_output") as command,
                contextlib.redirect_stdout(printed),
            ):
                benchmark.main()
            jobs = json.loads(printed.getvalue())
            self.assertEqual(len(jobs), 4)
            for job in jobs:
                self.assertEqual(job[job.index("--result-dir") + 1], str(output / "diagnostic"))
                self.assertEqual(job[job.index("--model") + 1], "/weights")
                self.assertEqual(job[job.index("--tokenizer") + 1], "/tokenizer")
                self.assertEqual(job[job.index("--served-model-name") + 1], "example-model")
            run.assert_not_called()
            command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
