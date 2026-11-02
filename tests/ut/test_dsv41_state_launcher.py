# SPDX-License-Identifier: Apache-2.0
"""Standalone launcher regression; no model, server or accelerator is started."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("dsv41_state_launcher", ROOT / "examples/dsv41_state/launch.py")
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class TestStateLauncher(unittest.TestCase):
    def test_default_and_explicit_window_protocol(self):
        for requested, expected in [(None, "0"), ("0", "0"), ("1", "1")]:
            with self.subTest(requested=requested), tempfile.TemporaryDirectory() as tmp:
                environment = {} if requested is None else {"MENGZI_STATE_DECODE_DIRECT": requested}
                args = ["launch.py", "--results-dir", tmp, "--", "/model", "--tensor-parallel-size", "4"]
                with (
                    patch.dict(os.environ, environment, clear=True),
                    patch("sys.argv", args),
                    patch.object(launcher.subprocess, "call", return_value=0) as call,
                ):
                    with self.assertRaises(SystemExit) as result:
                        launcher.main()
                    self.assertEqual(result.exception.code, 0)
                    self.assertEqual(call.call_args.args[0], ["vllm", "serve", "/model", "--tensor-parallel-size", "4"])
                    self.assertEqual(call.call_args.kwargs["env"]["MENGZI_STATE_DECODE_DIRECT"], expected)
                    self.assertEqual(os.environ.get("MENGZI_STATE_DECODE_DIRECT"), requested)


if __name__ == "__main__":
    unittest.main()
