# SPDX-License-Identifier: Apache-2.0
"""Standalone launcher regression; no model, server or accelerator is started."""

import ast
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
        for requested, expected in [(None, "0"), ("0", "0")]:
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

    def test_rejects_unqualified_direct_ring_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = ["launch.py", "--results-dir", tmp, "--", "/model"]
            with (
                patch.dict(os.environ, {"MENGZI_STATE_DECODE_DIRECT": "1"}, clear=True),
                patch("sys.argv", args),
                patch.object(launcher.subprocess, "call") as call,
            ):
                with self.assertRaises(SystemExit) as result:
                    launcher.main()
                self.assertEqual(result.exception.code, 2)
                call.assert_not_called()

    def test_runtime_also_rejects_direct_ring_without_launcher(self):
        source = ROOT / "vllm_ascend/models/deepseek_v41/request_state.py"
        module = ast.parse(source.read_text())
        # Execute the actual module-level mode selection/guard without importing
        # torch or vLLM. This verifies the bypass-launcher contract on CPU only.
        nodes = [
            node
            for node in module.body
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "_direct_decode" for target in node.targets)
            )
            or (isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "_direct_decode")
        ]
        selection = compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec")
        for requested in [None, "0", "1"]:
            with self.subTest(requested=requested):
                environment = {} if requested is None else {"MENGZI_STATE_DECODE_DIRECT": requested}
                with patch.dict(os.environ, environment, clear=True):
                    scope = {"os": os}
                    if requested == "1":
                        with self.assertRaisesRegex(ValueError, "failed strict numerical equivalence"):
                            exec(selection, scope)
                    else:
                        exec(selection, scope)
                        self.assertFalse(scope["_direct_decode"])


if __name__ == "__main__":
    unittest.main()
