# SPDX-License-Identifier: Apache-2.0
"""Run the actual lifetime helper without torch, vLLM, or an accelerator."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


def load_helper(backend, paths):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/request_state.py"
    node = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_release_current_kv"
    )
    scope = {"torch": SimpleNamespace(npu=backend), "record_path": paths.append}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope["_release_current_kv"]


class TestCurrentKVLifetime(unittest.TestCase):
    def run_case(
        self, *, prefill=True, capture=False, copied=False, unknown=False, same_stream=False, packed=False, fail=False
    ):
        calls, paths = [], []

        class Stream:
            device = "npu:0"

            def wait_stream(self, consumer):
                self_outer.assertIs(attn._request_state_current_kv, value)
                calls.append(("producer_wait", consumer))
                if fail:
                    raise RuntimeError("event failure")

        class Tensor:
            def record_stream(self, stream):
                calls.append(("record", self, stream))

        self_outer = self
        consumer = Stream()
        producer = None if unknown else (consumer if same_stream else Stream())
        value = Tensor()
        current = Tensor() if copied else value
        attn = SimpleNamespace(_request_state_current_kv=value, _request_state_current_kv_stream=producer)
        if packed:
            attn.packed_cache_ops = object()
        backend = SimpleNamespace(current_stream=lambda: consumer, is_current_stream_capturing=lambda: capture)
        helper = load_helper(backend, paths)
        if fail:
            with self.assertRaisesRegex(RuntimeError, "event failure"):
                helper(attn, current, prefill=prefill)
            self.assertIs(attn._request_state_current_kv, value)
            self.assertIs(attn._request_state_current_kv_stream, producer)
        else:
            helper(attn, current, prefill=prefill)
            self.assertIsNone(attn._request_state_current_kv)
            self.assertIsNone(attn._request_state_current_kv_stream)
        return calls, paths

    def test_eager_prefill_returns_dependency_before_release(self):
        calls, paths = self.run_case()
        self.assertEqual([c[0] for c in calls], ["producer_wait"])
        self.assertEqual(paths, ["prefill_producer_return_dependency"])

    def test_decode_capture_unknown_same_and_packed_keep_original_protection(self):
        for options in [
            {"prefill": False},
            {"capture": True},
            {"unknown": True},
            {"same_stream": True},
            {"packed": True},
        ]:
            with self.subTest(options=options):
                calls, paths = self.run_case(**options)
                self.assertEqual([c[0] for c in calls], ["record"])
                self.assertEqual(paths, ["current_kv_record_stream_fallback"])

    def test_contiguous_copy_protects_both_tensors(self):
        calls, _ = self.run_case(copied=True)
        self.assertEqual([c[0] for c in calls], ["record", "record"])

    def test_cache_only_uses_swa_metadata_contract(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/request_state.py"
        node = next(
            n
            for n in ast.walk(ast.parse(source.read_text()))
            if isinstance(n, ast.FunctionDef) and n.name == "update_only"
        )
        # Supply the real wrapper's kernel import as a CPU spy. LayerMetadata
        # exposes query bounds through swa, never on the outer bundle.
        node.body = [n for n in node.body if not isinstance(n, ast.ImportFrom)]
        for item in ast.walk(node):
            if isinstance(item, ast.If):
                item.body = [n for n in item.body if not isinstance(n, ast.ImportFrom)]
        for query in [1, 4]:
            calls = []
            value = SimpleNamespace(contiguous=lambda: "current")
            attn = SimpleNamespace(
                _request_state_current_kv=value,
                dsa_attn=SimpleNamespace(swa_cache_layer=SimpleNamespace(kv_cache=["ring"])),
            )
            swa = SimpleNamespace(state_ids="ids", query_start_loc="qstart", seq_lens="seq", max_query_len=query)
            scope = {
                "original_update": lambda *a, sink=calls: sink.append("project"),
                "commit": lambda *a, sink=calls: sink.append(a),
                "_release_current_kv": lambda a, v, *, prefill, sink=calls: sink.append((v, prefill)),
            }
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
            scope["update_only"](object(), attn, SimpleNamespace(shape=(1,)), SimpleNamespace(swa=swa))
            self.assertEqual(calls, ["project", ("ring", "current", "ids", "qstart", "seq"), ("current", query > 1)])

    def test_failed_dependency_retains_tensor_and_producer(self):
        calls, paths = self.run_case(fail=True)
        self.assertEqual([c[0] for c in calls], ["producer_wait"])
        self.assertEqual(paths, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
