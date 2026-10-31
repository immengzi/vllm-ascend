# SPDX-License-Identifier: Apache-2.0
"""Run the production slot-address helper on CPU without NPU plugin imports."""

import ast
import unittest
from pathlib import Path

import torch


class TestRequestRingSlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram/hash_state.py"
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "request_ring_slots")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        cls.slots = staticmethod(namespace["request_ring_slots"])

    def test_prefill_commits_only_last_turn(self):
        positions = torch.arange(2048)
        slots = self.slots(
            positions, torch.tensor([0, 2048]), torch.zeros(2048, dtype=torch.int32), torch.tensor([3]), 128
        )
        self.assertTrue(torch.all(slots[:1920] == -1))
        torch.testing.assert_close(slots[1920:], 384 + torch.arange(128), rtol=0, atol=0)
        self.assertEqual(len(set(slots[slots >= 0].tolist())), 128)

    def test_absolute_position_wrap_and_slot_reuse(self):
        for position in [0, 127, 128, 129, 2048, 100001]:
            for slot in [1, 7, 1]:
                result = self.slots(
                    torch.tensor([position]), torch.tensor([0, 1]), torch.tensor([0]), torch.tensor([slot]), 128
                )
                self.assertEqual(result.tolist(), [slot * 128 + position % 128])

    def test_multiple_requests_and_query_padding(self):
        result = self.slots(
            torch.tensor([127, 129, 0, 0]),
            torch.tensor([0, 1, 2]),
            torch.tensor([0, 1, 1, 1]),
            torch.tensor([2, 3]),
            128,
        )
        self.assertEqual(result.tolist(), [383, 385, -1, -1])

    def test_null_state_never_writes(self):
        result = self.slots(torch.tensor([128]), torch.tensor([0, 1]), torch.tensor([0]), torch.tensor([0]), 128)
        self.assertEqual(result.tolist(), [-1])


if __name__ == "__main__":
    unittest.main()
