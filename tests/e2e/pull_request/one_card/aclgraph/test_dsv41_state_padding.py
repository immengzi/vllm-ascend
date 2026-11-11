# SPDX-License-Identifier: Apache-2.0
"""Padded positive-query rows must not write a ring's predecessor or null slot."""

import unittest

import torch

from vllm_ascend.models.deepseek_v41.state_kernels import assemble, assemble_cp, commit


class TestStatePaddingGraph(unittest.TestCase):
    @unittest.skipUnless(hasattr(torch, "npu") and torch.npu.is_available(), "NPU required")
    def test_protected_backing_and_dummy_staging(self):
        torch.npu.set_device(0)
        for cp in [False, True]:
            for dtype in [torch.bfloat16, torch.float16]:
                with self.subTest(cp=cp, dtype=dtype):
                    self._exercise(cp, dtype)

    def _exercise(self, cp, dtype):
        # The predecessor guard belongs to the same allocation;
        # a regression does not cause an illegal DMA access.
        backing = torch.full((4, 128, 1, 512), -31, device="npu", dtype=dtype)
        ring = backing[1:3]
        ring.fill_(17)
        current = torch.tensor([2, 7], device="npu", dtype=dtype)[:, None, None].expand(2, 1, 512).contiguous()
        ids = torch.tensor([1, 0], device="npu", dtype=torch.int32)
        starts = torch.tensor([0, 1, 2], device="npu", dtype=torch.int32)
        lengths = torch.tensor([100001, 0], device="npu", dtype=torch.int32)
        bases = torch.tensor([0, 128], device="npu", dtype=torch.int32)
        staging = torch.full((256, 512), 19, device="npu", dtype=dtype)

        def operation():
            if cp:
                assemble_cp(ring, current, ids, starts, lengths, starts, lengths, bases, staging, 1)
            else:
                assemble(ring, current, ids, starts, lengths, bases, staging, 1)
            commit(ring, current, ids, starts, lengths)

        def reset():
            backing.fill_(-31)
            ring.fill_(17)
            staging.fill_(19)

        def check():
            actual = backing.cpu()
            self.assertTrue(bool((actual[0] == -31).all()))
            self.assertTrue(bool((actual[3] == -31).all()))
            self.assertTrue(bool((actual[1] == 17).all()))
            expected = torch.full((128, 1, 512), 17, dtype=dtype)
            expected[100000 % 128].fill_(2)
            torch.testing.assert_close(actual[2], expected, rtol=0, atol=0)
            values = staging.cpu()
            self.assertTrue(bool((values[:127] == 17).all()))
            self.assertTrue(bool((values[127] == 2).all()))
            self.assertTrue(bool((values[128:] == 19).all()))

        operation()
        torch.npu.synchronize()
        check()
        for _ in range(3):
            operation()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            operation()
        torch.npu.synchronize()
        for _ in range(2):
            reset()
            graph.replay()
            torch.npu.synchronize()
            check()


if __name__ == "__main__":
    unittest.main()
