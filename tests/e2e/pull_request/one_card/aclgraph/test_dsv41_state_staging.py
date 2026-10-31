# SPDX-License-Identifier: Apache-2.0
"""Actual 64-head chronological staging graph with changing lengths and slots."""

import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from vllm_ascend.models.deepseek_v41 import request_state as state
from vllm_ascend.models.deepseek_v41.state_kernels import assemble, commit
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op


class TestChronologicalStateGraph(unittest.TestCase):
    @unittest.skipUnless(hasattr(torch, "npu") and torch.npu.is_available(), "NPU required")
    def test_lengths_slots_and_valid_window(self):
        # Keep graph inputs and retained output references in one test lifetime.
        bootstrap_custom_op_env()
        enable_custom_op()
        torch.npu.set_device(0)
        with patch.object(state, "_direct_decode", False), patch.object(state, "record_path"):
            b = NS(
                _cache_kind="swa",
                kv_cache_spec=object.__new__(state.RequestWindowSpec),
                _seq_lens=torch.ones(4, dtype=torch.int32, device="npu"),
                vllm_config=NS(scheduler_config=NS(max_num_batched_tokens=4, max_num_seqs=4)),
            )
            ids = torch.ones((1, 1), dtype=torch.int32, device="npu")
            starts = torch.tensor([0, 1], dtype=torch.int32, device="npu")
            ends = b._seq_lens[:1]

            def build(total):
                ends.fill_(total)
                c = NS(
                    block_table_tensor=ids,
                    query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
                    seq_lens_cpu=torch.tensor([total], dtype=torch.int32),
                )
                r = NS(num_reqs=1, max_query_len=1, seq_lens=ends)
                return state._prepare_state_metadata(b, c, r)

            first = build(128)
            ring = torch.zeros((3, 128, 1, 512), dtype=torch.bfloat16, device="npu")
            current = torch.zeros((1, 512), dtype=torch.bfloat16, device="npu")
            q = torch.zeros((1, 64, 512), dtype=torch.bfloat16, device="npu")
            staging = torch.zeros((1, 128, 1, 512), dtype=torch.bfloat16, device="npu")
            sinks = torch.zeros(64, dtype=torch.float32, device="npu")
            ptrs = {
                k: getattr(first, k).data_ptr()
                for k in ["state_ids", "state_stage_lengths", "state_stage_row_starts", "state_stage_block_table"]
            }

            def operation():
                assemble(ring, current, first.state_ids, starts, ends, first.state_stage_row_starts, staging, 1)
                meta = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
                    64,
                    1,
                    512,
                    cu_seqlens_q=starts,
                    seqused_ori_kv=first.state_stage_lengths,
                    batch_size=1,
                    max_seqlen_q=1,
                    max_seqlen_ori_kv=128,
                    max_seqlen_cmp_kv=0,
                    ori_topk=0,
                    cmp_topk=0,
                    cmp_ratio=0,
                    ori_mask_mode=4,
                    cmp_mask_mode=0,
                    ori_win_left=127,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_BBND",
                    has_ori_kv=True,
                    has_cmp_kv=False,
                )
                out = torch.ops._C_ascend.npu_sparse_flash_mla(
                    q,
                    ori_kv=staging,
                    ori_block_table=first.state_stage_block_table,
                    cu_seqlens_q=starts,
                    seqused_ori_kv=first.state_stage_lengths,
                    sinks=sinks,
                    metadata=meta,
                    softmax_scale=512**-0.5,
                    cmp_ratio=0,
                    ori_mask_mode=4,
                    cmp_mask_mode=0,
                    ori_win_left=127,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_BBND",
                    topk_value_mode=1,
                    return_softmax_lse=False,
                )[0]
                commit(ring, current, first.state_ids, starts, ends)
                return out

            for _ in range(3):
                operation()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                out = operation()
            torch.npu.synchronize()
            results = []
            for j, total in enumerate([1, 2, 127, 128, 129, 2067, 2068, 100001] * 2):
                torch.manual_seed(20261031 + j)
                size = min(total, 128)
                slot = 1 + j % 2
                values = (torch.randn(size, 512) * 0.1).to(torch.bfloat16)
                qc = (torch.randn(64, 512) * 0.1).to(torch.bfloat16)
                host = torch.full((3, 128, 1, 512), 42, dtype=torch.bfloat16)
                host[slot].zero_()
                for i in range(size - 1):
                    host[slot, (total - size + i) % 128, 0] = values[i]
                ring.copy_(host)
                current.copy_(values[-1:])
                q.copy_(qc.unsqueeze(0))
                ids.fill_(slot)
                new = build(total)
                assert {k: getattr(new, k).data_ptr() for k in ptrs} == ptrs
                assert new.state_stage_lengths.cpu().tolist() == [size]
                graph.replay()
                actual = out.cpu()[0].float()
                torch.testing.assert_close(staging.cpu().view(128, 512)[:size], values, rtol=0, atol=0)
                host[slot, (total - 1) % 128, 0] = values[-1]
                torch.testing.assert_close(ring.cpu(), host, rtol=0, atol=0)
                scores = qc.float() @ values.float().T * (512**-0.5)
                weights = scores.exp()
                expected = (weights @ values.float()) / (weights.sum(-1, keepdim=True) + 1)
                torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.003)
                results.append(
                    {
                        "total": total,
                        "slot": slot,
                        "heads": 64,
                        "window_copy_exact": True,
                        "ring_commit_and_isolation_exact": True,
                        "metadata_pointers_stable": True,
                        "max_abs_cpu_oracle_error": float((actual - expected).abs().max()),
                    }
                )
            self.assertEqual(len(results), 16)
            print(json.dumps({"passed": True, "cases": results}))


if __name__ == "__main__":
    unittest.main()
