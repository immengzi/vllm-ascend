# SPDX-License-Identifier: Apache-2.0
"""Exercise production metadata functions on CPU without importing NPU plugins.

A graph retains tensor references from capture. The regression checks those
references after later metadata builds, including a different local state ID.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/request_state.py"


class WindowSpec:
    pass


class TestRequestStateMetadata(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text())
        names = {"_prepare_state_metadata", "_cpu_list", "export_state"}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.namespace = {
            "torch": torch,
            "RequestWindowSpec": WindowSpec,
            "WINDOW": 128,
            "_direct_decode": False,
            "record_path": Mock(),
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), self.namespace)
        self.prepare = self.namespace["_prepare_state_metadata"]
        self.builder = SimpleNamespace(
            _cache_kind="swa",
            kv_cache_spec=WindowSpec(),
            _seq_lens=torch.zeros(4, dtype=torch.int32),
            vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=128, max_num_seqs=4)),
        )
        self.ids = torch.ones((4, 1), dtype=torch.int32)

    def build(self, lengths, queries=None, max_query_len=1):
        count = len(lengths)
        queries = queries if queries is not None else [1] * count
        starts = [0]
        for length in queries:
            starts.append(starts[-1] + length)
        self.builder._seq_lens[:count].copy_(torch.tensor(lengths, dtype=torch.int32))
        common = SimpleNamespace(
            block_table_tensor=self.ids[:count],
            query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
            seq_lens_cpu=torch.tensor(lengths, dtype=torch.int32),
        )
        result = SimpleNamespace(
            num_reqs=count,
            max_query_len=max_query_len,
            seq_lens=self.builder._seq_lens[:count],
        )
        return self.prepare(self.builder, common, result)

    def test_staged_replay_references_receive_new_length_and_slot(self):
        first = self.build([1])
        names = ["state_ids", "state_stage_lengths", "state_stage_row_starts", "state_stage_block_table"]
        captured = {name: getattr(first, name) for name in names}
        pointers = {name: value.data_ptr() for name, value in captured.items()}
        self.ids[0, 0] = 3
        later = self.build([129])
        self.assertEqual(pointers, {name: getattr(later, name).data_ptr() for name in names})
        self.assertEqual(captured["state_ids"].tolist(), [3])
        self.assertEqual(captured["state_stage_lengths"].tolist(), [128])
        self.assertEqual(captured["state_stage_block_table"].tolist(), [[0]])
        self.assertIs(later.request_state["state_stage_lengths"], later.state_stage_lengths)

    def test_empty_request_row_keeps_independent_page(self):
        first = self.build([0, 129], [0, 1])
        rows = first.state_stage_row_starts
        table = first.state_stage_block_table
        later = self.build([0, 2068], [0, 1])
        self.assertEqual(later.state_stage_lengths.tolist(), [0, 128])
        self.assertEqual(rows.tolist(), [0, 128])
        self.assertEqual(table.tolist(), [[0], [1]])
        self.assertEqual(rows.data_ptr(), later.state_stage_row_starts.data_ptr())
        self.assertEqual(table.data_ptr(), later.state_stage_block_table.data_ptr())
        self.assertLessEqual(later.state_stage_rows_used, later.state_staging_capacity)

    def test_cp_rank_without_query_uses_global_decode_protocol(self):
        self.builder._global_builder = SimpleNamespace(_request_state_last_result=SimpleNamespace(max_query_len=1))
        first = self.build([0], [0], max_query_len=0)
        later = self.build([0], [0], max_query_len=0)
        self.assertEqual(first.state_stage_lengths.tolist(), [0])
        self.assertEqual(first.state_stage_lengths.data_ptr(), later.state_stage_lengths.data_ptr())

    def test_direct_protocol_keeps_existing_persistent_references(self):
        self.namespace["_direct_decode"] = True
        first = self.build([1])
        later = self.build([100001])
        self.assertEqual(first.state_stage_lengths.data_ptr(), later.state_stage_lengths.data_ptr())
        self.assertEqual(first.state_stage_lengths.tolist(), [128])
        self.assertFalse(hasattr(later, "state_stage_block_table"))

    def test_multi_query_prefill_retains_chronological_lookback(self):
        value = self.build([129], [3], max_query_len=3)
        self.assertEqual(value.state_stage_lengths.tolist(), [129])
        self.assertEqual(value.state_stage_block_table.tolist(), [[0, 1]])
        self.assertEqual(value.state_stage_rows_used, 256)


if __name__ == "__main__":
    unittest.main()
