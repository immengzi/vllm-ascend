# SPDX-License-Identifier: Apache-2.0
"""Exercise production metadata functions on CPU without importing NPU plugins.

A graph retains tensor references from capture. The regression checks those
references after later metadata builds, including a different local state ID.
"""

import unittest
from types import SimpleNamespace

import torch

from vllm_ascend.models.deepseek_v41.window_state import (
    RequestWindowMetadataBuilder,
    prepare_request_window_metadata,
)


class TestRequestStateMetadata(unittest.TestCase):
    def setUp(self):
        self.prepare = prepare_request_window_metadata
        lengths = torch.zeros(4, dtype=torch.int32)
        self.builder = SimpleNamespace(
            _seq_lens=lengths,
            request_window_builder=RequestWindowMetadataBuilder(lengths, 1152),
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
        return self.prepare(self.builder, common, result).request_state

    def test_staged_replay_references_receive_new_length_and_slot(self):
        first = self.build([1])
        names = ["state_ids", "lengths", "row_starts", "block_table"]
        captured = {name: getattr(first, name) for name in names}
        pointers = {name: value.data_ptr() for name, value in captured.items()}
        self.ids[0, 0] = 3
        later = self.build([129])
        self.assertEqual(pointers, {name: getattr(later, name).data_ptr() for name in names})
        self.assertEqual(captured["state_ids"].tolist(), [3])
        self.assertEqual(captured["lengths"].tolist(), [128])
        self.assertEqual(captured["block_table"].tolist(), [[0]])

    def test_empty_request_row_keeps_independent_page(self):
        first = self.build([0, 129], [0, 1])
        rows = first.row_starts
        table = first.block_table
        later = self.build([0, 2068], [0, 1])
        self.assertEqual(later.lengths.tolist(), [0, 128])
        self.assertEqual(rows.tolist(), [0, 128])
        self.assertEqual(table.tolist(), [[0], [1]])
        self.assertEqual(rows.data_ptr(), later.row_starts.data_ptr())
        self.assertEqual(table.data_ptr(), later.block_table.data_ptr())
        self.assertLessEqual(later.rows_used, later.capacity)

    def test_cp_rank_without_query_uses_global_decode_protocol(self):
        self.builder._global_builder = SimpleNamespace(request_window_builder=SimpleNamespace(last_max_query_len=1))
        first = self.build([0], [0], max_query_len=0)
        later = self.build([0], [0], max_query_len=0)
        self.assertEqual(first.lengths.tolist(), [0])
        self.assertEqual(first.lengths.data_ptr(), later.lengths.data_ptr())

    def test_multi_query_prefill_retains_chronological_lookback(self):
        value = self.build([129], [3], max_query_len=3)
        self.assertEqual(value.lengths.tolist(), [129])
        self.assertEqual(value.block_table.tolist(), [[0, 1]])
        self.assertEqual(value.rows_used, 256)


if __name__ == "__main__":
    unittest.main()
