import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager, prepare_inputs_to_capture
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers


class TestAclGraphUtilsV2(unittest.TestCase):

    def test_make_prefill_dummy_keeps_prefill_shape(self):
        buffers = AscendInputBuffers(
            max_num_reqs=8,
            max_num_tokens=16,
            device=torch.device("cpu"),
        )

        input_batch = AscendInputBatch.make_prefill_dummy(
            num_tokens=4,
            num_reqs_after_padding=4,
            input_buffers=buffers,
        )

        self.assertEqual(input_batch.num_reqs, 2)
        self.assertEqual(input_batch.num_reqs_after_padding, 4)
        self.assertEqual(input_batch.num_tokens, 4)
        self.assertEqual(input_batch.query_start_loc_np.tolist(), [0, 2, 4, 4, 4])
        self.assertEqual(input_batch.seq_lens_np.tolist(), [2, 2, 0, 0])
        self.assertEqual(input_batch.attn_state, AscendAttentionState.PrefillNoCache)

    def test_prepare_inputs_to_capture_uses_full_capture_for_laps_prefill(self):
        buffers = AscendInputBuffers(
            max_num_reqs=8,
            max_num_tokens=16,
            device=torch.device("cpu"),
        )
        block_tables = MagicMock()
        block_tables.cp_size = 1
        block_tables.get_dummy_block_tables.return_value = (torch.zeros((4, 1), dtype=torch.int32),)
        block_tables.get_dummy_slot_mappings.return_value = torch.zeros((1, 4), dtype=torch.int32)
        attn_groups = [[MagicMock()]]
        kv_cache_config = MagicMock()

        model_state = MagicMock()
        model_state.prepare_attn.return_value = {"layer0": "metadata"}

        input_batch, attn_metadata, slot_mappings = prepare_inputs_to_capture(
            num_reqs=4,
            num_tokens=4,
            model_state=model_state,
            input_buffers=buffers,
            block_tables=block_tables,
            attn_groups=attn_groups,
            kv_cache_config=kv_cache_config,
            use_laps_prefill_graph=True,
        )

        self.assertIsInstance(input_batch, AscendInputBatch)
        self.assertEqual(input_batch.attn_state, AscendAttentionState.PrefillNoCache)
        self.assertEqual(attn_metadata, {"layer0": "metadata"})
        self.assertIn("layer0", slot_mappings)

        call_args = model_state.prepare_attn.call_args
        self.assertIs(call_args.args[0], input_batch)
        self.assertEqual(call_args.args[1], CUDAGraphMode.FULL)
        self.assertEqual(call_args.args[0].num_reqs_after_padding, 4)
        self.assertTrue(call_args.kwargs["for_capture"])

    def test_materialize_laps_prefill_input_batch_uses_stable_buffers(self):
        vllm_config = MagicMock()
        vllm_config.scheduler_config.max_num_seqs = 8
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.compilation_config.cudagraph_capture_sizes = [4]
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
        model_runner = SimpleNamespace(
            speculative_config=None,
            model_config=SimpleNamespace(is_encoder_decoder=False),
            use_dcp=False,
        )

        with unittest.mock.patch(
            "vllm_ascend.worker.v2.aclgraph_utils.envs.VLLM_ASCEND_LAPS_SCHEDULING",
            True,
        ):
            manager = ModelAclGraphManager(
                vllm_config=vllm_config,
                device=torch.device("cpu"),
                cudagraph_mode=CUDAGraphMode.FULL,
                decode_query_len=1,
                model_runner=model_runner,
            )

        desc = manager.laps_prefill_descs[
            next(iter(manager.laps_prefill_descs.keys()))
        ]
        state = manager._get_or_create_laps_prefill_state(desc)
        source_buffers = AscendInputBuffers(
            max_num_reqs=8,
            max_num_tokens=16,
            device=torch.device("cpu"),
        )
        input_batch = AscendInputBatch.make_prefill_dummy(
            num_tokens=4,
            num_reqs_after_padding=desc.num_reqs,
            input_buffers=source_buffers,
        )
        input_batch.input_ids[:4] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        input_batch.positions[:4] = torch.tensor([10, 11, 12, 13], dtype=torch.int64)

        materialized = manager.materialize_laps_prefill_input_batch(desc, input_batch)

        self.assertEqual(materialized.input_ids.data_ptr(), state.input_buffers.input_ids.data_ptr())
        self.assertEqual(materialized.positions.data_ptr(), state.input_buffers.positions.data_ptr())
        self.assertEqual(materialized.query_start_loc.data_ptr(), state.input_buffers.query_start_loc.data_ptr())
        self.assertEqual(materialized.input_ids[:4].tolist(), [1, 2, 3, 4])
        self.assertEqual(materialized.positions[:4].tolist(), [10, 11, 12, 13])
        self.assertEqual(materialized.seq_lens_np.tolist(), input_batch.seq_lens_np.tolist())
        self.assertEqual(materialized.query_start_loc_np.tolist(), input_batch.query_start_loc_np.tolist())


if __name__ == "__main__":
    unittest.main()
