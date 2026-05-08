import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.aclgraph_utils import (
    ModelAclGraphManager,
    PrefillGraphKey,
    prepare_inputs_to_capture,
)
from vllm_ascend.worker.v2.model_states.default import AscendModelState
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

    def test_prepare_laps_prefill_replay_input_batch_uses_stable_buffers(self):
        vllm_config = MagicMock()
        vllm_config.scheduler_config.max_num_seqs = 8
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.compilation_config.cudagraph_capture_sizes = [8]
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

        desc = manager.laps_prefill_descs[PrefillGraphKey(num_reqs=4, num_tokens=8)]
        state = manager._get_or_create_laps_prefill_state(desc)
        source_buffers = AscendInputBuffers(
            max_num_reqs=8,
            max_num_tokens=16,
            device=torch.device("cpu"),
        )
        input_batch = AscendInputBatch.make_prefill_dummy(
            num_tokens=4,
            num_reqs_after_padding=2,
            input_buffers=source_buffers,
        )
        input_batch.input_ids[:4] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        input_batch.positions[:4] = torch.tensor([10, 11, 12, 13], dtype=torch.int64)

        materialized = manager.prepare_laps_prefill_replay_input_batch(desc, input_batch)

        self.assertEqual(materialized.input_ids.data_ptr(), state.input_buffers.input_ids.data_ptr())
        self.assertEqual(materialized.positions.data_ptr(), state.input_buffers.positions.data_ptr())
        self.assertEqual(materialized.input_ids[:4].tolist(), [1, 2, 3, 4])
        self.assertEqual(materialized.positions[:4].tolist(), [10, 11, 12, 13])
        self.assertEqual(materialized.input_ids[4:8].tolist(), [0, 0, 0, 0])
        self.assertEqual(materialized.positions[4:8].tolist(), [0, 0, 0, 0])
        self.assertEqual(materialized.logits_indices.data_ptr(), state.logits_indices.data_ptr())
        self.assertEqual(materialized.replay_query_start_loc.data_ptr(), state.input_buffers.query_start_loc.data_ptr())
        self.assertEqual(materialized.replay_query_start_loc_np.tolist(), [0, 2, 4, 4, 8])
        self.assertEqual(materialized.replay_seq_lens_np.tolist(), [2, 2, 0, 4])

    def test_dispatch_laps_prefill_can_pick_padded_target_shape(self):
        vllm_config = MagicMock()
        vllm_config.scheduler_config.max_num_seqs = 8
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.compilation_config.cudagraph_capture_sizes = [8]
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

        desc = manager.laps_prefill_descs[PrefillGraphKey(num_reqs=2, num_tokens=8)]
        manager.graphs[desc] = MagicMock()
        self.assertEqual(manager.dispatch_laps_prefill(2, 6, 4), desc)

    def test_ascend_model_state_prefers_replay_overrides(self):
        state = AscendModelState.__new__(AscendModelState)
        state.max_model_len = 1024

        input_batch = MagicMock()
        input_batch.num_tokens = 4
        input_batch.num_reqs = 2
        input_batch.num_reqs_after_padding = 2
        input_batch.num_tokens_after_padding = 4
        input_batch.attn_state = AscendAttentionState.PrefillNoCache
        input_batch.num_scheduled_tokens = torch.tensor([2, 2], dtype=torch.int32).numpy()
        input_batch.query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
        input_batch.query_start_loc_np = torch.tensor([0, 2, 4], dtype=torch.int32).numpy()
        input_batch.seq_lens = torch.tensor([2, 2], dtype=torch.int32)
        input_batch.seq_lens_np = torch.tensor([2, 2], dtype=torch.int32).numpy()
        input_batch.dcp_local_seq_lens = None
        input_batch.replay_num_reqs = 4
        input_batch.replay_num_tokens = 8
        input_batch.replay_max_query_len = 4
        input_batch.replay_query_start_loc = torch.tensor([0, 2, 4, 4, 8], dtype=torch.int32)
        input_batch.replay_query_start_loc_np = torch.tensor([0, 2, 4, 4, 8], dtype=torch.int32).numpy()
        input_batch.replay_seq_lens = torch.tensor([2, 2, 0, 4], dtype=torch.int32)
        input_batch.replay_seq_lens_np = torch.tensor([2, 2, 0, 4], dtype=torch.int32).numpy()

        with unittest.mock.patch(
            "vllm_ascend.worker.v2.model_states.default.build_attn_metadata",
            return_value={"layer0": "metadata"},
        ) as build_attn_metadata:
            result = AscendModelState.prepare_attn(
                state,
                input_batch=input_batch,
                cudagraph_mode=CUDAGraphMode.FULL,
                block_tables=(torch.zeros((4, 1), dtype=torch.int32),),
                slot_mappings=torch.zeros((1, 8), dtype=torch.int32),
                attn_groups=[[MagicMock()]],
                kv_cache_config=MagicMock(),
            )

        self.assertEqual(result, {"layer0": "metadata"})
        self.assertEqual(build_attn_metadata.call_args.kwargs["num_reqs"], 4)
        self.assertEqual(build_attn_metadata.call_args.kwargs["num_tokens"], 8)
        self.assertTrue(torch.equal(build_attn_metadata.call_args.kwargs["query_start_loc_gpu"],
                                    input_batch.replay_query_start_loc))
        self.assertTrue(torch.equal(build_attn_metadata.call_args.kwargs["seq_lens"],
                                    input_batch.replay_seq_lens))


if __name__ == "__main__":
    unittest.main()
