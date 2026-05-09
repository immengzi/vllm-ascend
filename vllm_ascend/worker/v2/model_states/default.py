# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_states/default.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch


class AscendModelState(DefaultModelState):
    """Model state for Ascend NPUs."""

    @staticmethod
    def _assert_laps_prefill_replay_metadata_sources(
        input_batch: AscendInputBatch,
        attn_metadata: dict[str, Any],
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
    ) -> None:
        replay_query_start_loc = input_batch.replay_query_start_loc
        replay_query_start_loc_np = input_batch.replay_query_start_loc_np
        replay_seq_lens = input_batch.replay_seq_lens
        replay_seq_lens_np = input_batch.replay_seq_lens_np
        replay_num_tokens = input_batch.replay_num_tokens
        assert replay_query_start_loc is not None
        assert replay_query_start_loc_np is not None
        assert replay_seq_lens is not None
        assert replay_seq_lens_np is not None
        assert replay_num_tokens is not None
        stable_block_table_by_ptr = {
            block_table.data_ptr(): block_table for block_table in block_tables
        }
        stable_slot_mapping_by_ptr = {
            slot_mapping.data_ptr(): slot_mapping for slot_mapping in slot_mappings
        }

        for layer_name, metadata in attn_metadata.items():
            source_block_table = stable_block_table_by_ptr.get(
                metadata.block_tables.data_ptr()
            )
            assert source_block_table is not None, (
                f"LAPS prefill replay attn_metadata[{layer_name}].block_tables "
                "must reuse the stable block_tables buffers prepared for replay."
            )
            assert metadata.block_tables.shape == source_block_table.shape, (
                f"LAPS prefill replay attn_metadata[{layer_name}].block_tables "
                "must preserve the stable replay block_table shape."
            )
            assert metadata.query_start_loc.data_ptr() == replay_query_start_loc.data_ptr(), (
                f"LAPS prefill replay attn_metadata[{layer_name}].query_start_loc "
                "must reuse replay_query_start_loc."
            )
            assert metadata.seq_lens.data_ptr() == replay_seq_lens.data_ptr(), (
                f"LAPS prefill replay attn_metadata[{layer_name}].seq_lens "
                "must reuse replay_seq_lens."
            )
            source_slot_mapping = stable_slot_mapping_by_ptr.get(
                metadata.slot_mapping.data_ptr()
            )
            assert source_slot_mapping is not None, (
                f"LAPS prefill replay attn_metadata[{layer_name}].slot_mapping "
                "must reuse the replay slot_mappings state."
            )
            assert metadata.slot_mapping.shape == source_slot_mapping.shape, (
                f"LAPS prefill replay attn_metadata[{layer_name}].slot_mapping "
                "must preserve the stable replay slot_mapping shape."
            )
            assert metadata.slot_mapping.shape[0] == replay_num_tokens, (
                f"LAPS prefill replay attn_metadata[{layer_name}].slot_mapping "
                "must match the padded target token shape."
            )
            assert metadata.actual_seq_lengths_q == replay_query_start_loc_np[1:].tolist(), (
                f"LAPS prefill replay attn_metadata[{layer_name}].actual_seq_lengths_q "
                "must be rebuilt from replay_query_start_loc_np."
            )
            assert metadata.seq_lens_list == replay_seq_lens_np.tolist(), (
                f"LAPS prefill replay attn_metadata[{layer_name}].seq_lens_list "
                "must be rebuilt from replay_seq_lens_np."
            )

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        """Override prepare_attn method because `build_attn_metadata` is different from vllm."""
        graph_pad_size = -1
        num_input_tokens = input_batch.num_tokens
        attn_state = input_batch.attn_state
        replay_num_reqs = input_batch.replay_num_reqs
        replay_num_tokens = input_batch.replay_num_tokens
        replay_query_start_loc = input_batch.replay_query_start_loc
        replay_query_start_loc_np = input_batch.replay_query_start_loc_np
        replay_seq_lens = input_batch.replay_seq_lens
        replay_seq_lens_np = input_batch.replay_seq_lens_np
        if cudagraph_mode == CUDAGraphMode.FULL:
            # Use padded sizes - padding is handled by model_runner.prepare_attn.
            num_reqs = replay_num_reqs or input_batch.num_reqs_after_padding
            num_tokens = replay_num_tokens or input_batch.num_tokens_after_padding
            num_input_tokens = replay_num_tokens or input_batch.num_tokens_after_padding
            if for_capture:
                graph_pad_size = input_batch.num_reqs_after_padding
        else:
            # For piecewise cudagraphs and eager, use unpadded sizes.
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc = (
            replay_query_start_loc
            if replay_query_start_loc is not None
            else input_batch.query_start_loc
        )
        query_start_loc_np = (
            replay_query_start_loc_np
            if replay_query_start_loc_np is not None
            else input_batch.query_start_loc_np
        )
        seq_lens = replay_seq_lens if replay_seq_lens is not None else input_batch.seq_lens
        seq_lens_np = replay_seq_lens_np if replay_seq_lens_np is not None else input_batch.seq_lens_np
        query_start_loc_cpu = torch.from_numpy(query_start_loc_np)
        max_query_len = (
            input_batch.replay_max_query_len
            if input_batch.replay_max_query_len is not None
            else input_batch.num_scheduled_tokens.max().item()
        )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            # extra attributes for ascend npus.
            seq_lens_np=seq_lens_np,
            attn_state=attn_state,
            graph_pad_size=graph_pad_size,
            num_input_tokens=num_input_tokens,
        )
        if input_batch.replay_num_reqs is not None:
            self._assert_laps_prefill_replay_metadata_sources(
                input_batch,
                attn_metadata,
                block_tables,
                slot_mappings,
            )
        input_batch.attn_metadata = attn_metadata
        return attn_metadata
