# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/aclgraph_utils.py
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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, get_forward_context, set_forward_context
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, ModelCudaGraphManager
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend import envs
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import set_graph_params, update_full_graph_params
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers


@dataclass(frozen=True)
class PrefillGraphKey:
    num_reqs: int
    num_tokens: int


class ModelAclGraphManager(ModelCudaGraphManager):
    """ACL Model Cuda Graph Manager for Ascend NPUs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        model_runner: Any,
    ):
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
        )
        self.model_runner = model_runner
        self.capture_sizes = sorted(self.compilation_config.cudagraph_capture_sizes)
        self.laps_prefill_descs: dict[PrefillGraphKey, BatchExecutionDescriptor] = {}
        self._next_laps_prefill_request: tuple[int, int, int] | None = None
        if super().needs_capture():
            set_graph_params(self.capture_sizes)
        self._install_laps_prefill_capture_descs()

    def _laps_prefill_capture_batch_sizes(self, num_tokens: int) -> list[int]:
        max_num_reqs = min(self.max_num_reqs, max(1, num_tokens // 2))
        batch_sizes = {1, max_num_reqs}
        batch_size = 1
        while batch_size <= max_num_reqs:
            batch_sizes.add(batch_size)
            batch_size *= 2
        return sorted(batch_sizes)

    def _install_laps_prefill_capture_descs(self) -> None:
        if not self._use_laps_prefill_graph():
            return
        full_descs = self._capture_descs.get(CUDAGraphMode.FULL)
        if not full_descs:
            return

        existing_descs = set(full_descs)
        new_descs: list[BatchExecutionDescriptor] = []
        for desc in list(full_descs):
            for num_reqs in self._laps_prefill_capture_batch_sizes(desc.num_tokens):
                key = PrefillGraphKey(num_reqs=num_reqs, num_tokens=desc.num_tokens)
                laps_desc = BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.FULL,
                    num_tokens=desc.num_tokens,
                    num_reqs=num_reqs,
                )
                self.laps_prefill_descs[key] = laps_desc
                if laps_desc not in existing_descs:
                    existing_descs.add(laps_desc)
                    new_descs.append(laps_desc)

        full_descs.extend(new_descs)
        full_descs.sort(
            key=lambda d: (d.num_tokens, d.num_reqs or 0),
            reverse=True,
        )

    def _use_laps_prefill_graph(self) -> bool:
        if not envs.VLLM_ASCEND_LAPS_SCHEDULING:
            return False
        if self.compilation_config.cudagraph_mode != CUDAGraphMode.FULL:
            return False
        if self.model_runner.speculative_config is not None:
            return False
        if self.model_runner.model_config.is_encoder_decoder:
            return False
        if self.model_runner.use_dcp:
            return False
        return True

    def supports_laps_prefill_graph(self) -> bool:
        return self._use_laps_prefill_graph() and bool(self.laps_prefill_descs)

    def set_next_laps_prefill_request(
        self,
        num_reqs: int,
        num_tokens: int,
        max_query_len: int,
    ) -> None:
        self._next_laps_prefill_request = (num_reqs, num_tokens, max_query_len)

    def clear_next_laps_prefill_request(self) -> None:
        self._next_laps_prefill_request = None

    def _is_laps_prefill_desc(self, desc: BatchExecutionDescriptor) -> bool:
        if desc.num_reqs is None:
            return False
        key = PrefillGraphKey(num_reqs=desc.num_reqs, num_tokens=desc.num_tokens)
        return self.laps_prefill_descs.get(key) == desc

    def dispatch_laps_prefill(
        self,
        num_reqs: int,
        num_tokens: int,
        max_query_len: int,
    ) -> BatchExecutionDescriptor | None:
        del max_query_len
        key = PrefillGraphKey(num_reqs=num_reqs, num_tokens=num_tokens)
        desc = self.laps_prefill_descs.get(key)
        if desc is None or desc not in self.graphs:
            return None
        return desc

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
    ) -> BatchExecutionDescriptor:
        if self._next_laps_prefill_request is not None:
            hinted_num_reqs, hinted_num_tokens, hinted_max_query_len = (
                self._next_laps_prefill_request
            )
            if hinted_num_reqs == num_reqs and hinted_num_tokens == num_tokens:
                desc = self.dispatch_laps_prefill(
                    num_reqs,
                    num_tokens,
                    hinted_max_query_len,
                )
                if desc is not None:
                    return desc
                return BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.NONE,
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                )
        return super().dispatch(num_reqs, num_tokens, uniform_token_count)

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Override run_fullgraph to update full graph params in run_fullgraph."""
        num_tokens = desc.num_tokens
        logger.info_once(f"run_fullgraph with num_tokens={num_tokens}")
        ret = super().run_fullgraph(desc)

        positions = self.model_runner.input_buffers.positions[:num_tokens]
        num_tokens_across_dp = torch.full([self.model_runner.dp_size], num_tokens, device=self.device)
        with set_forward_context(
            self.model_runner.input_batch.attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=desc.cg_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            batch_descriptor=None,
            slot_mapping=self.model_runner.input_batch.slot_mappings,
        ):
            forward_context = get_forward_context()
            update_full_graph_params(
                list(self.model_runner.attn_backends.values())[0],
                self.model_runner.update_stream,
                forward_context,
                num_tokens,
                self.vllm_config,
                self.model_runner.speculative_config,
                positions.shape[0],
            )
        return ret

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs for model forward pass."""
        self.use_aux_hidden_state_outputs = use_aux_hidden_state_outputs
        model = ModelWithContext(model)

        def create_forward_fn(
            desc: BatchExecutionDescriptor,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )
            use_laps_prefill_graph = (
                self._use_laps_prefill_graph()
                and desc.cg_mode == CUDAGraphMode.FULL
                and self._is_laps_prefill_desc(desc)
            )
            input_batch, attn_metadata, slot_mappings = prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                use_laps_prefill_graph=use_laps_prefill_graph,
            )

            def forward_fn(cg_mode: CUDAGraphMode) -> None:
                batch_descriptor = (
                    BatchDescriptor(num_tokens=num_tokens)
                    if cg_mode == CUDAGraphMode.PIECEWISE
                    else None
                )
                self.model_runner.input_batch = input_batch
                self.model_runner.input_batch.attn_metadata = attn_metadata
                self.model_runner.input_batch.slot_mappings = slot_mappings
                with set_forward_context(
                    attn_metadata if cg_mode != CUDAGraphMode.PIECEWISE else None,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    cudagraph_runtime_mode=cg_mode,
                    num_tokens_across_dp=num_tokens_across_dp,
                    slot_mapping=slot_mappings,
                    batch_descriptor=batch_descriptor,
                ):
                    model_inputs = {
                        "input_ids": input_batch.input_ids,
                        "positions": input_batch.positions,
                        "intermediate_tensors": None,
                        **model_state.prepare_dummy_inputs(
                            input_batch.num_reqs_after_padding
                            if cg_mode == CUDAGraphMode.FULL
                            else input_batch.num_reqs,
                            num_tokens,
                        ),
                    }
                    model_output = model(**model_inputs)
                    if self.use_aux_hidden_state_outputs:
                        hidden_states, aux_hidden_states = model_output
                    else:
                        hidden_states = model_output
                        aux_hidden_states = []
                    if self.hidden_states is None:
                        self.hidden_states = torch.empty_like(hidden_states)
                    if self.use_aux_hidden_state_outputs and not self.aux_hidden_states:
                        self.aux_hidden_states = [
                            torch.empty_like(x) for x in aux_hidden_states
                        ]
                    self.hidden_states[:num_tokens] = hidden_states
                    for i, aux in enumerate(aux_hidden_states):
                        self.aux_hidden_states[i][:num_tokens] = aux

            return forward_fn

        super(ModelCudaGraphManager, self).capture(create_forward_fn, progress_bar_desc)


class ModelWithContext(nn.Module):
    """Define a wrapper model to inject forward context."""

    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model

    def forward(self, *args, **kwargs):
        if torch.npu.is_current_stream_capturing():
            _EXTRA_CTX.capturing = True
        return self.original_model(*args, **kwargs)


def prepare_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    model_state: ModelState,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    use_laps_prefill_graph: bool = False,
) -> tuple[InputBatch, dict[str, Any], dict[str, torch.Tensor]]:
    if use_laps_prefill_graph:
        assert isinstance(input_buffers, AscendInputBuffers)
        input_batch = AscendInputBatch.make_prefill_dummy(
            num_tokens,
            num_reqs_after_padding=num_reqs,
            input_buffers=input_buffers,
        )
    else:
        input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)

    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    if block_tables.cp_size > 1:
        prepare_dcp_local_seq_lens(
            input_buffers.dcp_local_seq_lens,
            input_batch.seq_lens,
            num_reqs,
            block_tables.cp_size,
            block_tables.cp_rank,
            block_tables.cp_interleave,
        )
        input_batch.dcp_local_seq_lens = input_buffers.dcp_local_seq_lens[:num_reqs]

    attn_metadata = model_state.prepare_attn(
        input_batch,
        CUDAGraphMode.FULL if use_laps_prefill_graph else CUDAGraphMode.NONE,
        input_block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=True,
    )
    return input_batch, attn_metadata, slot_mappings_by_layer
