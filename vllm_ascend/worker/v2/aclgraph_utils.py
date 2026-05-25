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
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, get_forward_context, set_forward_context
from vllm.logger import logger
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
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
from vllm_ascend.worker.v2.block_table import AscendBlockTables
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers


@dataclass(frozen=True)
class PrefillGraphKey:
    num_reqs: int
    num_tokens: int
    max_query_len: int


@dataclass(frozen=True)
class LAPSPrefillGraphStats:
    candidates: int = 0
    hits: int = 0
    misses: int = 0
    unsupported_mode_misses: int = 0
    no_graph_key_misses: int = 0
    shape_overflow_misses: int = 0
    abi_guard_misses: int = 0
    fallback_to_none_misses: int = 0
    replay_tokens: int = 0
    eager_tokens: int = 0
    replay_us: int = 0
    eager_us: int = 0


@dataclass(frozen=True)
class LAPSPrefillCUDAGraphStat(CUDAGraphStat):
    laps_prefill_graph_stats: LAPSPrefillGraphStats | None = None


@dataclass
class LAPSPrefillGraphState:
    desc: BatchExecutionDescriptor
    input_buffers: AscendInputBuffers
    query_start_loc_np: np.ndarray
    logits_indices: torch.Tensor
    slot_mappings: torch.Tensor
    slot_mappings_by_layer: dict[str, torch.Tensor] | None = None
    attn_metadata_ptrs: dict[str, tuple[int, int, int, int]] | None = None
    last_replay_seq_lens_summary: dict[str, Any] | None = None


@dataclass(frozen=True)
class LAPSPrefillReplayPlan:
    desc: BatchExecutionDescriptor
    num_reqs: int
    num_tokens: int
    max_query_len: int
    target_num_reqs: int
    target_num_tokens: int
    right_align: bool = False


def assert_laps_prefill_replay_metadata_sources(
    input_batch: AscendInputBatch,
    attn_metadata: dict[str, Any],
    block_tables: tuple[torch.Tensor, ...],
    slot_mappings: torch.Tensor,
    *,
    graph_state: LAPSPrefillGraphState | None = None,
    on_error: Callable[[], None] | None = None,
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

    def fail(message: str) -> None:
        if on_error is not None:
            on_error()
        raise AssertionError(message)

    for layer_name, metadata in attn_metadata.items():
        current_ptrs = (
            metadata.block_tables.data_ptr(),
            metadata.query_start_loc.data_ptr(),
            metadata.seq_lens.data_ptr(),
            metadata.slot_mapping.data_ptr(),
        )
        if graph_state is not None and graph_state.attn_metadata_ptrs is not None:
            expected_ptrs = graph_state.attn_metadata_ptrs.get(layer_name)
            if expected_ptrs is not None and expected_ptrs != current_ptrs:
                fail(
                    f"LAPS prefill replay attn_metadata[{layer_name}] pointer signature "
                    "must match the capture-time stable ABI."
                )
        source_block_table = stable_block_table_by_ptr.get(
            metadata.block_tables.data_ptr()
        )
        if source_block_table is None:
            fail(
                f"LAPS prefill replay attn_metadata[{layer_name}].block_tables "
                "must reuse the stable block_tables buffers prepared for replay."
            )
        if metadata.block_tables.shape != source_block_table.shape:
            fail(
                f"LAPS prefill replay attn_metadata[{layer_name}].block_tables "
                "must preserve the stable replay block_table shape."
            )
        if metadata.query_start_loc.data_ptr() != replay_query_start_loc.data_ptr():
            fail(
                f"LAPS prefill replay attn_metadata[{layer_name}].query_start_loc "
                "must reuse replay_query_start_loc."
            )
        if metadata.seq_lens.data_ptr() != replay_seq_lens.data_ptr():
            fail(
                f"LAPS prefill replay attn_metadata[{layer_name}].seq_lens "
                "must reuse replay_seq_lens."
            )
        source_slot_mapping = stable_slot_mapping_by_ptr.get(
            metadata.slot_mapping.data_ptr()
        )
        if source_slot_mapping is None:
            fail(
                f"LAPS prefill replay attn_metadata[{layer_name}].slot_mapping "
                "must reuse the replay slot_mappings state."
            )
        if metadata.slot_mapping.shape != source_slot_mapping.shape:
            fail(
                f"LAPS prefill replay attn_metadata[{layer_name}].slot_mapping "
                "must preserve the stable replay slot_mapping shape."
            )
        if metadata.slot_mapping.shape[0] != replay_num_tokens:
            fail(
                f"LAPS prefill replay attn_metadata[{layer_name}].slot_mapping "
                "must match the padded target token shape."
            )
        if metadata.actual_seq_lengths_q != replay_query_start_loc_np[1:].tolist():
            fail(
                f"LAPS prefill replay attn_metadata[{layer_name}].actual_seq_lengths_q "
                "must be rebuilt from replay_query_start_loc_np."
            )
        if metadata.seq_lens_list != replay_seq_lens_np.tolist():
            fail(
                f"LAPS prefill replay attn_metadata[{layer_name}].seq_lens_list "
                "must be rebuilt from replay_seq_lens_np."
            )

        if graph_state is not None:
            if graph_state.attn_metadata_ptrs is None:
                graph_state.attn_metadata_ptrs = {}
            graph_state.attn_metadata_ptrs[layer_name] = current_ptrs


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
        self.laps_prefill_states: dict[BatchExecutionDescriptor, LAPSPrefillGraphState] = {}
        self._next_laps_prefill_request: tuple[int, int, int] | None = None
        self.laps_prefill_stats = LAPSPrefillGraphStats()
        self._laps_prefill_stats_last_log_at = time.monotonic()
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
            max_query_len = desc.num_tokens
            for num_reqs in self._laps_prefill_capture_batch_sizes(desc.num_tokens):
                key = PrefillGraphKey(
                    num_reqs=num_reqs,
                    num_tokens=desc.num_tokens,
                    max_query_len=max_query_len,
                )
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

    def _get_or_create_laps_prefill_state(
        self,
        desc: BatchExecutionDescriptor,
    ) -> LAPSPrefillGraphState:
        state = self.laps_prefill_states.get(desc)
        if state is not None:
            return state
        assert desc.num_reqs is not None
        state = LAPSPrefillGraphState(
            desc=desc,
            input_buffers=AscendInputBuffers(
                max_num_reqs=desc.num_reqs,
                max_num_tokens=desc.num_tokens,
                device=self.device,
            ),
            query_start_loc_np=np.zeros(desc.num_reqs + 1, dtype=np.int32),
            logits_indices=torch.zeros(desc.num_reqs, dtype=torch.int32, device=self.device),
            slot_mappings=torch.zeros(
                len(self.vllm_config.kv_cache_config.kv_cache_groups),
                desc.num_tokens,
                dtype=torch.int32,
                device=self.device,
            ),
            attn_metadata_ptrs=None,
        )
        self.laps_prefill_states[desc] = state
        return state

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

    def is_laps_prefill_desc(self, desc: BatchExecutionDescriptor) -> bool:
        return self._is_laps_prefill_desc(desc)

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
        key = PrefillGraphKey(
            num_reqs=desc.num_reqs,
            num_tokens=desc.num_tokens,
            max_query_len=desc.num_tokens,
        )
        return self.laps_prefill_descs.get(key) == desc

    def dispatch_laps_prefill(
        self,
        num_reqs: int,
        num_tokens: int,
        max_query_len: int,
    ) -> BatchExecutionDescriptor | None:
        self.laps_prefill_stats = replace(
            self.laps_prefill_stats,
            candidates=self.laps_prefill_stats.candidates + 1,
        )
        best_desc = None
        best_key = None
        has_available_graph = False
        for key, desc in self.laps_prefill_descs.items():
            if desc not in self.graphs:
                continue
            has_available_graph = True
            if key.num_reqs < num_reqs or key.num_tokens < num_tokens or key.max_query_len < max_query_len:
                continue
            if best_key is None or (key.num_tokens, key.num_reqs, key.max_query_len) < (
                best_key.num_tokens,
                best_key.num_reqs,
                best_key.max_query_len,
            ):
                best_key = key
                best_desc = desc
        if best_desc is not None:
            self.laps_prefill_stats = replace(
                self.laps_prefill_stats,
                hits=self.laps_prefill_stats.hits + 1,
            )
        else:
            if has_available_graph:
                self.record_laps_prefill_miss("shape_overflow")
            else:
                self.record_laps_prefill_miss("no_graph_key")
        return best_desc

    def _build_laps_prefill_replay_plan(
        self,
        desc: BatchExecutionDescriptor,
        input_batch: AscendInputBatch,
    ) -> LAPSPrefillReplayPlan:
        assert desc.num_reqs is not None
        # Ascend's prefill attention path already uses packed TND + sparse_mode=3
        # causal semantics, so we keep replay left-packed and only pad the tail.
        max_query_len = int(np.max(input_batch.num_scheduled_tokens))
        return LAPSPrefillReplayPlan(
            desc=desc,
            num_reqs=input_batch.num_reqs,
            num_tokens=input_batch.num_tokens,
            max_query_len=max_query_len,
            target_num_reqs=desc.num_reqs,
            target_num_tokens=desc.num_tokens,
            right_align=False,
        )

    def _update_laps_prefill_replay_inputs(
        self,
        state: LAPSPrefillGraphState,
        plan: LAPSPrefillReplayPlan,
        input_batch: AscendInputBatch,
    ) -> None:
        input_buffers = state.input_buffers

        input_buffers.input_ids[: plan.target_num_tokens].zero_()
        input_buffers.positions[: plan.target_num_tokens].zero_()
        input_buffers.input_ids[: plan.num_tokens].copy_(input_batch.input_ids[: plan.num_tokens])
        input_buffers.positions[: plan.num_tokens].copy_(input_batch.positions[: plan.num_tokens])

        query_start_loc_np = state.query_start_loc_np
        query_start_loc_np[0] = 0
        np.cumsum(
            input_batch.num_scheduled_tokens,
            out=query_start_loc_np[1 : plan.num_reqs + 1],
        )
        if plan.target_num_reqs > plan.num_reqs:
            query_start_loc_np[plan.num_reqs + 1 : plan.target_num_reqs] = plan.num_tokens
        query_start_loc_np[plan.target_num_reqs] = plan.target_num_tokens
        input_buffers.query_start_loc[: plan.target_num_reqs + 1].copy_(
            torch.from_numpy(query_start_loc_np[: plan.target_num_reqs + 1]).to(
                device=self.device
            )
        )

        replay_seq_lens_np = input_buffers.seq_lens_np
        replay_seq_lens_np[:plan.target_num_reqs] = np.diff(
            query_start_loc_np[: plan.target_num_reqs + 1]
        )
        input_buffers.seq_lens[: plan.target_num_reqs].copy_(
            torch.as_tensor(
                replay_seq_lens_np[:plan.target_num_reqs],
                dtype=torch.int32,
                device=self.device,
            )
        )
        input_buffers.seq_lens_cpu[: plan.target_num_reqs].copy_(
            torch.as_tensor(
                replay_seq_lens_np[:plan.target_num_reqs],
                dtype=torch.int32,
            )
        )

        state.logits_indices[: input_batch.logits_indices.shape[0]].copy_(
            input_batch.logits_indices
        )

    def prepare_laps_prefill_replay_input_batch(
        self,
        desc: BatchExecutionDescriptor,
        input_batch: AscendInputBatch,
    ) -> AscendInputBatch:
        state = self._get_or_create_laps_prefill_state(desc)
        plan = self._build_laps_prefill_replay_plan(desc, input_batch)
        self._update_laps_prefill_replay_inputs(state, plan, input_batch)

        input_batch.num_reqs_after_padding = plan.target_num_reqs
        input_batch.num_tokens_after_padding = plan.target_num_tokens
        input_batch.input_ids = state.input_buffers.input_ids[: plan.target_num_tokens]
        input_batch.positions = state.input_buffers.positions[: plan.target_num_tokens]
        input_batch.logits_indices = state.logits_indices[: input_batch.logits_indices.shape[0]]
        input_batch.replay_num_reqs = plan.target_num_reqs
        input_batch.replay_num_tokens = plan.target_num_tokens
        input_batch.replay_max_query_len = plan.max_query_len
        input_batch.replay_desc = desc
        input_batch.replay_query_start_loc = state.input_buffers.query_start_loc[
            : plan.target_num_reqs + 1
        ]
        input_batch.replay_query_start_loc_np = state.query_start_loc_np[
            : plan.target_num_reqs + 1
        ].copy()
        input_batch.replay_seq_lens = state.input_buffers.seq_lens[: plan.target_num_reqs]
        input_batch.replay_seq_lens_np = state.input_buffers.seq_lens_np[
            : plan.target_num_reqs
        ]
        input_batch.replay_seq_lens_summary = self._summarize_seq_lens(
            input_batch.replay_seq_lens_np,
            plan.target_num_reqs,
        )
        self._update_seq_lens_summary(state, input_batch.replay_seq_lens_summary)
        return input_batch

    @staticmethod
    def _summarize_seq_lens(seq_lens_np: np.ndarray, num_reqs: int) -> dict[str, Any]:
        active = seq_lens_np[:num_reqs]
        if active.size == 0:
            return {"min": 0, "max": 0, "nonzero": 0, "buckets": {}}
        return {
            "min": int(active.min()),
            "max": int(active.max()),
            "nonzero": int(np.count_nonzero(active)),
            "buckets": {
                "1": int(np.sum(active == 1)),
                "2_4": int(np.sum((active >= 2) & (active <= 4))),
                "5_16": int(np.sum((active >= 5) & (active <= 16))),
                "17_plus": int(np.sum(active >= 17)),
            },
        }

    def _update_seq_lens_summary(self, state: LAPSPrefillGraphState, summary: dict[str, Any]) -> None:
        if state.last_replay_seq_lens_summary is not None and state.last_replay_seq_lens_summary != summary:
            logger.debug(
                "LAPS prefill replay seq_lens summary changed for %s: previous=%s current=%s",
                state.desc,
                state.last_replay_seq_lens_summary,
                summary,
            )
        state.last_replay_seq_lens_summary = summary

    def _validate_laps_replay_abi(
        self,
        input_batch: AscendInputBatch,
        attn_metadata: dict[str, Any],
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
    ) -> None:
        state = self._get_or_create_laps_prefill_state(input_batch.replay_desc)
        assert_laps_prefill_replay_metadata_sources(
            input_batch,
            attn_metadata,
            block_tables,
            slot_mappings,
            graph_state=state,
            on_error=lambda: setattr(
                self,
                "laps_prefill_stats",
                replace(
                    self.laps_prefill_stats,
                    abi_guard_misses=self.laps_prefill_stats.abi_guard_misses + 1,
                ),
            )
        )

    def record_laps_prefill_execution(
        self,
        *,
        replay: bool,
        num_tokens: int,
        elapsed_us: int,
    ) -> None:
        if replay:
            self.laps_prefill_stats = replace(
                self.laps_prefill_stats,
                replay_tokens=self.laps_prefill_stats.replay_tokens + num_tokens,
                replay_us=self.laps_prefill_stats.replay_us + elapsed_us,
            )
        else:
            self.laps_prefill_stats = replace(
                self.laps_prefill_stats,
                eager_tokens=self.laps_prefill_stats.eager_tokens + num_tokens,
                eager_us=self.laps_prefill_stats.eager_us + elapsed_us,
            )

    def record_laps_prefill_miss(self, reason: str, *, count_miss: bool = True) -> None:
        updates: dict[str, int] = {}
        if count_miss:
            updates["misses"] = self.laps_prefill_stats.misses + 1
        if reason == "unsupported_mode":
            updates["unsupported_mode_misses"] = self.laps_prefill_stats.unsupported_mode_misses + 1
        elif reason == "no_graph_key":
            updates["no_graph_key_misses"] = self.laps_prefill_stats.no_graph_key_misses + 1
        elif reason == "shape_overflow":
            updates["shape_overflow_misses"] = self.laps_prefill_stats.shape_overflow_misses + 1
        elif reason == "abi_guard_failed":
            updates["abi_guard_misses"] = self.laps_prefill_stats.abi_guard_misses + 1
        elif reason == "fallback_to_none":
            updates["fallback_to_none_misses"] = self.laps_prefill_stats.fallback_to_none_misses + 1
        self.laps_prefill_stats = replace(self.laps_prefill_stats, **updates)

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
                if not self.supports_laps_prefill_graph():
                    self.record_laps_prefill_miss("unsupported_mode")
                    return BatchExecutionDescriptor(
                        cg_mode=CUDAGraphMode.NONE,
                        num_tokens=num_tokens,
                        num_reqs=num_reqs,
                    )
                desc = self.dispatch_laps_prefill(
                    num_reqs,
                    num_tokens,
                    hinted_max_query_len,
                )
                if desc is not None:
                    return desc
                self.record_laps_prefill_miss("fallback_to_none", count_miss=False)
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

        positions = self.model_runner.input_batch.positions[:num_tokens]
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

    def prepare_laps_prefill_replay_slot_mappings(
        self,
        desc: BatchExecutionDescriptor,
        block_tables: AscendBlockTables,
        input_batch: AscendInputBatch,
        kv_cache_config: KVCacheConfig,
    ) -> dict[str, torch.Tensor]:
        state = self._get_or_create_laps_prefill_state(desc)
        slot_mappings = block_tables.compute_slot_mappings(
            input_batch.idx_mapping,
            input_batch.replay_query_start_loc
            if input_batch.replay_query_start_loc is not None
            else input_batch.query_start_loc,
            input_batch.positions,
            num_tokens_padded=input_batch.replay_num_tokens
            if input_batch.replay_num_tokens is not None
            else input_batch.num_tokens_after_padding,
            out=state.slot_mappings,
        )
        del slot_mappings
        return self._get_or_create_laps_prefill_slot_mappings_by_layer(
            state,
            kv_cache_config,
        )

    def prepare_laps_prefill_capture_slot_mappings(
        self,
        desc: BatchExecutionDescriptor,
        kv_cache_config: KVCacheConfig,
    ) -> dict[str, torch.Tensor]:
        state = self._get_or_create_laps_prefill_state(desc)
        state.slot_mappings.fill_(PAD_SLOT_ID)
        return self._get_or_create_laps_prefill_slot_mappings_by_layer(
            state,
            kv_cache_config,
        )

    def _get_or_create_laps_prefill_slot_mappings_by_layer(
        self,
        state: LAPSPrefillGraphState,
        kv_cache_config: KVCacheConfig,
    ) -> dict[str, torch.Tensor]:
        if state.slot_mappings_by_layer is None:
            state.slot_mappings_by_layer = build_slot_mappings_by_layer(
                state.slot_mappings,
                kv_cache_config,
            )
        return state.slot_mappings_by_layer

    def get_laps_prefill_graph_stats(self) -> LAPSPrefillGraphStats:
        return self.laps_prefill_stats

    def maybe_log_laps_prefill_graph_stats(self, force: bool = False) -> None:
        interval_s = envs.VLLM_ASCEND_LAPS_PREFILL_GRAPH_STATS_LOG_INTERVAL_S
        if interval_s <= 0:
            return
        now = time.monotonic()
        if not force and (now - self._laps_prefill_stats_last_log_at) < interval_s:
            return
        self._laps_prefill_stats_last_log_at = now
        stats = self.laps_prefill_stats
        replay_tps = (stats.replay_tokens / stats.replay_us * 1e6) if stats.replay_us else 0.0
        eager_tps = (stats.eager_tokens / stats.eager_us * 1e6) if stats.eager_us else 0.0
        logger.info(
            "LAPS prefill graph stats: candidates=%d hits=%d misses=%d unsupported=%d "
            "no_graph_key=%d shape_overflow=%d abi_guard=%d fallback_none=%d "
            "replay_tokens=%d eager_tokens=%d replay_us=%d eager_us=%d "
            "replay_tps=%.3f eager_tps=%.3f",
            stats.candidates,
            stats.hits,
            stats.misses,
            stats.unsupported_mode_misses,
            stats.no_graph_key_misses,
            stats.shape_overflow_misses,
            stats.abi_guard_misses,
            stats.fallback_to_none_misses,
            stats.replay_tokens,
            stats.eager_tokens,
            stats.replay_us,
            stats.eager_us,
            replay_tps,
            eager_tps,
        )

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
            capture_input_buffers = input_buffers
            laps_prefill_state = None
            if use_laps_prefill_graph:
                laps_prefill_state = self._get_or_create_laps_prefill_state(desc)
                capture_input_buffers = laps_prefill_state.input_buffers
            input_batch, attn_metadata, slot_mappings = prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                model_state,
                capture_input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                use_laps_prefill_graph=use_laps_prefill_graph,
                laps_prefill_state=laps_prefill_state,
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
    laps_prefill_state: LAPSPrefillGraphState | None = None,
) -> tuple[InputBatch, dict[str, Any], dict[str, torch.Tensor]]:
    if use_laps_prefill_graph:
        assert isinstance(input_buffers, AscendInputBuffers)
        assert laps_prefill_state is not None
        input_batch = AscendInputBatch.make_prefill_dummy(
            num_tokens,
            num_reqs_after_padding=num_reqs,
            input_buffers=input_buffers,
        )
    else:
        input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)

    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    if use_laps_prefill_graph:
        slot_mappings = laps_prefill_state.slot_mappings[:, :num_tokens]
        laps_prefill_state.slot_mappings.fill_(PAD_SLOT_ID)
        if laps_prefill_state.slot_mappings_by_layer is None:
            laps_prefill_state.slot_mappings_by_layer = build_slot_mappings_by_layer(
                laps_prefill_state.slot_mappings,
                kv_cache_config,
            )
        slot_mappings_by_layer = laps_prefill_state.slot_mappings_by_layer
    else:
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
