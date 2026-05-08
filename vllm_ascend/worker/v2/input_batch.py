# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/input_batch.py
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
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


class AscendInputBuffers(InputBuffers):
    """Input buffers for Ascend NPUs."""

    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        device: torch.device,
    ):
        super().__init__(
            max_num_reqs,
            max_num_tokens,
            device,
        )
        del self.query_start_loc

        # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
        # See _pad_query_start_loc_for_fia.
        self.query_start_loc: torch.Tensor = torch.zeros(
            max_num_reqs + 2,
            dtype=torch.int32,
            device=device,
        )

        # Create seq_lens_cpu and seq_lens_np.
        # npu's attention backend still needs seq_lens on CPU side.
        self.seq_lens_cpu: torch.Tensor = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device="cpu",
        )
        # seq_len_np and seq_lens_cpu share the same memory.
        # define seq_lens_np for easier calculation with numpy.
        self.seq_lens_np: np.ndarray = self.seq_lens_cpu.numpy()


@dataclass
class AscendInputBatch(InputBatch):
    """Input batch for Ascend NPUs."""

    # Create seq_lens_np.
    # npu's attention backend still needs seq_lens on CPU side.
    seq_lens_np: np.ndarray
    # attn_state is used to build attention metadata.
    attn_state: AscendAttentionState | None = None
    # Optional replay-time overrides used by LAPS prefill full-graph replay.
    replay_num_reqs: int | None = None
    replay_num_tokens: int | None = None
    replay_max_query_len: int | None = None
    replay_query_start_loc: torch.Tensor | None = None
    replay_query_start_loc_np: np.ndarray | None = None
    replay_seq_lens: torch.Tensor | None = None
    replay_seq_lens_np: np.ndarray | None = None
    replay_desc: BatchExecutionDescriptor | None = None
    attn_metadata: dict[str, Any] | None = None
    slot_mappings: dict[str, torch.Tensor] | None = None

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        num_tokens: int,
        input_buffers: AscendInputBuffers,
    ) -> "AscendInputBatch":
        """Override the make_dummy method to calculate seq_lens_np."""
        input_batch = InputBatch.make_dummy(
            num_reqs,
            num_tokens,
            input_buffers,
        )
        # seq_len equals to query_len
        input_buffers.seq_lens_np[:num_reqs] = num_tokens // num_reqs
        input_buffers.seq_lens_np[num_reqs - 1] += num_tokens % num_reqs
        # Pad for full CUDA graph mode.
        input_buffers.seq_lens_np[num_reqs:] = 0
        seq_lens_np = input_buffers.seq_lens_np[:num_reqs]
        input_batch.seq_lens_np = seq_lens_np
        # A dummy run for dp or memory profiling.
        # When dummy run for dp, num_tokens is set to 1,
        # so attn_state is set to DecodeOnly.
        # when dummy run for memory profiling,
        # attention metadata isn't needed,
        # we can also set attn_state to AscendAttentionState.DecodeOnly.
        input_batch.attn_state = AscendAttentionState.DecodeOnly
        return cls(**asdict(input_batch), seq_lens_np=seq_lens_np)

    @classmethod
    def make_prefill_dummy(
        cls,
        num_tokens: int,
        num_reqs_after_padding: int,
        input_buffers: AscendInputBuffers,
    ) -> "AscendInputBatch":
        """Build a padded pure-prefill dummy batch for full-graph capture.

        The upstream dummy batch generator spreads tokens across all requests,
        which collapses small FULL capture shapes into decode-only metadata
        when `num_tokens == num_reqs`. For the LAPS prefill capture path we
        intentionally keep fewer active requests and pad the remaining request
        slots so that capture uses real prefill-shaped attention metadata.
        """
        assert 0 < num_tokens <= input_buffers.max_num_tokens
        assert 0 < num_reqs_after_padding <= input_buffers.max_num_reqs
        device = input_buffers.device

        if num_tokens == 1:
            num_reqs = 1
        else:
            num_reqs = min(num_reqs_after_padding, max(1, num_tokens // 2))

        req_ids = [f"prefill_req_{i}" for i in range(num_reqs)]
        idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
        expanded_idx_mapping = idx_mapping
        expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=device)

        num_scheduled_tokens = np.full(num_reqs, num_tokens // num_reqs, dtype=np.int32)
        num_scheduled_tokens[-1] += num_tokens % num_reqs
        assert int(num_scheduled_tokens.sum()) == num_tokens

        input_buffers.seq_lens[:num_reqs] = torch.from_numpy(num_scheduled_tokens).to(device=device)
        input_buffers.seq_lens[num_reqs:num_reqs_after_padding] = 0
        input_buffers.seq_lens_np[:num_reqs] = num_scheduled_tokens
        input_buffers.seq_lens_np[num_reqs:num_reqs_after_padding] = 0
        seq_lens = input_buffers.seq_lens[:num_reqs_after_padding]
        seq_lens_np = input_buffers.seq_lens_np[:num_reqs_after_padding]

        query_start_loc_np = np.empty(num_reqs_after_padding + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1 : num_reqs + 1])
        query_start_loc_np[num_reqs + 1 :] = num_tokens
        input_buffers.query_start_loc[: num_reqs_after_padding + 1] = torch.from_numpy(query_start_loc_np).to(
            device=device
        )
        query_start_loc = input_buffers.query_start_loc[: num_reqs_after_padding + 1]

        input_ids = input_buffers.input_ids[:num_tokens].zero_()
        positions = input_buffers.positions[:num_tokens].zero_()

        logits_indices = query_start_loc[1 : num_reqs + 1] - 1
        cu_num_logits = torch.arange(num_reqs + 1, device=device, dtype=torch.int32)
        cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)

        return cls(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs_after_padding,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            num_draft_tokens=0,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            dcp_local_seq_lens=None,
            input_ids=input_ids,
            positions=positions,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=False,
            seq_lens_np=seq_lens_np,
            attn_state=AscendAttentionState.PrefillNoCache,
        )


@triton.jit
def _post_update_kernel(
    idx_mapping_ptr,
    idx_mapping_stride,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    sampled_tokens_ptr,
    sampled_tokens_stride,
    num_rows,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    rows_per_program = (num_rows + n_programs - 1) // n_programs
    start_row = pid * rows_per_program
    end_row = tl.minimum(start_row + rows_per_program, num_rows)

    for row_idx in range(start_row, end_row):
        req_state_idx = tl.load(idx_mapping_ptr + row_idx * idx_mapping_stride)
        total_len = tl.load(total_len_ptr + req_state_idx)
        num_sampled = tl.load(num_sampled_ptr + row_idx)

        if num_sampled > 0:
            token_id = tl.load(sampled_tokens_ptr + row_idx * sampled_tokens_stride + num_sampled - 1)
            tl.store(last_sampled_tokens_ptr + req_state_idx, token_id)
            tl.store(total_len_ptr + req_state_idx, total_len + num_sampled)

        for i in range(num_sampled):
            token_id = tl.load(sampled_tokens_ptr + row_idx * sampled_tokens_stride + i)

            token_ptr = output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + token_id
            count = tl.load(token_ptr)
            count += 1
            tl.store(token_ptr, count)

            tl.store(
                all_token_ids_ptr + req_state_idx * all_token_ids_stride + total_len + i,
                token_id,
            )

        query_start = tl.load(query_start_loc_ptr + row_idx)
        query_end = tl.load(query_start_loc_ptr + row_idx + 1)
        query_len = query_end - query_start
        num_rejected = tl.load(num_rejected_ptr + row_idx)

        num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
        num_computed += query_len - num_rejected
        tl.store(num_computed_tokens_ptr + req_state_idx, num_computed)


def post_update(
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [max_num_reqs]
    last_sampled_tokens: torch.Tensor,
    # [max_num_reqs, vocab_size]
    output_bin_counts: torch.Tensor,
    # [num_reqs, num_speculative_steps + 1]
    sampled_tokens: torch.Tensor,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor,
    # [max_num_reqs, max_model_len]
    all_token_ids: torch.Tensor,
    # [max_num_reqs]
    total_len: torch.Tensor,
) -> None:
    num_rows = idx_mapping.shape[0]

    core_num = get_vectorcore_num()

    grid = (min(num_rows, core_num),)
    _post_update_kernel[grid](
        idx_mapping,
        idx_mapping.stride(0),
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        output_bin_counts.stride(0),
        sampled_tokens,
        sampled_tokens.stride(0),
        num_rows,
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
    )
