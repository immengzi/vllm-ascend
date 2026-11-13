# SPDX-License-Identifier: Apache-2.0
"""Chronological window execution; metadata and workspace ownership are explicit."""

from dataclasses import replace

import torch

from vllm_ascend.models.deepseek_v41.cache_config import WINDOW


def _release_current_kv(attn, current, *, prefill):
    """Return eager-prefill uses to the producing stream before KV reuse.

    The supported unpacked producer invokes defer_write on the stream that
    allocates normalized KV. Joining the consumer back to that producer lets
    its caching allocator reuse the block with stream ordering, instead of
    parking one projected KV per layer on record_stream events. Keep the
    original protection for decode/capture and unknown or copied producers.
    """
    value = attn._request_state_current_kv
    producer = getattr(attn, "_request_state_current_kv_stream", None)
    consumer = torch.npu.current_stream()
    if (
        prefill
        and producer is not None
        and producer != consumer
        and producer.device == consumer.device
        and current is value
        and getattr(attn, "packed_cache_ops", None) is None
        and not torch.npu.is_current_stream_capturing()
    ):
        # This is a GPU dependency, not a CPU/device synchronize. It is queued
        # before either reference is dropped; failures retain both references.
        producer.wait_stream(consumer)
    else:
        current.record_stream(consumer)
        if value is not current:
            value.record_stream(consumer)
    attn._request_state_current_kv = None
    attn._request_state_current_kv_stream = None


def state_swa_lengths(seq_lens, query_start):
    query = (query_start[1 : seq_lens.shape[0] + 1] - query_start[: seq_lens.shape[0]]).to(seq_lens.dtype)
    return torch.minimum(seq_lens, query + WINDOW - 1)


def global_swa(metadata):
    value = metadata.swa.global_metadata
    return value if value is not None else metadata.swa


def _stage(attn, metadata):
    from vllm_ascend.models.deepseek_v41.state_kernels import assemble

    swa = metadata.swa
    state = swa.request_state
    assert state is not None
    owner = global_swa(metadata)
    ring = attn.dsa_attn.swa_cache_layer.kv_cache[0]
    values = attn._request_state_current_kv
    assert values is not None and values.shape[0] == owner.num_actual_tokens
    current = values.contiguous()
    workspace = attn.dsa_attn.swa_cache_layer.request_state_workspace
    workspace.validate(ring, state.capacity)
    staging = workspace.tensor
    assert state.rows_used <= staging.shape[0] * WINDOW
    if owner is swa:
        assemble(
            ring,
            current,
            state.state_ids,
            swa.query_start_loc,
            swa.seq_lens,
            state.row_starts,
            staging,
            swa.max_query_len,
        )
    else:
        from vllm_ascend.models.deepseek_v41.state_kernels import assemble_cp

        assemble_cp(
            ring,
            current,
            state.state_ids,
            swa.query_start_loc,
            swa.seq_lens,
            owner.query_start_loc,
            owner.seq_lens,
            state.row_starts,
            staging,
            swa.max_query_len,
        )
    return ring, current, staging


def defer_window_write(attn, metadata, value):
    """Retain projected rows on their producer stream until stage and commit."""
    attn._request_state_current_kv = value
    attn._request_state_current_kv_stream = torch.npu.current_stream()


def forward_request_window(impl, attn, q, metadata, compressed_indices, **kwargs):
    from vllm_ascend.models.deepseek_v41.state_kernels import commit

    owner = global_swa(metadata)
    state = metadata.swa.request_state
    assert state is not None and owner.request_state is not None
    ring, current, staging = _stage(attn, metadata)
    layer = attn.dsa_attn.swa_cache_layer
    old_cache = layer.kv_cache
    replacement = replace(
        metadata,
        swa=replace(
            metadata.swa,
            block_table=state.block_table,
            seq_lens=state.lengths,
        ),
    )
    # These replay fields are optional on the pinned metadata type.
    replacement.swa.replay_block_table = None
    replacement.swa.replay_cache_blocks = None
    replacement.swa.replay_cache_offsets = None
    layer.kv_cache = [staging]
    try:
        output = impl._forward_attention_impl(attn, q, replacement, compressed_indices, **kwargs)
    finally:
        layer.kv_cache = old_cache
    commit(ring, current, owner.request_state.state_ids, owner.query_start_loc, owner.seq_lens)
    _release_current_kv(attn, current, prefill=owner.max_query_len > 1)
    return output


def commit_request_window(attn, metadata):
    """Cache-only CP ranks commit global KV even when they have no local query."""
    from vllm_ascend.models.deepseek_v41.state_kernels import commit

    state = metadata.swa.request_state
    assert state is not None
    value = attn._request_state_current_kv.contiguous()
    ring = attn.dsa_attn.swa_cache_layer.kv_cache[0]
    commit(ring, value, state.state_ids, metadata.swa.query_start_loc, metadata.swa.seq_lens)
    _release_current_kv(attn, value, prefill=metadata.swa.max_query_len > 1)
