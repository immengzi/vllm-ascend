# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 history/request-state separation on the pinned runtime.

No MTP or prefix reuse. DSA CP stages local query windows from replicated
global KV; graph decode uses persistent lengths. Pool ownership and admission
live in core.request_state_manager; this module handles window execution.
"""

import copy
from dataclasses import replace

import torch

from vllm_ascend.models.deepseek_v41.cache_config import RequestWindowSpec

WINDOW = 128
_attention_installed = False
_workspace = {}


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


def _cpu_list(value, count):
    assert value.device.type == "cpu", "Do not synchronize NPU metadata into a hot CPU path"
    return value[:count].tolist()


def hydrate(metadata):
    state = getattr(metadata, "request_state", None)
    if state is not None and not hasattr(metadata, "state_ids"):
        for key, value in state.items():
            setattr(metadata, key, value)
    return metadata


def export_state(builder, result):
    # A declared dataclass field preserves this contract through CP replace().
    result.request_state = {key: value for key, value in vars(result).items() if key.startswith("state_")}
    builder._request_state_last_result = result
    return result


def global_swa(metadata):
    value = getattr(metadata.swa, "global_metadata", None)
    return hydrate(value if value is not None else metadata.swa)


def _prepare_state_metadata(builder, common, result):
    if builder._cache_kind != "swa" or not isinstance(builder.kv_cache_spec, RequestWindowSpec):
        return result
    n = result.num_reqs
    global_owner = getattr(getattr(builder, "_global_builder", None), "_request_state_last_result", None)
    global_query_max = global_owner.max_query_len if global_owner is not None else result.max_query_len
    if result.max_query_len <= 1 and global_query_max == 1:
        # The graph must retain addresses whose contents are updated per step.
        # The one-column state block table already belongs to the runner.
        assert common.block_table_tensor.shape[1] == 1
        if not hasattr(builder, "_state_decode_lengths"):
            builder._state_decode_lengths = torch.empty_like(builder._seq_lens)
        lengths = builder._state_decode_lengths[:n]
        torch.clamp(result.seq_lens, max=WINDOW, out=lengths)
        result.state_ids = common.block_table_tensor[:n, 0]
        result.state_stage_lengths = lengths
        # Decode staging owns one fixed page per padded request row.
        # Graph replay must keep lengths, row offsets and table addresses.
        if not hasattr(builder, "_state_decode_rows"):
            rows = torch.arange(builder._seq_lens.numel(), dtype=torch.int32, device=builder._seq_lens.device)
            builder._state_decode_rows = rows * WINDOW
            builder._state_decode_table = rows.view(-1, 1)
        result.state_stage_row_starts = builder._state_decode_rows[:n]
        result.state_stage_block_table = builder._state_decode_table[:n]
        result.state_stage_rows_used = n * WINDOW
        result.state_stage_max_seq_len = WINDOW
        result.state_staging_capacity = (
            (
                builder.vllm_config.scheduler_config.max_num_batched_tokens
                + builder.vllm_config.scheduler_config.max_num_seqs * 256
                + 127
            )
            // 128
        ) * 128
        assert result.state_stage_rows_used <= result.state_staging_capacity
        return export_state(builder, result)
    qcpu = common.query_start_loc_cpu
    seqcpu = getattr(common, "seq_lens_cpu", None)
    if seqcpu is None:
        seqcpu = common._seq_lens_cpu
    q = _cpu_list(qcpu, n + 1)
    seq = _cpu_list(seqcpu, n)
    bases, lengths, cursor = [], [], 0
    for i in range(n):
        query = q[i + 1] - q[i]
        old = min(max(0, seq[i] - query), WINDOW - 1)
        length = old + query if query else 0
        bases.append(cursor)
        lengths.append(length)
        cursor += ((length + WINDOW - 1) // WINDOW) * WINDOW
    device = result.seq_lens.device
    max_pages = max(1, max(((n + 127) // 128 for n in lengths), default=0))
    table = [
        [base // 128 + j if j < (length + 127) // 128 else 0 for j in range(max_pages)]
        for base, length in zip(bases, lengths)
    ]
    result.state_ids = common.block_table_tensor[:n, 0].contiguous()
    result.state_stage_row_starts = torch.tensor(bases, dtype=torch.int32, device=device)
    result.state_stage_lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
    result.state_stage_block_table = torch.tensor(table, dtype=torch.int32, device=device)
    result.state_stage_rows_used = cursor
    result.state_stage_max_seq_len = max(lengths, default=0)
    result.state_staging_capacity = (
        (
            builder.vllm_config.scheduler_config.max_num_batched_tokens
            + builder.vllm_config.scheduler_config.max_num_seqs * 256
            + 127
        )
        // 128
    ) * 128
    return export_state(builder, result)


def _stage(attn, metadata):
    from vllm_ascend.models.deepseek_v41.state_kernels import assemble

    swa = hydrate(metadata.swa)
    owner = global_swa(metadata)
    ring = attn.dsa_attn.swa_cache_layer.kv_cache[0]
    values = attn._request_state_current_kv
    assert values is not None and values.shape[0] == owner.num_actual_tokens
    current = values.contiguous()
    key = str(ring.device)
    if key not in _workspace:
        _workspace[key] = torch.empty(
            (swa.state_staging_capacity // 128, 128, 1, 512), dtype=ring.dtype, device=ring.device
        )
    staging = _workspace[key]
    assert swa.state_stage_rows_used <= staging.shape[0] * 128
    if owner is swa:
        assemble(
            ring,
            current,
            swa.state_ids,
            swa.query_start_loc,
            swa.seq_lens,
            swa.state_stage_row_starts,
            staging,
            swa.max_query_len,
        )
    else:
        from vllm_ascend.models.deepseek_v41.state_kernels import assemble_cp

        assemble_cp(
            ring,
            current,
            swa.state_ids,
            swa.query_start_loc,
            swa.seq_lens,
            owner.query_start_loc,
            owner.seq_lens,
            swa.state_stage_row_starts,
            staging,
            swa.max_query_len,
        )
    return ring, current, staging


def install_attention():
    global _attention_installed
    if _attention_installed:
        return
    _attention_installed = True
    from vllm_ascend.attention import dsa_v41 as mod

    original_build = mod.AscendDSAV41MetadataBuilder.build

    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        if isinstance(self.kv_cache_spec, RequestWindowSpec):
            common_attn_metadata = copy.copy(common_attn_metadata)
            size = int(getattr(common_attn_metadata, "num_input_tokens", common_attn_metadata.slot_mapping.shape[0]))
            common_attn_metadata.slot_mapping = self._flat_slot_mapping[:size]
            common_attn_metadata.slot_mapping.fill_(-1)
        value = original_build(self, common_prefix_len, common_attn_metadata, *args, **kwargs)
        return _prepare_state_metadata(self, common_attn_metadata, value)

    mod.AscendDSAV41MetadataBuilder.build = build
    original_write = mod.AscendDSAV41Impl._write_swa_cache

    def defer_write(attn, metadata, value):
        hydrate(metadata)
        if hasattr(metadata, "state_ids"):
            # Existing projection code joins its auxiliary stream before the
            # attention call. Hold the projected rows until stage and commit.
            attn._request_state_current_kv = value
            # In the supported unpacked path this callback executes on the
            # stream that creates the normalized KV tensor.
            attn._request_state_current_kv_stream = torch.npu.current_stream()
        else:
            original_write(attn, metadata, value)

    mod.AscendDSAV41Impl._write_swa_cache = staticmethod(defer_write)
    original_forward = mod.AscendDSAV41Impl._forward_attention

    def forward(self, attn, q, metadata, compressed_indices, **kwargs):
        hydrate(metadata.swa)
        if not hasattr(metadata.swa, "state_ids"):
            return original_forward(self, attn, q, metadata, compressed_indices, **kwargs)
        from vllm_ascend.models.deepseek_v41.state_kernels import commit

        owner = global_swa(metadata)
        ring, current, staging = _stage(attn, metadata)
        layer = attn.dsa_attn.swa_cache_layer
        old_cache = layer.kv_cache
        replacement = replace(metadata, swa=copy.copy(metadata.swa))
        replacement.swa.block_table = metadata.swa.state_stage_block_table
        replacement.swa.seq_lens = metadata.swa.state_stage_lengths
        replacement.swa.replay_block_table = None
        replacement.swa.replay_cache_blocks = None
        replacement.swa.replay_cache_offsets = None
        layer.kv_cache = [staging]
        try:
            output = original_forward(self, attn, q, replacement, compressed_indices, **kwargs)
        finally:
            layer.kv_cache = old_cache
        commit(ring, current, owner.state_ids, owner.query_start_loc, owner.seq_lens)
        _release_current_kv(attn, current, prefill=owner.max_query_len > 1)
        return output

    mod.AscendDSAV41Impl._forward_attention = forward
    original_update = mod.AscendDSAV41Impl._update_caches

    def update_only(self, attn, hidden_states, metadata):
        original_update(self, attn, hidden_states, metadata)
        if hidden_states.shape[0] and hasattr(metadata.swa, "state_ids"):
            from vllm_ascend.models.deepseek_v41.state_kernels import commit

            value = attn._request_state_current_kv.contiguous()
            ring = attn.dsa_attn.swa_cache_layer.kv_cache[0]
            commit(ring, value, metadata.swa.state_ids, metadata.swa.query_start_loc, metadata.swa.seq_lens)
            _release_current_kv(attn, value, prefill=metadata.swa.max_query_len > 1)

    mod.AscendDSAV41Impl._update_caches = update_only
