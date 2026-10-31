# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend KV-slot adapter for the upstream Engram hash state.

Upstream ``NgramHashState`` reads exactly two things from its
``swa_cache_module``: ``block_size`` once at construction, and ``kv_cache`` on
every ``ensure_cache()`` to size (and re-size) the slot-keyed history. The
Ascend SWA cache differs in ways the adapter absorbs:

* the storage block size comes from the Ascend block-size table
  (``DSV4_BLOCK_SIZES``) and is not the logical block size the scheduler uses;
* ``kv_cache`` is a one-element list and is absent until the KV allocator binds
  it, so ``numel()``/``shape[0]`` must not be called on the raw attribute.

The subclass retains upstream history allocation and lifecycle. Its hash
launch is split into request search and hashing for the Ascend compiler.
"""

import torch

# Upstream #56741 normalized the V4.1 model package name.
from vllm.models.deepseek_v41.common.engram import (
    DEAD_ID,
    EngramLayout,
    NgramHashState,
    _write_hash_cache_kernel,
)
from vllm.triton_utils import triton


class AscendEngramSlotCache:
    """``swa_cache_module`` view for upstream ``NgramHashState``."""

    def __init__(self, swa_cache_layer) -> None:
        self._layer = swa_cache_layer
        self.block_size = int(swa_cache_layer.block_size)

    @property
    def kv_cache(self) -> torch.Tensor:
        """The SWA KV cache as a single tensor, or an empty one when unbound."""
        cache = getattr(self._layer, "kv_cache", None)
        while isinstance(cache, (list, tuple)) and len(cache) == 1:
            cache = cache[0]
        if cache is None or isinstance(cache, (list, tuple)):
            # Unbound (profiling pass) or a multi-plane cache we do not index.
            return torch.empty(0, dtype=torch.int32)
        return cache


def engram_dead_mask(
    token_ids: torch.Tensor,
    image_token_id: int,
    image_pad_token_id: int,
) -> torch.Tensor:
    """Positions that must not take part in an n-gram.

    Covers both image sentinels, for the current chunk and for the lookback
    window alike. A ``-1`` lookback padding entry is not an image: callers pass
    the window through the same sentinel check and rely on the negative value
    staying a non-match.
    """
    return (token_ids == image_token_id) | (token_ids == image_pad_token_id)


def request_ring_slots(positions, query_start_loc, req_ids, ring_ids, block_size):
    """Commit only each request chunk's final ring turn; padding writes nothing."""
    req = req_ids.long()
    begin = query_start_loc.index_select(0, req).long()
    end = query_start_loc.index_select(0, req + 1).long()
    first = positions.index_select(0, begin.clamp(max=positions.numel() - 1))
    last = first + (end - begin) - 1
    token = torch.arange(positions.numel(), device=positions.device)
    blocks = ring_ids.index_select(0, req).long()
    keep = (token >= begin) & (token < end) & (positions >= last - block_size + 1) & (blocks > 0)
    slots = blocks * block_size + positions.remainder(block_size)
    return torch.where(keep, slots, -1)


def _commit_hash_cache(state, input_ids, dead_mask, slots):
    _write_hash_cache_kernel[(triton.cdiv(input_ids.numel(), 256),)](
        input_ids,
        state.token_map,
        dead_mask,
        slots,
        state._cache,
        input_ids.numel(),
        input_ids.stride(0),
        dead_mask.stride(0),
        slots.stride(0),
        256,
        DEAD_ID,
    )


def create_engram_hash_state(
    vllm_config, config, swa_cache_layer, *, request_state_ring: bool = False
) -> NgramHashState:
    """Bind the upstream hash state to the Ascend SWA cache."""
    layout = EngramLayout.from_config(config)
    assert layout is not None, "Engram hash state needs at least one Engram layer"
    state = AscendNgramHashState(vllm_config, layout, AscendEngramSlotCache(swa_cache_layer))
    state.request_state_virtual_blocks = None
    if request_state_ring and state.use_slot_cache:
        if state.lookback_depth > state.block_size:
            raise ValueError("Engram lookback exceeds the request-state ring capacity")
        state.request_state_virtual_blocks = (
            vllm_config.model_config.max_model_len + state.block_size - 1
        ) // state.block_size
    return state


class AscendNgramHashState(NgramHashState):
    """Reuse upstream history lifecycle, splitting only the NPU hash launch.

    Triton-Ascend cannot compile the combined request-search/hash kernel.
    Keep that compiler workaround in this subclass, without changing vLLM.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        dead_mask: torch.Tensor,
        lookback_token_ids: torch.Tensor,
        lookback_dead_mask: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute [tokens, layers, hash columns] int32 n-gram hashes.

        History comes from the current chunk, then the runner's lookback
        window, then the optional V1 slot cache.
        """
        cache = self._cache if self.use_slot_cache else None
        num_tokens = input_ids.shape[0]
        num_layers, max_ngram = self.multipliers.shape
        num_heads = self.primes.shape[-1]
        output = input_ids.new_empty((num_tokens, num_layers, (max_ngram - 1) * num_heads), dtype=torch.int32)
        if num_tokens == 0:
            return output
        # Ascend needs the request search out of the hash body (E3); one launch
        # still covers every layer.
        req_ids = input_ids.new_empty(num_tokens, dtype=torch.int32)
        from vllm_ascend.ops.triton.engram_hash import _engram_req_index_kernel

        _engram_req_index_kernel[(triton.cdiv(num_tokens, 32),)](
            query_start_loc,
            req_ids,
            num_tokens,
            query_start_loc.numel() - 1,
            query_start_loc.stride(0),
            BLOCK_T=32,
        )
        ring_blocks = getattr(self, "request_state_virtual_blocks", None)
        ring_ids = None
        if self.use_slot_cache:
            assert cache is not None and block_table is not None
            if ring_blocks is None:
                assert slot_mapping is not None
                _commit_hash_cache(self, input_ids, dead_mask, slot_mapping)
            else:
                if block_table.shape[1] != 1:
                    raise ValueError("Request-state Engram requires one state ID per request")
                ring_ids = block_table[:, 0]
                # Logical absolute pages alias one request ring. The existing
                # hash kernel retains absolute position modulo block_size.
                block_table = block_table.expand(-1, ring_blocks)
        from vllm_ascend.ops.triton.engram_hash import _hash_ids_kernel

        _hash_ids_kernel[(triton.cdiv(num_tokens, 32), num_layers)](
            input_ids,
            self.token_map,
            dead_mask,
            positions,
            block_table,
            query_start_loc,
            req_ids,
            self.multipliers,
            self.primes,
            self.offsets,
            cache,
            lookback_token_ids,
            lookback_dead_mask,
            output,
            num_tokens,
            cache.shape[0] if cache is not None else 0,
            self.pad_id,
            input_stride=input_ids.stride(0),
            mask_stride=dead_mask.stride(0),
            position_stride=positions.stride(0),
            table_stride=block_table.stride(0) if block_table is not None else 0,
            table_col_stride=block_table.stride(1) if block_table is not None else 0,
            query_stride=query_start_loc.stride(0),
            num_table_rows=block_table.shape[0] if block_table is not None else 0,
            max_blocks=block_table.shape[1] if block_table is not None else 0,
            cache_block_size=self.block_size,
            MAX_NGRAM=max_ngram,
            num_heads=num_heads,
            BLOCK_T=32,
            BLOCK_H=triton.next_power_of_2(num_heads),
            dead_id=DEAD_ID,
            lookback_depth=lookback_token_ids.shape[1],
            lookback_row_stride=lookback_token_ids.stride(0),
            lookback_col_stride=lookback_token_ids.stride(1),
            lookback_mask_row_stride=lookback_dead_mask.stride(0),
            lookback_mask_col_stride=lookback_dead_mask.stride(1),
            num_warps=4,
        )
        if ring_ids is not None:
            # Reads must precede ring reuse: a long prefill's final turn must
            # not overwrite the previous chunk's ngram tail before hashing.
            slots = request_ring_slots(positions, query_start_loc, req_ids, ring_ids, self.block_size)
            _commit_hash_cache(self, input_ids, dead_mask, slots)
        return output
