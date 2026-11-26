# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend allocation notifications for circular state sharing physical pages."""

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager, FullAttentionManager
from vllm.v1.kv_cache_interface import KVCacheSpec


class AscendCircularBufferManager(CircularBufferManager):
    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        scheduler_block_size: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        needs_kv_cache_zeroing: bool = False,
        max_admission_blocks_per_request: int | None = None,
    ) -> None:
        super().__init__(
            kv_cache_spec,
            block_pool,
            enable_caching,
            kv_cache_group_id,
            scheduler_block_size,
            dcp_world_size,
            pcp_world_size,
            needs_kv_cache_zeroing,
            max_admission_blocks_per_request,
        )
        # Ascend zeroes every physical backing of a recycled block ID. Ring
        # allocations must report IDs too, even when the last owner was an
        # attention group. The upstream ring manager excludes these IDs.
        self._record_new_block_ids = needs_kv_cache_zeroing


class LogicalQuotaMixin:
    """Admission on top of native page ownership; all groups still share one BlockPool."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._detached = {}

    def pop_blocks_for_free(self, request_id):
        blocks = super().pop_blocks_for_free(request_id)
        self._detached.update((b.block_id, b) for b in blocks if not b.is_null)
        return blocks

    def get_num_blocks_to_allocate(self, *args, **kwargs):
        # All managers are queried before any group obtains new IDs. Retain
        # detached entries until the native deferred free drops its reference.
        if self._detached:
            self._detached = {i: b for i, b in self._detached.items() if b.ref_cnt}
        need = super().get_num_blocks_to_allocate(*args, **kwargs)
        if need:
            occupied = sum(map(len, self.req_to_blocks.values())) + len(self._detached)
            if occupied + need > self.logical_limit:
                return self.block_pool.num_gpu_blocks + 1
        return need


class LogicalHistoryManager(LogicalQuotaMixin, FullAttentionManager):
    @property
    def logical_limit(self):
        return self.block_pool.num_gpu_blocks - 1 - self.kv_cache_spec.reserved_state_tiles


class LogicalRingManager(LogicalQuotaMixin, AscendCircularBufferManager):
    @property
    def logical_limit(self):
        return self.kv_cache_spec.slot_limit
