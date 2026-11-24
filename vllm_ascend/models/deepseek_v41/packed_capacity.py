# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report the quota-limited capacity after workers agree on the native pool size."""

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs

from .packed_cache import LogicalCompressorSpec, LogicalHistorySpec, LogicalWindowSpec, uses_packed_history

_native_max_concurrency = kv_cache_utils.get_max_concurrency_for_kv_cache_config


def _max_concurrency(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> float:
    if not uses_packed_history(kv_cache_config):
        return _native_max_concurrency(vllm_config, kv_cache_config)

    # Scheduler projection unwraps the worker's per-layer spec collections.
    specs = [
        group.kv_cache_spec.first_spec
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        else group.kv_cache_spec
        for group in kv_cache_config.kv_cache_groups
    ]
    history = specs[0]
    compressor_limit = min(s.slot_limit for s in specs if isinstance(s, LogicalCompressorSpec))
    if not isinstance(history, LogicalHistorySpec):
        # Prefix hits can share pages. Report a conservative cold-request capacity
        # using the native window manager's in-flight chunk allowance.
        per_request = sum(cdiv(s.max_memory_usage_bytes(vllm_config), s.page_size_bytes) for s in specs)
        return float(min((kv_cache_config.num_blocks - 1) // per_request, compressor_limit))
    history_blocks = kv_cache_config.num_blocks - 1 - history.reserved_state_tiles
    length = vllm_config.model_config.max_model_len
    history_limit = history_blocks // cdiv(length, history.block_size)
    swa_limit = min(s.slot_limit for s in specs if isinstance(s, LogicalWindowSpec))
    logger.info(
        "Packed KV capacity at %d tokens/request: history %d requests (%d blocks), SWA %d, compressor %d.",
        length,
        history_limit,
        history_blocks,
        swa_limit,
        compressor_limit,
    )
    return float(min(history_limit, swa_limit, compressor_limit))


def install_packed_capacity_reporting() -> None:
    # update_kv_cache_capacity uses this lookup for both the startup log and
    # the values returned to the frontend. Allocation and admission use neither.
    kv_cache_utils.get_max_concurrency_for_kv_cache_config = _max_concurrency
