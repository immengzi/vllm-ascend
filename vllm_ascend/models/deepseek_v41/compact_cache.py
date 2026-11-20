# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pack Flash history and bounded state into equal-cost native block IDs."""

from collections.abc import Callable

import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec

CONFIG_KEY = "enable_deepseek_v41_compact_kv_pages"
# Each physical bank holds a dense BF16 window (or dense FP32 C2 ring) per ID.
BANK_BYTES = 128 * 1024
BANKS = 3
BLOCK_BYTES = BANKS * BANK_BYTES


class CompactPageGroup(KVCacheGroupSpec):
    """Native cache group whose block IDs use the compact DSV4.1 placement."""


def compact_groups_if_enabled(config: VllmConfig, specs: dict[str, KVCacheSpec]) -> list[KVCacheGroupSpec] | None:
    enabled = (getattr(config, "additional_config", None) or {}).get(CONFIG_KEY, False)
    if type(enabled) is not bool:
        raise ValueError(f"{CONFIG_KEY} must be a boolean")
    if not enabled:
        return None
    if (config.additional_config or {}).get("enable_kvpp") or config.cache_config.enable_extensible_kv_cache:
        raise ValueError("Compact KV pages require static placement with KVPP disabled")
    parallel = config.parallel_config
    if (
        config.model_config.architecture != "DeepseekV41ForCausalLM"
        or not config.use_v2_model_runner
        or config.scheduler_config.disable_hybrid_kv_cache_manager
        or (
            parallel.pipeline_parallel_size,
            parallel.prefill_context_parallel_size,
            parallel.decode_context_parallel_size,
        )
        != (1, 1, 1)
        or config.cache_config.enable_prefix_caching
        or config.cache_config.swa_bounded_replay
        or config.speculative_config is not None
    ):
        raise ValueError(
            "Compact KV pages require DSV4.1 Flash MRV2 hybrid cache with PP=PCP=DCP=1 "
            "without prefix cache, replay or draft"
        )

    history, windows, states = [], [], []
    for name, spec in sorted(specs.items()):
        if isinstance(spec, AscendSlidingWindowMLASpec) and (spec.block_size, spec.sliding_window, spec.dtype) == (
            128,
            128,
            torch.bfloat16,
        ):
            windows.append(name)
        elif isinstance(spec, CircularBufferSpec) and spec.dtype == torch.float32 and spec.block_size == 32:
            states.append(name)
        elif isinstance(spec, AscendMLAAttentionSpec) and spec.block_size == 128:
            if (name.endswith(".long_kv_cache") and spec.dtype == torch.bfloat16 and not spec.scale_dim) or (
                name.endswith(".indexer.k_cache")
                and spec.dtype == torch.int8
                and spec.scale_dim == 1
                and spec.scale_dtype == torch.float16
            ):
                history.append(name)
                continue
            raise ValueError(f"Unsupported compact history format for {name}")
        else:
            raise ValueError(f"Unsupported compact KV spec for {name}")
    if (len(history), len(windows), len(states)) != (8, 40, 3) or any(
        specs[n].unpadded_page_size_bytes != BANK_BYTES for n in windows + states
    ):
        raise ValueError("Compact KV pages require the unpacked Flash history, 40 windows and three C2 rings")
    # All ranks must derive the same group order, independently of dict insertion order.
    main = {name.removesuffix(".long_kv_cache"): specs[name] for name in history if name.endswith(".long_kv_cache")}
    index = {
        name.removesuffix(".indexer.k_cache"): specs[name] for name in history if name.endswith(".indexer.k_cache")
    }
    if (
        main.keys() != index.keys()
        or sorted(s.tokens_per_state for s in main.values()) != [1, 2, 2, 2]
        or any(s.num_kv_heads != 1 or s.head_size != 512 for s in main.values())
        or any(
            s.num_kv_heads != 1 or s.head_size != 128 or s.tokens_per_state != main[n].tokens_per_state
            for n, s in index.items()
        )
    ):
        raise ValueError("Compact history requires four matching Flash KV/index pairs")
    windows.sort(key=lambda name: int(name.rsplit(".layers.", 1)[1].split(".", 1)[0]))
    members = [history, states, *(windows[i : i + BANKS] for i in range(0, len(windows), BANKS))]
    groups = []
    for names in members:
        spec = UniformTypeKVCacheSpecs.from_specs({n: specs[n] for n in names})
        if spec is None:
            raise ValueError(f"Incompatible cache specs in compact group {names}")
        groups.append(CompactPageGroup(layer_names=names, kv_cache_spec=spec))
    return groups


def compact_cache_config(
    groups: list[KVCacheGroupSpec], num_blocks: int, retention_interval: int | None
) -> KVCacheConfig:
    descriptor = KVCacheTensor(
        size=num_blocks * BLOCK_BYTES,
        layers=[name for group in groups for name in group.layer_names],
        offset=0,
        block_stride=BANK_BYTES,
        layer_stride=0,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[descriptor],
        kv_cache_groups=groups,
        prefix_cache_retention_interval=retention_interval,
    )


def allocate_compact_views(config: KVCacheConfig, allocate: Callable[[int], torch.Tensor]) -> dict[str, torch.Tensor]:
    """Give each layer a view of its bank in the single aligned backing."""
    backing = allocate(config.kv_cache_tensors[0].size)
    bank_size = config.num_blocks * BANK_BYTES
    views = {}
    for group in config.kv_cache_groups:
        specs = group.kv_cache_spec.kv_cache_specs
        used = [0] * BANKS
        # Only layers in the same group can own the same block ID simultaneously.
        for name in sorted(group.layer_names, key=lambda n: (-specs[n].unpadded_page_size_bytes, n)):
            size = specs[name].unpadded_page_size_bytes
            bank = next((i for i, occupied in enumerate(used) if occupied + size <= BANK_BYTES), None)
            if bank is None:
                raise ValueError(f"Compact block cannot fit {name} ({size} bytes)")
            views[name] = backing[bank * bank_size + used[bank] :]
            used[bank] += size
    return views
