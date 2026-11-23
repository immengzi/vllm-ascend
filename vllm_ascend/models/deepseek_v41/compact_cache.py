# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pack Flash history and bounded state into equal-cost native block IDs."""

from collections.abc import Callable
from dataclasses import dataclass, fields

import torch
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
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


WINDOW = 128
RING_CONFIG_KEY = "enable_deepseek_v41_fixed_swa_ring"


@dataclass(frozen=True, kw_only=True)
class SinglePoolWindowSpec(AscendSlidingWindowMLASpec):
    @property
    def uses_slot_mapping(self) -> bool:
        return False

    @property
    def block_table_token_alignment(self) -> int | None:
        return None

    @property
    def prefix_cacheable(self) -> bool:
        return False

    def max_num_blocks_per_req(self, vllm_config, max_len) -> int:
        return 1

    def max_admission_blocks_per_request(self, **kwargs) -> int:
        return 1

    def max_memory_usage_bytes(self, vllm_config) -> int:
        return self.page_size_bytes

    @property
    def is_circular(self) -> bool:
        return True

    @property
    def num_retained_tokens(self) -> int:
        # The MRV2 gather kernel nulls old position-indexed table entries.
        # Our only entry addresses the whole ring at every sequence position.
        return 0


def get_window_staging_rows(max_tokens: int, max_requests: int) -> int:
    return cdiv(max_tokens + max_requests * 2 * WINDOW, WINDOW) * WINDOW


@dataclass(frozen=True, kw_only=True)
class LogicalHistorySpec(AscendMLAAttentionSpec):
    reserved_state_tiles: int
    ids_per_history_block: int = BANKS

    def history_capacity(self, num_blocks: int) -> int:
        return (num_blocks - 1 - self.reserved_state_tiles) // self.ids_per_history_block


@dataclass(frozen=True, kw_only=True)
class LogicalWindowSpec(SinglePoolWindowSpec):
    slot_limit: int


@dataclass(frozen=True, kw_only=True)
class LogicalCompressorSpec(CircularBufferSpec):
    slot_limit: int


class CompactPageGroup(KVCacheGroupSpec):
    """Native cache group whose block IDs use the compact DSV4.1 placement."""


class ReservedRingSpecs(UniformTypeKVCacheSpecs):
    """Reserve all configured ring slots during Core's max-model-length fit."""

    def max_memory_usage_pages(self, vllm_config):
        return self.first_spec.slot_limit


class LogicalPoolGroup(CompactPageGroup):
    """One native ID addresses one tile in the shared slab."""


class PackedHistoryGroup(LogicalPoolGroup):
    """One ID owns a slice in each of the three history planes."""


PACKED_HISTORY_KEY = "enable_deepseek_v41_packed_history"
LOGICAL_CONFIG_KEY = "enable_deepseek_v41_logical_state_pools"
SWA_SLOTS_KEY = "deepseek_v41_swa_slots"
C2_SLOTS_KEY = "deepseek_v41_compressor_slots"


def _slot_limit(config, key):
    value = config.additional_config.get(key)
    if value is None:
        value = config.scheduler_config.max_num_seqs
    if type(value) is not int or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _as_spec(cls, spec, **quota):
    return cls(**{f.name: getattr(spec, f.name) for f in fields(spec) if f.init}, **quota)


def _history_banks(specs):
    banks = []
    sizes = []
    for name in sorted(specs, key=lambda n: (-specs[n].unpadded_page_size_bytes, n)):
        size = specs[name].unpadded_page_size_bytes
        bank = next((i for i, used in enumerate(sizes) if used + size <= BANK_BYTES), len(banks))
        if bank == len(banks):
            banks.append([])
            sizes.append(0)
        banks[bank].append(name)
        sizes[bank] += size
    if len(banks) != BANKS or max(sizes) > BANK_BYTES:
        raise ValueError("Flash history must fit in three native tiles per token block")
    return banks, sizes


def _logical_groups(config, specs, history, windows, states, packed):
    swa_slots = _slot_limit(config, SWA_SLOTS_KEY)
    compressor_slots = _slot_limit(config, C2_SLOTS_KEY)
    banks, sizes = _history_banks({n: specs[n] for n in history})
    state_layers_per_id = sizes.count(BANK_BYTES) if packed else 1
    if not state_layers_per_id:
        raise ValueError("Packed history needs at least one full-size plane for request state")
    reserved = (
        cdiv(len(windows), state_layers_per_id) * swa_slots + cdiv(len(states), state_layers_per_id) * compressor_slots
    )
    history_members = [history] if packed else banks
    history_groups = len(history_members)
    bins = [
        {
            n: _as_spec(
                LogicalHistorySpec, specs[n], reserved_state_tiles=reserved, ids_per_history_block=history_groups
            )
            for n in names
        }
        for names in history_members
    ]
    for names, cls, slots in (
        (states, LogicalCompressorSpec, compressor_slots),
        (windows, LogicalWindowSpec, swa_slots),
    ):
        for start in range(0, len(names), state_layers_per_id):
            bins.append(
                {n: _as_spec(cls, specs[n], slot_limit=slots) for n in names[start : start + state_layers_per_id]}
            )
    group_type = PackedHistoryGroup if packed else LogicalPoolGroup
    groups = []
    for i, layers in enumerate(bins):
        wrapper = UniformTypeKVCacheSpecs if i < history_groups else ReservedRingSpecs
        uniform = wrapper.from_specs(layers)
        if uniform is None:
            raise ValueError(f"Incompatible logical pool layers {list(layers)}")
        groups.append(group_type(layer_names=list(layers), kv_cache_spec=uniform))
    return groups


def compact_groups_if_enabled(config: VllmConfig, specs: dict[str, KVCacheSpec]) -> list[KVCacheGroupSpec] | None:
    enabled = (getattr(config, "additional_config", None) or {}).get(CONFIG_KEY, False)
    if type(enabled) is not bool:
        raise ValueError(f"{CONFIG_KEY} must be a boolean")
    ring = (getattr(config, "additional_config", None) or {}).get(RING_CONFIG_KEY, False)
    logical = (getattr(config, "additional_config", None) or {}).get(LOGICAL_CONFIG_KEY, False)
    packed = (getattr(config, "additional_config", None) or {}).get(PACKED_HISTORY_KEY, False)
    if type(packed) is not bool or (packed and not logical):
        raise ValueError(f"{PACKED_HISTORY_KEY} must be boolean and requires {LOGICAL_CONFIG_KEY} when enabled")
    if type(logical) is not bool:
        raise ValueError(f"{LOGICAL_CONFIG_KEY} must be a boolean")
    if type(ring) is not bool or (ring and not (enabled or logical)):
        raise ValueError(f"{RING_CONFIG_KEY} must be boolean and requires {CONFIG_KEY} when enabled")
    ring = ring or logical
    if not (enabled or logical):
        return None
    if (config.additional_config or {}).get("enable_kvpp") or getattr(
        config.cache_config, "enable_extensible_kv_cache", False
    ):
        raise ValueError("Compact KV pages require static placement with KVPP disabled")
    parallel = config.parallel_config
    if ring and getattr(parallel, "use_ubatching", False):
        raise ValueError("Fixed SWA rings share a staging workspace and require non-overlapping forwards")
    if (
        ring
        and config.kv_transfer_config is not None
        and config.kv_transfer_config.kv_connector != "MooncakePullConnector"
    ):
        raise ValueError("Fixed SWA rings support MooncakePullConnector for disaggregated serving")
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
    if logical:
        return _logical_groups(config, specs, history, windows, states, packed)
    members = [history, states, *(windows[i : i + BANKS] for i in range(0, len(windows), BANKS))]
    if ring:
        specs = dict(specs)
        for name in windows:
            original = specs[name]
            specs[name] = SinglePoolWindowSpec(
                **{field.name: getattr(original, field.name) for field in fields(original) if field.init}
            )
    groups = []
    for names in members:
        spec = UniformTypeKVCacheSpecs.from_specs({n: specs[n] for n in names})
        if spec is None:
            raise ValueError(f"Incompatible cache specs in compact group {names}")
        groups.append(CompactPageGroup(layer_names=names, kv_cache_spec=spec))
    return groups


def uses_fixed_swa_ring(config: KVCacheConfig | None) -> bool:
    if config is None or not config.kv_cache_groups or not isinstance(config.kv_cache_groups[0], CompactPageGroup):
        return False
    return any(isinstance(group.kv_cache_spec.first_spec, SinglePoolWindowSpec) for group in config.kv_cache_groups)


def _packed_descriptors(groups, count):
    history_specs = groups[0].kv_cache_spec.kv_cache_specs
    banks, widths = _history_banks(history_specs)
    size = sum(widths) * count
    descriptors = []
    plane_starts = []
    start = 0
    for names, width in zip(banks, widths, strict=True):
        plane_starts.append(start)
        offset = start
        for name in names:
            descriptors.append(
                KVCacheTensor(size=size, layers=[name], offset=offset, block_stride=width, layer_stride=0)
            )
            offset += history_specs[name].unpadded_page_size_bytes
        start += count * width
    state_starts = [offset for offset, width in zip(plane_starts, widths, strict=True) if width == BANK_BYTES]
    for group in groups[1:]:
        for name, offset in zip(group.layer_names, state_starts[: len(group.layer_names)], strict=True):
            descriptors.append(
                KVCacheTensor(size=size, layers=[name], offset=offset, block_stride=BANK_BYTES, layer_stride=0)
            )
    return descriptors


def compact_cache_config(
    groups: list[KVCacheGroupSpec], num_blocks: int, retention_interval: int | None
) -> KVCacheConfig:
    if isinstance(groups[0], PackedHistoryGroup):
        descriptors = _packed_descriptors(groups, num_blocks)
    else:
        descriptor = KVCacheTensor(
            size=num_blocks * (BANK_BYTES if isinstance(groups[0], LogicalPoolGroup) else BLOCK_BYTES),
            layers=[name for group in groups for name in group.layer_names],
            offset=0,
            block_stride=BANK_BYTES,
            layer_stride=0,
        )
        descriptors = [descriptor]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=descriptors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=retention_interval,
    )


def allocate_compact_views(config: KVCacheConfig, allocate: Callable[[int], torch.Tensor]) -> dict[str, torch.Tensor]:
    """Give each layer a view of its bank in the single aligned backing."""
    backing = allocate(config.kv_cache_tensors[0].size)
    if isinstance(config.kv_cache_groups[0], PackedHistoryGroup):
        return {d.layers[0]: backing[d.offset :] for d in config.kv_cache_tensors}
    bank_size = config.num_blocks * BANK_BYTES
    views = {}
    for group in config.kv_cache_groups:
        specs = group.kv_cache_spec.kv_cache_specs
        used = [0] * (1 if isinstance(group, LogicalPoolGroup) else BANKS)
        # Only layers in the same group can own the same block ID simultaneously.
        for name in sorted(group.layer_names, key=lambda n: (-specs[n].unpadded_page_size_bytes, n)):
            size = specs[name].unpadded_page_size_bytes
            bank = next((i for i, occupied in enumerate(used) if occupied + size <= BANK_BYTES), None)
            if bank is None:
                raise ValueError(f"Compact block cannot fit {name} ({size} bytes)")
            views[name] = backing[bank * bank_size + used[bank] :]
            used[bank] += size
    return views
