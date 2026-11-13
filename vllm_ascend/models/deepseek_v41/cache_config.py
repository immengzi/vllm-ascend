# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 cache geometry and history/request-state allocation."""

from dataclasses import dataclass, fields, replace

import torch
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import (
    AscendKVCacheConfig,
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    CachePoolPlan,
    RequestStateMemoryBudget,
    RequestStatePlan,
)
from vllm_ascend.device.hardware_profile import (
    HardwareCapability,
    get_current_hardware_profile,
)

WINDOW = 128
ROW_BYTES = 1024
STATE_RING_ROWS = 32

# Fixed dimensions accepted by the packaged mixed-quant attention ABI.
# The actual head sizes come from the model config; these bounds reject
# unsupported layouts before allocating caches, rather than override config.
MLA_HEAD_DIM = 512
INDEX_HEAD_DIM = 128
INDEX_FOLD_ROWS = 8


def uses_a5_packed_cache() -> bool:
    """Select the V4.1 cache ABI from the detected hardware profile."""
    return get_current_hardware_profile().supports(HardwareCapability.DSV41_PACKED_CACHE)


def make_swa_cache_spec(*, block_size, window_size, head_size, dtype, cache_dtype):
    if uses_a5_packed_cache():
        if head_size != MLA_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {MLA_HEAD_DIM}, got {head_size}")
        # FP8 values followed by one BF16 scale per 32 values.
        head_size += (head_size // 32) * torch.bfloat16.itemsize
        dtype = torch.uint8
        cache_dtype = "a5_mxfp8_bf16_scale"
    return AscendSlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        sliding_window=window_size,
        cache_dtype_str=cache_dtype,
        model_version="deepseek_v41",
        alignment=None,
    )


def make_mla_cache_spec(*, block_size, head_size, compress_ratio):
    if uses_a5_packed_cache():
        if head_size != MLA_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {MLA_HEAD_DIM}, got {head_size}")
        # Two FP4 values per byte, followed by one BF16 scale per 16 values.
        head_size = head_size // 2 + (head_size // 16) * torch.bfloat16.itemsize
        dtype = torch.uint8
        scale_dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16
        scale_dtype = torch.int8
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        tokens_per_state=compress_ratio,
        model_version="deepseek_v41",
        storage_block_size=block_size // compress_ratio,
        scale_dtype=scale_dtype,
    )


def make_index_cache_spec(*, block_size, head_size, compress_ratio):
    if uses_a5_packed_cache():
        if head_size != INDEX_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {INDEX_HEAD_DIM}, got {head_size}")
        scale_dim = head_size // 32
        head_size //= 2
        dtype = torch.uint8
        scale_dtype = torch.uint8
    else:
        dtype = torch.int8
        scale_dim = 1
        scale_dtype = torch.float16
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        tokens_per_state=compress_ratio,
        model_version="deepseek_v41",
        storage_block_size=block_size // compress_ratio,
        scale_dim=scale_dim,
        scale_dtype=scale_dtype,
    )


def make_folded_index_cache_spec(*, block_size):
    """QSLI candidate-source view: eight K/scale rows per 544-byte row."""
    if block_size % INDEX_FOLD_ROWS:
        raise ValueError("A5 folded index page must contain full 8-token groups")
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=INDEX_FOLD_ROWS * (INDEX_HEAD_DIM // 2 + INDEX_HEAD_DIM // 32),
        dtype=torch.uint8,
        tokens_per_state=INDEX_FOLD_ROWS,
        model_version="deepseek_v41",
        storage_block_size=block_size // INDEX_FOLD_ROWS,
    )


def is_deepseek_v41_cache(specs_or_groups):
    if isinstance(specs_or_groups, dict):
        specs = list(specs_or_groups.values())
    else:
        specs = []
        for item in specs_or_groups:
            spec = getattr(item, "kv_cache_spec", item)
            if isinstance(spec, UniformTypeKVCacheSpecs):
                specs.extend(spec.kv_cache_specs.values())
            else:
                specs.append(spec)
    return any(getattr(spec, "model_version", None) == "deepseek_v41" for spec in specs)


def _layer_number(name):
    return int(name.rsplit(".layers.", 1)[1].split(".", 1)[0])


def _draft_layer_number(name):
    return int(("." + name).rsplit(".mtp.", 1)[1].split(".", 1)[0])


def get_layer_tuples(specs):
    """Return DSV4-style ordered layer tuples and their physical page sizes."""
    mla = {name for name, spec in specs.items() if isinstance(spec, AscendMLAAttentionSpec)}
    state = sorted((name for name, spec in specs.items() if isinstance(spec, CircularBufferSpec)), key=_layer_number)
    swa = {name for name, spec in specs.items() if isinstance(spec, AscendSlidingWindowMLASpec)}

    full = sorted((name for name in mla if name.endswith(".long_kv_cache")), key=_layer_number)
    target_swa = sorted((name for name in swa if ".mtp." not in f".{name}"), key=_layer_number)
    draft_swa = sorted((name for name in swa if ".mtp." in f".{name}"), key=_draft_layer_number)

    layer_tuples: list[tuple[str, ...]] = []
    page_sizes: list[int] = []
    for slot_idx, kv_name in enumerate(full):
        prefix = kv_name.rsplit(".", 1)[0]
        index_name = prefix + ".indexer.k_cache"
        folded_name = index_name + "_folded"
        folded_spec = specs.get(folded_name)
        index_spec = specs[index_name]
        kv_spec = specs[kv_name]
        aliases = ([state[slot_idx]] if slot_idx < len(state) else []) + target_swa[slot_idx :: len(full)]
        kv_bytes = kv_spec.unpadded_page_size_bytes
        index_bytes = index_spec.unpadded_page_size_bytes
        folded_bytes = folded_spec.unpadded_page_size_bytes if folded_spec is not None else 0
        if slot_idx < len(draft_swa):
            aliases.append(draft_swa[slot_idx])
        capacity = max(
            kv_bytes + index_bytes + folded_bytes,
            *(specs[name].unpadded_page_size_bytes for name in aliases),
        )
        layer_tuples.append((kv_name, index_name, *((folded_name,) if folded_spec is not None else ()), *aliases))
        page_sizes.append(capacity)
    return page_sizes, layer_tuples


def make_cache_groups(grouped_specs):
    return [KVCacheGroupSpec(layer_names=list(s.kv_cache_specs), kv_cache_spec=s) for s in grouped_specs]


def _specs_from_groups(groups):
    specs = {}
    for group in groups:
        for name in group.layer_names:
            specs[name] = group.kv_cache_spec.kv_cache_specs[name]
    return specs


@dataclass(frozen=True, kw_only=True)
class RequestWindowSpec(AscendSlidingWindowMLASpec):
    @property
    def uses_slot_mapping(self):
        return False

    @property
    def block_table_token_alignment(self):
        return None

    def max_num_blocks_per_req(self, vllm_config, max_len):
        return 1

    def max_admission_blocks_per_request(self, **kwargs):
        return 1

    def max_memory_usage_bytes(self, vllm_config):
        return self.page_size_bytes

    @property
    def prefix_cacheable(self):
        return False


def unwrap(spec):
    return spec.kv_cache_specs if isinstance(spec, UniformTypeKVCacheSpecs) else {"representative": spec}


def classify(spec):
    values = list(unwrap(spec).values())
    if all(isinstance(x, RequestWindowSpec) for x in values):
        return "swa"
    if all(isinstance(x, CircularBufferSpec) for x in values):
        return "compressor"
    return "history"


def group_cache_specs(specs):
    groups = [dict(), dict(), dict()]
    for name, spec in specs.items():
        spec = replace(spec, page_size_padded=None)
        if isinstance(spec, AscendSlidingWindowMLASpec):
            assert spec.block_size == WINDOW and spec.sliding_window == WINDOW and spec.dtype == torch.bfloat16
            attrs = {f.name: getattr(spec, f.name) for f in fields(spec)}
            spec = RequestWindowSpec(**attrs)
            groups[1][name] = spec
        elif isinstance(spec, CircularBufferSpec):
            groups[2][name] = spec
        else:
            assert name.endswith((".long_kv_cache", ".indexer.k_cache")), name
            groups[0][name] = spec
    assert [len(g) for g in groups] == [8, 40, 3]
    result = [UniformTypeKVCacheSpecs.from_specs(g) for g in groups]
    assert all(g is not None for g in result)
    return result


def all_specs(groups):
    return {n: s for g in groups for n, s in unwrap(g.kv_cache_spec).items()}


def history_bytes(groups):
    return sum(
        spec.unpadded_page_size_bytes
        for group in groups
        if classify(group.kv_cache_spec) == "history"
        for spec in unwrap(group.kv_cache_spec).values()
    )


def get_window_staging_rows(max_tokens: int, max_requests: int) -> int:
    """Keep the planner and metadata builder on the same window allowance."""
    return ((max_tokens + max_requests * (2 * WINDOW) + WINDOW - 1) // WINDOW) * WINDOW


def get_deepseek_v41_memory_budget(vllm_config: VllmConfig, groups: list[KVCacheGroupSpec]) -> RequestStateMemoryBudget:
    capacity = vllm_config.scheduler_config.max_num_seqs + 1
    fixed = sum(
        capacity * spec.unpadded_page_size_bytes
        for spec in all_specs(groups).values()
        if isinstance(spec, (RequestWindowSpec, CircularBufferSpec))
    )
    rows = get_window_staging_rows(
        vllm_config.scheduler_config.max_num_batched_tokens, vllm_config.scheduler_config.max_num_seqs
    )
    return RequestStateMemoryBudget(history_bytes(groups), fixed, rows * ROW_BYTES + 2**20)


def get_deepseek_v41_max_memory_usage(vllm_config: VllmConfig, groups: list[KVCacheGroupSpec]) -> int:
    """One full-length request plus all fixed reservations, excluding history null.

    The upstream admission/auto-fit caller already removes one history page
    from available memory. State capacities include their own null slots.
    """
    budget = get_deepseek_v41_memory_budget(vllm_config, groups)
    pages = max(
        (spec.max_memory_usage_bytes(vllm_config) + spec.page_size_bytes - 1) // spec.page_size_bytes
        for group in groups
        if classify(group.kv_cache_spec) == "history"
        for spec in unwrap(group.kv_cache_spec).values()
    )
    return budget.bytes_for_history_blocks(pages)


def get_deepseek_v41_kv_cache_config(
    vllm_config: VllmConfig, groups: list[KVCacheGroupSpec], available_memory: int
) -> AscendKVCacheConfig:
    """Reserve request state and staging, then budget the remaining history pages."""
    assert vllm_config.speculative_config is None
    if vllm_config.kv_transfer_config is not None:
        from vllm import envs

        from vllm_ascend.distributed.kv_transfer.kv_p2p.dsv41_state import validate_pd_config

        validate_pd_config(vllm_config)
        if envs.VLLM_USE_V2_MODEL_RUNNER:
            raise ValueError("State PD initially requires Model Runner V1")
    assert not vllm_config.cache_config.enable_prefix_caching
    assert vllm_config.parallel_config.pipeline_parallel_size == 1
    # The framework/CLI-provided budget is authoritative. No historical
    # experiment file or diagnostic environment can change admission capacity.
    maximum = vllm_config.scheduler_config.max_num_seqs
    batch = vllm_config.scheduler_config.max_num_batched_tokens
    capacity = maximum + 1
    specs = all_specs(groups)
    budget = get_deepseek_v41_memory_budget(vllm_config, groups)
    page = budget.history_page_bytes
    fixed = budget.request_state_bytes
    staging_rows = get_window_staging_rows(batch, maximum)
    staging = budget.workspace_bytes
    count = budget.history_capacity(available_memory)
    if count < 2:
        raise ValueError("Cache budget must cover fixed request state, staging, a null history page and a usable page")
    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        if override < 2 or override > count:
            raise ValueError(
                "Explicit override exceeds the state-separated byte budget or leaves no usable history page"
            )
        count = override
    tensors = []
    for name, spec in specs.items():
        n = capacity if isinstance(spec, (RequestWindowSpec, CircularBufferSpec)) else count
        stride = spec.unpadded_page_size_bytes
        tensors.append(KVCacheTensor(size=n * stride, layers=[name], offset=0, layer_stride=0, block_stride=stride))
    plan = RequestStatePlan(
        pools=tuple(
            CachePoolPlan(
                pool_id=classify(group.kv_cache_spec),
                capacity=count if classify(group.kv_cache_spec) == "history" else capacity,
                group_ids=(group_id,),
                allocation_unit="token_block" if classify(group.kv_cache_spec) == "history" else "request_slot",
            )
            for group_id, group in enumerate(groups)
        ),
        staging_rows=staging_rows,
        staging_allowance_bytes=staging,
    )
    result = AscendKVCacheConfig(
        request_state_plan=plan,
        num_blocks=count,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )
    plan.validate(result)
    assert sum(x.size for x in tensors) + staging <= available_memory
    logger.info(
        "DeepSeek V4.1 cache budget: %d bytes, history: %d blocks (%d bytes/block), "
        "request slots: %d per state pool (including null), state: %d bytes, staging: %d bytes",
        available_memory,
        count,
        page,
        capacity,
        fixed,
        staging,
    )
    return result


def get_deepseek_v41_pool_bytes_per_block(groups):
    return history_bytes(groups)
