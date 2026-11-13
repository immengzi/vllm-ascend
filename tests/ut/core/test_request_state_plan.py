# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for explicit pool ownership and the DSV4.1 planner."""

import copy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor

from vllm_ascend.core.kv_cache_interface import (
    AscendKVCacheConfig,
    CachePoolPlan,
    RequestStatePlan,
    get_cache_group_capacity,
    get_request_state_plan,
)
from vllm_ascend.models.deepseek_v41 import cache_config as planner


def sample_config():
    pools = (
        CachePoolPlan("history", 8877, (0,), "token_block"),
        CachePoolPlan("swa", 49, (1,), "request_slot"),
        CachePoolPlan("compressor", 33, (2,), "request_slot"),
    )
    # Deliberately avoid model-specific names and equal state capacities.
    groups = [SimpleNamespace(layer_names=[name], kv_cache_spec=object()) for name in ("a", "b", "c")]
    tensors = [
        KVCacheTensor(size=pool.capacity * stride, layers=[name], offset=0, layer_stride=0, block_stride=stride)
        for pool, name, stride in zip(pools, ("a", "b", "c"), (65536, 131072, 65536))
    ]
    return AscendKVCacheConfig(
        num_blocks=8877,
        kv_cache_groups=groups,
        kv_cache_tensors=tensors,
        request_state_plan=RequestStatePlan(pools, 20480, 21 * 2**20),
    )


def model_fixture(maximum=48, batch=8192, override=None):
    specs = {}
    for layer, ratio in ((2, 2), (8, 2), (14, 2), (20, 1)):
        specs[f"m.layers.{layer}.long_kv_cache"] = planner.make_mla_cache_spec(
            block_size=128, head_size=512, compress_ratio=ratio
        )
        specs[f"m.layers.{layer}.indexer.k_cache"] = planner.make_index_cache_spec(
            block_size=128, head_size=128, compress_ratio=ratio
        )
    for layer in range(40):
        specs[f"m.layers.{layer}.swa_cache"] = planner.make_swa_cache_spec(
            block_size=128, window_size=128, head_size=512, dtype=torch.bfloat16, cache_dtype="auto"
        )
    for layer in (2, 8, 14):
        specs[f"m.layers.{layer}.compressor.state_cache"] = planner.CircularBufferSpec(
            block_size=32, num_kv_heads=1, head_size=1024, head_size_v=0, dtype=torch.float32
        )
    cfg = SimpleNamespace(
        speculative_config=None,
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=maximum, max_num_batched_tokens=batch),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False, num_gpu_blocks_override=override, prefix_cache_retention_interval=None
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )
    return cfg, planner.make_cache_groups(planner.group_cache_specs(specs))


def test_capacity_is_explicit_and_native_fallback_is_unchanged():
    cfg = sample_config()
    cfg.request_state_plan.validate(cfg)
    assert [get_cache_group_capacity(cfg, i) for i in range(3)] == [8877, 49, 33]
    native = KVCacheConfig(num_blocks=100, kv_cache_groups=cfg.kv_cache_groups, kv_cache_tensors=[])
    assert get_request_state_plan(native) is None
    assert [get_cache_group_capacity(native, i) for i in range(3)] == [100, 100, 100]
    with pytest.raises(ValueError, match="No pool"):
        get_cache_group_capacity(cfg, 3)


@pytest.mark.parametrize("transform", [copy.copy, copy.deepcopy, lambda x: replace(x)])
def test_plan_survives_config_copy_and_replace(transform):
    cfg = transform(sample_config())
    assert cfg.request_state_plan == sample_config().request_state_plan
    assert [get_cache_group_capacity(cfg, i) for i in range(3)] == [8877, 49, 33]
    cfg.request_state_plan.validate(cfg)


def test_scheduler_representative_conversion_preserves_plan():
    cfg = copy.deepcopy(sample_config())
    cfg.kv_cache_groups = [SimpleNamespace(layer_names=[], kv_cache_spec=object()) for _ in cfg.kv_cache_groups]
    assert [get_cache_group_capacity(cfg, i) for i in range(3)] == [8877, 49, 33]


@pytest.mark.parametrize(
    "fault", ["missing_group", "wrong_history", "wrong_size", "missing_tensor", "duplicate_tensor"]
)
def test_invalid_layout_fails_before_allocation(fault):
    cfg = sample_config()
    if fault == "missing_group":
        cfg.kv_cache_groups.pop()
    elif fault == "wrong_history":
        cfg.num_blocks -= 1
    elif fault == "wrong_size":
        cfg.kv_cache_tensors[1].size -= 1
    elif fault == "missing_tensor":
        cfg.kv_cache_tensors.pop()
    else:
        cfg.kv_cache_tensors.append(cfg.kv_cache_tensors[0])
    with pytest.raises(ValueError):
        cfg.request_state_plan.validate(cfg)


def test_duplicate_group_and_invalid_marker_are_rejected():
    pool = CachePoolPlan("history", 3, (0,), "token_block")
    with pytest.raises(ValueError, match="unique"):
        RequestStatePlan((pool, replace(pool, pool_id="swa")), 0, 0)
    cfg = sample_config()
    cfg.request_state_plan = "state"
    with pytest.raises(TypeError):
        get_request_state_plan(cfg)


def test_budget_and_all_backing_sizes_match_measured_geometry_without_results_dir():
    with patch.object(planner, "uses_a5_packed_cache", return_value=False), patch.dict("os.environ", {}, clear=True):
        cfg, groups = model_fixture()
        result = planner.get_deepseek_v41_kv_cache_config(cfg, groups, 3576615936)
    assert result.num_blocks == 8877
    assert result.request_state_plan.staging_rows == 20480
    assert result.request_state_plan.staging_allowance_bytes == 21 * 2**20
    assert planner.get_deepseek_v41_pool_bytes_per_block(groups) == 369280
    assert len(result.kv_cache_tensors) == 51
    assert sum(t.size for t in result.kv_cache_tensors if t.layers[0].endswith(".swa_cache")) == 49 * 40 * 131072
    for group_id, group in enumerate(groups):
        capacity = get_cache_group_capacity(result, group_id)
        for name in group.layer_names:
            tensor = next(t for t in result.kv_cache_tensors if t.layers == [name])
            spec = group.kv_cache_spec.kv_cache_specs[name]
            assert tensor.size == capacity * spec.unpadded_page_size_bytes
            assert tensor.block_stride == spec.unpadded_page_size_bytes
            assert tensor.offset == tensor.layer_stride == 0


@pytest.mark.parametrize("maximum,batch,override", [(1, 128, None), (12, 8192, None), (48, 8192, 100)])
def test_budget_variants_and_explicit_override(maximum, batch, override):
    with patch.object(planner, "uses_a5_packed_cache", return_value=False):
        cfg, groups = model_fixture(maximum, batch, override)
        budget = 3576615936
        result = planner.get_deepseek_v41_kv_cache_config(cfg, groups, budget)
    plan = result.request_state_plan
    assert [pool.capacity for pool in plan.pools[1:]] == [maximum + 1, maximum + 1]
    assert sum(t.size for t in result.kv_cache_tensors) + plan.staging_allowance_bytes <= budget
    if override is not None:
        assert result.num_blocks == override


def test_insufficient_budget_and_oversized_override_are_rejected():
    with patch.object(planner, "uses_a5_packed_cache", return_value=False):
        cfg, groups = model_fixture()
        with pytest.raises(ValueError):
            planner.get_deepseek_v41_kv_cache_config(cfg, groups, 1)
        cfg.cache_config.num_gpu_blocks_override = 1000000
        with pytest.raises(ValueError, match="exceeds"):
            planner.get_deepseek_v41_kv_cache_config(cfg, groups, 3576615936)


def test_actual_scheduler_conversion_preserves_capacity_contract():
    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config

    with patch.object(planner, "uses_a5_packed_cache", return_value=False):
        cfg, groups = model_fixture()
        worker_config = planner.get_deepseek_v41_kv_cache_config(cfg, groups, 3576615936)
    scheduler_config = generate_scheduler_kv_cache_config([worker_config, copy.deepcopy(worker_config)])
    assert scheduler_config.request_state_plan == worker_config.request_state_plan
    assert [get_cache_group_capacity(scheduler_config, i) for i in range(3)] == [8877, 49, 49]
    assert isinstance(worker_config.kv_cache_groups[0].kv_cache_spec, planner.UniformTypeKVCacheSpecs)
    assert not isinstance(scheduler_config.kv_cache_groups[0].kv_cache_spec, planner.UniformTypeKVCacheSpecs)


def test_multi_group_pool_is_explicitly_out_of_scope():
    with pytest.raises(ValueError, match="one cache group"):
        RequestStatePlan((CachePoolPlan("shared", 4, (0, 1), "request_slot"),), 0, 0)
