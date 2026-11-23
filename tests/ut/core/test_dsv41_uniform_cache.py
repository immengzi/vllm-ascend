# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Storage, admission and release for the 128 KiB Flash KV layout."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_planning import get_global_kv_cache_groups, get_kv_cache_configs
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import CircularBufferSpec
from vllm.v1.request import Request

from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.core.kv_cache_interface import register_ascend_kv_cache_specs
from vllm_ascend.models.deepseek_v41.cache_config import make_index_cache_spec, make_mla_cache_spec, make_swa_cache_spec
from vllm_ascend.models.deepseek_v41.uniform_cache import (
    COMPRESSOR_SLOT_OPTION,
    LAYOUT_OPTION,
    SWA_SLOT_OPTION,
    TILE_BYTES,
    LogicalPoolGroup,
)


@pytest.fixture
def flash():
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    c = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="DeepseekV41ForCausalLM",
            hf_config=SimpleNamespace(model_type="deepseek_v41"),
            max_model_len=1048576,
            original_max_model_len=1048576,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=2,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False, max_num_batched_tokens=4096, max_num_seqs=32
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            swa_bounded_replay=False,
            enable_extensible_kv_cache=False,
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
        ),
        use_v2_model_runner=True,
        speculative_config=None,
        num_prefill_lookahead_tokens=1,
        max_in_flight_tokens=4096,
        kv_transfer_config=None,
        additional_config={},
    )
    specs = {}
    with patch("vllm_ascend.models.deepseek_v41.cache_config.uses_a5_packed_cache", return_value=False):
        for i in range(40):
            name = f"model.layers.{i}.self_attn"
            specs[name + ".swa_cache"] = make_swa_cache_spec(
                block_size=128, window_size=128, head_size=512, dtype=torch.bfloat16, cache_dtype="auto"
            )
            if i in (2, 8, 14, 20):
                ratio = 1 if i == 20 else 2
                specs[name + ".long_kv_cache"] = make_mla_cache_spec(
                    block_size=128, head_size=512, compress_ratio=ratio
                )
                specs[name + ".indexer.k_cache"] = make_index_cache_spec(
                    block_size=128, head_size=128, compress_ratio=ratio
                )
                if ratio == 2:
                    specs[name + ".compressor.state_cache"] = CircularBufferSpec(
                        block_size=32, num_kv_heads=1, head_size=1024, dtype=torch.float32, head_size_v=0
                    )
    return c, specs


def worker_views(c, specs, count):
    from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheBackend
    from vllm_ascend.patch.platform import patch_kv_cache_utils  # noqa: F401
    from vllm_ascend.worker.v2 import attn_utils

    c.kv_transfer_config = SimpleNamespace(kv_connector="MooncakePullConnector")
    builder = AscendKVCacheConfigBuilder()
    groups = builder.get_kv_cache_groups(c, specs)
    cfg = builder.get_kv_cache_config_from_groups(c, groups, count)
    attn = [
        SimpleNamespace(kv_cache_group_id=i, kv_cache_spec=s, layer_names=[n], backend=DeepseekV41CacheBackend)
        for i, g in enumerate(groups)
        for n, s in g.kv_cache_spec.kv_cache_specs.items()
    ]
    with patch.object(attn_utils, "get_current_vllm_config", return_value=c):
        raw = attn_utils._allocate_kv_cache(cfg, {}, torch.device("cpu"))
        views = attn_utils._reshape_kv_cache_v2(attn, raw, "auto", [128] * len(groups), {}, cfg)
    return cfg, views


def manager_for(cfg):
    from vllm_ascend.patch.platform import patch_kv_cache_coordinator

    with patch(
        "vllm.v1.core.kv_cache_manager.get_kv_cache_coordinator", patch_kv_cache_coordinator.get_kv_cache_coordinator
    ):
        manager = KVCacheManager(generate_scheduler_kv_cache_config([cfg]), 4096, 128, 128, enable_caching=False)
    pool = manager.coordinator.block_pool
    assert all(m.block_pool is pool for m in manager.coordinator.single_type_managers)
    return manager


def configure(config, swa, c2, length):
    config.additional_config = {LAYOUT_OPTION: True, SWA_SLOT_OPTION: swa, COMPRESSOR_SLOT_OPTION: c2}
    config.model_config.max_model_len = length


def plan(config, specs, budget):
    from vllm_ascend.patch.platform import patch_kv_cache_utils  # noqa: F401

    with patch(
        "vllm.v1.core.kv_cache_config_builder.get_kv_cache_config_builder", return_value=AscendKVCacheConfigBuilder()
    ):
        groups = get_global_kv_cache_groups(config, [specs, specs])
        return get_kv_cache_configs(config, [specs, specs], [budget + 2 * TILE_BYTES, budget], groups)


@pytest.mark.parametrize("hist,swa,c2,tiles", [(6, 2, 3, 108), (6, 3, 3, 148), (6, 2, 4, 111), (7, 2, 3, 111)])
def test_actual_storage(flash, hist, swa, c2, tiles):
    config, specs = flash
    configure(config, swa, c2, 128 * hist)
    for cfg in plan(config, specs, tiles * TILE_BYTES):
        assert cfg.num_blocks == tiles
        assert len(cfg.kv_cache_tensors) == 1
        assert cfg.kv_cache_tensors[0].size == tiles * TILE_BYTES
    cfg, views = worker_views(config, specs, cfg.num_blocks)
    storages = {
        t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
        for view in views.values()
        for t in (view if isinstance(view, tuple) else (view,))
    }
    assert list(storages.values()) == [tiles * TILE_BYTES + 2 * 1024**2]
    for view in views.values():
        for tensor in view if isinstance(view, tuple) else (view,):
            assert tensor.shape[0] == tiles
            assert tensor.stride(0) * tensor.element_size() == TILE_BYTES
    managers = manager_for(cfg).coordinator.single_type_managers
    assert [m.logical_limit for m in managers] == [hist] * 3 + [c2] * 3 + [swa] * 40


def test_auto_fit_reserves_state_and_null(flash):
    config, specs = flash
    configure(config, 2, 3, 2048)
    config.device_config = SimpleNamespace(device_type="cpu")
    config.model_config.original_max_model_len = -1
    plan(config, specs, 108 * TILE_BYTES)
    assert config.model_config.max_model_len == 6 * 128


@pytest.mark.parametrize("hist,swa,c2", [(3, 1, 2), (3, 2, 1), (1, 2, 2)])
def test_quota_atomicity_and_deferred_free(flash, hist, swa, c2):
    config, specs = flash
    configure(config, swa, c2, 128)
    cfg, _ = worker_views(config, specs, 1 + 3 * hist + 40 * swa + 3 * c2)
    mgr = manager_for(cfg)
    pool = mgr.coordinator.block_pool

    def req(name):
        return Request(name, [0], SamplingParams(max_tokens=1), pooling_params=None)

    a, b = req("first"), req("second")
    assert mgr.allocate_slots(a, 1) is not None
    ids = mgr.get_blocks(a.request_id).get_block_ids()
    assert len({i for group in ids for i in group}) == 46
    free = pool.get_num_free_blocks()
    assert free > 0 and mgr.allocate_slots(b, 1) is None
    assert pool.get_num_free_blocks() == free
    # The scheduler has removed the request, but a simulated in-flight
    # transfer still owns the native blocks until its completion fence.
    detached = mgr.pop_blocks_for_free(a)
    assert len(detached) == 46
    assert mgr.allocate_slots(b, 1) is None
    assert pool.get_num_free_blocks() == free
    pool.free_blocks(reversed(detached))
    for i in range(3):
        assert mgr.allocate_slots(b, 1) is not None
        mgr.free(b)
        assert pool.get_num_free_blocks() == cfg.num_blocks - 1


def test_same_budget_returns_unused_slots_to_history_at_startup(flash):
    config, specs = flash
    budget = 148 * TILE_BYTES
    for swa_slots, expected_history in ((3, 6), (2, 19)):
        configure(config, swa_slots, 3, 512)
        cfg, _ = plan(config, specs, budget)
        mgr = manager_for(cfg)
        assert [m.logical_limit for m in mgr.coordinator.single_type_managers[:3]] == [expected_history] * 3
        assert cfg.kv_cache_tensors[0].size == budget


@pytest.mark.parametrize("key,value", [(SWA_SLOT_OPTION, 0), (COMPRESSOR_SLOT_OPTION, False), (SWA_SLOT_OPTION, 1.5)])
def test_invalid_logical_slot_count(flash, key, value):
    config, specs = flash
    configure(config, 2, 3, 128)
    config.additional_config[key] = value
    with pytest.raises(ValueError, match=key):
        AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs)


@pytest.mark.parametrize("additional", [{}, {LAYOUT_OPTION: False}])
def test_native_layout_is_default(flash, additional):
    config, specs = flash
    config.additional_config = additional
    groups = AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs)
    assert len(groups) == 12
    assert not any(isinstance(group, LogicalPoolGroup) for group in groups)


@pytest.mark.parametrize("length", [128, 129, 257])
def test_decode_uses_last_window_at_page_boundaries(length):
    from vllm_ascend.models.deepseek_v41.window_state import RequestWindowMetadataBuilder

    lengths = torch.tensor([length], dtype=torch.int32)
    common = SimpleNamespace(block_table_tensor=torch.tensor([[17]]), query_start_loc=torch.tensor([0, 1]))
    result = SimpleNamespace(num_reqs=1, max_query_len=1, seq_lens=lengths)
    metadata = RequestWindowMetadataBuilder(lengths, capacity=128).build(common, result)
    assert metadata.state_ids.tolist() == [17]
    assert metadata.lengths.tolist() == [128]
    assert metadata.block_table.tolist() == [[0]]
