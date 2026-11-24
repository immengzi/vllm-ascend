# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Storage, admission and release for the 128 KiB Flash KV layout."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import CircularBufferSpec
from vllm.v1.request import Request

from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.core.kv_cache_interface import register_ascend_kv_cache_specs
from vllm_ascend.models.deepseek_v41.cache_config import make_index_cache_spec, make_mla_cache_spec, make_swa_cache_spec
from vllm_ascend.models.deepseek_v41.packed_cache import PackedHistoryGroup


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
            num_gpu_blocks_override=None,
            enable_extensible_kv_cache=False,
            resolved_gpu_memory_utilization=0.95,
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
    from vllm_ascend.worker.v2 import attn_utils

    c.kv_transfer_config = SimpleNamespace(
        kv_connector="MooncakePullConnector", kv_connector_extra_config={}, is_kv_producer=False
    )
    builder = AscendKVCacheConfigBuilder()
    groups = builder.get_kv_cache_groups(c, specs)
    cfg = builder.get_kv_cache_config_from_groups(c, groups, count)
    attn = [
        SimpleNamespace(kv_cache_group_id=i, kv_cache_spec=s, layer_names=[n], backend=DeepseekV41CacheBackend)
        for i, g in enumerate(groups)
        for n, s in g.kv_cache_spec.kv_cache_specs.items()
    ]
    with patch.object(attn_utils, "get_current_vllm_config", return_value=c):
        backing = attn_utils._allocate_int8_cache_tensor(cfg.kv_cache_tensors[0].size, 2 * 1024**2, torch.device("cpu"))
    views = attn_utils.create_kv_cache_runtime_views(cfg, backing, attn, [128] * len(groups), {})
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
    config.additional_config = {"packed_history_cache": {"enabled": True, "swa_slots": swa, "compressor_slots": c2}}
    config.model_config.max_model_len = length


@pytest.mark.parametrize("swa,c2", [(1, 2), (3, 2)])
def test_packed_views_share_one_allocation(flash, swa, c2):
    config, specs = flash
    configure(config, swa, c2, 6 * 128)
    count = 1 + 20 * swa + 2 * c2 + 6
    cfg, views = worker_views(config, specs, count)
    assert len(cfg.kv_cache_groups) == 23
    assert len(cfg.kv_cache_tensors) == len(views) == 51
    assert {d.size for d in cfg.kv_cache_tensors} == {count * 369280}
    storages = set()
    strides = set()
    for desc in cfg.kv_cache_tensors:
        view = views[desc.layers[0]]
        for part in view if isinstance(view, tuple) else (view,):
            storages.add(part.untyped_storage().data_ptr())
            assert part.shape[0] == count
            assert part.stride(0) * part.element_size() == desc.block_stride
        strides.add(desc.block_stride)
    assert len(storages) == 1
    assert strides == {131072, 107136}
    mgr = manager_for(cfg)
    assert [m.logical_limit for m in mgr.coordinator.single_type_managers] == [6] + [c2] * 2 + [swa] * 20


@pytest.mark.parametrize("hist,swa,c2", [(3, 1, 2), (3, 2, 1), (1, 2, 2)])
def test_quota_atomicity_and_deferred_free(flash, hist, swa, c2):
    config, specs = flash
    configure(config, swa, c2, 128)
    cfg, _ = worker_views(config, specs, 1 + hist + 20 * swa + 2 * c2)
    mgr = manager_for(cfg)
    pool = mgr.coordinator.block_pool

    def req(name):
        return Request(name, [0], SamplingParams(max_tokens=1), pooling_params=None)

    a, b = req("first"), req("second")
    assert mgr.allocate_slots(a, 1) is not None
    ids = mgr.get_blocks(a.request_id).get_block_ids()
    assert len({i for group in ids for i in group}) == 23
    free = pool.get_num_free_blocks()
    assert free > 0 and mgr.allocate_slots(b, 1) is None
    assert pool.get_num_free_blocks() == free
    # The scheduler has removed the request, but a simulated in-flight
    # transfer still owns the native blocks until its completion fence.
    detached = mgr.pop_blocks_for_free(a)
    assert len(detached) == 23
    assert mgr.allocate_slots(b, 1) is None
    assert pool.get_num_free_blocks() == free
    pool.free_blocks(reversed(detached))
    for i in range(3):
        assert mgr.allocate_slots(b, 1) is not None
        mgr.free(b)
        assert pool.get_num_free_blocks() == cfg.num_blocks - 1


@pytest.mark.parametrize("key,value", [("swa_slots", 0), ("compressor_slots", False), ("swa_slots", 1.5)])
def test_invalid_logical_slot_count(flash, key, value):
    config, specs = flash
    configure(config, 2, 3, 128)
    config.additional_config["packed_history_cache"][key] = value
    with pytest.raises(ValueError, match=key):
        AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs)


@pytest.mark.parametrize("additional", [{}, {"packed_history_cache": {"enabled": False}}])
def test_native_layout_is_default(flash, additional):
    config, specs = flash
    config.additional_config = additional
    groups = AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs)
    assert len(groups) == 12
    assert not any(isinstance(group, PackedHistoryGroup) for group in groups)


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


def test_extensible_cache_rejected_when_present(flash):
    config, specs = flash
    configure(config, 2, 2, 128)
    config.cache_config.enable_extensible_kv_cache = True
    with pytest.raises(ValueError, match="static placement"):
        AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs)


def test_planner_auto_fit_reserves_state_and_null(flash):
    from vllm.v1.core.kv_cache_planning import get_global_kv_cache_groups, get_kv_cache_configs

    config, specs = flash
    configure(config, 2, 3, 2048)
    config.device_config = SimpleNamespace(device_type="cpu")
    config.model_config.original_max_model_len = -1
    budget = (1 + 20 * 2 + 2 * 3 + 6) * 369280
    with patch("vllm.v1.core.kv_cache_planning.get_kv_cache_config_builder", return_value=AscendKVCacheConfigBuilder()):
        groups = get_global_kv_cache_groups(config, [specs, specs])
        configs = get_kv_cache_configs(config, [specs, specs], [budget + 369280, budget], groups)
    assert config.model_config.max_model_len == 768
    for cfg in configs:
        assert cfg.num_blocks == 53
        assert {d.size for d in cfg.kv_cache_tensors} == {budget}


def test_stock_pull_registers_one_backing_with_per_plane_strides(flash):
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker

    config, specs = flash
    configure(config, 2, 2, 128)
    cfg, views = worker_views(config, specs, 50)
    worker = object.__new__(MooncakeBaseConnectorWorker)
    worker.vllm_config, worker.kv_cache_config = config, cfg
    worker.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    worker.block_size, worker.tp_size, worker.tp_rank, worker.pcp_rank = 128, 2, 0, 0
    worker.engine_id, worker.te_rpc_port, worker.side_channel_host, worker.handshake_port = "test", 1, "127.0.0.1", 2
    with patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker.global_te") as engine:
        worker.register_kv_caches(views)
    pointers, sizes = engine.register_buffer.call_args.args
    assert len(pointers) == 1 and pointers[0] % (2 * 1024**2) == 0
    assert sizes == [50 * 369280]
    meta = worker.transfer_metadata
    assert {stride for layer in meta.block_strides for stride in layer} == {131072, 107136}
    assert all(scale == 1 for layer in meta.block_size_scales for scale in layer)


@pytest.mark.parametrize("bad_value", ["false", 1])
def test_enable_requires_json_boolean(flash, bad_value):
    config, specs = flash
    config.additional_config = {"packed_history_cache": {"enabled": bad_value}}
    with pytest.raises(ValueError, match="enabled"):
        AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs)


@pytest.mark.parametrize("feature", ["prefix_cache", "pcp", "other_model", "spec_decode", "connector", "hisparse"])
def test_explicit_opt_in_rejects_unsupported_combinations(flash, feature):
    config, specs = flash
    configure(config, 1, 2, 128)
    if feature == "prefix_cache":
        config.cache_config.enable_prefix_caching = True
    elif feature == "pcp":
        config.parallel_config.prefill_context_parallel_size = 4
    elif feature == "other_model":
        config.model_config.architecture = "OtherForCausalLM"
    elif feature == "spec_decode":
        config.speculative_config = SimpleNamespace()
    elif feature == "connector":
        config.kv_transfer_config = SimpleNamespace(kv_connector="MooncakeHybridConnector")
    else:
        config.attention_config.hisparse_config = SimpleNamespace()
    with pytest.raises(ValueError, match="Packed history|Fixed SWA rings"):
        AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs)


def test_request_window_prefill_uses_previous_tail_and_current_chunk():
    from vllm_ascend.models.deepseek_v41.window_state import RequestWindowMetadataBuilder

    # A 2-row continuation crossing the window boundary, followed by a new 3-row prompt.
    queries = torch.tensor([0, 2, 5], dtype=torch.int32)
    lengths = torch.tensor([130, 3], dtype=torch.int32)
    common = SimpleNamespace(
        query_start_loc_cpu=queries,
        seq_lens_cpu=lengths,
        block_table_tensor=torch.tensor([[19], [7]], dtype=torch.int32),
    )
    req = SimpleNamespace(num_reqs=2, max_query_len=3, seq_lens=lengths)
    metadata = RequestWindowMetadataBuilder(lengths, capacity=384).build(common, req)
    assert metadata.state_ids.tolist() == [19, 7]
    assert metadata.lengths.tolist() == [129, 3]
    assert metadata.row_starts.tolist() == [0, 256]
    assert metadata.block_table.tolist() == [[0, 1], [2, 0]]
