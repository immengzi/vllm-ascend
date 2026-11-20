# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Exercise Flash page placement with the native planner, worker, allocator and Pull."""

import ctypes
from collections import defaultdict
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
from vllm_ascend.models.deepseek_v41.compact_cache import CONFIG_KEY, CompactPageGroup


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
        additional_config={CONFIG_KEY: True},
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


def test_unsupported_or_disabled(flash):
    c, specs = flash
    builder = AscendKVCacheConfigBuilder()
    for value in (False, None):
        c.additional_config = {} if value is None else {CONFIG_KEY: value}
        assert not any(isinstance(g, CompactPageGroup) for g in builder.get_kv_cache_groups(c, specs))
    c.additional_config = {CONFIG_KEY: True}
    with pytest.raises(ValueError, match="history"):
        builder.get_kv_cache_groups(c, {n: s for n, s in specs.items() if not n.endswith("long_kv_cache")})
    c.cache_config.enable_prefix_caching = True
    with pytest.raises(ValueError, match="prefix"):
        builder.get_kv_cache_groups(c, specs)


@pytest.mark.parametrize("enabled,stride,num_groups", [(False, 540928, 12), (True, 393216, 16)])
@pytest.mark.parametrize("external", [False, True], ids=["chunked", "decode-handoff"])
def test_plan_and_retain_700k(flash, enabled, stride, num_groups, external):
    from vllm_ascend.patch.platform import patch_kv_cache_coordinator, patch_kv_cache_utils  # noqa: F401

    c, specs = flash
    c.additional_config[CONFIG_KEY] = enabled
    budget = int(13.23 * 2**30)
    with patch(
        "vllm.v1.core.kv_cache_config_builder.get_kv_cache_config_builder", return_value=AscendKVCacheConfigBuilder()
    ):
        groups = get_global_kv_cache_groups(c, [specs, specs])
        configs = get_kv_cache_configs(c, [specs, specs], [budget + 2 * stride, budget], groups)
    assert all(cfg.num_blocks == budget // stride for cfg in configs)
    assert len(groups) == num_groups
    assert sum(t.size for t in configs[0].kv_cache_tensors) == (budget // stride) * stride
    with patch(
        "vllm.v1.core.kv_cache_manager.get_kv_cache_coordinator", patch_kv_cache_coordinator.get_kv_cache_coordinator
    ):
        manager = KVCacheManager(
            generate_scheduler_kv_cache_config(configs),
            1048576,
            128,
            128,
            max_in_flight_tokens=4096,
            enable_caching=False,
        )
    pool = manager.coordinator.block_pool
    assert all(m.block_pool is pool for m in manager.coordinator.single_type_managers)
    retained = []
    expected_residents = 6 if enabled else 4
    for i in range(expected_residents + 1):
        req = Request(str(i), [1] * 700001, SamplingParams(max_tokens=4), pooling_params=None)
        # A failed chunk can leave earlier chunks resident until the request is freed.
        if external:
            allocated = manager.allocate_slots(req, 0, num_external_computed_tokens=700000, delay_cache_blocks=True)
            if allocated is not None:
                req.num_computed_tokens = 700000
                assert manager.allocate_slots(req, 1) is not None
                req.num_computed_tokens += 1
        else:
            while req.num_computed_tokens < 700001:
                chunk = min(4096, 700000 - req.num_computed_tokens) or 1
                if manager.allocate_slots(req, chunk) is None:
                    break
                req.num_computed_tokens += chunk
        if i == expected_residents:
            assert req.num_computed_tokens < 700001
            manager.free(req)
            break
        assert req.num_computed_tokens == 700001
        live = [
            sum(b != pool.null_block.block_id for b in ids)
            for ids in manager.get_blocks(req.request_id).get_block_ids()
        ]
        assert sorted(live) == sorted([5469, 1] + [2] * (num_groups - 2))
        retained.append(req)
        assert pool.get_num_free_blocks() == configs[0].num_blocks - 1 - len(retained) * sum(live)
    for req in retained:
        manager.free(req)
    assert pool.get_num_free_blocks() == configs[0].num_blocks - 1


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


def register(c, cfg, views):
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker

    worker = object.__new__(MooncakeBaseConnectorWorker)
    worker.vllm_config, worker.kv_cache_config = c, cfg
    worker.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    worker.block_size, worker.tp_size, worker.tp_rank, worker.pcp_rank = 128, 2, 0, 0
    worker.engine_id, worker.te_rpc_port, worker.side_channel_host, worker.handshake_port = "cpu", 1, "127.0.0.1", 2
    with patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker.global_te") as engine:
        worker.register_kv_caches(views)
    pointers, sizes = engine.register_buffer.call_args.args
    assert len(pointers) == 1 and pointers[0] % (2 * 1024**2) == 0
    assert sizes == [cfg.num_blocks * 393216]
    return worker


def test_worker_views_and_stock_pull(flash):
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread

    c, specs = flash
    # Different capacities ensure no local offset or bank boundary can stand in for the peer's.
    p_cfg, p = worker_views(c, specs, 39)
    d_cfg, d = worker_views(c, dict(reversed(specs.items())), 37)
    producer, consumer = register(c, p_cfg, p), register(c, d_cfg, d)
    remote, local = producer.transfer_metadata, consumer.transfer_metadata
    assert remote.layer_names == local.layer_names
    thread = object.__new__(MooncakePullRecvingThread)
    thread.kv_cache_specs = consumer.kv_cache_specs
    thread.tp_size, thread.pcp_size, thread.dcp_size, thread.pcp_rank = 2, 1, 1, 0
    for field in ("layer_names", "block_shapes", "block_lens", "block_strides", "kv_caches_base_addr"):
        setattr(thread, field, getattr(local, field))
    peer = SimpleNamespace(
        metadata_by_pcp_rank=[SimpleNamespace(metadata_by_tp_rank=[remote] * 4)],
        dcp_sharded=remote.dcp_sharded,
        block_lens=remote.block_lens,
        block_strides=remote.block_strides,
        block_shapes=remote.block_shapes,
    )
    entries = defaultdict(dict)
    expected_bytes = 0
    for layer, name in enumerate(local.layer_names):
        group = local.group_indices[layer]
        for thread.tp_rank in range(2):
            eligible = thread._get_layer_remote_tp_rank_groups(layer, layer, specs[name], peer, 1, 4, 1)
            assert eligible == [[0, 1, 2, 3]]
        source_ids = [2 * group + 7, 2 * group + 8]
        dest_ids = [2 * group + 5, 2 * group + 6]
        entries[consumer.layer_name_to_spec_index[name]][layer, layer] = [("req", dest_ids, source_ids)]
        planes = p[name] if isinstance(p[name], tuple) else (p[name],)
        for component, plane in enumerate(planes):
            if "state_cache" in name or "swa_cache" in name:
                assert plane.is_contiguous()
            for j, block in enumerate(source_ids):
                plane[block].fill_(1 + 2 * layer + component + j)
            expected_bytes += 2 * plane[0].numel() * plane.element_size()
    src, dst, lengths = [], [], []
    for idx, layers in entries.items():
        thread._append_spec_transfer_addresses(idx, 0, 3, 1, 4, 1, layers, peer, src, dst, lengths)
    assert sum(lengths) == expected_bytes
    for laddr, raddr, nbytes in zip(src, dst, lengths, strict=True):
        for addr, cache in ((laddr, d), (raddr, p)):
            first = next(iter(cache.values()))
            if isinstance(first, tuple):
                first = first[0]
            storage = first.untyped_storage()
            assert storage.data_ptr() <= addr < addr + nbytes <= storage.data_ptr() + storage.nbytes()
        ctypes.memmove(laddr, raddr, nbytes)
    for layer, name in enumerate(local.layer_names):
        group = local.group_indices[layer]
        planes = d[name] if isinstance(d[name], tuple) else (d[name],)
        for component, plane in enumerate(planes):
            assert not plane[0].count_nonzero()
            for j, block in enumerate([2 * group + 5, 2 * group + 6]):
                assert torch.all(plane[block] == 1 + 2 * layer + component + j)
