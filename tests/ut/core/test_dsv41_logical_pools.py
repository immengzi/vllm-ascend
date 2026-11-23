# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure actual backing bytes and exercise independent upper-pool quotas."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_planning import get_global_kv_cache_groups, get_kv_cache_configs
from vllm.v1.request import Request

from tests.ut.core import test_dsv41_compact_pages as pages
from tests.ut.core.test_dsv41_compact_ring_handoff import cache_parts, manager_for, stock_read_batch
from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.models.deepseek_v41.compact_cache import BANK_BYTES, C2_SLOTS_KEY, LOGICAL_CONFIG_KEY, SWA_SLOTS_KEY

flash = pages.flash


def configure(config, swa, c2, length):
    config.additional_config = {LOGICAL_CONFIG_KEY: True, SWA_SLOTS_KEY: swa, C2_SLOTS_KEY: c2}
    config.model_config.max_model_len = length


def plan(config, specs, budget):
    from vllm_ascend.patch.platform import patch_kv_cache_utils  # noqa: F401

    with patch(
        "vllm.v1.core.kv_cache_config_builder.get_kv_cache_config_builder", return_value=AscendKVCacheConfigBuilder()
    ):
        groups = get_global_kv_cache_groups(config, [specs, specs])
        return get_kv_cache_configs(config, [specs, specs], [budget + 2 * BANK_BYTES, budget], groups)


@pytest.mark.parametrize("hist,swa,c2,tiles", [(6, 2, 3, 108), (6, 3, 3, 148), (6, 2, 4, 111), (7, 2, 3, 111)])
def test_actual_storage(flash, hist, swa, c2, tiles):
    config, specs = flash
    configure(config, swa, c2, 128 * hist)
    for cfg in plan(config, specs, tiles * BANK_BYTES):
        assert cfg.num_blocks == tiles
        assert len(cfg.kv_cache_tensors) == 1
        assert cfg.kv_cache_tensors[0].size == tiles * BANK_BYTES
    cfg, views = pages.worker_views(config, specs, cfg.num_blocks)
    storages = {
        t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
        for view in views.values()
        for t in cache_parts(view)
    }
    assert list(storages.values()) == [tiles * BANK_BYTES + 2 * 1024**2]
    worker = pages.register(config, cfg, views)
    assert all(s == 1 for scales in worker.transfer_metadata.block_size_scales for s in scales)
    managers = manager_for(cfg).coordinator.single_type_managers
    assert [m.logical_limit for m in managers] == [hist] * 3 + [c2] * 3 + [swa] * 40
    print(f"H={hist} SWA={swa} C2={c2} registered={tiles * BANK_BYTES} allocated={next(iter(storages.values()))}")


def test_auto_fit_reserves_state_and_null(flash):
    config, specs = flash
    configure(config, 2, 3, 2048)
    config.device_config = SimpleNamespace(device_type="cpu")
    config.model_config.original_max_model_len = -1
    plan(config, specs, 108 * BANK_BYTES)
    assert config.model_config.max_model_len == 6 * 128


@pytest.mark.parametrize("hist,swa,c2", [(3, 1, 2), (3, 2, 1), (1, 2, 2)])
def test_quota_atomicity_and_deferred_free(flash, hist, swa, c2):
    config, specs = flash
    configure(config, swa, c2, 128)
    cfg, _ = pages.worker_views(config, specs, 1 + 3 * hist + 40 * swa + 3 * c2)
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


def test_four_producer_sources_both_decode_ranks(flash):
    config, specs = flash
    configure(config, 4, 4, 512)

    def workers(count, tp_size):
        result = []
        for rank in range(tp_size):
            cfg, views = pages.worker_views(config, specs, count)
            result.append(pages.register(config, cfg, views, tp_size, rank))
        return result

    producers = workers(197, 4)
    configure(config, 5, 5, 512)
    consumers = workers(246, 2)
    pm = manager_for(producers[0].kv_cache_config)
    dm = manager_for(consumers[0].kv_cache_config)
    held = Request("held", [0], SamplingParams(max_tokens=1), pooling_params=None)
    assert dm.allocate_slots(held, 1) is not None
    jobs = {}
    for index, n in enumerate((128, 129, 257, 130)):
        key = str(index)
        req = Request(key, [0] * n, SamplingParams(max_tokens=1), pooling_params=None)
        assert pm.allocate_slots(req, n - 1) is not None
        assert dm.allocate_slots(req, 0, num_external_computed_tokens=n - 1, delay_cache_blocks=True) is not None
        jobs[key] = pm.get_blocks(key).get_block_ids(), dm.get_blocks(key).get_block_ids(), n
    for rank, worker in enumerate(producers):
        for key, (pids, _, _) in jobs.items():
            for layer, values in worker.kv_caches.items():
                group = worker.layer_name_to_group_index[layer]
                for plane in cache_parts(values):
                    for page in pids[group]:
                        plane[page].fill_(1 + rank * 10 + int(key))
    for consumer in consumers:
        total = stock_read_batch(producers, consumer, jobs)
        assert total == 2 * 6005376 + 2 * 6374656
        for key, (_, dids, _) in jobs.items():
            # Native Pull round-robins interchangeable replicas by batch order.
            source_rank = int(key)
            for layer, values in consumer.kv_caches.items():
                group = consumer.layer_name_to_group_index[layer]
                for plane in cache_parts(values):
                    for page in dids[group]:
                        assert torch.all(plane[page] == 1 + source_rank * 10 + int(key))
        print(f"D{consumer.tp_rank} P_source_tags=[0,1,2,3] total_bytes={total}")


def test_same_budget_returns_unused_slots_to_history_at_startup(flash):
    config, specs = flash
    budget = 148 * BANK_BYTES
    for swa_slots, expected_history in ((3, 6), (2, 19)):
        configure(config, swa_slots, 3, 512)
        cfg, _ = plan(config, specs, budget)
        mgr = manager_for(cfg)
        assert [m.logical_limit for m in mgr.coordinator.single_type_managers[:3]] == [expected_history] * 3
        assert cfg.kv_cache_tensors[0].size == budget


@pytest.mark.parametrize("key,value", [(SWA_SLOTS_KEY, 0), (C2_SLOTS_KEY, False), (SWA_SLOTS_KEY, 1.5)])
def test_invalid_logical_slot_count(flash, key, value):
    config, specs = flash
    configure(config, 2, 3, 128)
    config.additional_config[key] = value
    with pytest.raises(ValueError, match=key):
        AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs)
