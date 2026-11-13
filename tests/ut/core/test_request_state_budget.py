# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full rank planning must reserve fixed state/workspace exactly once."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.core import kv_cache_utils

from tests.ut.core.test_request_state_plan import model_fixture
from vllm_ascend.core.kv_cache_interface import RequestStateMemoryBudget
from vllm_ascend.models.deepseek_v41 import cache_config as planner
from vllm_ascend.patch.platform import patch_kv_cache_utils as adapter


def rank_fixture(length=128, override=None, auto=False, dcp=1):
    cfg, groups = model_fixture(override=override)
    cfg.model_config = SimpleNamespace(max_model_len=length, original_max_model_len=-1 if auto else length)
    cfg.attention_config = SimpleNamespace(hisparse_config=None)
    cfg.parallel_config.decode_context_parallel_size = dcp
    return cfg, planner.all_specs(groups), planner.get_deepseek_v41_memory_budget(cfg, groups)


def plan_ranks(cfg, specs, memory):
    return adapter._ascend_get_kv_cache_configs(cfg, [copy.deepcopy(specs) for _ in memory], memory)


@pytest.mark.parametrize("offset,capacity", [(-1, 99), (0, 100), (1, 100)])
def test_full_budget_round_trip(offset, capacity):
    _, _, budget = rank_fixture()
    assert budget.history_page_bytes == 369280
    assert budget.request_state_bytes == 49 * (40 * 131072 + 3 * 131072)
    assert budget.workspace_bytes == 21 * 2**20
    assert budget.history_capacity(budget.bytes_for_history_blocks(100) + offset) == capacity
    assert budget.history_capacity(budget.fixed_bytes - 1) == 0


@pytest.mark.parametrize("args", [(0, 1, 1), (1, -1, 0), (1, 0, -1)])
def test_invalid_byte_geometry(args):
    with pytest.raises(ValueError):
        RequestStateMemoryBudget(*args)


def test_negative_history_count():
    with pytest.raises(ValueError):
        RequestStateMemoryBudget(1, 0, 0).bytes_for_history_blocks(-1)


def test_unequal_rank_memory_preserves_state_and_avoids_history_only_replan():
    cfg, specs, budget = rank_fixture()
    memory = [budget.bytes_for_history_blocks(n) + extra for n, extra in ((100, 0), (101, 0), (200, 17), (100, 99))]
    before = copy.deepcopy((cfg, specs, memory))
    with patch.object(
        kv_cache_utils, "get_kv_cache_config_from_groups", wraps=kv_cache_utils.get_kv_cache_config_from_groups
    ) as build:
        results = plan_ranks(cfg, specs, memory)
    assert build.call_count == 4  # No second history-only rank rebuild.
    assert [r.num_blocks for r in results] == [100] * 4
    assert (cfg, specs, memory) == before
    for result, available in zip(results, memory):
        result.request_state_plan.validate(result)
        assert [pool.capacity for pool in result.request_state_plan.pools] == [100, 49, 49]
        assert (
            sum(t.size for t in result.kv_cache_tensors) + result.request_state_plan.staging_allowance_bytes
            <= available
        )
        assert result.kv_transfer_config is None


@pytest.mark.parametrize("override", [2, 100])
def test_override_uses_full_budget_without_mutating_config(override):
    cfg, specs, budget = rank_fixture(override=override)
    cache_config = cfg.cache_config
    result = plan_ranks(cfg, specs, [budget.bytes_for_history_blocks(110), budget.bytes_for_history_blocks(200)])
    assert [r.num_blocks for r in result] == [override, override]
    assert cfg.cache_config is cache_config
    assert cfg.cache_config.num_gpu_blocks_override == override


@pytest.mark.parametrize("override", [-1, 0, 1, 101])
def test_override_must_fit_smallest_worker(override):
    cfg, specs, budget = rank_fixture(override=override)
    with pytest.raises(ValueError, match="override"):
        plan_ranks(cfg, specs, [budget.bytes_for_history_blocks(100), budget.bytes_for_history_blocks(200)])
    assert cfg.cache_config.num_gpu_blocks_override == override


@pytest.mark.parametrize("dcp", [1, 2])
def test_admission_reserves_one_history_null_and_fixed_slots(dcp):
    cfg, specs, budget = rank_fixture(length=99 * 128 * dcp, dcp=dcp)
    available = budget.bytes_for_history_blocks(100)
    assert plan_ranks(cfg, specs, [available])[0].num_blocks == 100
    with pytest.raises(ValueError, match="max seq len"):
        plan_ranks(cfg, specs, [available - 1])
    cfg.model_config.max_model_len += 1
    with pytest.raises(ValueError, match="max seq len"):
        plan_ranks(cfg, specs, [available])


@pytest.mark.parametrize("override", [None, 80])
def test_auto_fit_uses_effective_history_capacity_and_updates_original_model(override):
    cfg, specs, budget = rank_fixture(length=100000, auto=True, override=override)
    results = plan_ranks(cfg, specs, [budget.bytes_for_history_blocks(100), budget.bytes_for_history_blocks(200)])
    count = override or 100
    assert cfg.model_config.max_model_len == (count - 1) * 128
    assert cfg.cache_config.num_gpu_blocks_override == override
    assert [r.num_blocks for r in results] == [count, count]


def test_override_admission_cannot_use_unallocated_physical_budget():
    cfg, specs, budget = rank_fixture(length=100 * 128, override=100)
    with pytest.raises(ValueError, match="max seq len"):
        plan_ranks(cfg, specs, [budget.bytes_for_history_blocks(200)])
    assert cfg.model_config.max_model_len == 100 * 128


@pytest.mark.parametrize("auto", [False, True])
def test_no_usable_history_page_fails_before_materialization(auto):
    cfg, specs, budget = rank_fixture(auto=auto)
    with patch.object(
        kv_cache_utils, "get_kv_cache_config_from_groups", wraps=kv_cache_utils.get_kv_cache_config_from_groups
    ) as build:
        with pytest.raises(ValueError):
            plan_ranks(cfg, specs, [budget.bytes_for_history_blocks(2) - 1])
        build.assert_not_called()


@pytest.mark.parametrize("fault", ["pp", "specs", "missing_budget"])
def test_unsupported_rank_layout_fails_explicitly(fault):
    cfg, specs, budget = rank_fixture()
    ranks = [specs, copy.deepcopy(specs)]
    memory = [budget.bytes_for_history_blocks(100)] * 2
    if fault == "pp":
        cfg.parallel_config.pipeline_parallel_size = 2
    elif fault == "specs":
        ranks[1].pop(next(iter(specs)))
    else:
        memory.pop()
    with pytest.raises(ValueError):
        adapter._ascend_get_kv_cache_configs(cfg, ranks, memory)


def test_other_models_delegate_with_original_arguments():
    cfg, specs, memory = object(), [{"native": object()}], [123]
    expected = object()
    with patch.object(adapter, "_orig_get_kv_cache_configs", return_value=expected) as original:
        assert adapter._ascend_get_kv_cache_configs(cfg, specs, memory) is expected
    assert original.call_args.args[0] is cfg
    assert original.call_args.args[1] is specs
    assert original.call_args.args[2] is memory
