# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pool ownership, admission rollback and lifecycle tests on CPU."""

from collections import defaultdict, deque
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core import request_state_manager as mod
from vllm_ascend.core.kv_cache_interface import AscendKVCacheConfig, CachePoolPlan, RequestStatePlan
from vllm_ascend.patch.platform import patch_kv_cache_coordinator as coordinator_adapter
from vllm_ascend.patch.platform import patch_request_state_manager as adapter


class Pool:
    def __init__(self, capacity, enable_caching=False, block_size=128):
        self.capacity = capacity
        self.enable_caching = enable_caching
        self.free = deque(range(1, capacity))
        self.returned = []

    def get_num_free_blocks(self):
        return len(self.free)

    def get_new_blocks(self, count):
        if count > len(self.free):
            raise ValueError("pool exhausted")
        return [self.free.popleft() for _ in range(count)]

    def free_blocks(self, blocks):
        blocks = list(blocks)
        assert 0 not in blocks
        assert not (set(blocks) & set(self.free))
        self.free.extend(blocks)
        self.returned.extend(blocks)


class Manager:
    def __init__(self, pool, *, fixed=False, group_id=0):
        self.block_pool = pool
        self.req_to_blocks = defaultdict(list)
        self.num_cached_block = {}
        self.new_block_ids = []
        self.fixed = fixed
        self.kv_cache_group_id = group_id
        self.block_size = 128
        self.fail_after_append = False

    def get_num_blocks_to_allocate(self, request_id, num_tokens, *args, **kwargs):
        return max(0, (num_tokens + 127) // 128 - len(self.req_to_blocks.get(request_id, ())))

    def allocate_new_blocks(self, request_id, num_tokens, *args, **kwargs):
        target = 1 if self.fixed else (num_tokens + 127) // 128
        count = max(0, target - len(self.req_to_blocks[request_id]))
        blocks = self.block_pool.get_new_blocks(count)
        self.req_to_blocks[request_id].extend(blocks)
        self.new_block_ids.extend(blocks)
        if self.fail_after_append:
            raise RuntimeError("injected after append")
        return blocks

    def free(self, request_id):
        self.block_pool.free_blocks(reversed(self.req_to_blocks.pop(request_id, ())))
        self.num_cached_block.pop(request_id, None)

    def add_local_computed_blocks(self, request_id, blocks, local_tokens, external_tokens):
        assert not blocks and local_tokens == 0
        self.num_cached_block[request_id] = 0

    def allocate_external_computed_blocks(self, request_id, local_tokens, external_tokens):
        self.allocate_new_blocks(request_id, local_tokens + external_tokens)


class Fixed(Manager, mod.FixedStateManager):
    # Execute the production fixed-slot demand override in the inherited
    # upstream coordinator's scalar history gate.
    get_num_blocks_to_allocate = mod.FixedStateManager.get_num_blocks_to_allocate


class Coordinator:
    get_num_blocks_to_allocate = KVCacheCoordinator.get_num_blocks_to_allocate
    allocate_new_computed_blocks = KVCacheCoordinator.allocate_new_computed_blocks
    allocate_new_blocks = KVCacheCoordinator.allocate_new_blocks
    free = KVCacheCoordinator.free

    def __init__(self, managers):
        self.single_type_managers = tuple(managers)
        self.remove_skipped_blocks = Mock()
        self.cache_blocks = Mock()


def config():
    spec = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
    return AscendKVCacheConfig(
        num_blocks=5,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([name], spec) for name in ("history", "window", "compressor")],
        request_state_plan=RequestStatePlan(
            (
                CachePoolPlan("history", 5, (0,), "token_block"),
                CachePoolPlan("window", 2, (1,), "request_slot"),
                CachePoolPlan("compressor", 3, (2,), "request_slot"),
            ),
            128,
            131072,
        ),
    )


def managers():
    return [Manager(Pool(5)), Fixed(Pool(2), fixed=True, group_id=1), Fixed(Pool(3), fixed=True, group_id=2)]


def runtime(items):
    return SimpleNamespace(
        kv_cache_config=config(),
        coordinator=Coordinator(items),
        block_pool=items[0].block_pool,
        empty_kv_cache_blocks=KVCacheBlocks(([], [], [])),
        retained_hit_group_ids=(),
        max_model_len=4096,
        watermark_blocks=0,
        create_kv_cache_blocks=KVCacheBlocks,
    )


def request(name="r", computed=0):
    return SimpleNamespace(
        request_id=name,
        num_computed_tokens=computed,
        num_in_flight_tokens=0,
        num_prompt_tokens=256,
        num_tokens=512,
    )


def allocate(owner, req, count=128):
    # Exercise the platform bridge and the actual upstream allocate_slots.
    return adapter.allocate_slots(owner, req, count, has_scheduled_reqs=False)


def test_complete_cancel_and_reuse_keep_independent_pool_counts():
    items = managers()
    owner = runtime(items)
    initial = [m.block_pool.get_num_free_blocks() for m in items]
    result = allocate(owner, request())
    assert result is not None
    first_states = [m.req_to_blocks["r"][:] for m in items[1:]]
    assert [m.block_pool.get_num_free_blocks() for m in items] == [3, 0, 1]
    assert allocate(owner, request(computed=128)) is not None
    assert [m.req_to_blocks["r"] for m in items[1:]] == first_states
    assert len(items[0].req_to_blocks["r"]) == 2
    KVCacheManager.free(owner, request())
    assert [m.block_pool.get_num_free_blocks() for m in items] == initial
    assert all("r" not in m.req_to_blocks for m in items)
    assert allocate(owner, request("next")) is not None
    assert items[1].req_to_blocks["next"] == first_states[0]
    # Cancellation before allocation, completion, and repeated cleanup all
    # use upstream free(), without a new independently maintained free path.
    KVCacheManager.free(owner, request("never-admitted"))
    KVCacheManager.free(owner, request("next"))
    KVCacheManager.free(owner, request("next"))
    assert [m.block_pool.get_num_free_blocks() for m in items] == initial


@pytest.mark.parametrize("index", [0, 1, 2])
def test_any_exhausted_pool_defers_without_partial_reservation(index):
    items = managers()
    owner = runtime(items)
    items[index].block_pool.get_new_blocks(items[index].block_pool.get_num_free_blocks())
    before = [m.block_pool.get_num_free_blocks() for m in items]
    assert allocate(owner, request()) is None
    assert [m.block_pool.get_num_free_blocks() for m in items] == before
    assert all("r" not in m.req_to_blocks for m in items)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_exception_at_each_allocation_stage_rolls_back_and_allows_retry(index):
    items = managers()
    owner = runtime(items)
    before = [m.block_pool.get_num_free_blocks() for m in items]
    items[index].fail_after_append = True
    with pytest.raises(RuntimeError, match="injected"):
        allocate(owner, request())
    assert [m.block_pool.get_num_free_blocks() for m in items] == before
    assert all("r" not in m.req_to_blocks and not m.new_block_ids for m in items)
    items[index].fail_after_append = False
    assert allocate(owner, request()) is not None


@pytest.mark.parametrize("failure", ["none", "exception"])
def test_failed_attempt_preserves_prior_resources_and_unrelated_request(failure):
    items = managers()
    items[0].allocate_new_blocks("r", 128)
    items[2].allocate_new_blocks("other", 128)
    items[0].num_cached_block["r"] = 7
    before = [
        (dict(m.req_to_blocks), dict(m.num_cached_block), m.new_block_ids[:], m.block_pool.get_num_free_blocks())
        for m in items
    ]

    def attempt():
        for m in items:
            m.allocate_new_blocks("r", 256)
            m.num_cached_block["r"] = 100
        if failure == "exception":
            raise RuntimeError("injected")
        return None

    # Freeze list values before the callback appends to them.
    before = [({k: v[:] for k, v in blocks.items()}, cached, ids, free) for blocks, cached, ids, free in before]
    if failure == "exception":
        with pytest.raises(RuntimeError, match="injected"):
            mod.reserve_append_only(items, "r", attempt)
    else:
        assert mod.reserve_append_only(items, "r", attempt) is None
    for m, (blocks, cached, ids, free) in zip(items, before):
        assert dict(m.req_to_blocks) == blocks
        assert m.num_cached_block == cached
        assert m.new_block_ids == ids
        assert m.block_pool.get_num_free_blocks() == free


def test_falsey_success_is_not_rejection():
    items = managers()
    result = []

    def attempt():
        items[0].allocate_new_blocks("r", 128)
        return result

    assert mod.reserve_append_only(items, "r", attempt) is result
    assert len(items[0].req_to_blocks["r"]) == 1


def test_non_append_mutation_fails_before_any_pool_is_freed():
    items = managers()
    items[0].allocate_new_blocks("r", 128)

    def attempt():
        items[0].req_to_blocks["r"][0] = 99
        items[1].allocate_new_blocks("r", 128)
        return None

    with pytest.raises(RuntimeError, match="append-only"):
        mod.reserve_append_only(items, "r", attempt)
    assert all(not m.block_pool.returned for m in items)


def test_resident_state_fast_path_does_not_snapshot_long_history():
    items = managers()
    for m in items[1:]:
        m.allocate_new_blocks("r", 128)
    callback = Mock(return_value="existing")
    with patch.object(mod, "reserve_append_only", side_effect=AssertionError("unexpected snapshot")):
        assert mod.allocate_request_slots(items, "r", callback) == "existing"
    callback.assert_called_once_with()


def test_native_bridge_preserves_arguments_and_result():
    native = KVCacheConfig(num_blocks=5, kv_cache_groups=[], kv_cache_tensors=[])
    owner = SimpleNamespace(kv_cache_config=native)
    req = request()
    callback = Mock(return_value="native")
    with patch.object(adapter, "_original_allocate_slots", callback):
        assert adapter.allocate_slots(owner, req, 4, reserved_blocks=2) == "native"
    callback.assert_called_once_with(owner, req, 4, reserved_blocks=2)


@pytest.mark.parametrize("uniform", [False, True])
def test_factory_creates_each_manager_once_on_declared_pool(uniform):
    cfg = config()
    history_spec = cfg.kv_cache_groups[0].kv_cache_spec
    if uniform:
        for group in cfg.kv_cache_groups:
            group.kv_cache_spec = UniformTypeKVCacheSpecs.from_specs({group.layer_names[0]: group.kv_cache_spec})
    history_pool = Pool(5)
    history = Manager(history_pool)
    history_factory = Mock(return_value=history)

    class FixedFactory(Manager):
        def __init__(self, spec, **kwargs):
            super().__init__(kwargs["block_pool"], fixed=True, group_id=kwargs["kv_cache_group_id"])
            self.kwargs = kwargs

    with (
        patch.object(mod, "BlockPool", Pool),
        patch.object(mod, "FullAttentionManager", Manager),
        patch.object(mod, "FixedStateManager", FixedFactory),
        patch.object(mod, "get_manager_for_kv_cache_spec", history_factory),
    ):
        result = mod.create_request_state_managers(
            cfg,
            history_pool,
            scheduler_block_size=128,
            dcp_world_size=1,
            pcp_world_size=1,
            max_model_len=4096,
            max_in_flight_tokens=8192,
        )
    assert result[0] is history
    assert [m.block_pool.get_num_free_blocks() for m in result] == [4, 1, 2]
    assert len({id(m.block_pool) for m in result}) == 3
    assert [m.kv_cache_group_id for m in result] == [0, 1, 2]
    assert all(not m.kwargs["needs_kv_cache_zeroing"] for m in result[1:])
    history_factory.assert_called_once()
    assert history_factory.call_args.kwargs["kv_cache_spec"] is history_spec


def test_factory_rejects_prefix_caching():
    with pytest.raises(ValueError, match="prefix caching"):
        mod.create_request_state_managers(
            config(),
            Pool(5, True),
            scheduler_block_size=128,
            dcp_world_size=1,
            pcp_world_size=1,
            max_model_len=4096,
            max_in_flight_tokens=8192,
        )


def test_coordinator_factory_initializes_declared_pools_directly():
    cfg = config()
    # Distinct specs keep the test independent of DSV4.1 name heuristics.
    for i, group in enumerate(cfg.kv_cache_groups):
        group.kv_cache_spec = replace(group.kv_cache_spec, head_size=8 + i)
    history_pool = Pool(5)
    items = managers()
    items[0].block_pool = history_pool
    with (
        patch.object(coordinator_adapter, "BlockPool", return_value=history_pool) as pool_factory,
        patch.object(coordinator_adapter, "create_request_state_managers", return_value=tuple(items)) as factory,
        patch.object(coordinator_adapter, "get_manager_for_kv_cache_spec") as native_factory,
        patch.object(coordinator_adapter, "_orig_get_kv_cache_coordinator") as upstream_factory,
    ):
        result = coordinator_adapter.get_kv_cache_coordinator(
            cfg, max_model_len=4096, enable_caching=False, scheduler_block_size=128, hash_block_size=128
        )
    assert isinstance(result, coordinator_adapter.AscendHybridKVCacheCoordinator)
    assert result.single_type_managers == tuple(items)
    assert result.block_pool is history_pool
    assert result.group_block_sizes == (128, 128, 128)
    pool_factory.assert_called_once()
    factory.assert_called_once_with(
        cfg,
        history_pool,
        scheduler_block_size=128,
        dcp_world_size=1,
        pcp_world_size=1,
        max_model_len=4096,
        max_in_flight_tokens=4096,
    )
    native_factory.assert_not_called()
    upstream_factory.assert_not_called()


def test_native_coordinator_factory_still_delegates_to_upstream():
    cfg = config()
    native = KVCacheConfig(num_blocks=5, kv_cache_groups=cfg.kv_cache_groups[:1], kv_cache_tensors=[])
    with patch.object(coordinator_adapter, "_orig_get_kv_cache_coordinator", return_value="native") as factory:
        assert coordinator_adapter.get_kv_cache_coordinator(native, 4096, enable_caching=False) == "native"
    factory.assert_called_once()


@pytest.mark.parametrize("fail_index", [None, 0, 1, 2])
def test_external_reservation_uses_same_transaction(fail_index):
    items = managers()
    owner = runtime(items)
    initial = [m.block_pool.get_num_free_blocks() for m in items]
    if fail_index is not None:
        items[fail_index].fail_after_append = True
        with pytest.raises(RuntimeError, match="injected"):
            adapter.allocate_slots(owner, request(), 0, num_external_computed_tokens=256, has_scheduled_reqs=False)
        assert all("r" not in m.req_to_blocks and "r" not in m.num_cached_block for m in items)
        assert [m.block_pool.get_num_free_blocks() for m in items] == initial
    else:
        assert (
            adapter.allocate_slots(owner, request(), 0, num_external_computed_tokens=256, has_scheduled_reqs=False)
            is not None
        )
        assert [len(m.req_to_blocks["r"]) for m in items] == [2, 1, 1]
        KVCacheManager.free(owner, request())
        assert [m.block_pool.get_num_free_blocks() for m in items] == initial
