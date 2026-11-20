# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Exercise DSV4.1 scheduling and worker geometry on CPU."""

import copy
import pickle
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import CircularBufferSpec
from vllm.v1.request import Request

from vllm_ascend.core.kv_cache_interface import (
    KV_CACHE_BUFFER_ALIGNMENT,
    get_cache_group_capacity,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.models.deepseek_v41 import request_state
from vllm_ascend.models.deepseek_v41.cache_config import (
    REQUEST_STATE_POOLS_CONFIG_KEY,
    SINGLE_POOL_SWA_CONFIG_KEY,
    get_deepseek_v41_kv_cache_config,
    get_deepseek_v41_pool_bytes_per_block,
    get_request_state_groups,
    get_request_state_pd_alignment_bytes,
    get_request_state_pool_plan,
    is_request_state_cache,
    make_index_cache_spec,
    make_mla_cache_spec,
    make_swa_cache_spec,
    materialize_request_state_pools,
    uses_single_pool_windows,
)
from vllm_ascend.models.deepseek_v41.window_state import (
    RequestWindowMetadataBuilder,
    bind_window_staging_workspace,
)
from vllm_ascend.patch.platform import patch_kv_cache_coordinator, patch_request_state_manager


@pytest.fixture
def geometry():
    config = SimpleNamespace(
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        model_config=SimpleNamespace(
            max_model_len=512,
            architecture="DeepseekV41ForCausalLM",
            hf_config=SimpleNamespace(model_type="deepseek_v41"),
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=512, max_num_seqs=2, disable_hybrid_kv_cache_manager=False
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            enable_extensible_kv_cache=False,
            swa_bounded_replay=False,
            prefix_cache_retention_interval=None,
        ),
        speculative_config=None,
        additional_config={REQUEST_STATE_POOLS_CONFIG_KEY: True},
        num_prefill_lookahead_tokens=1,
        kv_transfer_config=None,
    )
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    with patch("vllm_ascend.models.deepseek_v41.cache_config.uses_a5_packed_cache", return_value=False):
        specs = {}
        for i in range(40):
            specs[f"m.layers.{i}.swa_cache"] = make_swa_cache_spec(
                block_size=128, window_size=128, head_size=512, dtype=torch.bfloat16, cache_dtype="auto"
            )
        for i in range(4):
            ratio = 1 if i == 0 else 2
            specs[f"m.layers.{i}.long_kv_cache"] = make_mla_cache_spec(
                block_size=128, head_size=512, compress_ratio=ratio
            )
            specs[f"m.layers.{i}.indexer.k_cache"] = make_index_cache_spec(
                block_size=128, head_size=128, compress_ratio=ratio
            )
        for i in range(3):
            specs[f"m.layers.{i}.compressor.state_cache"] = CircularBufferSpec(
                block_size=32, num_kv_heads=1, head_size=1024, dtype=torch.float32, head_size_v=0
            )
        groups = get_request_state_groups(config, specs)
        yield config, specs, groups


def test_fixed_buffers_survive_rank_replan_and_config_serialization(geometry):
    config, _, groups = geometry
    wide = materialize_request_state_pools(config, groups, 12)
    narrow = materialize_request_state_pools(config, groups, 7)
    assert len(narrow.kv_cache_tensors) == 51
    assert [get_cache_group_capacity(narrow, i) for i in range(3)] == [7, 3, 3]
    assert narrow.request_state_plan.budget.history_page_bytes == 369280
    assert wide.kv_cache_tensors[8:] == narrow.kv_cache_tensors[8:]
    assert generate_scheduler_kv_cache_config([narrow]).request_state_plan == narrow.request_state_plan
    assert pickle.loads(pickle.dumps(narrow)) == copy.deepcopy(narrow)
    assert get_request_state_pd_alignment_bytes(config, groups) == 0
    config.kv_transfer_config = object()
    assert get_request_state_pd_alignment_bytes(config, groups) == 51 * KV_CACHE_BUFFER_ALIGNMENT
    assert get_request_state_pool_plan(config, groups, 7) == narrow.request_state_plan


@pytest.mark.parametrize("role", [None, "kv_producer", "kv_consumer"])
def test_engine_uses_request_state_pools_when_enabled(geometry, role):
    from vllm.v1.core.kv_cache_planning import get_global_kv_cache_groups

    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder

    config, specs, expected = geometry
    if role is not None:
        config.kv_transfer_config = SimpleNamespace(
            kv_connector="MooncakePullConnector",
            kv_role=role,
            kv_load_failure_policy="fail",
            get_from_extra_config={"state_model_id": "test-weights"}.get,
        )
    with patch(
        "vllm.v1.core.kv_cache_config_builder.get_kv_cache_config_builder",
        return_value=AscendKVCacheConfigBuilder(),
    ):
        assert get_global_kv_cache_groups(config, [specs]) == expected


@pytest.mark.parametrize(
    ("section", "flag", "value"),
    [
        (None, "use_v2_model_runner", False),
        (None, "speculative_config", object()),
        ("cache_config", "enable_prefix_caching", True),
        ("cache_config", "swa_bounded_replay", True),
        ("parallel_config", "prefill_context_parallel_size", 2),
        ("model_config", "architecture", "Qwen3ForCausalLM"),
        ("kv_transfer_config", "kv_connector", "MooncakeHybridConnector"),
        ("kv_transfer_config", "get_from_extra_config", {}.get),
    ],
)
def test_ineligible_opt_in_is_rejected(geometry, section, flag, value):
    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
    from vllm_ascend.models.deepseek_v41.cache_config import group_cache_specs, make_cache_groups

    config, specs, _ = geometry
    if section == "kv_transfer_config":
        config.kv_transfer_config = SimpleNamespace(
            kv_connector="MooncakePullConnector",
            kv_role="kv_consumer",
            kv_load_failure_policy="fail",
            get_from_extra_config={"state_model_id": "test-weights"}.get,
        )
    setattr(getattr(config, section) if section is not None else config, flag, value)
    with pytest.raises(ValueError):
        AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs)
    config.additional_config[REQUEST_STATE_POOLS_CONFIG_KEY] = False
    groups = AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs)
    assert groups == make_cache_groups(group_cache_specs(specs))
    assert not is_request_state_cache(groups)


def test_eligible_geometry_defaults_to_existing_layout(geometry):
    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
    from vllm_ascend.models.deepseek_v41.cache_config import group_cache_specs, make_cache_groups

    config, specs, _ = geometry
    config.additional_config = {}
    assert AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs) == make_cache_groups(
        group_cache_specs(specs)
    )
    config.additional_config[REQUEST_STATE_POOLS_CONFIG_KEY] = "false"
    with pytest.raises(ValueError, match="must be a boolean"):
        AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs)


def make_manager(config, groups):
    worker = materialize_request_state_pools(config, groups, 7)
    scheduler = generate_scheduler_kv_cache_config([worker])
    with patch(
        "vllm.v1.core.kv_cache_manager.get_kv_cache_coordinator",
        patch_kv_cache_coordinator.get_kv_cache_coordinator,
    ):
        return KVCacheManager(scheduler, 512, 128, 128, enable_caching=False)


def make_request(rid):
    return Request(rid, list(range(257)), SamplingParams(max_tokens=4), pooling_params=None)


def available(manager):
    return tuple(m.block_pool.get_num_free_blocks() for m in manager.coordinator.single_type_managers)


def test_joint_admission_external_read_and_reuse(geometry):
    config, _, groups = geometry
    manager = make_manager(config, groups)
    assert available(manager) == (6, 2, 2)
    p, d, waiting = [make_request(rid) for rid in ("local", "remote", "waiting")]
    slots = patch_request_state_manager.allocate_slots
    assert slots(manager, p, 129) is not None
    remote = slots(
        manager, d, 0, num_external_computed_tokens=256, delay_cache_blocks=True, skip_zeroing_group_ids=(0, 1, 2)
    )
    assert remote is not None
    assert tuple(map(len, manager.get_blocks(d.request_id).get_block_ids())) == (2, 1, 1)
    assert available(manager) == (2, 0, 0)
    assert slots(manager, waiting, 1) is None
    assert available(manager) == (2, 0, 0)
    manager.free(p)
    assert slots(manager, waiting, 1) is not None
    manager.free(waiting)
    manager.free(d)
    assert available(manager) == (6, 2, 2)


def test_partial_reservation_is_rolled_back(geometry):
    config, _, groups = geometry
    manager = make_manager(config, groups)
    request = make_request("rollback")
    compressor = manager.coordinator.single_type_managers[-1]
    claim = compressor.allocate_new_blocks

    def fail_after_claim(*args):
        claim(*args)
        raise RuntimeError("injected after final claim")

    with (
        patch.object(compressor, "allocate_new_blocks", side_effect=fail_after_claim),
        pytest.raises(RuntimeError, match="injected"),
    ):
        patch_request_state_manager.allocate_slots(manager, request, 129)
    assert available(manager) == (6, 2, 2)
    assert manager.get_blocks(request.request_id).get_block_ids() == ([], [], [])
    assert patch_request_state_manager.allocate_slots(manager, request, 129) is not None
    manager.free(request)
    assert available(manager) == (6, 2, 2)


def test_mixed_requests_keep_their_slots_until_released(geometry):
    config, _, groups = geometry
    manager = make_manager(config, groups)
    prefill, decode, next_prefill = [make_request(rid) for rid in ("prefill", "decode", "next")]
    slots = patch_request_state_manager.allocate_slots
    assert slots(manager, prefill, 128) is not None
    assert slots(manager, decode, 256) is not None
    prefill.num_computed_tokens = 128
    decode.num_computed_tokens = 256
    before = manager.get_blocks(decode.request_id).get_block_ids()
    assert slots(manager, prefill, 128) is not None
    assert slots(manager, decode, 1) is not None
    after = manager.get_blocks(decode.request_id).get_block_ids()
    assert before[1:] == after[1:]
    assert after[0][:-1] == before[0] and len(after[0]) == 3
    assert available(manager) == (1, 0, 0)
    assert slots(manager, next_prefill, 256) is None
    assert available(manager) == (1, 0, 0)

    previous_slots = manager.get_blocks(prefill.request_id).get_block_ids()[1:]
    manager.free(prefill)
    assert slots(manager, next_prefill, 256) is not None
    assert manager.get_blocks(next_prefill.request_id).get_block_ids()[1:] == previous_slots
    assert manager.get_blocks(decode.request_id).get_block_ids() == after
    manager.free(next_prefill)
    manager.free(decode)
    assert available(manager) == (6, 2, 2)


def test_staging_uses_one_workspace_and_pads_only_staging_rows(geometry):
    config, _, groups = geometry
    worker = materialize_request_state_pools(config, groups, 7)
    ring = torch.empty((3, 128, 1, 512), dtype=torch.bfloat16)
    layers = {name: SimpleNamespace(kv_cache=[ring]) for name in groups[1].layer_names}
    workspace = bind_window_staging_workspace(worker, layers)
    assert all(layer.request_state_workspace is workspace for layer in layers.values())
    builder = RequestWindowMetadataBuilder(torch.zeros(2, dtype=torch.int32), worker.staging_rows)
    common = SimpleNamespace(
        block_table_tensor=torch.tensor([[1], [2]], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 3]),
        seq_lens_cpu=torch.tensor([129, 0]),
    )
    result = SimpleNamespace(num_reqs=2, seq_lens=common.seq_lens_cpu, max_query_len=2)
    metadata = builder.build(common, result)
    assert metadata.state_ids.tolist() == [1, 2]
    assert metadata.lengths.tolist() == [129, 0]
    assert metadata.rows_used == 256


def test_decode_staging_reuses_graph_buffers_with_new_slot_assignment():
    seq_lens = torch.ones(4, dtype=torch.int32)
    block_table = torch.zeros((4, 1), dtype=torch.int32)
    starts = torch.arange(5, dtype=torch.int32)
    builder = RequestWindowMetadataBuilder(seq_lens, 4 * 128)
    common = SimpleNamespace(block_table_tensor=block_table, query_start_loc=starts)
    result = SimpleNamespace(num_reqs=4, seq_lens=seq_lens, max_query_len=1)
    captured = builder.build(common, result)
    assert captured.lengths.tolist() == [1, 1, 1, 1]

    seq_lens[:] = torch.tensor([9, 129, 0, 0])
    block_table[:, 0] = torch.tensor([1, 2, 0, 0])
    replay = builder.build(common, result)
    for name in ("lengths", "state_ids", "row_starts", "block_table"):
        assert getattr(captured, name).data_ptr() == getattr(replay, name).data_ptr()
    assert captured.state_ids.tolist() == [1, 2, 0, 0]
    assert captured.lengths.tolist() == [9, 128, 0, 0]
    assert captured.row_starts.tolist() == [0, 128, 256, 384]


@pytest.mark.parametrize(
    ("state_id", "seq_len", "expected_offsets"),
    [(0, 1, []), (0, 0, []), (2, 0, []), (-1, 1, []), (2, 129, [256])],
)
def test_swa_ring_commit_masks_reserved_and_padded_rows(monkeypatch, state_id, seq_len, expected_offsets):
    from vllm_ascend.models.deepseek_v41 import state_kernels

    # Run the actual mask and address expressions on CPU tensors. The first
    # four loads are the request's query bounds, sequence length and slot ID.
    controls = [torch.tensor(x) for x in (0, 1, seq_len, state_id)]
    tl = SimpleNamespace(
        int64=torch.int64,
        arange=torch.arange,
        program_id=lambda _axis: 0,
        load=Mock(side_effect=[*controls, torch.zeros(16, 1)]),
        minimum=lambda value, bound: value.clamp_max(bound),
        where=torch.where,
        store=Mock(),
    )
    monkeypatch.setattr(state_kernels, "tl", tl)
    kernel = getattr(state_kernels._commit, "fn", state_kernels._commit)
    kernel(0, 0, 0, 0, 0, WIDTH=1, WINDOW=128, ROWS=16, PAGE_STRIDE=128)
    args, kwargs = tl.store.call_args
    assert args[0][kwargs["mask"]].tolist() == expected_offsets


def test_staged_attention_keeps_registered_ring_bound(monkeypatch):
    import sys

    ring, projected, staging = (object() for _ in range(3))
    layer = SimpleNamespace(kv_cache=[ring])
    attn = SimpleNamespace(dsa_attn=SimpleNamespace(swa_cache_layer=layer))
    state = SimpleNamespace(state_ids=object())
    swa = SimpleNamespace(request_state=state, query_start_loc=object(), seq_lens=object(), max_query_len=1)
    metadata = SimpleNamespace(swa=swa)
    stages = Mock()
    monkeypatch.setattr(request_state, "_stage", Mock(return_value=(ring, projected, staging)))
    monkeypatch.setattr(request_state, "_release_current_kv", stages.release)
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.models.deepseek_v41.state_kernels", SimpleNamespace(commit=stages.commit)
    )

    def forward(attn_arg, q, step_metadata, indices, *, source_cache, staged_window):
        assert attn_arg is attn and layer.kv_cache == [ring]
        assert step_metadata is metadata and staged_window is staging
        return q

    stages.forward.side_effect = forward
    impl = SimpleNamespace(_forward_attention_impl=stages.forward)
    q = object()
    assert request_state.forward_request_window(impl, attn, q, metadata, None) is q
    assert [entry[0] for entry in stages.mock_calls] == ["forward", "commit", "release"]
    stages.commit.assert_called_once_with(ring, projected, state.state_ids, swa.query_start_loc, swa.seq_lens)
    assert layer.kv_cache == [ring]


def test_core_resolves_rank_minimum_after_reserving_fixed_pools(geometry):
    from vllm.v1.core import kv_cache_planning, kv_cache_utils

    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
    from vllm_ascend.patch.platform import patch_kv_cache_utils

    config, specs, groups = geometry
    config.model_config.original_max_model_len = config.model_config.max_model_len
    config.model_config.hf_config = SimpleNamespace(model_type="deepseek_v41")
    config.parallel_config.tensor_parallel_size = 2
    config.attention_config = SimpleNamespace(hisparse_config=None)
    config.cache_config.num_gpu_blocks_override = None
    config.kv_transfer_config = object()
    budget = get_request_state_pool_plan(config, groups, 1).budget
    fixed = budget.fixed_bytes + get_request_state_pd_alignment_bytes(config, groups)
    budgets = [fixed + n * budget.history_page_bytes for n in (11, 7)]
    with patch(
        "vllm.v1.core.kv_cache_config_builder.get_kv_cache_config_builder",
        return_value=AscendKVCacheConfigBuilder(),
    ):
        configs = kv_cache_planning.get_kv_cache_configs(config, [specs, specs], budgets, groups)
        assert len(configs) == 2 and [cfg.num_blocks for cfg in configs] == [7, 7]
        assert [[get_cache_group_capacity(cfg, i) for i in range(3)] for cfg in configs] == [[7, 3, 3]] * 2
        for worker in configs:
            scheduler = generate_scheduler_kv_cache_config([worker])
            assert kv_cache_utils.get_max_concurrency_for_kv_cache_config(config, worker) == 1.5
            assert kv_cache_utils.get_max_concurrency_for_kv_cache_config(config, scheduler) == 1.5
        config.cache_config.num_gpu_blocks_override = 8
        with pytest.raises(ValueError, match="only .* bytes are available"):
            patch_kv_cache_utils._ascend_get_kv_cache_configs(config, [specs, specs], budgets, groups)


def test_mooncake_registers_each_backing_independently_of_history_capacity(geometry):
    from vllm_ascend.core.kv_cache_interface import get_storage_block_size
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.dsv41 import pool_capacity
    from vllm_ascend.distributed.kv_transfer.utils.utils import align_memory
    from vllm_ascend.models.deepseek_v41.cache_config import unwrap

    config, _, groups = geometry
    config.parallel_config.cp_kv_cache_interleave_size = 1
    config.additional_config = {}
    config.model_config.hf_config = SimpleNamespace(to_dict=lambda: {"model_type": "deepseek_v41"})
    config.model_config.dtype = torch.bfloat16
    config.model_config.quantization = "ascend"
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="MooncakePullConnector",
        kv_role="kv_producer",
        kv_load_failure_policy="fail",
        get_from_extra_config={"state_model_id": "test-weights"}.get,
    )
    cache_config = materialize_request_state_pools(config, groups, 7)
    specs = {name: spec for group in groups for name, spec in unwrap(group.kv_cache_spec).items()}
    caches = {}
    for allocation in cache_config.kv_cache_tensors:
        name = allocation.layers[0]
        spec = specs[name]
        capacity = pool_capacity(cache_config, name)
        rows = get_storage_block_size(spec)
        backing = align_memory(
            torch.empty(allocation.size + KV_CACHE_BUFFER_ALIGNMENT, dtype=torch.int8),
            KV_CACHE_BUFFER_ALIGNMENT,
        )[: allocation.size]
        keys = backing.view(spec.dtype).as_strided(
            (capacity, rows, 1, spec.head_size),
            (allocation.block_stride // spec.dtype.itemsize, spec.head_size, spec.head_size, 1),
            backing.storage_offset() // spec.dtype.itemsize,
        )
        if name.endswith(".indexer.k_cache"):
            scales = backing.view(spec.scale_dtype).as_strided(
                (capacity, rows, 1, 1),
                (allocation.block_stride // 2, 1, 1, 1),
                (backing.storage_offset() + rows * spec.head_size) // 2,
            )
            caches[name] = (keys, scales)
        else:
            caches[name] = keys

    worker = object.__new__(MooncakeBaseConnectorWorker)
    worker.vllm_config, worker.kv_cache_config = config, cache_config
    worker.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    worker.block_size, worker.tp_size, worker.tp_rank, worker.pcp_rank = 128, 4, 0, 0
    worker.engine_id, worker.te_rpc_port, worker.side_channel_host, worker.handshake_port = (
        "p",
        10000,
        "127.0.0.1",
        10020,
    )
    with patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker.global_te") as engine:
        worker.register_kv_caches(caches)
    addresses, sizes = engine.register_buffer.call_args.args
    assert len(addresses) == len(sizes) == 51
    assert all(address % KV_CACHE_BUFFER_ALIGNMENT == 0 for address in addresses)
    assert sizes == [entry.size for entry in cache_config.kv_cache_tensors]
    planes = worker.get_handshake_metadata().state_layout["planes"]
    assert len(planes) == 55
    assert {plane["capacity"] for plane in planes if plane["group"] == 0} == {7}
    assert {plane["capacity"] for plane in planes if plane["group"] != 0} == {3}
    assert {plane["rows"] for plane in planes if plane["group"] == 0} == {64, 128}


def test_single_pool_window_allocation_and_reuse(geometry):
    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
    from vllm_ascend.core.single_type_kv_cache_manager import AscendCircularBufferManager

    config, specs, _ = geometry
    config.cache_config.num_gpu_blocks_override = None
    config.additional_config.clear()
    builder = AscendKVCacheConfigBuilder()
    native = builder.get_kv_cache_groups(config, specs)
    config.additional_config[SINGLE_POOL_SWA_CONFIG_KEY] = True
    groups = builder.get_kv_cache_groups(config, specs)
    assert uses_single_pool_windows(groups) and not is_request_state_cache(groups)
    assert [g.layer_names for g in groups] == [g.layer_names for g in native]
    stride = get_deepseek_v41_pool_bytes_per_block(groups)
    assert stride == get_deepseek_v41_pool_bytes_per_block(native) == 540928
    # Null page + three history + one compressor + ten SWA leaves no spare IDs.
    worker = get_deepseek_v41_kv_cache_config(config, groups, 15 * stride)
    assert worker.request_state_plan is None and worker.num_blocks == 15
    assert worker.staging_rows == 1024
    for group in groups[2:]:
        ring = group.kv_cache_spec.first_spec
        assert ring.sliding_window == 128 and ring.num_retained_tokens == 0
        assert ring.max_num_blocks_per_req(config, 700_001) == 1
        assert not ring.uses_slot_mapping
    assert [t.size for t in worker.kv_cache_tensors] == [
        t.size for t in get_deepseek_v41_kv_cache_config(config, native, 15 * stride).kv_cache_tensors
    ]
    scheduler = generate_scheduler_kv_cache_config([worker])
    with patch(
        "vllm.v1.core.kv_cache_manager.get_kv_cache_coordinator", patch_kv_cache_coordinator.get_kv_cache_coordinator
    ):
        manager = KVCacheManager(scheduler, 512, 128, 128, enable_caching=False)
    managers = manager.coordinator.single_type_managers
    assert len(managers) == 12 and all(m.block_pool is manager.coordinator.block_pool for m in managers)
    assert all(isinstance(m, AscendCircularBufferManager) for m in managers[2:])
    first, second = make_request("first"), make_request("after-cancel")
    assert manager.allocate_slots(first, 128) is not None
    initial = manager.get_blocks(first.request_id).get_block_ids()
    assert [len(ids) for ids in initial] == [1] * 12
    first.num_computed_tokens = 128
    assert manager.allocate_slots(first, 129) is not None
    final = manager.get_blocks(first.request_id).get_block_ids()
    assert len(final[0]) == 3 and final[1:] == initial[1:]
    assert manager.allocate_slots(second, 1) is None
    manager.free(first)
    assert manager.allocate_slots(second, 257) is not None
    recycled = manager.get_blocks(second.request_id).get_block_ids()
    assert set(sum((list(ids) for ids in recycled), [])) == set(range(1, 15))
    manager.free(second)
    assert manager.coordinator.block_pool.get_num_free_blocks() == 14


def test_single_pool_rejects_incompatible_modes(geometry):
    config, specs, _ = geometry
    config.additional_config[SINGLE_POOL_SWA_CONFIG_KEY] = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        get_request_state_groups(config, specs)
    config.additional_config[REQUEST_STATE_POOLS_CONFIG_KEY] = False
    config.kv_transfer_config = object()
    with pytest.raises(ValueError, match="KV transfer"):
        get_request_state_groups(config, specs)


def test_single_pool_reserves_workspace_once_per_rank(geometry):
    from vllm_ascend.patch.platform import patch_kv_cache_utils as planning

    config, specs, _ = geometry
    config.additional_config = {SINGLE_POOL_SWA_CONFIG_KEY: True}
    config.cache_config.num_gpu_blocks_override = None
    groups = get_request_state_groups(config, specs)
    page = get_deepseek_v41_pool_bytes_per_block(groups)
    workspace = 1024 * 1024 + 2**20
    available = [workspace + 20 * page, workspace + 24 * page]
    with patch.object(planning, "_orig_get_kv_cache_configs") as delegate:
        planning._ascend_get_kv_cache_configs(config, [specs, specs], available, groups)
        delegate.assert_called_once_with(config, [specs, specs], [20 * page, 24 * page], groups)
        config.cache_config.num_gpu_blocks_override = 21
        with pytest.raises(ValueError, match="profiled KV memory budget"):
            planning._ascend_get_kv_cache_configs(config, [specs, specs], available, groups)
        assert delegate.call_count == 1


def test_single_pool_worker_replanning_through_core(geometry):
    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
    from vllm_ascend.patch.platform import patch_kv_cache_utils as planning

    config, specs, _ = geometry
    config.additional_config = {SINGLE_POOL_SWA_CONFIG_KEY: True}
    config.parallel_config.tensor_parallel_size = 2
    config.cache_config.num_gpu_blocks_override = None
    config.model_config.original_max_model_len = 512
    config.attention_config = SimpleNamespace(hisparse_config=None)
    builder = AscendKVCacheConfigBuilder()
    groups = builder.get_kv_cache_groups(config, specs)
    page = get_deepseek_v41_pool_bytes_per_block(groups)
    workspace = 2 * 1024**2
    with patch("vllm.v1.core.kv_cache_config_builder.get_kv_cache_config_builder", return_value=builder):
        workers = planning._ascend_get_kv_cache_configs(
            config, [specs, specs], [workspace + 20 * page, workspace + 24 * page], groups
        )
    for worker in workers:
        assert worker.num_blocks == 20 and worker.staging_rows == 1024
        assert worker.request_state_plan is None
        assert [tensor.block_stride for tensor in worker.kv_cache_tensors] == [147712, 131072, 131072, 131072]
        assert sum(tensor.size for tensor in worker.kv_cache_tensors) == 20 * page


def test_single_pool_staging_reads_each_swa_groups_own_table(geometry):
    config, specs, _ = geometry
    config.cache_config.num_gpu_blocks_override = None
    config.additional_config = {SINGLE_POOL_SWA_CONFIG_KEY: True}
    groups = get_request_state_groups(config, specs)
    worker = get_deepseek_v41_kv_cache_config(config, groups, get_deepseek_v41_pool_bytes_per_block(groups) * 12)
    layers = {}
    for tensor in worker.kv_cache_tensors:
        storage = torch.empty(tensor.size // 2, dtype=torch.bfloat16)
        ring = storage.as_strided((12, 128, 1, 512), (tensor.block_stride // 2, 512, 512, 1))
        for name in tensor.layers:
            if name.endswith(".swa_cache"):
                layers[name] = SimpleNamespace(kv_cache=[ring])
    workspace = bind_window_staging_workspace(worker, layers)
    assert len(layers) == 40 and all(layer.request_state_workspace is workspace for layer in layers.values())
    assert workspace.tensor.shape == (8, 128, 1, 512)
    result = SimpleNamespace(num_reqs=1, max_query_len=1, seq_lens=torch.tensor([4097]))
    for block_id, group in enumerate(groups[2:], 1):
        table = torch.tensor([[block_id]], dtype=torch.int32)
        metadata = RequestWindowMetadataBuilder(result.seq_lens, worker.staging_rows).build(
            SimpleNamespace(block_table_tensor=table, query_start_loc=torch.tensor([0, 1])), result
        )
        assert metadata.state_ids.data_ptr() == table.data_ptr()
        assert metadata.lengths.tolist() == [128]
        assert all(layers[name].kv_cache[0].shape[0] == worker.num_blocks for name in group.layer_names)
