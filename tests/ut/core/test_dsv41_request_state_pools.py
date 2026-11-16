# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Exercise DSV4.1 scheduling and worker geometry on CPU."""

import copy
import pickle
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

from vllm_ascend.core.kv_cache_interface import (
    KV_CACHE_BUFFER_ALIGNMENT,
    get_cache_group_capacity,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.models.deepseek_v41.cache_config import (
    get_request_state_groups,
    get_request_state_pd_alignment_bytes,
    get_request_state_pool_plan,
    is_request_state_cache,
    make_index_cache_spec,
    make_mla_cache_spec,
    make_swa_cache_spec,
    materialize_request_state_pools,
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
        model_config=SimpleNamespace(max_model_len=512, hf_config=SimpleNamespace(model_type="deepseek_v41")),
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
        additional_config={},
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
def test_engine_uses_request_state_pools_by_default(geometry, role):
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
        ("kv_transfer_config", "kv_connector", "MooncakeHybridConnector"),
        ("kv_transfer_config", "get_from_extra_config", {}.get),
    ],
)
def test_ineligible_configurations_use_existing_layout(geometry, section, flag, value):
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
    groups = AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs)
    assert groups == make_cache_groups(group_cache_specs(specs))
    assert not is_request_state_cache(groups)


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
