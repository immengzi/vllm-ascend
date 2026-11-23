# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU contract for compact fixed rings, native block ownership and stock Pull."""

import ctypes
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm.v1.request import Request

from tests.ut.core import test_dsv41_compact_pages as compact_tests
from vllm_ascend.attention.dsa_v41 import AscendDSAV41Impl
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import MooncakeBaseConnectorScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread
from vllm_ascend.models.deepseek_v41 import request_state, state_kernels
from vllm_ascend.models.deepseek_v41.compact_cache import RING_CONFIG_KEY, get_window_staging_rows
from vllm_ascend.models.deepseek_v41.window_state import RequestWindowMetadataBuilder, bind_window_staging_workspace

flash = compact_tests.flash


@dataclass
class Pointer:
    """A contiguous CPU tensor and offsets in elements, like a Triton pointer."""

    flat: torch.Tensor
    offset: object = 0

    def __add__(self, offset):
        return Pointer(self.flat, self.offset + offset)


class CpuTile:
    """Only interpret the memory/ALU primitives used by the two production kernels."""

    int64 = torch.int64
    arange = staticmethod(torch.arange)
    minimum = staticmethod(lambda x, bound: x.clamp_max(bound))
    where = staticmethod(torch.where)
    program_id = staticmethod(lambda axis: 0)

    @staticmethod
    def load(ptr, mask=None, other=0):
        if mask is None:
            return ptr.flat[ptr.offset]
        offsets, mask = torch.broadcast_tensors(ptr.offset, mask)
        values = torch.full_like(offsets, other, dtype=ptr.flat.dtype)
        values[mask] = ptr.flat[offsets[mask]]
        return values

    @staticmethod
    def store(ptr, value, mask):
        offsets, mask = torch.broadcast_tensors(ptr.offset, mask)
        ptr.flat[offsets[mask]] = value[mask]


class CpuLaunch:
    def __init__(self, kernel):
        self.body = getattr(kernel, "fn", kernel)

    def __getitem__(self, grid):
        def run(*args, **constants):
            assert all(x.is_contiguous() for x in args)
            pointers = [Pointer(x.view(-1)) for x in args]
            for i in range(grid[0]):
                for j in range(grid[1]):
                    with patch.object(CpuTile, "program_id", staticmethod((i, j).__getitem__)):
                        self.body(*pointers, **constants)

        return run


@pytest.fixture
def cpu_ring_kernels(monkeypatch):
    monkeypatch.setattr(state_kernels, "tl", CpuTile)
    monkeypatch.setattr(state_kernels, "triton", SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y))
    for name in ("_commit", "_assemble"):
        monkeypatch.setattr(state_kernels, name, CpuLaunch(getattr(state_kernels, name)))


def manager_for(cfg):
    from vllm_ascend.patch.platform import patch_kv_cache_coordinator

    with patch(
        "vllm.v1.core.kv_cache_manager.get_kv_cache_coordinator", patch_kv_cache_coordinator.get_kv_cache_coordinator
    ):
        manager = KVCacheManager(generate_scheduler_kv_cache_config([cfg]), 4096, 128, 128, enable_caching=False)
    pool = manager.coordinator.block_pool
    assert all(m.block_pool is pool for m in manager.coordinator.single_type_managers)
    return manager


def cache_parts(value):
    return value if isinstance(value, tuple) else (value,)


def stock_read(producer, consumer, pids, dids, n):
    remote, local = producer.transfer_metadata, consumer.transfer_metadata
    thread = object.__new__(MooncakePullRecvingThread)
    thread.tp_size, thread.tp_rank, thread.pcp_size, thread.pcp_rank, thread.dcp_size = 2, 1, 1, 0, 1
    thread.kv_cache_specs = consumer.kv_cache_specs
    thread.spec_indices = [consumer.layer_name_to_spec_index[name] for name in local.layer_names]
    for field in (
        "layer_names",
        "group_indices",
        "layer_block_sizes",
        "block_size_scales",
        "block_shapes",
        "block_strides",
        "block_lens",
        "kv_caches_base_addr",
    ):
        setattr(thread, field, getattr(local, field))
    peer = SimpleNamespace(
        **{
            key: getattr(remote, key)
            for key in (
                "group_indices",
                "block_shapes",
                "block_lens",
                "block_strides",
                "dcp_sharded",
                "layer_block_sizes",
                "block_size_scales",
                "indexer_replication_sizes",
                "cp_kv_cache_interleave_size",
            )
        },
        metadata_by_pcp_rank=[SimpleNamespace(metadata_by_tp_rank=[remote] * 4)],
    )
    pg = producer.kv_cache_config.kv_cache_groups
    adv = SimpleNamespace(
        group_unique_specs=[list(g.kv_cache_spec.kv_cache_specs.values()) for g in pg],
        group_block_size=[g.kv_cache_spec.block_size for g in pg],
        dcp_size=1,
    )
    advertised = MooncakeBaseConnectorScheduler._get_transfer_block_ids(adv, pids, n - 1)
    assert all(len(ids) == 1 for ids in advertised[1:])
    req = SimpleNamespace(
        local_block_ids=dids,
        local_full_block_ids=dids,
        remote_block_ids=advertised,
        local_num_prompt_tokens=n,
        remote_num_prompt_tokens=n - 1,
        num_computed_tokens=0,
    )
    pairs = [(i, remote.layer_names.index(name)) for i, name in enumerate(local.layer_names)]
    ranks = {
        (local_idx, remote_idx): thread._get_layer_remote_tp_rank_groups(
            local_idx, remote_idx, thread.kv_cache_specs[thread.spec_indices[local_idx]], peer, 1, 4, 1
        )
        for local_idx, remote_idx in pairs
    }
    buckets, _ = thread._build_transfer_block_buckets(peer, pairs, ranks, 1, 1, {"target": req}, {})
    local_addr, remote_addr, lengths = [], [], []
    for (pcp, tp), by_spec in buckets.items():
        for spec, layers in by_spec.items():
            thread._append_spec_transfer_addresses(
                spec, pcp, tp, 1, 4, 1, layers, peer, local_addr, remote_addr, lengths
            )
    for dst, src, size in zip(local_addr, remote_addr, lengths, strict=True):
        for worker, address in ((producer, src), (consumer, dst)):
            view = cache_parts(next(iter(worker.kv_caches.values())))[0]
            storage = view.untyped_storage()
            assert storage.data_ptr() <= address < address + size <= storage.data_ptr() + storage.nbytes()
        ctypes.memmove(dst, src, size)
    return sum(lengths)


def fixture_probabilities(windows, query, head):
    # Deterministic auxiliary vocabulary. Checks a numerical consumer of the window.
    batch = torch.stack(windows).float()
    weights = torch.softmax(batch @ query, dim=1)
    hidden = (weights.unsqueeze(-1) * batch).sum(dim=1).mean(dim=0)
    return torch.softmax(hidden @ head, dim=-1)


@pytest.mark.parametrize("n", [128, 129, 257])
def test_compact_ring_pd_boundary(flash, cpu_ring_kernels, n):
    config, specs = flash
    config.additional_config[RING_CONFIG_KEY] = True
    p_cfg, p = compact_tests.worker_views(config, specs, 73)
    d_cfg, d = compact_tests.worker_views(config, specs, 71)
    pm, dm = manager_for(p_cfg), manager_for(d_cfg)
    p_req = Request("producer", [0] * (n - 1), SamplingParams(max_tokens=1), pooling_params=None)
    d_req = Request("consumer", [0] * n, SamplingParams(max_tokens=1), pooling_params=None)
    held = Request("held", [0], SamplingParams(max_tokens=1), pooling_params=None)
    assert dm.allocate_slots(held, 1) is not None
    # Split P across a non-page-aligned chunk. Stable ring IDs must survive the split.
    assert pm.allocate_slots(p_req, 65) is not None
    before = pm.get_blocks(p_req.request_id).get_block_ids()
    p_req.num_computed_tokens = 65
    assert pm.allocate_slots(p_req, n - 1 - 65) is not None
    pids = pm.get_blocks(p_req.request_id).get_block_ids()
    assert before[1:] == pids[1:]
    assert dm.allocate_slots(d_req, 0, num_external_computed_tokens=n - 1, delay_cache_blocks=True) is not None
    dids = dm.get_blocks(d_req.request_id).get_block_ids()
    assert len(pids) == len(dids) == 16 and all(len(ids) == 1 for ids in (*pids[1:], *dids[1:]))
    assert all(a != b for a, b in zip(pids[1:], dids[1:]))
    pro, con = compact_tests.register(config, p_cfg, p), compact_tests.register(config, d_cfg, d)
    names = [name for name in specs if name.endswith(".swa_cache")]
    gen = torch.Generator().manual_seed(621)
    history = {name: torch.randn(n, 512, generator=gen, dtype=torch.bfloat16) for name in names}
    for name, rows in history.items():
        group = pro.layer_name_to_group_index[name]
        p[name][pids[group][0]].fill_(-300)  # Unwritten rows must never enter attention.
        for begin, end in [(0, 65), (65, n - 1)]:
            state_kernels.commit(
                p[name], rows[begin:end], torch.tensor(pids[group]), torch.tensor([0, end - begin]), torch.tensor([end])
            )
    for name in specs.keys() - history.keys():
        group = pro.layer_name_to_group_index[name]
        for component, plane in enumerate(cache_parts(p[name])):
            for j, block in enumerate(pids[group]):
                plane[block].fill_(3 * j + component + pro.transfer_metadata.layer_names.index(name) + 1)
    count = stock_read(pro, con, pids, dids, n)
    assert count == len(pids[0]) * 369280 + 43 * 128 * 1024
    for name in specs:
        group = pro.layer_name_to_group_index[name]
        for src, dst in zip(cache_parts(p[name]), cache_parts(d[name]), strict=True):
            for source_id, destination_id in zip(pids[group], dids[group], strict=True):
                assert torch.equal(src[source_id], dst[destination_id])
    d_req.num_computed_tokens = n - 1
    assert dm.allocate_slots(d_req, 1) is not None
    assert dm.get_blocks(d_req.request_id).get_block_ids()[1:] == dids[1:]
    layers = [
        SimpleNamespace(self_attn=SimpleNamespace(dsa_attn=SimpleNamespace(swa_cache_layer=SimpleNamespace())))
        for _ in names
    ]
    bind_window_staging_workspace(layers, config.scheduler_config, torch.device("cpu"))
    workspaces = [layer.self_attn.dsa_attn.swa_cache_layer.request_state_workspace for layer in layers]
    assert len({id(w) for w in workspaces}) == 1
    builder = RequestWindowMetadataBuilder(
        torch.zeros(1, dtype=torch.int32),
        get_window_staging_rows(config.scheduler_config.max_num_batched_tokens, config.scheduler_config.max_num_seqs),
    )
    observed = []
    for name, workspace in zip(names, workspaces, strict=True):
        rows = history[name]
        group = con.layer_name_to_group_index[name]
        block = dids[group][0]
        absolute = torch.arange(max(0, n - 1 - 128), n - 1)
        assert torch.equal(d[name][block, absolute % 128, 0], rows[absolute])
        qstart = torch.tensor([0, 1], dtype=torch.int32)
        seq = torch.tensor([n], dtype=torch.int32)
        state = builder.build(
            SimpleNamespace(block_table_tensor=torch.tensor([[block]]), query_start_loc=qstart),
            SimpleNamespace(num_reqs=1, max_query_len=1, seq_lens=seq),
        )
        layer = SimpleNamespace(kv_cache=[d[name]], request_state_workspace=workspace)
        attention = SimpleNamespace(
            dsa_attn=SimpleNamespace(swa_cache_layer=layer), _request_state_current_kv=rows[-1:]
        )
        swa = SimpleNamespace(
            request_state=state, query_start_loc=qstart, seq_lens=seq, num_actual_tokens=1, max_query_len=1
        )

        def read_staged(attn, q, step, indices, *, source_cache, staged_window):
            state = step.swa.request_state
            return staged_window[state.block_table[0, 0], : state.lengths.item(), 0].clone()

        # Device stream ownership and the fused attention call are the hardware boundaries.
        with patch.object(request_state, "_release_current_kv"):
            window = AscendDSAV41Impl._forward_attention(
                SimpleNamespace(_forward_attention_impl=read_staged), attention, None, SimpleNamespace(swa=swa), None
            )
        assert torch.equal(window, rows[-128:])
        observed.append(window)
        assert torch.equal(d[name][block, (n - 1) % 128, 0], rows[-1])
    q = torch.randn(512, generator=gen) / 512**0.5
    head = torch.randn(512, 64, generator=gen) / 512**0.5
    expected = fixture_probabilities([history[k][-128:] for k in names], q, head)
    actual = fixture_probabilities(observed, q, head)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Negative control: repeating a stale row must change the distribution.
    wrong = [w.clone() for w in observed]
    for w in wrong:
        w[-1] = w[0]
    stale_error = (fixture_probabilities(wrong, q, head) - expected).abs().max().item()
    assert stale_error > 1e-7
    print(
        f"N={n} transferred_bytes={count} exact_SWA_layers={len(observed)} "
        f"fixture_max_probability_error={(actual - expected).abs().max().item():.8g} "
        f"stale_row_control_error={stale_error:.8g}"
    )
    for manager, requests, cfg in [(pm, [p_req], p_cfg), (dm, [d_req, held], d_cfg)]:
        for req in requests:
            manager.free(req)
        assert manager.coordinator.block_pool.get_num_free_blocks() == cfg.num_blocks - 1


def test_dummy_commit_preserves_null_and_live_ring(cpu_ring_kernels):
    ring = torch.full((3, 128, 1, 512), -2.0, dtype=torch.bfloat16)
    current = torch.ones((1, 512), dtype=torch.bfloat16)
    state_kernels.commit(ring, current, torch.tensor([0]), torch.tensor([0, 1]), torch.tensor([1]))
    assert torch.all(ring == -2.0)


def test_ring_requires_compact_and_stock_pull(flash):
    from vllm_ascend.models.deepseek_v41.compact_cache import CONFIG_KEY, compact_groups_if_enabled

    config, specs = flash
    config.additional_config = {RING_CONFIG_KEY: True}
    with pytest.raises(ValueError, match=CONFIG_KEY):
        compact_groups_if_enabled(config, specs)
    config.additional_config[CONFIG_KEY] = True
    config.kv_transfer_config = SimpleNamespace(kv_connector="MooncakeHybridConnector")
    with pytest.raises(ValueError, match="MooncakePullConnector"):
        compact_groups_if_enabled(config, specs)
