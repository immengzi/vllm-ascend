# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Use the frozen stock Pull path. Only device allocation, data and the SDK execution are on CPU."""

import ctypes
from unittest.mock import patch

import pytest
import torch

from tests.ut.core import test_dsv41_compact_pages as pages
from tests.ut.core import test_dsv41_compact_ring_handoff as handoff
from tests.ut.core.dsv41_request_state_fixture import Cache, make_leases
from vllm_ascend.models.deepseek_v41 import state_kernels

flash = pages.flash
cpu_ring_kernels = handoff.cpu_ring_kernels
MIB = 1024**2


def endpoint(config, specs, counts, rank=0, tp=2):
    c = Cache(config, specs, *counts)
    worker = pages.register(config, c.cfg, c.registration, tp, rank)
    assert all(scale == 1 for scales in worker.transfer_metadata.block_size_scales for scale in scales)
    assert all(
        part.shape[0] == c.slots[c.layout[name][0]] for name, v in c.compute.items() for part in handoff.cache_parts(v)
    )
    assert all(
        t.untyped_storage().data_ptr() == c.slab.untyped_storage().data_ptr()
        for views in (c.compute, c.registration)
        for v in views.values()
        for t in handoff.cache_parts(v)
    )
    h, s, k = counts
    assert c.slab.numel() == (h + 1) * 369280 + (s + 1) * 5242880 + (k + 1) * 393216
    assert c.slab.untyped_storage().nbytes() == c.slab.numel() + 2 * MIB
    return c, worker


def ranges(p, d, pids, dids):
    # Derive exact (destination, source, length) pairs from the small model views,
    # independent of all transport metadata and registration views. Merge only
    # immediately adjacent records with the same translation on both peers.
    expected = []
    for name, (group, _, _, _) in p.layout.items():
        for src, dst in zip(handoff.cache_parts(p.compute[name]), handoff.cache_parts(d.compute[name]), strict=True):
            for pi, di in zip(pids[group], dids[group], strict=True):
                a, b = src[pi], dst[di]
                assert a.is_contiguous() and b.is_contiguous()
                expected.append((b.data_ptr(), a.data_ptr(), a.numel() * a.element_size()))
    merged = []
    for dst, src, size in sorted(expected):
        if merged:
            a, b, c = merged[-1]
            assert a + c <= dst
            if (a + c, b + c) == (dst, src):
                merged[-1] = (a, b, c + size)
                continue
        merged.append((dst, src, size))
    return merged


@pytest.mark.parametrize("n", [128, 129, 130, 257])
def test_slot3_to_slot7(flash, cpu_ring_kernels, n):
    config, specs = flash
    p, pro = endpoint(config, specs, (47, 5, 4), tp=4)
    d, con = endpoint(config, dict(reversed(specs.items())), (53, 9, 8), rank=1)
    for arena, count in ((p, 2), (d, 6)):
        for i in range(count):
            assert arena.requests.admit(f"neighbor-{i}", 2 if i == 0 else 1) is not None
    target = p.requests.admit("target", 1)
    h = (n - 2) // 128 + 1
    if h > 1:
        assert p.requests.extend(target, h - 1)
    incoming = d.requests.admit("target", h)
    pt, pids = p.requests.begin_read(target)
    dt, dids = d.requests.begin_read(incoming)
    assert (pids[1:], dids[1:]) == (([3], [3]), ([7], [7]))
    assert pids[0][0] != 3 and dids[0][0] != 7
    rng = torch.Generator().manual_seed(4100 + n)
    p.slab.random_(0, 127, generator=rng)
    d.slab.fill_(0x55)
    references = {}
    for name, (group, *_) in p.layout.items():
        if group == 1:
            rows = torch.randn(n, 512, dtype=torch.bfloat16, generator=rng)
            references[name] = rows
            for start, end in ((0, 63), (63, n - 1)):
                state_kernels.commit(
                    p.compute[name],
                    rows[start:end],
                    torch.tensor([3]),
                    torch.tensor([0, end - start]),
                    torch.tensor([end]),
                )
    original_p = p.slab.clone()
    exact = ranges(p, d, pids, dids)
    original_copy = ctypes.memmove
    copied = []

    def checked_copy(dst, src, size):
        assert any(a <= dst and dst + size <= a + c and src - b == dst - a for a, b, c in exact)
        copied.append((dst, dst + size))
        return original_copy(dst, src, size)

    with patch.object(handoff.ctypes, "memmove", checked_copy):
        total = handoff.stock_read_batch([pro] * 4, con, {"target": (pids, dids, n)})
    expected_bytes = h * 369280 + 43 * 131072
    assert total == expected_bytes == sum(c for _, _, c in exact)
    ordered = sorted(copied)
    assert all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:]))
    assert torch.equal(original_p, p.slab)
    outside = torch.ones(d.slab.numel(), dtype=torch.bool)
    for a, b, c in exact:
        off = a - d.slab.data_ptr()
        outside[off : off + c] = False
        start = b - p.slab.data_ptr()
        assert torch.equal(d.slab[off : off + c], original_p[start : start + c])
    assert torch.all(d.slab[outside] == 0x55)
    for name, rows in references.items():
        staged = torch.empty(128, 512, dtype=rows.dtype)
        args = (d.compute[name], rows[-1:], torch.tensor([7]), torch.tensor([0, 1]), torch.tensor([n]))
        state_kernels.assemble(*args, torch.tensor([0]), staged, 1)
        assert torch.equal(staged, rows[-128:])
        state_kernels.commit(*args)
        assert torch.equal(d.compute[name][7, (n - 1) % 128, 0], rows[-1])
    assert torch.all(d.slab[outside] == 0x55)
    p.requests.complete(target, pt)
    d.requests.complete(incoming, dt)
    print(
        f"N={n} src=3 dst=7 read_bytes={total} exact_windows={len(references)} "
        f"P_storage={p.slab.untyped_storage().nbytes()} D_storage={d.slab.untyped_storage().nbytes()}"
    )


def test_actual_footprint(flash):
    cfg, specs = flash

    def measure(h, w, k):
        arena, _ = endpoint(cfg, specs, (h, w, k))
        return arena.slab.untyped_storage().nbytes()

    base = measure(40, 4, 4)
    diffs = [measure(41, 4, 4) - base, measure(40, 5, 4) - base, measure(40, 4, 5) - base]
    assert diffs == [369280, 5242880, 393216]
    print(f"extra_storage_bytes history,SWA,C2={diffs}")


@pytest.mark.parametrize("counts", [(1, 2, 2), (5, 1, 2), (5, 2, 1)])
def test_separate_admission_and_dma_retirement(counts):
    h, s, c = counts
    mgr = make_leases(h, s, c)
    first = mgr.admit("r", 1)
    t0, _ = mgr.begin_read(first)
    t1, _ = mgr.begin_read(first)
    mgr.retire(first)
    before = mgr.history.get_num_free_blocks()
    assert mgr.admit("new", 1) is None
    mgr.complete(first, t0)
    with pytest.raises(ValueError, match="unmatched"):
        mgr.complete(first, t0)
    assert mgr.admit("new", 1) is None and mgr.history.get_num_free_blocks() == before
    mgr.complete(first, t1)
    second = mgr.admit("r", 1)
    assert second is not None
    with pytest.raises(ValueError, match="stale"):
        mgr.begin_read(first)
    mgr.retire(second)
    assert mgr.history.get_num_free_blocks() == h and len(mgr.windows) == s and len(mgr.compressors) == c


def test_independent_group_ids_and_tp_sources(flash):
    config, specs = flash
    producers = [endpoint(config, specs, (29, 5, 4), rank, 4) for rank in range(4)]
    decoders = [endpoint(config, specs, (37, 9, 8), rank, 2) for rank in range(2)]
    # One scheduler per DP replica. All TP workers consume its group-local IDs.
    pm, dm = producers[0][0].requests, decoders[0][0].requests
    for manager, steps in ((pm, 7), (dm, 15)):
        for i in range(steps):
            manager.retire(manager.admit(str(i), 1))
    jobs = {}
    leases = []
    for i, n in enumerate((128, 129, 257, 130)):
        key = str(i)
        h = (n - 2) // 128 + 1
        a, b = pm.admit(key, h), dm.admit(key, h)
        pt, pids = pm.begin_read(a)
        dt, dids = dm.begin_read(b)
        leases.append((a, b, pt, dt))
        jobs[key] = (pids, dids, n)
    pi, di, _ = jobs["0"]
    assert (pi[1], pi[2], di[1], di[2]) == ([3], [4], [7], [8])
    assert len({pi[0][0], pi[1][0], pi[2][0]}) == len({di[0][0], di[1][0], di[2][0]}) == 3
    for rank, (p, _) in enumerate(producers):
        p.slab.random_(0, 127, generator=torch.Generator().manual_seed(2300 + rank))
    for d, con in decoders:
        d.slab.fill_(0x5A)
        # Stock Pull chooses one of the four equivalent P replicas per request.
        # Distinct CPU buffers and byte patterns expose wrong source selection.
        expected = []
        for i, (pids, dids, n) in enumerate(jobs.values()):
            expected.extend((i, a, b, c) for a, b, c in ranges(producers[i][0], d, pids, dids))
        physical_copy = ctypes.memmove

        def bounded(dst, src, size, expected=expected, physical_copy=physical_copy):
            assert any(a <= dst and dst + size <= a + c and src - b == dst - a for _, a, b, c in expected)
            return physical_copy(dst, src, size)

        with patch.object(handoff.ctypes, "memmove", bounded):
            copied = handoff.stock_read_batch([worker for _, worker in producers], con, jobs)
        assert copied == sum(c for _, _, _, c in expected) == 24760064
        outside = torch.ones(d.slab.numel(), dtype=torch.bool)
        for rank, a, b, c in expected:
            dest = a - d.slab.data_ptr()
            source = producers[rank][0].slab
            start = b - source.data_ptr()
            assert outside[dest : dest + c].all()
            outside[dest : dest + c] = False
            assert torch.equal(source[start : start + c], d.slab[dest : dest + c])
        assert (d.slab[outside] == 0x5A).all()
        print(f"D_rank={con.tp_rank} source_ranks=0,1,2,3 group_ids={pi}->{di} checked_bytes={copied}")
    for a, b, pt, dt in leases:
        for mgr, lease, ticket in ((pm, a, pt), (dm, b, dt)):
            mgr.retire(lease)
            mgr.complete(lease, ticket)
    assert pm.history.get_num_free_blocks() == 29 and dm.history.get_num_free_blocks() == 37


@pytest.mark.parametrize("group", [0, 1, 2])
def test_zero_only_new_slots_in_named_group(flash, group):
    cfg, specs = flash
    cache = Cache(cfg, specs, 11, 4, 3)
    cache.slab.fill_(85)
    expected = cache.slab.clone()
    for name, (owner, _, _, _) in cache.layout.items():
        if owner == group:
            for view in handoff.cache_parts(cache.compute[name]):
                for slot in [1, 3]:
                    part = view[slot]
                    start = part.data_ptr() - cache.slab.data_ptr()
                    expected[start : start + part.numel() * part.element_size()].zero_()
    cache.zero_slots(group, [1, 3])
    assert torch.equal(cache.slab, expected)
    for bad in (0, cache.slots[group], -1):
        with pytest.raises(ValueError, match="capacity"):
            cache.zero_slots(group, [1, bad])
    assert torch.equal(cache.slab, expected)
