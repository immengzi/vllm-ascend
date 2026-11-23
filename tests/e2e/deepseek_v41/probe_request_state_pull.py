# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Drive the frozen Pull registration, transfer and release code on dedicated NPUs.

The fixture supplies layer specs, rank identity and four deterministic requests.
The real stock worker generates the READ lists directly from live tensors.
"""

import argparse
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import msgspec
import torch
import torch_npu  # noqa: F401

from tests.ut.core.dsv41_request_state_fixture import Cache
from tests.ut.core.test_dsv41_compact_pages import flash
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import (
    MooncakeBaseConnectorScheduler,
    MooncakeMetadataThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker, global_te
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeTransferMetadata,
    MooncakeTransferMetadataGroups,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread
from vllm_ascend.models.deepseek_v41 import state_kernels

P_SIZE = (29, 5, 4)
D_SIZE = (37, 9, 8)
LENGTHS = (128, 129, 257, 130)
ENGINE = "request-state-producer"


def publish(file, data):
    tmp = file.with_suffix(".partial")
    tmp.write_bytes(data)
    tmp.rename(file)


def read(file):
    end = time.monotonic() + 180
    while not file.exists():
        if time.monotonic() > end:
            raise TimeoutError(str(file))
        time.sleep(0.2)
    return file.read_bytes()


def allocation(cache, warm):
    manager = cache.requests
    for i in range(warm):
        manager.retire(manager.admit(str(i), 1))
    jobs = []
    for i, n in enumerate(LENGTHS):
        lease = manager.admit(str(i), (n - 2) // 128 + 1)
        ticket, ids = manager.begin_read(lease)
        jobs.append((lease, ticket, ids))
    return jobs


def register(ip, rank, size, cache, cfg):
    w = object.__new__(MooncakeBaseConnectorWorker)
    w.vllm_config, w.kv_cache_config = cfg, cache.cfg
    w.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    w.block_size, w.tp_size, w.tp_rank, w.pcp_rank = 128, size, rank, 0
    w.engine_id, w.side_channel_host, w.handshake_port = ENGINE if size == 4 else "request-state-consumer", ip, 19760
    w.engine = global_te.get_transfer_engine(ip, None)
    w.te_rpc_port = w.engine.get_rpc_port()
    w.register_kv_caches(cache.registration)
    assert global_te.registered_buffers == [(cache.slab.data_ptr(), cache.slab.numel())]
    return w


def producer_pattern(cache, rank):
    return torch.randint(
        0, 127, (cache.slab.numel(),), dtype=torch.int8, generator=torch.Generator().manual_seed(2300 + rank)
    )


def rows(rank, index, layer):
    rng = torch.Generator().manual_seed(71000 + 1000 * rank + 100 * index + layer)
    return torch.randn((LENGTHS[index], 512), generator=rng).to(torch.bfloat16)


def vector(*values):
    return torch.tensor(values, dtype=torch.int32, device="npu")


def source_windows(cache, rank, ids, *, gpu):
    names = cache.cfg.kv_cache_groups[1].layer_names
    for i, (n, groups) in enumerate(zip(LENGTHS, ids, strict=True)):
        slot = groups[1][0]
        for layer, name in enumerate(names):
            values = rows(rank, i, layer)
            ring = cache.compute[name]
            if gpu:
                inp = values.to("npu")
                for a, b in ((0, 63), (63, n - 1)):
                    state_kernels.commit(ring, inp[a:b], vector(slot), vector(0, b - a), vector(b))
            else:
                positions = torch.arange(max(0, n - 1 - 128), n - 1)
                ring[slot, positions % 128, 0] = values[positions]


def finish_windows(cache, expected, jobs):
    for i, (n, (_, _, ids)) in enumerate(zip(LENGTHS, jobs, strict=True)):
        slot = ids[1][0]
        for layer, name in enumerate(cache.cfg.kv_cache_groups[1].layer_names):
            values = rows(i, i, layer)
            last = values[-1:].to("npu")
            ring = cache.compute[name]
            out = torch.empty((128, 512), dtype=torch.bfloat16, device="npu")
            state_kernels.assemble(ring, last, vector(slot), vector(0, 1), vector(n), vector(0), out, 1)
            assert torch.equal(out.cpu(), values[-128:])
            state_kernels.commit(ring, last, vector(slot), vector(0, 1), vector(n))
            expected.compute[name][slot, (n - 1) % 128, 0].copy_(values[-1])
    assert torch.equal(cache.slab.cpu(), expected.slab)


def schedule(cache, ids, n):
    groups = cache.cfg.kv_cache_groups
    scheduler = SimpleNamespace(
        group_unique_specs=[list(g.kv_cache_spec.kv_cache_specs.values()) for g in groups],
        group_block_size=[g.kv_cache_spec.block_size for g in groups],
        dcp_size=1,
    )
    return MooncakeBaseConnectorScheduler._get_transfer_block_ids(scheduler, ids, n - 1)


def receiving_thread(w):
    t = object.__new__(MooncakePullRecvingThread)
    t.tp_size, t.tp_rank, t.pcp_size, t.pcp_rank, t.dcp_size = 2, w.tp_rank, 1, 0, 1
    t.local_metadata, t.kv_cache_specs, t.engine = w.transfer_metadata, w.kv_cache_specs, w.engine
    t.spec_indices = [w.layer_name_to_spec_index[name] for name in t.local_metadata.layer_names]
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
        setattr(t, field, getattr(t.local_metadata, field))
    return t


def expected_data(cfg, specs, jobs, pids, count_jobs=4):
    """Build the full expected destination using only bounded compute views and independent source bytes."""
    src, dst = Cache(cfg, specs, *P_SIZE), Cache(cfg, specs, *D_SIZE)
    dst.slab.fill_(0x5A)
    copied = 0
    for rank, ((_, _, dids), sids) in enumerate(zip(jobs[:count_jobs], pids[:count_jobs], strict=True)):
        src.slab.copy_(producer_pattern(src, rank))
        source_windows(src, rank, pids, gpu=False)
        for name, (group, *_) in src.layout.items():
            sv, dv = src.compute[name], dst.compute[name]
            for s, d in zip(sv if isinstance(sv, tuple) else [sv], dv if isinstance(dv, tuple) else [dv], strict=True):
                for si, di in zip(sids[group], dids[group], strict=True):
                    d[di].view(torch.uint8).copy_(s[si].view(torch.uint8))
                    copied += s[si].numel() * s.element_size()
    assert copied == (24760064 if count_jobs == 4 else 6005376)
    return dst, copied


def main(a):
    torch.npu.set_device(a.rank)
    cfg, specs = flash.__wrapped__()
    is_p = a.role == "P"
    cache = Cache(cfg, specs, *(P_SIZE if is_p else D_SIZE), device="npu")
    jobs = allocation(cache, 7 if is_p else 15)
    # Retire the first request after issuing the read ticket. Other allocations
    # must never reclaim its group IDs until both D ranks have acknowledged.
    old, old_ticket, old_ids = jobs[0]
    cache.requests.retire(old)
    assert old_ticket in old.pending
    guard = cache.requests.admit("while-transfer-pending", 1)
    if guard is not None:
        guard_ticket, guard_ids = cache.requests.begin_read(guard)
        assert all(set(a).isdisjoint(b) for a, b in zip(old_ids, guard_ids, strict=True))
        cache.requests.retire(guard)
        cache.requests.complete(guard, guard_ticket)
    w = register(a.ip, a.rank, 4 if is_p else 2, cache, cfg)

    def f(name):
        return a.exchange / name

    if is_p:
        ids = [entry[2] for entry in jobs]
        cache.slab.copy_(producer_pattern(cache, a.rank))
        source_windows(cache, a.rank, ids, gpu=True)
        reference = Cache(cfg, specs, *P_SIZE)
        reference.slab.copy_(producer_pattern(cache, a.rank))
        source_windows(reference, a.rank, ids, gpu=False)
        assert torch.equal(cache.slab.cpu(), reference.slab)
        torch.npu.synchronize()
        publish(f(f"p{a.rank}.bin"), msgspec.msgpack.encode(w.transfer_metadata))
        if a.rank == 0:
            metas = {r: msgspec.msgpack.decode(read(f(f"p{r}.bin")), type=MooncakeTransferMetadata) for r in range(4)}
            metadata = MooncakeMetadataThread(
                a.ip, 19760, ENGINE, metas, 4, 1, 1, 1, False, threading.Event()
            ).encoded_metadata
            ids = [entry[2] for entry in jobs]
            advertised = [schedule(cache, si, n) for si, n in zip(ids, LENGTHS, strict=True)]
            publish(f("peer.bin"), msgspec.msgpack.encode((metadata, ids, advertised)))
        if a.partial:
            read(f("partial.bin"))
            for lease, ticket, _ in jobs:
                if not lease.retired:
                    cache.requests.retire(lease)
                assert ticket in lease.pending
            assert cache.requests.admit("after-first-d-ack", 1) is None
            assert all(cache.requests.live[lease.owner] is lease for lease, _, _ in jobs)
            print("P_QUARANTINE_PASS", a.rank, "pending=4, no slot reused", flush=True)
            publish(f(f"hold{a.rank}.bin"), b"retained")
            if a.rank == 0:
                for rank in range(4):
                    read(f(f"hold{rank}.bin"))
                publish(f("held.bin"), b"all four producer ranks retain their sources")
        read(f("done.bin"))
        assert torch.equal(cache.slab.cpu(), reference.slab)
    else:
        cache.slab.fill_(0x5A)
        torch.npu.synchronize()
        metadata, ids, advertised = msgspec.msgpack.decode(read(f("peer.bin")))
        hs = msgspec.msgpack.decode(metadata, type=MooncakeTransferMetadataGroups)
        t = receiving_thread(w)
        t._validate_remote_metadata(hs, ENGINE)
        peer = hs.metadata_by_pp_rank[0]
        ranks, pairs = t._build_remote_transfer_layout(hs)
        reqs = {}
        for i, ((_, _, dids), pids, n) in enumerate(zip(jobs, advertised, LENGTHS, strict=True)):
            reqs[str(i)] = SimpleNamespace(
                local_block_ids=dids,
                local_full_block_ids=dids,
                remote_block_ids=pids,
                local_num_prompt_tokens=n,
                remote_num_prompt_tokens=n - 1,
                num_computed_tokens=0,
            )
        buckets, _ = t._build_transfer_block_buckets(peer, pairs[0], ranks[0], 1, 1, reqs, {})
        assert set(buckets) == {(0, r) for r in range(4)}
        for (pcp, tp), entries in buckets.items():
            if a.partial and a.rank == 1 and tp != 0:
                continue
            t._execute_tp_transfer_bucket(0, pcp, tp, 1, 4, 1, peer, entries)
        torch.npu.synchronize()
        partial_failure = a.partial and a.rank == 1
        expected, count = expected_data(cfg, specs, jobs, ids, 1 if partial_failure else 4)
        assert torch.equal(cache.slab.cpu(), expected.slab)
        if partial_failure:
            read(f("d0.bin"))
            publish(f("partial.bin"), b"D0 complete; D1 aborted after first source")
            read(f("held.bin"))
            print("D_PARTIAL_CANCEL_PASS", a.rank, "read 6005376 bytes; three unsent requests unchanged", flush=True)
        else:
            finish_windows(cache, expected, jobs)
            print(
                "LIVE_STOCK_PULL_PASS",
                a.rank,
                "sources=0,1,2,3",
                "bytes",
                count,
                "40_layers_per_request=pass",
                "full_slab_canaries=pass",
                flush=True,
            )
    global_te.unregister_buffer()
    for lease, ticket, _ in jobs:
        if not lease.retired:
            cache.requests.retire(lease)
        cache.requests.complete(lease, ticket)
    assert cache.requests.history.get_num_free_blocks() == (P_SIZE if is_p else D_SIZE)[0]
    assert not cache.requests.live
    recycled = []
    for i in range(P_SIZE[2] if is_p else D_SIZE[2]):
        recycled.append(cache.requests.admit(f"after-completion-{i}", 1))
    assert all(recycled)
    assert old.compressor in {lease.compressor for lease in recycled}
    for lease in recycled:
        cache.requests.retire(lease)
    print("RETIRE_REUSE_PASS", a.role, a.rank, flush=True)
    # Close the SDK while its device context and registered backing tensors are still alive.
    w.engine = None
    if not is_p:
        t.engine = None
    global_te.transfer_engine = None
    print("RELEASED", a.role, a.rank, flush=True)
    if not is_p:
        publish(f(f"d{a.rank}.bin"), b"unregistered")
        if a.rank == 0:
            read(f("d1.bin"))
            publish(f("done.bin"), b"unregistered")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("P", "D"))
    parser.add_argument("rank", type=int)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--ip", required=True)
    parser.add_argument("--exchange", type=Path, required=True)
    main(parser.parse_args())
