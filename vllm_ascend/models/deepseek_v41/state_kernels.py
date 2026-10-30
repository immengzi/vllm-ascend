# SPDX-License-Identifier: Apache-2.0
"""NPU copies for request-owned SWA rings and packed temporary PA pages."""

from vllm.triton_utils import tl, triton


@triton.jit
def _assemble(
    RING, CURRENT, IDS, QSTART, SEQLEN, BASE, OUT, WIDTH: tl.constexpr, WINDOW: tl.constexpr, ROWS: tl.constexpr
):
    req = tl.program_id(0)
    row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    dim = tl.arange(0, WIDTH)
    first = tl.load(QSTART + req)
    end = tl.load(QSTART + req + 1)
    length = end - first
    computed = tl.load(SEQLEN + req) - length
    old = tl.minimum(computed, WINDOW - 1)
    state = tl.load(IDS + req)
    base = tl.load(BASE + req)
    old_slot = (computed - old + row) % WINDOW
    previous = tl.load(
        RING + (state * WINDOW + old_slot[:, None]) * WIDTH + dim[None, :], mask=(row < old)[:, None], other=0
    )
    new = tl.load(
        CURRENT + (first + row[:, None] - old) * WIDTH + dim[None, :],
        mask=((row >= old) & (row < old + length))[:, None],
        other=0,
    )
    value = tl.where((row < old)[:, None], previous, new)
    tl.store(OUT + (base + row[:, None]) * WIDTH + dim[None, :], value, mask=(row < old + length)[:, None])


@triton.jit
def _commit(RING, CURRENT, IDS, QSTART, SEQLEN, WIDTH: tl.constexpr, WINDOW: tl.constexpr, ROWS: tl.constexpr):
    req = tl.program_id(0)
    row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    dim = tl.arange(0, WIDTH)
    first = tl.load(QSTART + req)
    end = tl.load(QSTART + req + 1)
    length = end - first
    count = tl.minimum(length, WINDOW)
    computed = tl.load(SEQLEN + req) - length
    absolute = computed + length - count + row
    state = tl.load(IDS + req)
    value = tl.load(CURRENT + (end - count + row[:, None]) * WIDTH + dim[None, :], mask=(row < count)[:, None], other=0)
    tl.store(
        RING + (state * WINDOW + absolute[:, None] % WINDOW) * WIDTH + dim[None, :], value, mask=(row < count)[:, None]
    )


def assemble(ring, current, state_ids, query_start, seq_lens, row_starts, output, max_query_len):
    assert ring.is_contiguous() and current.is_contiguous() and output.is_contiguous()
    nreq = state_ids.numel()
    width = ring.shape[-1]
    _assemble[(nreq, triton.cdiv(max_query_len + 127, 16))](
        ring, current, state_ids, query_start, seq_lens, row_starts, output, WIDTH=width, WINDOW=128, ROWS=16
    )


def commit(ring, current, state_ids, query_start, seq_lens):
    assert ring.is_contiguous() and current.is_contiguous()
    _commit[(state_ids.numel(), 8)](
        ring, current, state_ids, query_start, seq_lens, WIDTH=ring.shape[-1], WINDOW=128, ROWS=16
    )


@triton.jit
def _assemble_cp(
    RING, CURRENT, IDS, LQ, LSEQ, GQ, GSEQ, BASE, OUT, WIDTH: tl.constexpr, WINDOW: tl.constexpr, ROWS: tl.constexpr
):
    req = tl.program_id(0)
    row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    dim = tl.arange(0, WIDTH)
    lfirst = tl.load(LQ + req)
    lend = tl.load(LQ + req + 1)
    length = lend - lfirst
    if length > 0:
        local_computed = tl.load(LSEQ + req) - length
        old = tl.minimum(local_computed, WINDOW - 1)
        global_first = tl.load(GQ + req)
        global_end = tl.load(GQ + req + 1)
        global_computed = tl.load(GSEQ + req) - (global_end - global_first)
        absolute = local_computed - old + row
        live = (length > 0) & (row < old + length)
        from_old = absolute < global_computed
        state = tl.load(IDS + req)
        base = tl.load(BASE + req)
        previous = tl.load(
            RING + (state * WINDOW + absolute[:, None] % WINDOW) * WIDTH + dim[None, :],
            mask=(live & from_old)[:, None],
            other=0,
        )
        current_row = tl.minimum(tl.maximum(absolute - global_computed, 0), global_end - global_first - 1)
        new = tl.load(
            CURRENT + (global_first + current_row[:, None]) * WIDTH + dim[None, :],
            mask=(live & ~from_old)[:, None],
            other=0,
        )
        value = tl.where(from_old[:, None], previous, new)
        safe_output = tl.where(live, base + row, 0)
        tl.store(OUT + safe_output[:, None] * WIDTH + dim[None, :], value, mask=live[:, None])


def assemble_cp(
    ring,
    current,
    state_ids,
    local_query_start,
    local_seq_lens,
    global_query_start,
    global_seq_lens,
    row_starts,
    output,
    max_query_len,
):
    assert ring.is_contiguous() and current.is_contiguous() and output.is_contiguous()
    if max_query_len == 0 or state_ids.numel() == 0:
        return
    _assemble_cp[(state_ids.numel(), triton.cdiv(max_query_len + 127, 16))](
        ring,
        current,
        state_ids,
        local_query_start,
        local_seq_lens,
        global_query_start,
        global_seq_lens,
        row_starts,
        output,
        WIDTH=ring.shape[-1],
        WINDOW=128,
        ROWS=16,
    )
