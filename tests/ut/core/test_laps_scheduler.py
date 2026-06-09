#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the LAPS waiting queue (LapsRequestQueue) and LapsConfig.

These are pure-Python tests: requests are lightweight ``SimpleNamespace``
objects (the queue only reads ``request_id`` / ``num_prompt_tokens``), and the
aging clock is driven by patching ``vllm_ascend.core.laps_scheduler.time``.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                              create_request_queue)

from vllm_ascend.core.laps_scheduler import LapsRequestQueue

THRESHOLD = 256


def make_request(request_id: str, prompt_len: int, **extra) -> SimpleNamespace:
    return SimpleNamespace(request_id=request_id,
                           num_prompt_tokens=prompt_len,
                           **extra)


def make_queue(long_max_wait_ms: float = 0.0,
               long_token_reservation: float = 0.0,
               threshold: int = THRESHOLD,
               immediate_predicate=None) -> LapsRequestQueue:
    return LapsRequestQueue(
        policy=SchedulingPolicy.FCFS,
        threshold=threshold,
        long_max_wait_ms=long_max_wait_ms,
        long_token_reservation=long_token_reservation,
        immediate_predicate=immediate_predicate,
    )


def short_req(rid="short", n=10):
    return make_request(rid, n)


def long_req(rid="long", n=THRESHOLD + 1000):
    return make_request(rid, n)


class FakeClock:
    """Monotonic clock stub; advance with ``.advance(seconds)``."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def advance(self, seconds: float):
        self.now += seconds


# --------------------------------------------------------------------------- #
# Classification & dispatch priority
# --------------------------------------------------------------------------- #
def test_classify_short_and_long_by_threshold():
    q = make_queue(threshold=256)
    q.add_request(short_req("s", 256))      # <= threshold -> short
    q.add_request(long_req("l", 257))       # > threshold -> long
    assert q.num_short_requests == 1
    assert q.num_long_requests == 1
    assert q.num_immediate_requests == 0


def test_immediate_predicate_routes_to_immediate_queue():
    q = make_queue(immediate_predicate=lambda r: r.request_id == "hot")
    q.add_request(make_request("hot", 5))
    assert q.num_immediate_requests == 1
    assert q.num_short_requests == 0


def test_dispatch_priority_immediate_then_short_then_long():
    q = make_queue()  # aging disabled
    q.add_request(long_req("l"))
    q.add_request(short_req("s"))
    q.prepend_request(make_request("imm", 5), force_immediate=True)

    assert q.pop_request().request_id == "imm"
    assert q.pop_request().request_id == "s"
    assert q.pop_request().request_id == "l"


def test_mark_force_immediate_routes_next_add_to_immediate():
    q = make_queue()
    # A short-by-length request marked force-immediate lands in immediate.
    q.mark_force_immediate("p")
    q.add_request(make_request("p", 5))
    assert q.num_immediate_requests == 1
    assert q.num_short_requests == 0


def test_short_runs_before_long_without_aging():
    q = make_queue()
    q.add_request(long_req("l"))
    q.add_request(short_req("s"))
    assert q._select_schedulable_queue() is q._short_queue


# --------------------------------------------------------------------------- #
# Basic queue operations
# --------------------------------------------------------------------------- #
def test_len_contains_iter_and_peek():
    q = make_queue()
    a, b = short_req("a"), long_req("b")
    q.add_request(a)
    q.add_request(b)
    assert len(q) == 2
    assert a in q and b in q
    assert {r.request_id for r in q} == {"a", "b"}
    # Short has priority, so peek returns the short request.
    assert q.peek_request().request_id == "a"


def test_remove_request_uses_request_id_not_identity():
    q = make_queue()
    q.add_request(short_req("shared"))
    # A different object with the same id must still be removable.
    q.remove_request(make_request("shared", 10))
    assert len(q) == 0


def test_prepend_request_can_force_immediate_queue():
    q = make_queue()
    q.prepend_request(short_req("s"), force_immediate=True)
    assert q.num_immediate_requests == 1
    assert q._prepend_counters["immediate"] == 1


def test_owns_queue_only_true_for_internal_subqueues():
    q = make_queue()
    assert q.owns_queue(q._immediate_queue)
    assert q.owns_queue(q._short_queue)
    assert q.owns_queue(q._long_queue)
    external = create_request_queue(SchedulingPolicy.FCFS)
    assert not q.owns_queue(external)
    assert not q.owns_queue(None)


# --------------------------------------------------------------------------- #
# Skip-or-requeue accounting
# --------------------------------------------------------------------------- #
def test_skip_or_requeue_counts_reason():
    q = make_queue()
    q.add_request(short_req("s"))
    q.pop_request_from_queue(
        q._short_queue,
        count_as_removal=True,
        skip_or_requeue_reason="blocked_waiting_status",
    )
    assert q._skip_or_requeue_counters["blocked_waiting_status"]["short"] == 1
    # A skip is not a dispatch.
    assert q._dispatch_counters["short"] == 0


def test_unknown_skip_or_requeue_reason_raises():
    q = make_queue()
    q.add_request(short_req("s"))
    with pytest.raises(ValueError):
        q.pop_request_from_queue(
            q._short_queue,
            count_as_removal=True,
            skip_or_requeue_reason="bogus",
        )


# --------------------------------------------------------------------------- #
# Two-tier anti-starvation aging
# --------------------------------------------------------------------------- #
def test_soft_phase_promotes_long_only_while_bucket_has_credit():
    clock = FakeClock()
    with patch("vllm_ascend.core.laps_scheduler.time", clock):
        q = make_queue(long_max_wait_ms=100.0, long_token_reservation=0.5)
        q.add_request(long_req("l"))     # enqueued now
        q.add_request(short_req("s"))
        q.begin_step(100)                # bucket refills to 50 (0.5 * 100)

        # 150ms is in [soft=100ms, hard=200ms): soft phase, hard not yet reached.
        clock.advance(0.15)
        # Bucket has credit -> aged-long jumps the short queue.
        assert q._select_schedulable_queue() is q._long_queue

        # Exhaust the bucket while still below the hard bound -> prefer short.
        q._long_bucket = 0.0
        assert q._select_schedulable_queue() is q._short_queue


def test_hard_deadline_forces_long_even_with_empty_bucket():
    clock = FakeClock()
    with patch("vllm_ascend.core.laps_scheduler.time", clock):
        q = make_queue(long_max_wait_ms=100.0, long_token_reservation=0.5)
        q.add_request(long_req("l"))
        q.add_request(short_req("s"))
        q.begin_step(100)
        q._long_bucket = 0.0             # no soft-phase credit

        clock.advance(0.5)               # 500ms >= hard bound (~200ms)
        assert q._select_schedulable_queue() is q._long_queue


def test_zero_reservation_is_pure_deadline_at_long_max_wait():
    """Regression: with reservation=0 and a sustained short stream, an aged
    long MUST still be admitted once it passes long_max_wait_ms. The old
    cumulative-quota implementation failed this and let long TTFT blow up."""
    clock = FakeClock()
    with patch("vllm_ascend.core.laps_scheduler.time", clock):
        q = make_queue(long_max_wait_ms=100.0, long_token_reservation=0.0)
        q.add_request(long_req("l"))
        q.add_request(short_req("s"))    # short queue stays non-empty
        q.begin_step(4096)

        clock.advance(0.05)              # 50ms < 100ms -> short still preferred
        assert q._select_schedulable_queue() is q._short_queue

        clock.advance(0.10)             # total 150ms >= 100ms -> long forced in
        assert q._select_schedulable_queue() is q._long_queue


def test_stall_avoidance_admits_long_without_charge_or_promotion():
    clock = FakeClock()
    with patch("vllm_ascend.core.laps_scheduler.time", clock):
        q = make_queue(long_max_wait_ms=100.0, long_token_reservation=0.5)
        q.add_request(long_req("l"))     # no short waiting
        q.begin_step(100)

        clock.advance(0.5)
        assert q._select_schedulable_queue() is q._long_queue
        # Dispatch via the queue path; no short was waiting -> not counted/charged.
        bucket_before = q._long_bucket
        q.pop_request_from_queue(q._long_queue, scheduled_tokens=4096)
        assert q._long_starvation_promotions == 0
        assert q._long_hard_deadline_promotions == 0
        assert q._long_bucket == bucket_before  # not charged


def test_aged_long_dispatch_counts_and_charges_when_jumping_shorts():
    clock = FakeClock()
    with patch("vllm_ascend.core.laps_scheduler.time", clock):
        q = make_queue(long_max_wait_ms=100.0, long_token_reservation=0.5)
        q.add_request(long_req("l"))
        q.add_request(short_req("s"))    # short waiting -> long jumps it
        q.begin_step(100)                # bucket = 50

        clock.advance(0.5)               # past hard bound
        q.pop_request_from_queue(q._long_queue, scheduled_tokens=40)
        assert q._long_starvation_promotions == 1
        # Hard-deadline path (bucket was > 0 here actually -> not hard). Re-check
        # via a dedicated hard scenario below; here just assert charge applied.
        assert q._long_tokens_charged == 40
        assert q._long_bucket == 50 - 40


def test_hard_deadline_promotion_counter_increments_on_forced_admission():
    clock = FakeClock()
    with patch("vllm_ascend.core.laps_scheduler.time", clock):
        q = make_queue(long_max_wait_ms=100.0, long_token_reservation=0.5)
        q.add_request(long_req("l"))
        q.add_request(short_req("s"))
        q.begin_step(100)
        q._long_bucket = 0.0             # bucket empty -> only hard path admits

        clock.advance(0.5)               # past hard bound
        q.pop_request_from_queue(q._long_queue, scheduled_tokens=40)
        assert q._long_starvation_promotions == 1
        assert q._long_hard_deadline_promotions == 1


def test_no_aging_when_long_max_wait_is_zero():
    clock = FakeClock()
    with patch("vllm_ascend.core.laps_scheduler.time", clock):
        q = make_queue(long_max_wait_ms=0.0, long_token_reservation=0.5)
        q.add_request(long_req("l"))
        q.add_request(short_req("s"))
        q.begin_step(100)
        clock.advance(10_000)            # arbitrarily long wait
        # Aging disabled -> short keeps priority forever.
        assert q._select_schedulable_queue() is q._short_queue


# --------------------------------------------------------------------------- #
# Token-bucket budget control (configurable burst / hard-deadline multiplier)
# --------------------------------------------------------------------------- #
def test_begin_step_caps_bucket_at_burst_capacity():
    q = make_queue(long_max_wait_ms=100.0, long_token_reservation=0.5,
                   long_burst_steps=2)
    # refill_per_step = 0.5 * 100 = 50 ; capacity = 50 * 2 = 100.
    for _ in range(10):
        q.begin_step(100)
    assert q._long_bucket_capacity == 100.0
    assert q._long_bucket == 100.0


def test_long_hard_deadline_mult_is_configurable():
    q = make_queue(long_max_wait_ms=100.0, long_token_reservation=0.5,
                   long_hard_deadline_mult=3.0)
    assert q._long_hard_deadline_ms() == 300.0
    # With reservation 0 the hard bound collapses to long_max_wait_ms.
    q0 = make_queue(long_max_wait_ms=100.0, long_token_reservation=0.0,
                    long_hard_deadline_mult=3.0)
    assert q0._long_hard_deadline_ms() == 100.0


def test_hard_deadline_bypass_does_not_charge_bucket():
    """Regression: a burst of hard-deadline promotions must NOT drive the bucket
    into unbounded negative debt (which would then block soft promotions for many
    steps). Only soft-phase promotions (taken with credit) are charged."""
    clock = FakeClock()
    with patch("vllm_ascend.core.laps_scheduler.time", clock):
        q = make_queue(long_max_wait_ms=100.0, long_token_reservation=0.5)
        q.begin_step(100)
        q._long_bucket = 0.0             # empty -> only the hard path can admit
        for i in range(5):
            q.add_request(long_req(f"l{i}"))
        q.add_request(short_req("s"))    # a short stays waiting throughout
        clock.advance(0.5)               # all longs past the hard bound
        for _ in range(5):
            q.pop_request_from_queue(q._long_queue, scheduled_tokens=4096)
        assert q._long_bucket == 0.0     # never went negative
        assert q._long_hard_deadline_promotions == 5
        assert q._long_tokens_charged == 0


def test_soft_charge_debt_is_bounded_by_one_chunk():
    """A soft promotion may take the bucket below zero by at most one chunk; the
    next refill recovers it, so debt is bounded (no cliff)."""
    clock = FakeClock()
    with patch("vllm_ascend.core.laps_scheduler.time", clock):
        q = make_queue(long_max_wait_ms=100.0, long_token_reservation=0.5)
        q.add_request(long_req("l"))
        q.add_request(short_req("s"))
        q.begin_step(100)                # bucket = 50
        clock.advance(0.15)              # soft window [100, 200): not yet hard
        q.pop_request_from_queue(q._long_queue, scheduled_tokens=80)
        assert q._long_hard_deadline_promotions == 0  # soft, not hard
        assert q._long_bucket == -30.0   # dipped below zero by < one chunk
        q.begin_step(100)                # one refill (+50) recovers it
        assert q._long_bucket == 20.0
