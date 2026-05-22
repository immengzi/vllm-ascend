from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.core.sched.request_queue import SchedulingPolicy

from vllm_ascend.core.laps_scheduler import (
    LAPSBudgetContext,
    LAPSRequestQueue,
    LAPSSchedulerMixin,
    _LAPSRequestClass,
)


def _make_request(
    request_id: str,
    prompt_len: int,
    computed: int = 0,
    num_tokens: int | None = None,
):
    return SimpleNamespace(
        request_id=request_id,
        num_prompt_tokens=prompt_len,
        num_computed_tokens=computed,
        num_tokens=num_tokens if num_tokens is not None else prompt_len,
    )


def _make_mixin(
    threshold: int = 256,
    cap: int = 0,
    ratio: float = 0.0,
    max_tokens: int = 4096,
    short_requests: bool = False,
):
    queue = LAPSRequestQueue(
        policy=SchedulingPolicy.FCFS,
        threshold=threshold,
        wait_window_ms=0,
        wait_max_batch=4,
    )
    if short_requests:
        queue.add_request(_make_request("_short_filler", threshold))

    mixin = SimpleNamespace(
        laps_long_prefill_cap=cap,
        laps_short_reserved_ratio=ratio,
        max_num_scheduled_tokens=max_tokens,
        _laps_waiting_queue=lambda: queue,
    )
    # Bind mixin methods from the real class
    for name in (
        "_laps_threshold",
        "_classify_laps_request",
        "_laps_short_reserved_tokens",
        "_compute_long_budget_remaining",
        "_apply_long_prefill_cap",
        "_apply_long_budget_limit",
        "_record_laps_step_usage",
        "_laps_long_budgeting_enabled",
    ):
        method = getattr(LAPSSchedulerMixin, name)
        setattr(mixin, name, method.__get__(mixin, type(mixin)))
    return mixin, queue


# --- _classify_laps_request ---


def test_classify_non_prefill():
    mixin, _ = _make_mixin(threshold=256)
    req = _make_request("r1", prompt_len=100, computed=100)
    assert mixin._classify_laps_request(req) is None


def test_classify_short_prefill():
    mixin, _ = _make_mixin(threshold=256)
    req = _make_request("r1", prompt_len=200, computed=0)
    assert mixin._classify_laps_request(req) is _LAPSRequestClass.SHORT_PREFILL


def test_classify_long_prefill():
    mixin, _ = _make_mixin(threshold=256)
    req = _make_request("r1", prompt_len=512, computed=0)
    assert mixin._classify_laps_request(req) is _LAPSRequestClass.LONG_PREFILL


def test_classify_with_explicit_computed_tokens():
    mixin, _ = _make_mixin(threshold=256)
    req = _make_request("r1", prompt_len=512, computed=512)
    assert mixin._classify_laps_request(req) is None
    assert (
        mixin._classify_laps_request(req, num_computed_tokens=0)
        is _LAPSRequestClass.LONG_PREFILL
    )


# --- LAPSBudgetContext: classify once ---


def test_budget_context_classifies_once():
    mixin, _ = _make_mixin(threshold=256, cap=512, ratio=0.0)
    ctx = LAPSBudgetContext(mixin, token_budget=4096)
    req = _make_request("r1", prompt_len=1024, computed=0)

    call_count = [0]
    original = mixin._classify_laps_request

    def counting_classify(*args, **kwargs):
        call_count[0] += 1
        return original(*args, **kwargs)

    mixin._classify_laps_request = counting_classify

    num_new_tokens, was_capped, request_class = ctx.adjust_tokens(
        req, 1024
    )
    assert request_class is _LAPSRequestClass.LONG_PREFILL
    assert call_count[0] == 1

    ctx.record_scheduled(req, request_class, num_new_tokens, was_capped)
    assert call_count[0] == 1


# --- LAPSBudgetContext: rollback uses cached class ---


def test_rollback_uses_cached_class():
    mixin, _ = _make_mixin(threshold=256, cap=0, ratio=0.0)
    ctx = LAPSBudgetContext(mixin, token_budget=4096)
    req = _make_request("r1", prompt_len=512, computed=0)

    _, _, request_class = ctx.adjust_tokens(req, 512)
    ctx.record_scheduled(req, request_class, 512, False)

    original = mixin._classify_laps_request
    classify_called = [False]

    def fail_classify(*args, **kwargs):
        classify_called[0] = True
        return original(*args, **kwargs)

    mixin._classify_laps_request = fail_classify
    ctx.rollback_scheduled(req, 512)
    assert not classify_called[0]


# --- LAPSBudgetContext: hot path avoids recompute ---


def test_hot_path_avoids_recompute():
    mixin, _ = _make_mixin(threshold=256, cap=0, ratio=0.0)

    call_count = [0]
    original = mixin._compute_long_budget_remaining

    def counting_compute(*args, **kwargs):
        call_count[0] += 1
        return original(*args, **kwargs)

    mixin._compute_long_budget_remaining = counting_compute

    ctx = LAPSBudgetContext(mixin, token_budget=4096)
    init_calls = call_count[0]

    non_prefill_req = _make_request("d1", prompt_len=100, computed=100)
    num_new, _, rc = ctx.adjust_tokens(non_prefill_req, 1)
    assert rc is None
    ctx.record_scheduled(non_prefill_req, rc, num_new, False)

    long_req = _make_request("l1", prompt_len=512, computed=0)
    num_new, _, rc = ctx.adjust_tokens(long_req, 512)
    ctx.record_scheduled(long_req, rc, num_new, False)

    assert call_count[0] == init_calls


# --- LAPSBudgetContext: short prefill overflow ---


def test_short_prefill_only_consumes_overflow():
    mixin, queue = _make_mixin(
        threshold=256, cap=0, ratio=0.25, max_tokens=1024,
        short_requests=True,
    )
    ctx = LAPSBudgetContext(mixin, token_budget=1024)

    reserved = ctx.short_reserved_tokens
    assert reserved == int(1024 * 0.25)

    initial_long_budget = ctx.long_budget_remaining

    short_req = _make_request("s1", prompt_len=200, computed=0)
    num_new, _, rc = ctx.adjust_tokens(short_req, 200)
    assert rc is _LAPSRequestClass.SHORT_PREFILL
    ctx.record_scheduled(short_req, rc, 200, False)

    if 200 <= reserved:
        assert ctx.long_budget_remaining == initial_long_budget
    else:
        overflow = 200 - reserved
        assert ctx.long_budget_remaining == initial_long_budget - overflow


# --- LAPSBudgetContext: recover_zero_budget ---


def test_recover_zero_budget_long_prefill_breaks():
    mixin, _ = _make_mixin(threshold=256)
    ctx = LAPSBudgetContext(mixin, token_budget=4096)

    num_new, should_break = ctx.recover_zero_budget(
        _make_request("r1", prompt_len=512, computed=0),
        _LAPSRequestClass.LONG_PREFILL,
        token_budget=100,
    )
    assert should_break is True
    assert num_new == 0


def test_recover_zero_budget_short_prefill_continues():
    mixin, _ = _make_mixin(threshold=256)
    ctx = LAPSBudgetContext(mixin, token_budget=4096)

    req = _make_request("r1", prompt_len=200, computed=0)
    num_new, should_break = ctx.recover_zero_budget(
        req,
        _LAPSRequestClass.SHORT_PREFILL,
        token_budget=100,
    )
    assert should_break is False
    assert num_new == min(200, 100)


# --- LAPSBudgetContext: finalize ---


def test_finalize_records_stats():
    mixin, queue = _make_mixin(threshold=256, cap=512, ratio=0.0)
    ctx = LAPSBudgetContext(mixin, token_budget=4096)

    long_req = _make_request("l1", prompt_len=512, computed=0)
    num_new, was_capped, rc = ctx.adjust_tokens(long_req, 600)
    assert was_capped is True
    ctx.record_scheduled(long_req, rc, num_new, was_capped)

    ctx.finalize(queue)
    assert queue._last_long_capped_count == 1
    assert queue._last_long_actual_used_tokens == num_new


# --- backward compatibility: _is_*_prefill_request wrappers ---


def test_is_prefill_request_wrapper():
    mixin, _ = _make_mixin(threshold=256)
    decode = _make_request("d1", prompt_len=100, computed=100)
    prefill = _make_request("p1", prompt_len=512, computed=0)
    assert mixin._is_prefill_request(decode) is False
    assert mixin._is_prefill_request(prefill) is True


def test_is_short_prefill_request_wrapper():
    mixin, _ = _make_mixin(threshold=256)
    short = _make_request("s1", prompt_len=200, computed=0)
    long = _make_request("l1", prompt_len=512, computed=0)
    assert mixin._is_short_prefill_request(short) is True
    assert mixin._is_short_prefill_request(long) is False


def test_is_long_prefill_request_wrapper():
    mixin, _ = _make_mixin(threshold=256)
    short = _make_request("s1", prompt_len=200, computed=0)
    long = _make_request("l1", prompt_len=512, computed=0)
    assert mixin._is_long_prefill_request(short) is False
    assert mixin._is_long_prefill_request(long) is True
