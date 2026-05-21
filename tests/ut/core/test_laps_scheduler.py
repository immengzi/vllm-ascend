import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_ascend.core.laps_scheduler import (
    AsyncLAPSScheduler,
    LAPSBudgetContext,
    LAPSScheduler,
)
from vllm_ascend.core.schedule_template import AscendSchedulerTemplateMixin


@pytest.mark.cpu_test
def test_laps_long_prefill_cap_limits_single_long_request(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "256")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "128")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    request = create_requests(num_requests=1, num_tokens=800)[0]
    scheduler.add_request(request)
    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] == 256
    waiting = scheduler.waiting
    assert waiting._last_long_capped_count == 1
    assert waiting._last_long_actual_used_tokens == 256
    assert waiting._last_short_reserved_tokens == 0


@pytest.mark.cpu_test
def test_laps_short_reserved_budget_reduces_long_prefill_share(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0.25")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "128")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    short_request = create_requests(
        num_requests=1,
        num_tokens=64,
        req_ids=["short"],
    )[0]
    long_request = create_requests(
        num_requests=1,
        num_tokens=800,
        req_ids=["long"],
    )[0]

    scheduler.add_request(short_request)
    scheduler.add_request(long_request)
    output = scheduler.schedule()

    assert output.num_scheduled_tokens[short_request.request_id] == 64
    assert output.num_scheduled_tokens[long_request.request_id] == 768
    waiting = scheduler.waiting
    assert waiting._last_short_reserved_tokens == 256
    assert waiting._last_short_actual_used_tokens == 64
    assert waiting._last_long_actual_used_tokens == 768


@pytest.mark.cpu_test
def test_laps_long_prefill_cap_does_not_limit_short_prefill(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "128")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "256")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    short_request = create_requests(
        num_requests=1,
        num_tokens=200,
        req_ids=["short-200"],
    )[0]

    scheduler.add_request(short_request)
    output = scheduler.schedule()

    assert output.num_scheduled_tokens[short_request.request_id] == 200
    waiting = scheduler.waiting
    assert waiting._last_long_capped_count == 0
    assert waiting._last_short_actual_used_tokens == 200
    assert waiting._last_long_actual_used_tokens == 0


@pytest.mark.cpu_test
def test_laps_long_cap_count_only_tracks_scheduled_requests(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "256")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "128")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    request_1 = create_requests(num_requests=1, num_tokens=800)[0]
    request_2 = create_requests(num_requests=1, num_tokens=800, req_ids=["long-2"])[0]
    scheduler.add_request(request_1)
    scheduler.add_request(request_2)

    request_1.status = RequestStatus.RUNNING
    request_2.status = RequestStatus.RUNNING
    scheduler.running = [request_1, request_2]
    scheduler.waiting.remove_requests([request_1, request_2])
    scheduler.kv_cache_manager.allocate_slots = lambda *args, **kwargs: None

    output = scheduler.schedule()

    assert output.preempted_req_ids
    assert scheduler.waiting._last_long_capped_count == 0


@pytest.mark.cpu_test
def test_laps_preempted_request_is_requeued_immediately(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "256")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_WAIT_WINDOW_MS", "10")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_WAIT_MAX_BATCH", "4")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    request = create_requests(num_requests=1, num_tokens=64)[0]
    request.status = RequestStatus.RUNNING

    scheduler._preempt_request(request, 0.0)

    selected_queue = scheduler._select_waiting_queue_for_scheduling()
    assert selected_queue is not None
    assert selected_queue.peek_request() is request


@pytest.mark.cpu_test
def test_laps_non_budget_path_records_short_usage(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "256")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_WAIT_WINDOW_MS", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_WAIT_MAX_BATCH", "4")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    short_request = create_requests(
        num_requests=1,
        num_tokens=64,
        req_ids=["short-stats"],
    )[0]

    scheduler.add_request(short_request)
    output = scheduler.schedule()

    assert output.num_scheduled_tokens[short_request.request_id] == 64
    waiting = scheduler.waiting
    assert waiting._last_short_actual_used_tokens == 64
    assert waiting._last_long_actual_used_tokens == 0


@pytest.mark.cpu_test
def test_async_laps_scheduler_installs_laps_waiting_queue(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "256")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_WAIT_WINDOW_MS", "10")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_WAIT_MAX_BATCH", "4")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    base_scheduler.vllm_config.scheduler_config.async_scheduling = True
    scheduler = AsyncLAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    request = create_requests(num_requests=1, num_tokens=64)[0]
    scheduler.add_request(request)

    assert scheduler.waiting.__class__.__name__ == "LAPSRequestQueue"


@pytest.mark.cpu_test
def test_laps_budget_path_uses_shared_schedule_template(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "128")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "256")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    called = False
    original = AscendSchedulerTemplateMixin._schedule_with_hooks

    def wrapped(self):
        nonlocal called
        called = True
        return original(self)

    monkeypatch.setattr(
        AscendSchedulerTemplateMixin,
        "_schedule_with_hooks",
        wrapped,
    )

    request = create_requests(num_requests=1, num_tokens=800)[0]
    scheduler.add_request(request)
    output = scheduler.schedule()

    assert called
    assert output.num_scheduled_tokens[request.request_id] == 256


@pytest.mark.cpu_test
def test_laps_budget_path_classifies_waiting_request_once(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "128")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "256")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    request = create_requests(num_requests=1, num_tokens=800, req_ids=["long-hot"])[0]
    scheduler.add_request(request)

    original = scheduler._classify_laps_request
    classify_calls: list[tuple[str, int | None]] = []

    def wrapped(req, num_computed_tokens=None):
        classify_calls.append((req.request_id, num_computed_tokens))
        return original(req, num_computed_tokens)

    monkeypatch.setattr(scheduler, "_classify_laps_request", wrapped)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] == 256
    assert classify_calls == [(request.request_id, 0)]


@pytest.mark.cpu_test
def test_laps_budget_context_rollback_reuses_cached_request_class(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "128")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "256")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    request = create_requests(num_requests=1, num_tokens=800, req_ids=["rollback-long"])[0]
    laps_ctx = LAPSBudgetContext(scheduler, token_budget=1024)

    num_new_tokens, was_capped, request_class = laps_ctx.adjust_tokens(
        request,
        num_new_tokens=800,
        token_budget=1024,
        num_computed_tokens=0,
    )
    laps_ctx.record_scheduled(
        request,
        request_class,
        num_new_tokens,
        was_capped,
    )

    classify_calls = 0

    def fail_if_classified(*args, **kwargs):
        nonlocal classify_calls
        classify_calls += 1
        raise AssertionError("rollback should reuse cached request class")

    monkeypatch.setattr(scheduler, "_classify_laps_request", fail_if_classified)

    laps_ctx.rollback_scheduled(request, num_new_tokens)

    assert classify_calls == 0
    assert laps_ctx.long_budget_remaining == 1024
    assert laps_ctx.long_actual_used_tokens == 0
    assert laps_ctx.short_actual_used_tokens == 0


@pytest.mark.cpu_test
def test_laps_budget_context_hot_path_avoids_recompute(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "128")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "256")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0.25")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    short_request = create_requests(
        num_requests=1,
        num_tokens=64,
        req_ids=["short-hot"],
    )[0]
    long_request = create_requests(
        num_requests=1,
        num_tokens=800,
        req_ids=["long-hot-2"],
    )[0]
    decode_request = create_requests(
        num_requests=1,
        num_tokens=64,
        req_ids=["decode-hot"],
    )[0]
    decode_request.num_computed_tokens = decode_request.num_prompt_tokens

    scheduler.add_request(short_request)
    scheduler.add_request(long_request)

    laps_ctx = LAPSBudgetContext(scheduler, token_budget=1024)
    compute_calls = 0
    original_compute = scheduler._compute_long_budget_remaining

    def wrapped_compute(*args, **kwargs):
        nonlocal compute_calls
        compute_calls += 1
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(scheduler, "_compute_long_budget_remaining", wrapped_compute)

    num_new_tokens, was_capped, request_class = laps_ctx.adjust_tokens(
        decode_request,
        num_new_tokens=1,
        token_budget=laps_ctx.token_budget_remaining,
        num_computed_tokens=decode_request.num_computed_tokens,
    )
    laps_ctx.record_scheduled(
        decode_request,
        request_class,
        num_new_tokens,
        was_capped,
    )

    num_new_tokens, was_capped, request_class = laps_ctx.adjust_tokens(
        long_request,
        num_new_tokens=800,
        token_budget=laps_ctx.token_budget_remaining,
        num_computed_tokens=0,
    )
    laps_ctx.record_scheduled(
        long_request,
        request_class,
        num_new_tokens,
        was_capped,
    )

    assert compute_calls == 0


@pytest.mark.cpu_test
def test_laps_budget_context_short_prefill_within_reserved_budget(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "128")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0.25")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    short_waiting = create_requests(
        num_requests=1,
        num_tokens=32,
        req_ids=["short-waiting-a"],
    )[0]
    short_scheduled = create_requests(
        num_requests=1,
        num_tokens=64,
        req_ids=["short-scheduled-a"],
    )[0]
    scheduler.add_request(short_waiting)
    scheduler.add_request(short_scheduled)

    laps_ctx = LAPSBudgetContext(scheduler, token_budget=1024)

    num_new_tokens, was_capped, request_class = laps_ctx.adjust_tokens(
        short_scheduled,
        num_new_tokens=64,
        token_budget=laps_ctx.token_budget_remaining,
        num_computed_tokens=0,
    )
    laps_ctx.record_scheduled(
        short_scheduled,
        request_class,
        num_new_tokens,
        was_capped,
    )

    assert laps_ctx.long_budget_remaining == 768
    assert laps_ctx.short_actual_used_tokens == 64


@pytest.mark.cpu_test
def test_laps_budget_context_short_prefill_only_consumes_overflow(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "128")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0.25")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    short_waiting = create_requests(
        num_requests=1,
        num_tokens=32,
        req_ids=["short-waiting-b"],
    )[0]
    short_scheduled = create_requests(
        num_requests=1,
        num_tokens=400,
        req_ids=["short-scheduled-b"],
    )[0]
    scheduler.add_request(short_waiting)
    scheduler.add_request(short_scheduled)

    laps_ctx = LAPSBudgetContext(scheduler, token_budget=1024)

    num_new_tokens, was_capped, request_class = laps_ctx.adjust_tokens(
        short_scheduled,
        num_new_tokens=400,
        token_budget=laps_ctx.token_budget_remaining,
        num_computed_tokens=0,
    )
    laps_ctx.record_scheduled(
        short_scheduled,
        request_class,
        num_new_tokens,
        was_capped,
    )

    assert laps_ctx.long_budget_remaining == 624
    assert laps_ctx.short_actual_used_tokens == 400


@pytest.mark.cpu_test
def test_laps_budget_context_releases_reserved_budget_after_last_short_waiting(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "128")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0.25")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    short_request = create_requests(
        num_requests=1,
        num_tokens=64,
        req_ids=["last-short"],
    )[0]
    scheduler.add_request(short_request)
    laps_ctx = LAPSBudgetContext(scheduler, token_budget=1024)

    scheduler.waiting.pop_request()
    num_new_tokens, was_capped, request_class = laps_ctx.adjust_tokens(
        short_request,
        num_new_tokens=64,
        token_budget=laps_ctx.token_budget_remaining,
        num_computed_tokens=0,
    )
    laps_ctx.record_scheduled(
        short_request,
        request_class,
        num_new_tokens,
        was_capped,
    )

    assert laps_ctx.has_short_waiting_requests is False
    assert laps_ctx.token_budget_remaining == 960
    assert laps_ctx.long_budget_remaining == 960


@pytest.mark.cpu_test
def test_laps_budget_context_rollback_resyncs_long_budget(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "128")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0.25")

    base_scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )
    scheduler = LAPSScheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )

    short_waiting = create_requests(
        num_requests=1,
        num_tokens=32,
        req_ids=["short-waiting-rollback"],
    )[0]
    short_request = create_requests(
        num_requests=1,
        num_tokens=64,
        req_ids=["short-rollback"],
    )[0]
    long_request = create_requests(
        num_requests=1,
        num_tokens=800,
        req_ids=["long-rollback-2"],
    )[0]
    scheduler.add_request(short_waiting)
    scheduler.add_request(short_request)
    scheduler.add_request(long_request)

    laps_ctx = LAPSBudgetContext(scheduler, token_budget=1024)

    num_new_tokens, was_capped, request_class = laps_ctx.adjust_tokens(
        short_request,
        num_new_tokens=64,
        token_budget=laps_ctx.token_budget_remaining,
        num_computed_tokens=0,
    )
    laps_ctx.record_scheduled(
        short_request,
        request_class,
        num_new_tokens,
        was_capped,
    )

    num_new_tokens, was_capped, request_class = laps_ctx.adjust_tokens(
        long_request,
        num_new_tokens=800,
        token_budget=laps_ctx.token_budget_remaining,
        num_computed_tokens=0,
    )
    laps_ctx.record_scheduled(
        long_request,
        request_class,
        num_new_tokens,
        was_capped,
    )

    laps_ctx.rollback_scheduled(long_request, num_new_tokens)

    assert laps_ctx.token_budget_remaining == 960
    assert laps_ctx.short_actual_used_tokens == 64
    assert laps_ctx.long_actual_used_tokens == 0
    assert laps_ctx.long_budget_remaining == 768
