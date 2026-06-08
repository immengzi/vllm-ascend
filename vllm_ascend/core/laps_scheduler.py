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

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from enum import Enum
from typing import Callable, cast

from vllm.logger import logger
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.request import Request
from vllm.v1.request import RequestStatus

from vllm_ascend import envs


class LAPSRequestQueue(RequestQueue):
    """Two-level waiting queue for short and long prefills."""

    _SKIP_OR_REQUEUE_REASONS = (
        "blocked_waiting_status",
        "max_loras",
        "remote_kv_not_ready",
    )

    def __init__(
        self,
        policy: SchedulingPolicy,
        threshold: int,
        long_max_wait_ms: float,
        long_token_reservation: float = 0.0,
        immediate_predicate: Callable[[Request], bool] | None = None,
    ) -> None:
        self.policy = policy
        self.threshold = threshold
        # Anti-starvation: a long request waiting longer than this many ms is
        # promoted ahead of short prefills. <= 0 disables aging.
        self.long_max_wait_ms = max(long_max_wait_ms, 0.0)
        # Fraction of the per-step token budget reserved for aged-long prefills.
        # 0.0 disables aging (strict short-priority); larger values reserve more
        # compute for the aged-long lane, tightening long-wait SLO bounds at the
        # cost of short-request latency. Clamped to [0.0, 1.0].
        self.long_token_reservation = max(0.0, min(long_token_reservation, 1.0))
        self.immediate_predicate = immediate_predicate
        self._immediate_queue = create_request_queue(policy)
        self._short_queue = create_request_queue(policy)
        self._long_queue = create_request_queue(policy)
        # Tracks when each request entered the long queue, for aging.
        self._long_enqueue_at: dict[str, float] = {}
        self._long_starvation_promotions = 0
        # Number of tokens scheduled for aged-long promotions this step; reset
        # by begin_step().
        self._long_tokens_this_step = 0
        # Token budget for aged-long promotions this step; set by begin_step().
        self._long_token_budget = 0.0
        self._stats_log_interval_s = max(
            envs.VLLM_ASCEND_LAPS_STATS_LOG_INTERVAL_S, 0.0
        )
        self._last_stats_log_at = time.monotonic()
        self._prepend_counters = {"immediate": 0, "short": 0, "long": 0}
        self._dispatch_counters = {"immediate": 0, "short": 0, "long": 0}
        self._skip_or_requeue_counters = {
            reason: {"immediate": 0, "short": 0, "long": 0}
            for reason in self._SKIP_OR_REQUEUE_REASONS
        }
        self._debug_logging_enabled = logger.isEnabledFor(logging.DEBUG)
        self._force_immediate_request_ids: set[str] = set()
        self._queue_index: dict[str, RequestQueue] = {}

    def _queues(self) -> tuple[RequestQueue, ...]:
        return (self._immediate_queue, self._short_queue, self._long_queue)

    def _queue_name(self, queue: RequestQueue) -> str:
        if queue is self._immediate_queue:
            return "immediate"
        if queue is self._short_queue:
            return "short"
        if queue is self._long_queue:
            return "long"
        return "unknown"

    def _maybe_log_stats(self, force: bool = False) -> None:
        if self._stats_log_interval_s <= 0:
            return
        now = time.monotonic()
        if not force and (now - self._last_stats_log_at) < self._stats_log_interval_s:
            return
        self._last_stats_log_at = now
        logger.info(
            "LAPS stats: threshold=%d long_max_wait_ms=%.3f "
            "long_token_reservation=%.3f "
            "sizes=(immediate=%d short=%d long=%d) "
            "prepends=%s dispatches=%s skip_or_requeues=%s "
            "long_starvation_promotions=%d",
            self.threshold,
            self.long_max_wait_ms,
            self.long_token_reservation,
            len(self._immediate_queue),
            len(self._short_queue),
            len(self._long_queue),
            self._prepend_counters,
            self._dispatch_counters,
            self._skip_or_requeue_counters,
            self._long_starvation_promotions,
        )

    def _increment_skip_or_requeue_counter(
        self, queue: RequestQueue, reason: str
    ) -> None:
        if reason not in self._skip_or_requeue_counters:
            raise ValueError(f"Unknown skip_or_requeue reason: {reason}")
        self._skip_or_requeue_counters[reason][self._queue_name(queue)] += 1

    def _debug_state(
        self,
        event: str,
        request: Request | None = None,
        queue: RequestQueue | None = None,
        extra: str = "",
    ) -> None:
        if not self._debug_logging_enabled:
            return
        request_id = "-" if request is None else request.request_id
        prompt_tokens = -1 if request is None else request.num_prompt_tokens
        queue_name = "-" if queue is None else self._queue_name(queue)
        extra_suffix = f", {extra}" if extra else ""
        logger.debug(
            "LAPS queue %s: request_id=%s, prompt_tokens=%d, target_queue=%s, "
            "sizes=(immediate=%d, short=%d, long=%d)%s",
            event,
            request_id,
            prompt_tokens,
            queue_name,
            len(self._immediate_queue),
            len(self._short_queue),
            len(self._long_queue),
            extra_suffix,
        )

    @property
    def num_immediate_requests(self) -> int:
        return len(self._immediate_queue)

    @property
    def num_short_requests(self) -> int:
        return len(self._short_queue)

    @property
    def num_long_requests(self) -> int:
        return len(self._long_queue)

    def has_short_requests(self) -> bool:
        return len(self._short_queue) > 0

    def has_long_requests(self) -> bool:
        return len(self._long_queue) > 0

    def has_immediate_requests(self) -> bool:
        return len(self._immediate_queue) > 0

    def _classify_queue(
        self, request: Request, *, force_immediate: bool = False
    ) -> RequestQueue:
        if force_immediate or request.request_id in self._force_immediate_request_ids:
            return self._immediate_queue
        if self.immediate_predicate is not None and self.immediate_predicate(request):
            return self._immediate_queue
        if request.num_prompt_tokens <= self.threshold:
            return self._short_queue
        return self._long_queue

    def _long_queue_starving(self) -> bool:
        """Whether the oldest long request has waited past its aging bound."""
        if self.long_max_wait_ms <= 0 or not self._long_queue:
            return False
        head = self._long_queue.peek_request()  # FCFS head = oldest long request
        enqueued_at = self._long_enqueue_at.get(head.request_id)
        if enqueued_at is None:
            return False
        return (time.monotonic() - enqueued_at) * 1000.0 >= self.long_max_wait_ms

    def begin_step(self, token_budget: int) -> None:
        """Reset per-step state and set the token reservation budget.
        Called once at the start of each schedule()."""
        self._long_tokens_this_step = 0
        self._long_token_budget = self.long_token_reservation * token_budget
        if self._debug_logging_enabled:
            logger.debug(
                "LAPS begin_step: token_budget=%d long_token_reservation=%.3f long_token_budget=%.0f",
                token_budget,
                self.long_token_reservation,
                self._long_token_budget,
            )

    def _select_schedulable_queue(self) -> RequestQueue | None:
        # Pure query (no side effects): called repeatedly per scheduling step.
        if self._immediate_queue:
            return self._immediate_queue
        if self._long_queue and self._long_queue_starving():
            # Token-based reservation: aged-long prefills may consume at most
            # long_token_reservation of the per-step token budget. This bounds
            # long-prefill compute without flipping the policy into long-first.
            # When short queue is non-empty, only promote long if budget remains.
            # When short queue is empty, always allow long (stall avoidance).
            if not self._short_queue:
                # Stall avoidance: allow long when no short is waiting
                self._debug_state(
                    "aged-long_selected_stall_avoidance",
                    extra=f"tokens={self._long_tokens_this_step:.0f} budget={self._long_token_budget:.0f}",
                )
                return self._long_queue
            if self._long_tokens_this_step < self._long_token_budget:
                # Reservation available: promote aged-long ahead of short
                self._debug_state(
                    "aged-long_selected_with_budget",
                    extra=f"tokens={self._long_tokens_this_step:.0f} budget={self._long_token_budget:.0f}",
                )
                return self._long_queue
            # Budget exhausted and short queue is non-empty: prefer short
            self._debug_state(
                "aged-long_blocked_budget_exhausted",
                extra=f"tokens={self._long_tokens_this_step:.0f} budget={self._long_token_budget:.0f}",
            )
            # (fall through to short queue check below)
        if self._short_queue:
            return self._short_queue
        if self._long_queue:
            return self._long_queue
        return None

    def has_schedulable_requests(self) -> bool:
        """Return whether a request can be dispatched right now."""
        return self._select_schedulable_queue() is not None

    def select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        return self._select_schedulable_queue()

    def mark_force_immediate(self, request_id: str) -> None:
        self._force_immediate_request_ids.add(request_id)

    @staticmethod
    def _request_id(request: Request | object) -> str | None:
        return getattr(request, "request_id", None)

    def _find_matching_request(
        self, queue: RequestQueue, request: Request | object
    ) -> Request | None:
        request_id = self._request_id(request)
        if request_id is None:
            return None
        for candidate in queue:
            if candidate.request_id == request_id:
                return cast(Request, candidate)
        return None

    def add_request(self, request: Request) -> None:
        queue = self._classify_queue(request)
        queue.add_request(request)
        self._queue_index[request.request_id] = queue
        if queue is self._long_queue:
            self._long_enqueue_at.setdefault(request.request_id, time.monotonic())
        if queue is not self._immediate_queue:
            self._force_immediate_request_ids.discard(request.request_id)
        self._debug_state("enqueue", request=request, queue=queue)
        self._maybe_log_stats()

    def pop_request(self) -> Request:
        queue = self._select_schedulable_queue()
        if queue is None:
            raise IndexError("pop from empty LAPS queue")
        return self.pop_request_from_queue(queue)

    def pop_request_from_queue(
        self,
        queue: RequestQueue,
        *,
        count_as_removal: bool = False,
        skip_or_requeue_reason: str | None = None,
        scheduled_tokens: int = 0,
    ) -> Request:
        request = queue.pop_request()
        self._queue_index.pop(request.request_id, None)
        enqueued_at = self._long_enqueue_at.pop(request.request_id, None)
        if count_as_removal:
            if skip_or_requeue_reason is not None:
                self._increment_skip_or_requeue_counter(
                    queue, skip_or_requeue_reason
                )
                event_name = f"skip_or_requeue:{skip_or_requeue_reason}"
            else:
                event_name = "remove"
        else:
            self._dispatch_counters[self._queue_name(queue)] += 1
            event_name = "dispatch"
            # Count long requests dispatched after aging past their bound.
            if (
                queue is self._long_queue
                and self.long_max_wait_ms > 0
                and enqueued_at is not None
                and (time.monotonic() - enqueued_at) * 1000.0
                >= self.long_max_wait_ms
            ):
                self._long_starvation_promotions += 1
                self._long_tokens_this_step += scheduled_tokens
                self._debug_state(
                    "aged-long_dispatched",
                    request=request,
                    queue=queue,
                    extra=f"scheduled_tokens={scheduled_tokens} "
                    f"total_tokens={self._long_tokens_this_step:.0f} "
                    f"budget={self._long_token_budget:.0f}",
                )
        self._force_immediate_request_ids.discard(request.request_id)
        self._debug_state(event_name, request=request, queue=queue)
        self._maybe_log_stats()
        return request

    def peek_request(self) -> Request:
        queue = self._select_schedulable_queue()
        if queue is None:
            raise IndexError("peek from an empty LAPS queue")
        return queue.peek_request()

    def prepend_request(self, request: Request, force_immediate: bool = False) -> None:
        if force_immediate:
            self._force_immediate_request_ids.add(request.request_id)
        queue = self._classify_queue(request, force_immediate=force_immediate)
        queue.prepend_request(request)
        self._queue_index[request.request_id] = queue
        if queue is self._long_queue:
            self._long_enqueue_at.setdefault(request.request_id, time.monotonic())
        self._prepend_counters[self._queue_name(queue)] += 1
        self._debug_state("prepend", request=request, queue=queue)
        self._maybe_log_stats()

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.prepend_request(cast(Request, request))

    def remove_request(self, request: Request) -> None:
        queue = self._queue_index.get(request.request_id)
        if queue is None:
            raise ValueError("request not found in LAPS queue")
        matched_request = self._find_matching_request(queue, request)
        if matched_request is None:
            raise ValueError("request not found in LAPS queue")
        queue.remove_request(matched_request)
        self._queue_index.pop(request.request_id, None)
        self._long_enqueue_at.pop(request.request_id, None)
        self._force_immediate_request_ids.discard(request.request_id)
        self._debug_state("remove", request=matched_request, queue=queue)
        self._maybe_log_stats()

    def remove_requests(self, requests: Iterable[Request]) -> None:
        queue_to_requests: dict[int, list[Request]] = {}
        queue_map = {id(q): q for q in self._queues()}
        removed_count = 0
        for request in requests:
            queue = self._queue_index.get(request.request_id)
            if queue is None:
                continue
            matched_request = self._find_matching_request(queue, request)
            if matched_request is not None:
                queue_to_requests.setdefault(id(queue), []).append(matched_request)
        for queue_id, matched_requests in queue_to_requests.items():
            removed_count += len(matched_requests)
            queue = queue_map[queue_id]
            queue.remove_requests(matched_requests)
            for matched in matched_requests:
                self._queue_index.pop(matched.request_id, None)
                self._long_enqueue_at.pop(matched.request_id, None)
                self._force_immediate_request_ids.discard(matched.request_id)
        if removed_count:
            self._debug_state("remove_batch", extra=f"count={removed_count}")
            self._maybe_log_stats()

    def __bool__(self) -> bool:
        return self._select_schedulable_queue() is not None

    def __len__(self) -> int:
        return (
            len(self._immediate_queue)
            + len(self._short_queue)
            + len(self._long_queue)
        )

    def __iter__(self) -> Iterator[Request]:
        yield from self._immediate_queue
        yield from self._short_queue
        yield from self._long_queue

    def __contains__(self, request: object) -> bool:
        request_id = self._request_id(request)
        return request_id is not None and request_id in self._queue_index


class _LAPSRequestClass(Enum):
    SHORT_PREFILL = "short_prefill"
    LONG_PREFILL = "long_prefill"


class LAPSSchedulerMixin:
    """Inject a LAPS-style waiting queue into vLLM's scheduler."""

    def _get_attached_waiting_computed_tokens(self, request: Request) -> int | None:
        return None

    def _is_recovery_request(self, request: Request) -> bool:
        """Recovery-style requests are prioritized via the immediate queue."""
        return (
            request.status == RequestStatus.PREEMPTED
            or request.num_computed_tokens > 0
            or request.num_external_computed_tokens > 0
            or self._get_attached_waiting_computed_tokens(request) is not None
            or request.num_output_tokens > 0
        )

    def _init_laps_waiting_queue(
        self,
        immediate_predicate: Callable[[Request], bool] | None = None,
    ) -> None:
        if self.policy != SchedulingPolicy.FCFS:
            logger.warning_once(
                "VLLM_ASCEND_LAPS_SCHEDULING currently supports only FCFS "
                "scheduler policy; keeping the default waiting queue."
            )
            return

        if immediate_predicate is None:
            immediate_predicate = self._is_recovery_request
        threshold = envs.VLLM_ASCEND_LAPS_THRESHOLD
        long_max_wait_ms = envs.VLLM_ASCEND_LAPS_LONG_MAX_WAIT_MS
        long_token_reservation = envs.VLLM_ASCEND_LAPS_LONG_TOKEN_RESERVATION
        self.waiting = LAPSRequestQueue(
            policy=self.policy,
            threshold=threshold,
            long_max_wait_ms=long_max_wait_ms,
            long_token_reservation=long_token_reservation,
            immediate_predicate=immediate_predicate,
        )
        logger.info(
            "LAPS scheduling enabled on Ascend: threshold=%d, "
            "long_max_wait_ms=%.3f, long_token_reservation=%.3f",
            threshold,
            long_max_wait_ms,
            long_token_reservation,
        )

    def _laps_waiting_queue(self) -> LAPSRequestQueue | None:
        if isinstance(self.waiting, LAPSRequestQueue):
            return self.waiting
        return None

    def _laps_threshold(self) -> int:
        laps_waiting = self._laps_waiting_queue()
        if laps_waiting is not None:
            return laps_waiting.threshold
        return envs.VLLM_ASCEND_LAPS_THRESHOLD

    def _classify_laps_request(
        self, request: Request, num_computed_tokens: int | None = None
    ) -> _LAPSRequestClass | None:
        computed = (
            request.num_computed_tokens
            if num_computed_tokens is None
            else num_computed_tokens
        )
        if computed >= request.num_prompt_tokens:
            return None
        if request.num_prompt_tokens <= self._laps_threshold():
            return _LAPSRequestClass.SHORT_PREFILL
        return _LAPSRequestClass.LONG_PREFILL

    def _is_prefill_request(
        self, request: Request, num_computed_tokens: int | None = None
    ) -> bool:
        return self._classify_laps_request(request, num_computed_tokens) is not None

    def _is_short_prefill_request(
        self, request: Request, num_computed_tokens: int | None = None
    ) -> bool:
        return (
            self._classify_laps_request(request, num_computed_tokens)
            is _LAPSRequestClass.SHORT_PREFILL
        )

    def _is_long_prefill_request(
        self, request: Request, num_computed_tokens: int | None = None
    ) -> bool:
        return (
            self._classify_laps_request(request, num_computed_tokens)
            is _LAPSRequestClass.LONG_PREFILL
        )

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        waiting = getattr(self, "waiting", None)
        if isinstance(waiting, LAPSRequestQueue):
            skipped_waiting = getattr(self, "skipped_waiting", None)
            if self.policy == SchedulingPolicy.FCFS and skipped_waiting:
                return skipped_waiting
            queue = waiting.select_waiting_queue_for_scheduling()
            if queue is not None:
                return queue
            return skipped_waiting or None
        return super()._select_waiting_queue_for_scheduling()

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        waiting = getattr(self, "waiting", None)
        if isinstance(waiting, LAPSRequestQueue):
            waiting.mark_force_immediate(request.request_id)
        super()._preempt_request(request, timestamp)
