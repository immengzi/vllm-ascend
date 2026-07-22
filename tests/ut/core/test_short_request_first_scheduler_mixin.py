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
"""Tests for the ShortRequestFirst scheduler injection glue."""

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import RequestStatus

from vllm_ascend.core.short_request_first_scheduler import (
    ShortRequestFirstRequestQueue,
    install_short_request_first_waiting_queue,
    is_recovery_request,
)

THRESHOLD = 256


def make_request(
    request_id: str,
    prompt_len: int,
    computed: int = 0,
    *,
    status: RequestStatus = RequestStatus.WAITING,
    num_output_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        num_prompt_tokens=prompt_len,
        num_computed_tokens=computed,
        status=status,
        num_output_tokens=num_output_tokens,
    )


class SchedulerStub:
    def __init__(self, waiting=None, skipped_waiting=None, policy=SchedulingPolicy.FCFS):
        self.waiting = waiting or create_request_queue(policy)
        self.skipped_waiting = skipped_waiting or create_request_queue(policy)
        self.policy = policy


def test_install_replaces_waiting_queue_and_preserves_skipped_waiting():
    scheduler = SchedulerStub()
    skipped = scheduler.skipped_waiting

    queue = install_short_request_first_waiting_queue(
        scheduler,
        threshold=THRESHOLD,
        long_max_wait_ms=100.0,
    )

    assert isinstance(queue, ShortRequestFirstRequestQueue)
    assert scheduler.waiting is queue
    assert scheduler.skipped_waiting is skipped


def test_install_is_idempotent():
    scheduler = SchedulerStub()
    first = install_short_request_first_waiting_queue(
        scheduler,
        threshold=THRESHOLD,
        long_max_wait_ms=0.0,
    )
    second = install_short_request_first_waiting_queue(
        scheduler,
        threshold=THRESHOLD + 1,
        long_max_wait_ms=100.0,
    )

    assert second is first
    assert second.threshold == THRESHOLD


def test_install_rejects_non_fcfs_policy():
    scheduler = SchedulerStub(policy=SchedulingPolicy.PRIORITY)
    with pytest.raises(ValueError, match="requires FCFS"):
        install_short_request_first_waiting_queue(
            scheduler,
            threshold=THRESHOLD,
            long_max_wait_ms=0.0,
        )


def test_install_rejects_nonempty_waiting_queue():
    scheduler = SchedulerStub()
    scheduler.waiting.add_request(make_request("already_waiting", 1))

    with pytest.raises(RuntimeError, match="before request admission"):
        install_short_request_first_waiting_queue(
            scheduler,
            threshold=THRESHOLD,
            long_max_wait_ms=0.0,
        )


def test_recovery_markers_route_long_prompt_to_immediate_queue():
    long_len = THRESHOLD + 100
    cases = {
        "preempted": make_request("preempted", long_len, status=RequestStatus.PREEMPTED),
        "computed": make_request("computed", long_len, computed=1),
        "output": make_request("output", long_len, num_output_tokens=1),
    }
    for name, request in cases.items():
        queue = ShortRequestFirstRequestQueue(
            policy=SchedulingPolicy.FCFS,
            threshold=THRESHOLD,
            long_max_wait_ms=0.0,
            immediate_predicate=is_recovery_request,
        )
        queue.add_request(request)
        assert queue.num_immediate_requests == 1, name
        assert queue.num_long_requests == 0, name
        assert queue.num_short_requests == 0, name


def test_non_recovery_requests_stay_in_length_based_queues():
    long_queue = ShortRequestFirstRequestQueue(
        policy=SchedulingPolicy.FCFS,
        threshold=THRESHOLD,
        long_max_wait_ms=0.0,
        immediate_predicate=is_recovery_request,
    )
    long_queue.add_request(make_request("plain_long", THRESHOLD + 100))
    assert long_queue.num_long_requests == 1
    assert long_queue.num_immediate_requests == 0

    short_queue = ShortRequestFirstRequestQueue(
        policy=SchedulingPolicy.FCFS,
        threshold=THRESHOLD,
        long_max_wait_ms=0.0,
        immediate_predicate=is_recovery_request,
    )
    short_queue.add_request(make_request("plain_short", 10))
    assert short_queue.num_short_requests == 1
    assert short_queue.num_immediate_requests == 0
