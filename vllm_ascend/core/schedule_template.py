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

import time
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, SchedulingPolicy, create_request_queue
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request, RequestStatus
from vllm.v1.utils import record_function_or_nullcontext


@dataclass
class AscendScheduleStepState:
    scheduled_new_reqs: list[Request] = field(default_factory=list)
    scheduled_resumed_reqs: list[Request] = field(default_factory=list)
    scheduled_running_reqs: list[Request] = field(default_factory=list)
    preempted_reqs: list[Request] = field(default_factory=list)
    req_to_new_blocks: dict[str, KVCacheBlocks] = field(default_factory=dict)
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    token_budget: int = 0
    scheduled_encoder_inputs: dict[str, list[int]] = field(default_factory=dict)
    encoder_compute_budget: int = 0
    scheduled_spec_decode_tokens: dict[str, list[int]] = field(default_factory=dict)
    scheduled_timestamp: float = 0.0
    laps_ctx: Any | None = None
    total_num_scheduled_tokens: int = 0
    num_common_prefix_blocks: list[int] = field(default_factory=list)
    new_reqs_data: list[NewRequestData] = field(default_factory=list)
    cached_reqs_data: Any = None
    new_block_ids_to_zero: Any = None


@dataclass
class RecomputeScheduleStepState(AscendScheduleStepState):
    recomputed_reqs: list[Any] = field(default_factory=list)


class AscendSchedulerTemplateMixin:
    """Shared schedule skeleton for Ascend scheduler variants."""

    def _make_step_state(
        self,
        token_budget: int,
        scheduled_timestamp: float,
    ) -> AscendScheduleStepState:
        return AscendScheduleStepState(
            token_budget=token_budget,
            encoder_compute_budget=self.max_num_encoder_input_tokens,
            scheduled_timestamp=scheduled_timestamp,
        )

    def _adjust_running_num_new_tokens(
        self,
        state: AscendScheduleStepState,
        request: Request,
        num_new_tokens: int,
    ) -> tuple[int, bool]:
        return num_new_tokens, False

    def _adjust_waiting_num_new_tokens(
        self,
        state: AscendScheduleStepState,
        request: Request,
        num_new_tokens: int,
        num_computed_tokens: int,
    ) -> tuple[int, bool, bool]:
        return num_new_tokens, False, False

    def _should_break_for_non_chunked_waiting_prefill(
        self,
        request: Request,
        num_new_tokens: int,
        token_budget: int,
        num_computed_tokens: int,
    ) -> bool:
        return (
            not self.scheduler_config.enable_chunked_prefill
            and num_new_tokens > token_budget
        )

    def _release_scheduled_running_request(
        self,
        state: AscendScheduleStepState,
        request: Request,
    ) -> None:
        request_id = request.request_id
        state.scheduled_running_reqs.remove(request)
        released_tokens = state.num_scheduled_tokens.pop(request_id)
        state.token_budget += released_tokens
        self._rollback_scheduled_request(state, request, released_tokens)
        state.req_to_new_blocks.pop(request_id, None)
        state.scheduled_spec_decode_tokens.pop(request_id, None)
        preempted_encoder_inputs = state.scheduled_encoder_inputs.pop(request_id, None)
        if preempted_encoder_inputs:
            num_embeds_to_restore = sum(
                request.get_num_encoder_embeds(i) for i in preempted_encoder_inputs
            )
            state.encoder_compute_budget += num_embeds_to_restore

    def _handle_running_allocation_failure(
        self,
        state: AscendScheduleStepState,
        request: Request,
        scheduled_timestamp: float,
        req_index: int,
    ) -> tuple[int, bool]:
        if self.policy == SchedulingPolicy.PRIORITY:
            preempted_req = max(
                self.running,
                key=lambda r: (r.priority, r.arrival_time),
            )
            self.running.remove(preempted_req)
            if preempted_req in state.scheduled_running_reqs:
                self._release_scheduled_running_request(state, preempted_req)
                req_index -= 1
        else:
            preempted_req = self.running.pop()

        self._preempt_request(preempted_req, scheduled_timestamp)
        state.preempted_reqs.append(preempted_req)
        return req_index, preempted_req == request

    def _should_schedule_waiting_phase(
        self,
        state: AscendScheduleStepState,
    ) -> bool:
        return (
            not state.preempted_reqs
            and self._pause_state == PauseState.UNPAUSED
        )

    def _get_waiting_num_new_tokens_base(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> int:
        return request.num_tokens - num_computed_tokens

    def _pop_waiting_request_for_schedule(
        self,
        request_queue: RequestQueue,
        *,
        count_as_removal: bool = False,
    ) -> Request:
        return request_queue.pop_request()

    def _record_scheduled_request(
        self,
        state: AscendScheduleStepState,
        request: Request,
        num_new_tokens: int,
        num_computed_tokens: int | None = None,
        was_capped: bool = False,
    ) -> None:
        return None

    def _rollback_scheduled_request(
        self,
        state: AscendScheduleStepState,
        request: Request,
        released_tokens: int,
    ) -> None:
        return None

    def _build_scheduler_output(
        self,
        state: AscendScheduleStepState,
        num_common_prefix_blocks: list[int],
    ) -> SchedulerOutput:
        return SchedulerOutput(
            scheduled_new_reqs=state.new_reqs_data,
            scheduled_cached_reqs=state.cached_reqs_data,
            num_scheduled_tokens=state.num_scheduled_tokens,
            total_num_scheduled_tokens=state.total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=state.scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=state.scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in state.preempted_reqs},
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=state.new_block_ids_to_zero,
        )

    def _finalize_step_state(self, state: AscendScheduleStepState) -> None:
        return None

    def _schedule_with_hooks(self) -> SchedulerOutput:
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            token_budget = 0

        scheduled_timestamp = time.monotonic()
        state = self._make_step_state(token_budget, scheduled_timestamp)

        self.kv_cache_manager.new_step_starts()

        req_index = 0
        while req_index < len(self.running) and state.token_budget > 0:
            request = self.running[req_index]

            if (
                request.num_output_placeholders > 0
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens, was_capped = self._adjust_running_num_new_tokens(
                state, request, num_new_tokens
            )
            if num_new_tokens == 0:
                req_index += 1
                continue
            num_new_tokens = min(num_new_tokens, state.token_budget)

            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )

            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = state.encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    state.encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            if num_new_tokens == 0:
                req_index += 1
                continue

            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        break

                    req_index, should_break = self._handle_running_allocation_failure(
                        state, request, scheduled_timestamp, req_index
                    )
                    if should_break:
                        break

            if new_blocks is None:
                break

            state.scheduled_running_reqs.append(request)
            request_id = request.request_id
            state.req_to_new_blocks[request_id] = new_blocks
            state.num_scheduled_tokens[request_id] = num_new_tokens
            state.token_budget -= num_new_tokens
            self._record_scheduled_request(
                state, request, num_new_tokens, was_capped=was_capped
            )
            req_index += 1

            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    state.scheduled_spec_decode_tokens[request.request_id] = (
                        spec_token_ids
                    )
                request.spec_token_ids = []

            if encoder_inputs_to_schedule:
                state.scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                state.encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in state.scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        if self._should_schedule_waiting_phase(state):
            step_skipped_waiting = create_request_queue(self.policy)

            while (self.waiting or self.skipped_waiting) and state.token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request_queue = self._select_waiting_queue_for_scheduling()
                if request_queue is None:
                    break

                request = request_queue.peek_request()
                request_id = request.request_id

                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request):
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                    request = self._pop_waiting_request_for_schedule(
                        request_queue, count_as_removal=True
                    )
                    step_skipped_waiting.prepend_request(request)
                    continue

                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    request = self._pop_waiting_request_for_schedule(
                        request_queue, count_as_removal=True
                    )
                    step_skipped_waiting.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0

                attached_computed_tokens = self._get_attached_waiting_computed_tokens(
                    request
                )
                if attached_computed_tokens is not None:
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = attached_computed_tokens
                elif request.num_computed_tokens == 0:
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:
                            request = self._pop_waiting_request_for_schedule(
                                request_queue, count_as_removal=True
                            )
                            step_skipped_waiting.prepend_request(request)
                            continue

                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens
                else:
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = state.encoder_compute_budget
                was_capped = False

                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    num_new_tokens = self._get_waiting_num_new_tokens_base(
                        request, num_computed_tokens
                    )
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
                    if self._should_break_for_non_chunked_waiting_prefill(
                        request,
                        num_new_tokens,
                        state.token_budget,
                        num_computed_tokens,
                    ):
                        break
                    (
                        num_new_tokens,
                        was_capped,
                        should_break,
                    ) = self._adjust_waiting_num_new_tokens(
                        state,
                        request,
                        num_new_tokens,
                        num_computed_tokens,
                    )
                    if should_break:
                        break
                    num_new_tokens = min(num_new_tokens, state.token_budget)
                    assert num_new_tokens > 0

                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            state.encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            break

                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        break

                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                num_encoder_tokens = 0
                if (
                    self.is_encoder_decoder
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
                        )

                request = self._pop_waiting_request_for_schedule(request_queue)
                if load_kv_async:
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
                    request.num_computed_tokens = num_computed_tokens
                    continue

                if getattr(self, "is_mtp_kv_consumer", False) and request.spec_token_ids:
                    num_scheduled_spec_tokens = (
                        num_new_tokens
                        + request.num_computed_tokens
                        - request.num_tokens
                        - request.num_output_placeholders
                    )
                    if num_scheduled_spec_tokens > 0:
                        spec_token_ids = request.spec_token_ids
                        if len(spec_token_ids) > num_scheduled_spec_tokens:
                            spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                        state.scheduled_spec_decode_tokens[request.request_id] = (
                            spec_token_ids
                        )
                    request.spec_token_ids = []

                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, state.scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    state.scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    state.scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                state.req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                state.num_scheduled_tokens[request_id] = num_new_tokens
                state.token_budget -= num_new_tokens
                self._record_scheduled_request(
                    state,
                    request,
                    num_new_tokens,
                    num_computed_tokens=num_computed_tokens,
                    was_capped=was_capped,
                )
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                if encoder_inputs_to_schedule:
                    state.scheduled_encoder_inputs[request_id] = (
                        encoder_inputs_to_schedule
                    )
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
                    state.encoder_compute_budget = new_encoder_compute_budget
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)

            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

        state.total_num_scheduled_tokens = sum(state.num_scheduled_tokens.values())
        assert state.total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert state.token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        assert (
            len(state.scheduled_new_reqs)
            + len(state.scheduled_resumed_reqs)
            + len(state.scheduled_running_reqs)
            <= len(self.running)
        )

        state.num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                state.num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        if self.use_v2_model_runner:
            scheduled_new_reqs = (
                state.scheduled_new_reqs + state.scheduled_resumed_reqs
            )
            scheduled_resumed_reqs = []
            state.new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    state.req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            scheduled_new_reqs = state.scheduled_new_reqs
            scheduled_resumed_reqs = state.scheduled_resumed_reqs
            state.new_reqs_data = [
                NewRequestData.from_request(
                    req, state.req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            state.cached_reqs_data = self._make_cached_request_data(
                state.scheduled_running_reqs,
                scheduled_resumed_reqs,
                state.num_scheduled_tokens,
                state.scheduled_spec_decode_tokens,
                state.req_to_new_blocks,
            )

        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(state.num_scheduled_tokens.keys())

        state.new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None)
            if self.needs_kv_cache_zeroing
            else None
        )

        scheduler_output = self._build_scheduler_output(
            state, state.num_common_prefix_blocks
        )

        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        if self.ec_connector is not None:
            ec_meta = self.ec_connector.build_connector_meta(scheduler_output)
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)

        self._finalize_step_state(state)
        return scheduler_output
