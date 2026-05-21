##
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/v1/core/sched/scheduler.py
#

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, fields

from vllm.config import SchedulerConfig, VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import (
    SchedulingPolicy,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.utils import ConstantList, record_function_or_nullcontext

from vllm_ascend import envs
from vllm_ascend.core.laps_scheduler import LAPSBudgetContext, LAPSSchedulerMixin
from vllm_ascend.core.schedule_template import (
    AscendScheduleStepState,
    RecomputeScheduleStepState,
)

# `spec_manager_map` in single_type_kv_cache_manager is a module-level dict
# whose keys are class objects bound at import time.  When the async
# recompute scheduler is enabled, `recompute_scheduler.py` is imported by
# `check_and_update_config()` (via AsyncScheduler → scheduler.py →
# kv_cache_coordinator → single_type_kv_cache_manager) *before*
# this patch file is executed a second time (e.g. triggered by
# unpickling an AscendMLAAttentionSpec in the EngineCoreProc subprocess).
# In that case the dict already contains the original MLAAttentionSpec
# class as a key, so a subsequent lookup with type(AscendMLAAttentionSpec
# instance) raises KeyError.
#
# Fix: whenever this patch is applied, register AscendMLAAttentionSpec as
# an additional key in spec_manager_map (if the module is already loaded).
def register_ascend_mla_spec_in_manager():
    import sys as _sys

    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    from vllm.v1.kv_cache_interface import MLAAttentionSpec as AscendMLAAttentionSpec

    _stm = _sys.modules.get("vllm.v1.core.single_type_kv_cache_manager")
    if _stm is not None and AscendMLAAttentionSpec not in _stm.spec_manager_map:
        _stm.spec_manager_map[AscendMLAAttentionSpec] = FullAttentionManager


@dataclass
class RecomputeSchedulerConfig(SchedulerConfig):
    scheduler_cls: str | type[object] = "vllm_ascend.core.recompute_scheduler.RecomputeScheduler"

    @classmethod
    def initialize_from_config(cls, vllm_config: VllmConfig):
        vllm_scheduler_config = vllm_config.scheduler_config
        scheduler_config = {
            field.name: getattr(vllm_scheduler_config, field.name)
            for field in fields(vllm_scheduler_config)
            if field.init
        }
        if vllm_scheduler_config.async_scheduling:
            scheduler_config["scheduler_cls"] = "vllm_ascend.core.recompute_scheduler.AsyncRecomputeScheduler"
        else:
            scheduler_config["scheduler_cls"] = "vllm_ascend.core.recompute_scheduler.RecomputeScheduler"
        scheduler_config["max_model_len"] = vllm_config.model_config.max_model_len
        scheduler_config["is_encoder_decoder"] = vllm_config.model_config.is_encoder_decoder
        return cls(**scheduler_config)


@dataclass
class RecomputeReqInfo:
    request_id: str
    output_token_ids: ConstantList
    client_index: int = 0


@dataclass
class RecomputeSchedulerOutput(SchedulerOutput):
    recomputed_reqs: list[RecomputeReqInfo] | None = None


class RecomputeScheduler(LAPSSchedulerMixin, Scheduler):
    running: list[Request]

    def __init__(self, *args, **kwargs):
        register_ascend_mla_spec_in_manager()

        super().__init__(*args, **kwargs)
        # When is_mtp_kv_consumer is true, we will fill request.spec_token_ids
        # with placeholder tokens to enable full graph when decode nodes pull
        # the KV cache of one request from prefill nodes.
        self.is_mtp_kv_consumer = (
            self.vllm_config.speculative_config
            and self.vllm_config.kv_transfer_config
            and self.vllm_config.kv_transfer_config.is_kv_consumer
        )
        self.is_kv_producer = self.vllm_config.kv_transfer_config and self.vllm_config.kv_transfer_config.is_kv_producer
        self.is_hybrid_model = (
            "qwen3_next" in self.vllm_config.model_config.hf_text_config.model_type
            or "qwen3_5" in self.vllm_config.model_config.hf_text_config.model_type
        )
        if envs.VLLM_ASCEND_LAPS_SCHEDULING:
            self._init_laps_waiting_queue()

    def _get_attached_waiting_computed_tokens(self, request: Request) -> int | None:
        """Return already attached computed tokens for a waiting request.

        In PD recovery flows a request can legitimately carry attached/cached KV
        blocks while ``num_computed_tokens`` has been reset to 0 for a retry.
        Re-running ``get_computed_blocks()`` for such a request violates the KV
        cache manager invariant for resumed requests because the blocks are
        already attached to the request.
        """
        if request.num_computed_tokens != 0:
            return None

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)
        if not any(block_ids):
            return None

        return max(request.num_cached_tokens, 0)

    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)
            elif update is not None:
                # Commence next input chunk.
                self._update_request_as_session(existing, update)
            else:
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            if request.resumable:
                request.streaming_queue = deque()
            # Fill in placeholder tokens to enable full graph compatibility. Without
            # placeholders, graph matching may fail, forcing eager mode execution.
            if self.is_kv_producer and self.is_hybrid_model and request.num_tokens > 1:
                request.prompt_token_ids.pop()
                request._all_token_ids.pop()
                request.num_prompt_tokens -= 1
            if self.is_mtp_kv_consumer:
                request.spec_token_ids = [PLACEHOLDER_TOKEN_ID] * self.num_spec_tokens
            self._enqueue_waiting_request(request)
            self.requests[request.request_id] = request
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.

        NOTE: The check for whether request.request_id is in
        finished_recving_kv_req_ids is now done by the caller
        (_try_promote_blocked_waiting_request in the parent Scheduler),
        so this method is only called when the recv is confirmed finished.
        """
        assert self.connector is not None

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            # Use Ascend-specific block_ids logic to handle multi-group KV
            # cache configurations (e.g. MLA) where len(block_ids) > 1.
            block_ids = self.kv_cache_manager.get_block_ids(request.request_id)
            if len(block_ids) == 1:
                num_computed_tokens = len(block_ids[0]) * self.block_size
                # Handle the case where num request tokens less than one block.
                num_computed_tokens = min(num_computed_tokens, request.num_tokens)
            else:
                num_computed_tokens = request.num_tokens
            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            if num_computed_tokens == request.num_tokens:
                num_computed_tokens -= 1
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

            # Update the request state for scheduling.
            request.num_computed_tokens = num_computed_tokens

            # Count the number of prefix cached tokens.
            if request.num_cached_tokens < 0:
                request.num_cached_tokens = request.num_computed_tokens

        self.finished_recving_kv_req_ids.remove(request.request_id)

    def _make_step_state(
        self,
        token_budget: int,
        scheduled_timestamp: float,
    ) -> RecomputeScheduleStepState:
        state = RecomputeScheduleStepState(
            token_budget=token_budget,
            encoder_compute_budget=self.max_num_encoder_input_tokens,
            scheduled_timestamp=scheduled_timestamp,
        )
        if self._laps_long_budgeting_enabled():
            state.laps_ctx = LAPSBudgetContext(self, token_budget)
        return state

    def _adjust_running_num_new_tokens(
        self,
        state: AscendScheduleStepState,
        request: Request,
        num_new_tokens: int,
    ) -> tuple[int, bool, object | None]:
        if state.laps_ctx is None:
            return min(num_new_tokens, state.token_budget), False, None
        return state.laps_ctx.adjust_tokens(request, num_new_tokens, state.token_budget)

    def _adjust_waiting_num_new_tokens(
        self,
        state: AscendScheduleStepState,
        request: Request,
        num_new_tokens: int,
        num_computed_tokens: int,
    ) -> tuple[int, bool, bool, object | None]:
        if state.laps_ctx is None:
            return min(num_new_tokens, state.token_budget), False, False, None
        num_new_tokens, was_capped, request_class = state.laps_ctx.adjust_tokens(
            request,
            num_new_tokens,
            state.token_budget,
            num_computed_tokens=num_computed_tokens,
        )
        if num_new_tokens != 0:
            return num_new_tokens, was_capped, False, request_class
        num_new_tokens, should_break = state.laps_ctx.recover_zero_budget(
            request,
            request_class,
            state.token_budget,
            num_computed_tokens,
        )
        return num_new_tokens, was_capped, should_break, request_class

    def _handle_running_allocation_failure(
        self,
        state: AscendScheduleStepState,
        request: Request,
        scheduled_timestamp: float,
        req_index: int,
    ) -> tuple[int, bool]:
        transfer_config = self.vllm_config.kv_transfer_config
        if transfer_config is not None and not transfer_config.is_kv_producer:
            recomputed_req = self.running.pop()
            self.kv_cache_manager.free(recomputed_req)
            if recomputed_req in state.scheduled_running_reqs:
                self._release_scheduled_running_request(state, recomputed_req)
            state.recomputed_reqs.append(
                RecomputeReqInfo(
                    recomputed_req.request_id,
                    recomputed_req.output_token_ids,
                    recomputed_req.client_index,
                )
            )
            return req_index, recomputed_req == request
        return super()._handle_running_allocation_failure(
            state, request, scheduled_timestamp, req_index
        )

    def _should_schedule_waiting_phase(
        self,
        state: AscendScheduleStepState,
    ) -> bool:
        return (
            not state.preempted_reqs
            and not state.recomputed_reqs
            and self._pause_state == PauseState.UNPAUSED
        )

    def _get_waiting_num_new_tokens_base(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> int:
        if self.is_mtp_kv_consumer:
            return request.num_tokens_with_spec - num_computed_tokens
        return super()._get_waiting_num_new_tokens_base(request, num_computed_tokens)

    def _record_scheduled_request(
        self,
        state: AscendScheduleStepState,
        request: Request,
        num_new_tokens: int,
        num_computed_tokens: int | None = None,
        was_capped: bool = False,
        request_class: object | None = None,
    ) -> None:
        if state.laps_ctx is not None:
            assert request_class is not None
            state.laps_ctx.record_scheduled(
                request,
                request_class,
                num_new_tokens,
                was_capped,
            )

    def _rollback_scheduled_request(
        self,
        state: AscendScheduleStepState,
        request: Request,
        released_tokens: int,
    ) -> None:
        if state.laps_ctx is not None:
            state.laps_ctx.rollback_scheduled(request, released_tokens)

    def _build_scheduler_output(
        self,
        state: AscendScheduleStepState,
        num_common_prefix_blocks: list[int],
    ) -> RecomputeSchedulerOutput:
        scheduler_output = super()._build_scheduler_output(
            state, num_common_prefix_blocks
        )
        return RecomputeSchedulerOutput(
            scheduled_new_reqs=scheduler_output.scheduled_new_reqs,
            scheduled_cached_reqs=scheduler_output.scheduled_cached_reqs,
            num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
            total_num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduler_output.scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduler_output.scheduled_encoder_inputs,
            num_common_prefix_blocks=scheduler_output.num_common_prefix_blocks,
            preempted_req_ids=scheduler_output.preempted_req_ids,
            finished_req_ids=scheduler_output.finished_req_ids,
            free_encoder_mm_hashes=scheduler_output.free_encoder_mm_hashes,
            new_block_ids_to_zero=scheduler_output.new_block_ids_to_zero,
            recomputed_reqs=state.recomputed_reqs,
        )

    def _finalize_step_state(self, state: AscendScheduleStepState) -> None:
        laps_waiting = self._laps_waiting_queue()
        if laps_waiting is not None and state.laps_ctx is not None:
            state.laps_ctx.finalize(laps_waiting)

    def schedule(self) -> RecomputeSchedulerOutput:
        return self._schedule_with_hooks()

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(kv_connector_output.invalid_block_ids)

        # return recomputed requests as EngineCoreOutput
        if scheduler_output.recomputed_reqs is not None:
            for req_info in scheduler_output.recomputed_reqs:
                outputs[req_info.client_index].append(
                    EngineCoreOutput(
                        request_id=req_info.request_id,
                        finish_reason=FinishReason.STOP,
                        new_token_ids=[],
                        stop_reason="recomputed",
                    )
                )

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                # cache transfer in KV connector), the aborted request will not
                # be set to None (in order to finish async KV transfer).
                # In this case, we use is_finished() to check.
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []

            scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            routed_experts = None
            finish_reason = None
            if stopped:
                routed_experts = self._get_routed_experts(request)

                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params = self._free_request(request)

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None and request.sampling_params.logprobs is not None and logprobs:
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                ok = struct_output_request.grammar.accept_tokens(req_id, new_token_ids)
                if not ok:
                    logger.warning(
                        "Unexpected: grammar rejected tokens %s for request %s.",
                        new_token_ids,
                        req_id,
                    )

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None or kv_transfer_params or stopped:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                        num_external_computed_tokens=request.num_external_computed_tokens,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {client_index: EngineCoreOutputs(outputs=outs) for client_index, outs in outputs.items()}

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(finished_requests=finished_set)
            finished_req_ids.clear()

        if (stats := self.make_stats(spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs


class AsyncRecomputeScheduler(AsyncScheduler, RecomputeScheduler):
    def __init__(self, *args, **kwargs):
        register_ascend_mla_spec_in_manager()

        super().__init__(*args, **kwargs)
