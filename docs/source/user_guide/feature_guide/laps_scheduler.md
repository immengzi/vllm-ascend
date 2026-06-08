# LAPS-Inspired Prefill Scheduling

`vllm-ascend` now provides an engine-side, NPU-oriented adaptation of the core
LAPS scheduling idea from the paper *Length-Aware Prefill Scheduling for LLM
Serving*.

## What Was Ported

The original LAPS artifact on top of SGLang contains three layers:

1. Dual-queue scheduling for short and long prefills.
2. A short-request waiting window to build better short-prefill batches.
3. Dynamic GPU allocation in the router/load balancer.

For `vllm-ascend`, the currently ported pieces are the **engine-local**
mechanisms:

- `triple-queue`: split the waiting queue into immediate, short, and long prompt
  classes.
- `anti-starvation (aging)`: bound how long a long prefill can be held behind
  short prefills, so a sustained short-request stream cannot starve long
  requests indefinitely.

The original LAPS short-request *waiting window* was **not** ported: vLLM v1
already performs continuous batching (it re-batches all running requests and
admits new ones up to the token/running budget every step, with chunked
prefill), so manually accumulating a short batch is redundant and only adds
latency.

The CUDA-specific `attention-in-graph` optimization from the LAPS SGLang branch
was intentionally **not** ported, because it is tightly coupled to CUDA graph
capture and FlashAttention internals (and is incompatible with the two-stage
sparse attention used by models such as DeepSeek-V4). Likewise, router-level
dynamic allocation has **not** been merged into the Ascend proxy layer yet.

## Why It Fits `vllm-ascend`

The vLLM scheduler already has a centralized waiting queue in the engine core,
which makes the LAPS queueing policy portable without changing model execution
code. This is a good match for NPU deployment because the main benefit here is
request isolation and batching behavior, not CUDA-only kernel machinery.

## Configuration

Enable the feature with environment variables before launching `vllm serve`:

```bash
export VLLM_ASCEND_LAPS_SCHEDULING=1
export VLLM_ASCEND_LAPS_THRESHOLD=256
export VLLM_ASCEND_LAPS_LONG_MAX_WAIT_MS=2000
```

### Variables

- `VLLM_ASCEND_LAPS_SCHEDULING`
  - `1` enables the Ascend LAPS scheduler.
  - `0` disables it.
- `VLLM_ASCEND_LAPS_THRESHOLD`
  - Prompts with `num_prompt_tokens <= threshold` are treated as short.
- `VLLM_ASCEND_LAPS_LONG_MAX_WAIT_MS`
  - Anti-starvation aging bound for long prefills, in milliseconds.
  - A long request that has waited longer than this is promoted ahead of short
    prefills, bounding its worst-case admission wait.
  - `0` disables aging (strict short-priority).

## How It Is Selected

`vllm-ascend` selects the scheduler at config time:

- `VLLM_ASCEND_LAPS_SCHEDULING=0`
  - Keep the normal scheduler path.
- `VLLM_ASCEND_LAPS_SCHEDULING=1` and `recompute_scheduler_enable=true`
  - Keep `RecomputeScheduler`, and install the LAPS waiting queue inside it.
- `VLLM_ASCEND_LAPS_SCHEDULING=1` and `recompute_scheduler_enable=false`
  - LAPS is not activated. The platform logs a warning and keeps the default
    scheduler path.
- `SLO_limits_for_dynamic_batch != -1`
  - `SchedulerDynamicBatch` takes precedence and LAPS is ignored.

In other words, the effective priority is:

`dynamic batch > recompute (+ optional LAPS) > default`

## Minimal Examples

Enable recompute scheduler together with LAPS:

```bash
export VLLM_ASCEND_LAPS_SCHEDULING=1
export VLLM_ASCEND_LAPS_THRESHOLD=256
export VLLM_ASCEND_LAPS_LONG_MAX_WAIT_MS=2000

vllm serve <model> \
  --additional-config '{"recompute_scheduler_enable": true}'
```

Disable LAPS explicitly:

```bash
export VLLM_ASCEND_LAPS_SCHEDULING=0
```

## Scheduling Semantics

The `LAPSRequestQueue` manages three queues:

- **immediate queue**: Requests that must be dispatched immediately, such as
  preempted requests or requests with already-computed tokens (recovery flows).
- **short queue**: Short prefills where `num_prompt_tokens <= threshold`.
- **long queue**: Long prefills where `num_prompt_tokens > threshold`.

Dispatch priority is: immediate > aged-long > short > long.

- Immediate requests are dispatched as soon as they arrive.
- Short requests are dispatched whenever the short queue is non-empty (vLLM's
  continuous batching groups them together each step).
- Long requests are normally only dispatched when no immediate or short
  requests are schedulable.
- **Anti-starvation:** if the oldest long request has waited longer than
  `VLLM_ASCEND_LAPS_LONG_MAX_WAIT_MS`, it is promoted ahead of short prefills.
  This bounds each long request's worst-case admission wait so a sustained
  short-request stream cannot starve long prefills. Aged long requests are
  drained one at a time, so this does not flood the batch with long prefills.

## Current Scope and Limitations

- The current implementation is enabled only for the **FCFS** scheduler policy.
- The adaptation is **engine-local**; it does not yet rebalance prefill and
  decode instances across nodes or proxies.
- The implementation targets the vLLM waiting-queue layer and is intended for
  PD / EPD style serving where prompt length skew is a dominant bottleneck.
- LAPS currently requires `recompute_scheduler_enable=true`; enabling only
  `VLLM_ASCEND_LAPS_SCHEDULING=1` is not sufficient.
- When dynamic batch is selected through `SLO_limits_for_dynamic_batch`,
  LAPS is not applied.
