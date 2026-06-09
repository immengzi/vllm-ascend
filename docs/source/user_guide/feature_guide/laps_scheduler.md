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
export VLLM_ASCEND_LAPS_LONG_TOKEN_RESERVATION=0.2
```

### Variables

- `VLLM_ASCEND_LAPS_SCHEDULING`
  - `1` enables the Ascend LAPS scheduler.
  - `0` disables it.
- `VLLM_ASCEND_LAPS_THRESHOLD`
  - Prompts with `num_prompt_tokens <= threshold` are treated as short.
- `VLLM_ASCEND_LAPS_LONG_MAX_WAIT_MS`
  - Anti-starvation aging bound for long prefills, in milliseconds.
  - `0` disables aging (strict short-priority).
  - Aging is two-tiered (see `LONG_TOKEN_RESERVATION`):
    - **soft** (waited `>= LONG_MAX_WAIT_MS`): the long becomes eligible to be
      promoted ahead of shorts, but the token bucket rate-limits the promotion.
    - **hard** (waited `>= hard bound`): the long is promoted unconditionally,
      bypassing the bucket, which makes the worst-case admission wait a true
      upper bound.
  - The hard bound (and thus the worst-case wait) is:
    - `LONG_MAX_WAIT_MS` when `LONG_TOKEN_RESERVATION == 0` (pure deadline
      aging — no soft phase, since the bucket never refills);
    - `~2 * LONG_MAX_WAIT_MS` when `LONG_TOKEN_RESERVATION > 0`, leaving the
      `[soft, hard)` window for bucket-smoothed promotion.
- `VLLM_ASCEND_LAPS_LONG_TOKEN_RESERVATION`
  - Average fraction of token throughput reserved for **admitting** aged-long
    prefills ahead of waiting shorts during the *soft* aging phase (default
    `0.0`, valid range `[0.0, 1.0]`).
  - Implemented as a token bucket: it refills by
    `reservation * per-step token budget` every scheduling step (burst capped at
    a few steps' worth), and admitting an aged-long ahead of shorts spends that
    credit. In the soft phase an aged-long is promoted only while the bucket has
    credit; once spent, shorts are preferred until the bucket refills (typically
    within a handful of steps). This smoothing prevents a backlog of aged longs
    from flipping the queue into long-first and starving shorts.
  - `0` disables soft-phase smoothing: aging then reduces to a pure deadline at
    `LONG_MAX_WAIT_MS` (the hard bound). Larger values drain aged-long requests
    faster during the soft phase at the cost of short-request latency.
  - The bucket bounds the *admission rate*, not total compute: once admitted, a
    long prefill proceeds chunk-by-chunk through the normal running loop. The
    reservation only affects how aggressively the soft phase drains; the hard
    deadline guarantees the worst-case wait regardless of reservation.
  - The bucket refills against `max_num_scheduled_tokens` (the full per-step
    budget), not the budget left after running requests. Under decode-heavy load
    the bucket therefore tends to stay saturated, so aged longs are admitted
    almost as soon as they reach the soft phase. This errs toward stronger
    anti-starvation and is an intentional trade-off.

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
export VLLM_ASCEND_LAPS_LONG_TOKEN_RESERVATION=0.2

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
- **Anti-starvation (two-tier aging):** aging kicks in once the oldest long
  request has waited longer than `VLLM_ASCEND_LAPS_LONG_MAX_WAIT_MS`.
  - **Soft phase** (`>= LONG_MAX_WAIT_MS`): the long is promoted ahead of shorts
    only while the token-reservation bucket has credit, so the promotion rate is
    smoothed and a backlog of aged longs cannot flip the queue into long-first.
  - **Hard phase** (`>= hard bound`): the long is promoted unconditionally,
    bypassing the bucket. This makes the worst-case admission wait a true upper
    bound — `LONG_MAX_WAIT_MS` when `LONG_TOKEN_RESERVATION == 0`, or
    `~2 * LONG_MAX_WAIT_MS` when it is `> 0` — so a sustained short-request
    stream cannot starve long prefills regardless of the reservation.
- **Token reservation (soft-phase smoothing):** aged-long admissions in the soft
  phase are rate-limited by a token bucket that refills
  `VLLM_ASCEND_LAPS_LONG_TOKEN_RESERVATION` of the per-step token budget each
  step (burst capped at a few steps' worth). While the bucket has credit an
  aged-long is promoted ahead of shorts; admitting it spends the first chunk's
  tokens from the bucket. When the bucket is empty shorts are preferred until it
  refills (a few steps) — unless the hard deadline has been reached, in which
  case the long is forced through. When no short request is waiting, long is
  dispatched regardless of the bucket (stall avoidance) and is **not** charged or
  counted as a starvation promotion, since it did not jump the queue. Larger
  reservations drain aged-long requests faster during the soft phase at the cost
  of short-request latency.

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
