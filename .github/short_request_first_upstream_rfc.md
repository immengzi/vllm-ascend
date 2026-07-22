# [RFC] Server-side prompt-length-aware waiting admission

## Motivation

Under a mixed prompt-length workload, a long prompt at the head of a server-side
waiting queue can delay many short prompts even when those short prompts could
otherwise be admitted immediately. Client-provided `priority` is not an
equivalent solution: it is an explicit static user contract, while prompt
length is server-observable request metadata. Similarly,
`long_prefill_token_threshold` limits the number of tokens scheduled in one
step; it does not choose which waiting request is admitted next or prevent
waiting-request starvation.

The proposed policy provides bounded, server-side prompt-length-aware waiting
admission while preserving a clear recovery path for preempted requests.

## Proposed Change

Add a generic waiting-admission policy. `SchedulingPolicy.SHORT_FIRST` is the
initial API proposal, but a generic request-queue policy or scheduler extension
point is equally acceptable if it better fits vLLM's scheduler architecture.

For a configured prompt-token threshold and optional `long_max_wait_ms`, the
policy maintains one outer `RequestQueue` with these internal lanes:

```text
skipped_waiting > recovery/preempted > aged long > short > long
```

- `skipped_waiting` remains scheduler-owned and retains its existing FCFS
  precedence over ordinary waiting admission.
- A fresh request with `num_prompt_tokens <= threshold` is `short`; other
  fresh requests are `long`.
- A long request becomes `aged` when the oldest long request has waited at
  least `long_max_wait_ms`. A value of zero disables aging and gives strict
  short-first admission.
- A preempted or otherwise resumable/recovery request is admitted before fresh
  work. The policy should use existing upstream request state and preemption
  transitions rather than add a second preemption path.
- `peek_request()` pins both the selected lane and request ID. `pop_request()`
  consumes that same request when it is still present, so a transition across
  the aging boundary cannot make one scheduler iteration peek a short request
  and pop a long request. Queue mutation, cancellation, and removal clear the
  pin and force reselection.

The initial implementation should stay in the standard `RequestQueue`
contract. It should not copy `Scheduler.schedule()` or require a separate
scheduler subclass solely to select a waiting lane.

## Observability and Evaluation

The implementation should expose dispatch order by class, queue sizes,
aged-long promotion count, short-request TTFT percentiles, long-request queue
wait tail latency, and token throughput. Evaluation should compare FCFS and
short-first with the same model, concurrency, total request rate, prompt-length
distribution, cache configuration, and arrival pattern. The published RFC
should attach reproducible mixed short/long benchmark data before requesting a
merge decision.

## Non-goals

- Replacing client-provided `priority`.
- Changing `long_prefill_token_threshold` semantics.
- Adding hardware, disaggregation, or deployment-specific APIs.
- Promising a latency improvement when the queue is not backlogged or another
  bottleneck dominates.

## Open Questions

- Should the API be a new `SchedulingPolicy` enum value, a generic waiting
  queue policy, or a scheduler extension point?
- Which configuration names and defaults best fit the existing public scheduler
  configuration surface?
- Which request states should qualify as recovery beyond `PREEMPTED`?
- Should the policy be limited to synchronous scheduling in the first version?

## Feedback Period

Request at least one week of feedback after posting, with benchmark data and a
small reference implementation linked from the issue.
