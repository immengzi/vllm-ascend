# Packed history and request-local state for DeepSeek V4.1 Flash

A Flash request keeps compressed history for its entire lifetime. Its sliding
window (SWA) and compressor need only their most recent rows. Packing these
resources by lifetime increases long-context capacity in a fixed KV budget.
The option is off by default. The measurements below also show why a service
with predominantly short requests may prefer the native layout.

## Placement and admission

Each TP worker uses the existing MRV2 shared backing allocation. All groups in
a DP replica obtain IDs from the same native block pool. For each ID, the
worker exposes three planes with strides of 128, 128 and 104.625 KiB. A group
owns the entire ID until release. The following examples show how its contents
depend on the owner.

| Owner of one ID | First 128 KiB | Second 128 KiB | Last 104.625 KiB |
| --- | --- | --- | --- |
| History, 128 input positions | One full KV page | Two half-size KV pages | Remaining half-size KV page, four index pages and their scales |
| A pair of SWA layers | First layer's 128-row ring | Second layer's 128-row ring | Unused |
| A pair of compressor layers | First layer's 32-row ring | Second layer's 32-row ring, if present | Unused |

Each history block occupies 369,280 bytes (360.625 KiB). The experimental
uniform-stride control spent 384 KiB on the same history, including 23.375 KiB
of padding. Packed state still has slack. Forty SWA layers need 20 IDs per
request and three compressor layers need two. This costs 7.748 MiB per request,
compared with 5.375 MiB for the uniform-stride control.

For a worker with room for G IDs and quotas S (SWA requests) and C (compressor
requests), the history capacity is G − 1 − 20S − 2C blocks. The native pool
reserves ID zero. Both quotas default to the local scheduler's `max_num_seqs`.
Quota checks precede allocation. Detached blocks still count while a transfer
owns a reference. The byte budget comes from the worker's KV profile.
Staging and base-address alignment have separate costs.

At startup the packed layout reports how many whole requests of the configured
maximum length fit per replica. Each request needs one history block per
128 input positions, rounded up. The limit is the smallest of the history-based
request count and the two state quotas. The frontend token estimate multiplies
this limit by the configured length. Shorter or mixed-length requests can
reach a different concurrency.

During attention, the worker assembles the visible tail of the old ring and
the current rows in chronological order. After attention it commits the new
rows to the ring. Sequential layers share the staging workspace. The packed
path requires non-overlapping forwards and excludes prefix caching, bounded
SWA replay, speculative decoding and extensible cache allocation. It supports
the unpacked BF16 Flash KV format, INT8 indices with FP16 scales, and FP32
compressor rings with PP, PCP and DCP all equal to one.

## Serving and P/D handoff

Enable the option on the standalone service or on **both** P and D:

```shell
--additional-config '{"packed_history_cache":{"enabled":true}}'
```

Each peer uses the same layout and layer formats. They can have different
budgets and local block IDs. For N input tokens, P supplies the history, index,
window and compressor state for the first N−1 input tokens.
D computes the final input token. The stock Mooncake Pull connector registers
the backing and addresses each layer using its base, the local or remote block
ID, and the stride published by the tensor view. The layout supplies normal
placement descriptors consumed by the upstream allocator. The deployment
must configure matching layouts on both sides.

Disaggregated serving currently accepts MooncakePullConnector only. Each layer
descriptor publishes the block stride and the size of its physical plane. This
lets a consumer reconstruct page boundaries even when a history component starts
in the middle of a page. CPU tests exercised the stock MooncakeHybridConnector's
address calculations at 128, 129, 130 and 257 tokens, including the final block
of the allocation. Hybrid still requires hardware validation.

## A3 measurements

These results were collected on a frozen vLLM 0.33 runtime. The native and
uniform controls used Ascend revision `6202a38`; packed history used
`1a18d73`. The community port uses main's descriptor-driven MRV2 allocator.
The NPU measurements apply to the frozen revisions. The port requires a
hardware rerun before claiming the same numbers for main.

One A3 hosted W4A8 DeepSeek-V4.1-Flash-1017. P used DP2/TP4 and D DP4/TP2,
eight logical devices each. Both used BF16 KV and a 1,048,576-token context
limit. Memory utilization was 0.90 on P and 0.95 on D. All layouts used the
same model, tokenizer, seed and scheduling limits. The official Ascend proxy
served the P/D frontend. Each timed `vllm bench serve` round used 100 requests.
The comparison covers 33 P/D rounds and 18 standalone rounds. The capacity row
records the frozen runtime's startup estimate. The streaming row counts requests
observed generating concurrently.

| P/D workload and metric | Native | Uniform 128 KiB control | Packed history |
| --- | ---: | ---: | ---: |
| D capacity per replica, millions of tokens reported at startup, about 12 GiB/rank | 3.02 | 4.19 | 4.45 |
| Long requests streaming on D0, out of 16 with 262,144-token prompts | 11 | 15 | 15 |
| D0 aggregate long output during short foreground traffic, tokens/s | 296.9 | 351.2 | 393.2 |
| Short foreground throughput under that load, requests/s | 35.51 | 29.14 | 32.70 |
| Warm 128 input / 16 output, concurrency 4, requests/s | 4.40 | 3.67 | 4.10 |
| 130 / 16, concurrency 4, requests/s | 4.32 | 3.61 | 4.17 |
| 8,192 / 16, concurrency 32, requests/s | 1.53 | 1.50 | 1.52 |
| 32,768 / 16, concurrency 8, requests/s | 0.39 | 0.38 | 0.38 |

**Long requests.** All 16 background requests targeted one D replica. Its
queue had 5 / 1 / 1 waiting requests. All 100 short foreground requests
were served by the other three replicas. The respective measurement windows
were 20.27 / 20.89 / 18.08 seconds, one run per layout. Relative to native,
packed history raised aggregate long output by 32.4% and reduced short
foreground throughput by 7.9%. Both experimental layouts averaged 15 tokens
per decode iteration. The measured long-stream token interval was 42.65 ms
for uniform and 38.16 ms for packed. The number of cache groups falls from
46 to 23; the experiment did not isolate time spent updating block tables
or executing individual kernels.

**Short requests.** Two warm rounds at 128 / 16 show a 6.9% throughput loss
for packed history relative to native. Mean client time to first token rose
from 431 to 500 ms. Mean P prefill rose from 218 to 248 ms and D queue time
from 86 to 91 ms. Mean time per subsequent token was 30.9 / 38.7 / 30.6 ms
for native / uniform / packed. At 130 input tokens, crossing the window
boundary, both experimental layouts reduced successful Pull reads from
22.16 to 12.16 MiB per request across the two D ranks. The reduction in
bytes did not offset the other service costs for this load. At 8K and 32K,
packed throughput was about 0.5% below native in two rounds each; the native
requests already spent about 13–14 s queuing on P.

**Standalone serving.** For warm 128 / 16 at concurrency 4, native / uniform /
packed throughput was 1.14 / 1.01 / 0.99 requests/s on DP2/TP4 and
3.73 / 4.01 / 4.11 on DP4/TP2. Each warm standalone case had one round.
The isolated throughput change depends on topology and scheduling parameters.

**Correctness.** Frozen-model tests covered standalone TP4 and TP2, and P/D
TP4 → TP2. For lengths 128, 129, 130 and 257, each tested decode step matched
the independent same-topology reference across the full vocabulary; the
130-token input generated 130 consecutive outputs. For the P/D reference,
P exported the prefix and D independently loaded it without the connector.
Checks of the real handoff covered the actual P source, both receiving ranks,
and D's local 1 / external N−1 / cache-hit 0 token counts. Clean services also
matched native tokens and API top-five probabilities at 8K, 32K and 256K for
eight output steps. Concurrent 8K runs showed some text variation even within
the native layout. Fixed-route replays of seven selected inputs matched across
layouts for all 16 output tokens and their top-five probabilities.
