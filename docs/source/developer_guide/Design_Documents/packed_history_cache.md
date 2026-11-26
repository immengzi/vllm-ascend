# Packed history and request-local state for DeepSeek V4.1 Flash

A Flash request keeps compressed history for its entire lifetime. Its sliding window (SWA) and compressor need only their most recent rows. Packing these resources by lifetime increases long-context capacity in a fixed KV budget. The option is off by default.

## Placement and admission

Each TP worker uses the existing MRV2 shared backing allocation. All groups in a DP replica obtain IDs from the same native block pool. For each ID, the worker exposes three planes with strides of 128, 128 and 104.625 KiB. A group owns the entire ID until release. The table and capacity formula below describe the fixed-ring layout used when prefix caching is disabled.

| Owner of one ID | First 128 KiB | Second 128 KiB | Last 104.625 KiB |
| --- | --- | --- | --- |
| History, 128 input positions | One full KV page | Two half-size KV pages | Remaining half-size KV page, four index pages and their scales |
| A pair of SWA layers | First layer's 128-row ring | Second layer's 128-row ring | Unused |
| A pair of compressor layers | First layer's 32-row ring | Second layer's 32-row ring, if present | Unused |

Each history block occupies 369,280 bytes (360.625 KiB). Forty SWA layers need 20 IDs per request, and three compressor layers need two. Together these state groups occupy 7.748 MiB per request.

For a worker with room for G IDs and quotas S (SWA requests) and C (compressor requests), the history capacity is G − 1 − 20S − 2C blocks. The native pool reserves ID zero. Both quotas default to the local scheduler's `max_num_seqs`. Quota checks precede allocation. Detached blocks still count while a transfer owns a reference. The byte budget comes from the worker's KV profile. Staging and base-address alignment have separate costs.

At startup the packed layout reports how many whole requests of the configured maximum length fit per replica. Each request needs one history block per 128 input positions, rounded up. The limit is the smallest of the history-based request count and the two state quotas. The frontend token estimate multiplies this limit by the configured length. Shorter or mixed-length requests can reach a different concurrency.

During attention, the worker assembles the visible tail of the old ring and the current rows in chronological order. After attention it commits the new rows to the ring. Sequential layers share the staging workspace. Packed history requires non-overlapping forwards and excludes bounded SWA replay, speculative decoding and extensible cache allocation. It supports the unpacked BF16 Flash KV format, INT8 indices with FP16 scales, and FP32 compressor rings with PP, PCP and DCP all equal to one.

## Prefix caching

Enable this mode with `--enable-prefix-caching --no-swa-bounded-replay`. Compressed history uses the same packed planes. A request's fixed SWA ring overwrites old rows as the window advances. With prefix caching enabled, SWA uses vLLM's position-indexed pages in the shared pool so a continuation can reuse the window tail at its cache-hit boundary.

Consider two requests with the same first 256 tokens and different suffixes. The second can start at token 257 when both the compressed history for the prefix and the SWA tail at token 256 are cached. The two requests can read the same prefix pages and write their own suffix pages. vLLM can evict a cached page once its last reader releases it. Hits end on a 128-token page boundary, which also completes a C2 compression pair. The continuation therefore starts its next pair in its own compressor ring.

A cold request draws its history pages, SWA pages for the visible tail and in-flight chunk, and private compressor rings from the shared pool. Startup sizing checks that this demand fits for one request at the configured maximum length. The reported cold-request concurrency also accounts for the runtime compressor limit, which is the smaller of `swa_slots` and `compressor_slots` in this mode. Prefix hits can raise the concurrency that fits within the same page budget.

## Serving and P/D handoff

Enable the option on the standalone service or on **both** P and D:

```shell
--additional-config '{"packed_history_cache":{"enabled":true}}'
```

Both peers must use the same layout, layer formats, prefix caching and SWA replay settings. Their budgets and local block IDs may differ. For N input tokens, P supplies the history, index, window and compressor state for the first N−1 tokens. D restores those groups under its own block IDs and computes the final input token.

The layout publishes standard KV cache groups, tensor views and placement descriptors. A connector must support the model's hybrid cache groups and preserve each physical plane's block stride when translating between P and D block IDs. The descriptors record the plane size and layer offsets so a connector can locate a page even when a layer starts in the middle of it. vLLM handles connector selection and hybrid cache capability checks.
