# Packed history and request-local state for DeepSeek V4.1 Flash

A Flash request keeps compressed history for its entire lifetime. Its sliding window (SWA) and compressor need only their most recent rows. Packing these resources by lifetime increases long-context capacity in a fixed KV budget. The option is off by default.

## Placement and admission

Each TP worker uses the existing MRV2 shared backing allocation. All groups in a DP replica obtain IDs from the same native block pool. For each ID, the worker exposes three planes with strides of 128, 128 and 104.625 KiB. A group owns the entire ID until release. The following examples show how its contents depend on the owner.

| Owner of one ID | First 128 KiB | Second 128 KiB | Last 104.625 KiB |
| --- | --- | --- | --- |
| History, 128 input positions | One full KV page | Two half-size KV pages | Remaining half-size KV page, four index pages and their scales |
| A pair of SWA layers | First layer's 128-row ring | Second layer's 128-row ring | Unused |
| A pair of compressor layers | First layer's 32-row ring | Second layer's 32-row ring, if present | Unused |

Each history block occupies 369,280 bytes (360.625 KiB). Forty SWA layers need 20 IDs per request, and three compressor layers need two. Together these state groups occupy 7.748 MiB per request.

For a worker with room for G IDs and quotas S (SWA requests) and C (compressor requests), the history capacity is G − 1 − 20S − 2C blocks. The native pool reserves ID zero. Both quotas default to the local scheduler's `max_num_seqs`. Quota checks precede allocation. Detached blocks still count while a transfer owns a reference. The byte budget comes from the worker's KV profile. Staging and base-address alignment have separate costs.

At startup the packed layout reports how many whole requests of the configured maximum length fit per replica. Each request needs one history block per 128 input positions, rounded up. The limit is the smallest of the history-based request count and the two state quotas. The frontend token estimate multiplies this limit by the configured length. Shorter or mixed-length requests can reach a different concurrency.

During attention, the worker assembles the visible tail of the old ring and the current rows in chronological order. After attention it commits the new rows to the ring. Sequential layers share the staging workspace. The packed path requires non-overlapping forwards and excludes prefix caching, bounded SWA replay, speculative decoding and extensible cache allocation. It supports the unpacked BF16 Flash KV format, INT8 indices with FP16 scales, and FP32 compressor rings with PP, PCP and DCP all equal to one.

## Serving and P/D handoff

Enable the option on the standalone service or on **both** P and D:

```shell
--additional-config '{"packed_history_cache":{"enabled":true}}'
```

Each peer uses the same layout and layer formats. They can have different budgets and local block IDs. For N input tokens, P supplies the history, index, window and compressor state for the first N−1 input tokens. D computes the final input token. The stock Mooncake Pull connector registers the backing and addresses each layer using its base, the local or remote block ID, and the stride published by the tensor view. The layout supplies normal placement descriptors consumed by the upstream allocator. The deployment must configure matching layouts on both sides.

Disaggregated serving currently accepts MooncakePullConnector only. Each layer descriptor publishes the block stride and the size of its physical plane. This lets a consumer reconstruct page boundaries even when a history component starts in the middle of a page.
