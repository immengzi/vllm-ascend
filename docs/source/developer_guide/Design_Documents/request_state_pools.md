# History-page and request-state pools

This implementation uses MRV1 and implements the capacity split proposed in
[RFC #21324](https://github.com/vllm-project/vllm-ascend/issues/21324).
DeepSeek V4.1 is the first model integration. The interfaces below are local
implementation contracts, not finalized vLLM core or MRV2 APIs.

## Motivation and scope

History KV and sparse indices grow with processed tokens. SWA and compressor
state have bounded storage per request. Giving all of them one shared physical
page stride leaves padding in history pages and couples their capacity limits.
The planner reserves fixed state slots and execution workspace first, then
uses the remaining budget for compact history pages.

There are three resource pools: history, SWA and compressor. Each state pool
contains one slot per configured request plus its own null slot. Pools do not
borrow physical memory from each other. Chronological window staging is a
runner workspace, not another request resource pool.

The current implementation targets vLLM
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`, MRV1 and homogeneous worker cache
specifications with PP=1. Prefix caching and speculative decoding are excluded.
The first core contract supports one append-only history group and one cache
group per pool; it does not yet implement arbitrary group sharing or eviction
policies. MRV2, KDA and GDN require separate integration and validation.

## Interfaces and ownership

| Component | Responsibility |
| --- | --- |
| `core/kv_cache_interface.py` | `CachePoolPlan` declares capacity, group ownership and allocation unit; `RequestStatePlan` validates the mapping and backing sizes; `RequestStateMemoryBudget` separates growing and fixed bytes |
| `models/deepseek_v41/cache_config.py` | Derive model-specific specs, fixed state capacity, staging allowance and history page geometry; build tensor descriptors |
| `core/request_state_manager.py` | Create managers on declared pools; check missing slots and roll back allocations made by a failed admission attempt |
| `patch/platform/patch_kv_cache_utils.py` | Adapt the pinned upstream planner to the full per-rank budget; preserve native dispatch for other models |
| `models/deepseek_v41/window_state.py` | Typed window metadata, stable Decode metadata buffers and a runner-owned staging workspace |
| `worker/model_runner_v1.py` | Allocate buffers and bind views from the plan; create and bind staging before warmup/capture |
| `models/deepseek_v41/request_state.py` | Assemble the chronological window, execute attention, commit state and release temporary KV references |
| Mooncake connector | Own transfer descriptors, TP completion, leases and DMA lifetime independently of scheduler allocation rollback |

Pool ownership is explicit in the configuration. Worker tensor views use
`KVCacheTensor` sizes, byte offsets and strides. Scheduler representative-spec
conversion preserves the pool plan. A history page ID and a request slot ID are
local to their respective pools; equality of the integers does not imply shared
storage.

## Budget and rank planning

```text
F = fixed request-state bytes + staging/workspace allowance
H = bytes for one history page ID across all history tensors
history capacity = floor((available bytes - F) / H)
full budget for B history pages = F + B * H
```

For the unpacked example with 48 requests and an 8192-token batch budget,
`F = 298,188,800 B` and `H = 369,280 B`. State capacities include null slots.
History capacity also includes its null page, which cannot serve a request.

The pinned upstream rank planner assumes that all cache bytes scale with block
count. Its final rank reconciliation rebuilds larger ranks using only
`min_blocks * bytes_per_block`. Passing that value to a fixed-state planner
would deduct `F` a second time and produce inconsistent rank capacities.

For identical worker specs, the adapter first normalizes the complete byte
budgets to their minimum, then delegates grouping, admission, automatic maximum
length and configuration generation to upstream. Equal full budgets produce
equal capacities without entering the history-only rebuild. Heterogeneous
specifications and PP are rejected on this path.

An explicit `num_gpu_blocks_override` is checked against every worker's physical
budget, then converted to `F + B * H`. A shallow config/cache-option copy clears
the override for the delegated call, preventing upstream from replacing that
budget with history-only bytes. The caller's override is unchanged. The model
config remains shared so automatic maximum-length fitting updates it normally.

Maximum-length admission uses the same fixed reservations plus the historical
pages needed by one full-length request. Upstream has already reserved one
history null page; the check must not subtract it again. The calculation uses
the cache spec's local history requirement, including its existing DCP sizing.
This arithmetic check does not qualify device-side CP execution.

## Admission and execution

Managers are constructed directly on the declared pools. Admission checks
missing request slots before invoking the existing history allocator. For a
new slot assignment, the transaction records the request's existing resources
and restores them if allocation rejects or raises. Resident Decode requests
keep the existing path without copying long history tables each step.

Rollback is limited to the single-threaded scheduler, disabled prefix reuse,
append-only history and fixed state slots. It does not cancel DMA or release
buffers still owned by a transfer. Completion, cancellation and preemption
must respect the worker and connector's outstanding accesses before reuse.

MRV1 owns one chronological staging workspace shared by sequential SWA layers.
It binds the workspace after cache allocation and before warmup/capture.
Separate runners own separate allocations. Typed metadata carries state IDs,
row starts, valid lengths and the staging block table; Decode buffers update
in place. No module-global device-to-workspace dictionary or import-time
replacement of attention methods is needed.

## Prefill/Decode integration

The Mooncake Pull path hands over history, indices, SWA and compressor
state at the same processed-token boundary. Prefill handles the first `N-1`
prompt tokens. Decode reserves local IDs, READs the valid ranges, waits for all
participating TP ranks and computes the last prompt token locally. Source
leases and destination state cannot be reused while DMA completion is uncertain.

## Validation requirements

CPU tests cover pool ownership, byte accounting, admission rollback and window
metadata. They are under `tests/ut/core/test_request_state_*.py` and
`tests/ut/models/test_dsv41_*.py`. Device regression entrypoints are under
`tests/e2e/pull_request/one_card/aclgraph/`.

Changes to these interfaces require full-runtime initialization and IPC checks,
unequal-budget rank allocation, override and automatic-length checks, staging
graph replay, first and continuous Decode numerical comparison, cancellation
and slot reuse. Applicable CP configurations and normal P/D handoff require
separate device regression. Full-model recovery after an in-flight SDK transfer
failure remains to be validated. Performance evaluation follows correctness
and resource-lifetime validation.
