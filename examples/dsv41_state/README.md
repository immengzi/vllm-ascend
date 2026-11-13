# DeepSeek V4.1 history and request-state caches

This implementation separates token-growing history from fixed SWA and compressor
state. The scheduler reserves one slot per request in each state pool. The
remaining cache budget supplies history pages. Decode assembles the window in
chronological order before attention.

The current integration targets vLLM
`ced6857afa0ea7b2e3f0846a62e1394e90f15607` and model runner V1. These commands are
configuration examples, not validation results for a new checkout or runtime.
Prefix caching, speculative decoding and pipeline parallelism are outside the
supported scope. Model runner V2 requires separate integration and validation.

## Start a colocated service

Use an isolated environment with the matching Ascend runtime and install this
checkout. Set the model path and parallelism for the available hardware. The
launcher selects model runner V1 and forwards arguments to `vllm serve`; it does
not modify `PYTHONPATH` or write diagnostic artifacts.

```bash
python3 examples/dsv41_state/launch.py -- \
  /path/to/DeepSeek-V4.1-Flash-W8A8 \
  --served-model-name dsv41 \
  --tensor-parallel-size 4 --data-parallel-size 3 --enable-expert-parallel \
  --block-size 128 --kv-cache-memory-bytes 3576615936 \
  --max-num-seqs 48 --max-num-batched-tokens 8192 --max-model-len 150000 \
  --quantization ascend --no-enable-prefix-caching --async-scheduling \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"enable_fused_mc2":1,"enable_dsa_cp":true,"enable_flashcomm1":true,"enable_shared_expert_dp":true}'
```

`--kv-cache-memory-bytes` specifies the per-worker cache budget.
`--max-num-seqs` determines the request-state reservation. Increasing it reserves
more state memory and leaves fewer bytes for history. The three pools have fixed
capacities and do not borrow memory from one another. The chronological staging
workspace is included in planning; it is not a fourth request resource pool.

## Benchmark artifacts

The optional `bench.py` wrapper requires explicit configuration and output paths:

```bash
python3 examples/dsv41_state/bench.py \
  --config /path/to/benchmark.json --results-dir /path/to/results \
  --suite core --plan-only
```

The configuration contains `port`, `model`, `served_model_name`, `repeats`,
`pressure_repeats`, and a timezone-qualified `deadline`. `tokenizer` defaults to
`model`; `served_model_name` defaults to `dsv41`. For example:

```json
{
  "port": 8000,
  "model": "/path/to/DeepSeek-V4.1-Flash-W8A8",
  "served_model_name": "dsv41",
  "repeats": 2,
  "pressure_repeats": 2,
  "deadline": "2030-01-01T09:00:00+08:00"
}
```

Replace the example deadline with the authorized test deadline. `--plan-only`
prints benchmark commands without contacting a server. Remove it to run against
an already running local service. Existing result files are not overwritten.
Short diagnostic cases are not throughput qualification; formal comparisons need
at least 100 requests per round and a stable measurement interval.

## Code and validation boundaries

The [pool design and validation status](../../docs/source/developer_guide/Design_Documents/request_state_pools.md)
describes the current explicit interfaces, full-budget rank planning and
remaining hardware gates.

- `models/deepseek_v41/cache_config.py`: model geometry and memory budget.
- `core/kv_cache_interface.py`: declared pool capacities and cache-group ownership.
- `core/request_state_manager.py`: pool creation, joint admission and rollback.
- `models/deepseek_v41/window_state.py`: typed window metadata and runner-owned staging.
- `models/deepseek_v41/request_state.py`: window assembly, attention and state commit.
- `models/deepseek_v41/state_kernels.py`: window assembly and state updates.

P/D adds transfer ownership and device synchronization to this local allocation
contract; see the [P/D design](../../docs/source/developer_guide/Design_Documents/dsv41_state_pd.md).
Local pool tests do not establish cross-machine transfer or numerical correctness.

Model runner V1 binds one staging workspace to all request-window layers before
warmup and graph capture. Sequential layers reuse that workspace; separate runners
own separate allocations. The workspace is not resized or freed per request.
Decode metadata buffers retain their addresses while their contents change.
Context-parallel metadata copies retain the typed window state and the global KV
owner. Device graph replay and stream ordering still require hardware validation
after changes to these ownership boundaries.

The other diagnostic scripts and `tested-source.json` in this directory describe
historical experiments. Some require the old container mount layout or historical
Git objects. They are not a portable acceptance suite for the refactored runtime.
The community cleanup of those tools is separate from the service entry above.
