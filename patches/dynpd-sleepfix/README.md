# dyn-pd sleepfix backports (vllm-ascend @ v0.18.0)

Source-only backports used by the llm-la dyn-pd warm-standby deployment
(`reg.local:32000/library/vllm-ascend:v0.18.0`). They were previously kept in
the llm-la repo's `patches/vllm/` working directory; this is their proper home.

Archive state: committed on local branch `wip/dynpd-sleepfix-v0.18.0`, **not
pushed**. Baseline: vllm-ascend `v0.18.0` = `c5959ec19`.

## 02-ascendstore-task-done-finally-v0.18.0.patch

Custom backport (mirrors the accounting half of upstream vLLM #43742,
"Release GPU pin on failed store in MooncakeStoreConnector", commit
`d68f0b220e`) for the AscendStore connector. Verified with `patch -p1
--dry-run` against the `v0.18.0` tree.

Why: `KVCacheStoreSendingThread._handle_request` decremented `stored_requests`
and called `request_queue.task_done()` only on the happy path. Early returns
(`if not keys`, `if not missing_indices`) skipped `task_done()`, and an
exception in `lookup`/`synchronize`/`m_store.put` skipped both. A missing
`task_done()` makes `request_queue.join()` (used by the connector reset
cascade before sleep) hang forever; a stuck `stored_requests` counter means
the scheduler never sees `finished_sending` and keeps the request's GPU blocks
pinned, so `reset_prefix_cache` fails and `/sleep` returns HTTP 500.

Validated remotely (dyn-pd rev 52+, sleep freed 8.19 -> 54.66 GiB after the
accounting fix).

Apply:

```bash
cd <vllm-ascend-v0.18.0-source>
git apply patches/dynpd-sleepfix/02-ascendstore-task-done-finally-v0.18.0.patch
```

## 03-camem-sleep-free-block-diagnostics-v0.18.0.patch

Diagnostics + gated cleanup for `vllm_ascend/device_allocator/camem.py`.
Verified with `git apply --check` and `patch -p1 --dry-run` against the
`v0.18.0` tree.

Why: the dyn-pd flip crashed when waking a long-slept engine, and the earlier
report that "sleep retains ~48.66 GiB of camem pool free blocks" could not be
reproduced. This patch is deliberately diagnostic-first:

- `sleep()` logs the per-tag active allocation inventory before unmapping and
  probes `torch.npu.memory_snapshot()` for pool blocks the NPU caching
  allocator still holds as cached/free (`state == "inactive"`). Measured on
  the deployed cluster: only 2 free blocks / 0.02 GiB, disproving the
  ~46.47 GiB free-block hypothesis; sleep already frees 54.66 GiB.
- The actual unmap of those free blocks is gated behind
  `VLLM_ASCEND_CAMEM_SLEEP_UNMAP_FREE_BLOCKS=1` (default off); the snapshot
  inventory is the only safe Python-level view because torch_npu 2.9.0's
  `NPUPluggableAllocator::emptyCacheImpl`/`releasePool` are no-ops.
- `wake_up()` logs the restore inventory and per-allocation debug lines so a
  mid-remap crash leaves the last attempted `ptr`/`tag`/`size` in the log.

Container overlay: copy the resulting `camem.py` to
`/vllm-workspace/vllm-ascend/vllm_ascend/device_allocator/camem.py` (same
mechanism as the previous `camem_sleep_fix.py` overlay). Do not carry the old
#34600 wake rollback; it is unreachable on Ascend because the C extension's
`std::terminate` bypasses Python `except`.

Apply:

```bash
cd <vllm-ascend-v0.18.0-source>
git apply patches/dynpd-sleepfix/03-camem-sleep-free-block-diagnostics-v0.18.0.patch
```
