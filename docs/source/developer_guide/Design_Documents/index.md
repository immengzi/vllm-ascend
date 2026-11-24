# Design Documents

This section provides an overview of the features implemented in vLLM Ascend. Developers can refer to this guide to understand how vLLM Ascend works.

- [KVPP: KV Cache Layer Parallelism](kvpp.md) — Physical cache placement, full-layer broadcast, and test design.
- [Persistent Incremental Csrc Build Cache](persistent_csrc_build_cache.md) — Fine-grained native build reuse, correctness identity, concurrency, and CI integration.
- [Triton JIT warmup on Ascend](triton_jit_warmup.md) — Kernel registration, startup compilation, and validation.
- [Packed history for DeepSeek V4.1 Flash](packed_history_cache.md) — Placement, request-state quotas and A3 measurements.
