# Design Documents

This section provides an overview of the features implemented in vLLM Ascend. Developers can refer to this guide to understand how vLLM Ascend works.

- [KVPP: KV Cache Layer Parallelism](kvpp.md) — Physical cache placement, full-layer broadcast, and test design.
- [Persistent Incremental Csrc Build Cache](persistent_csrc_build_cache.md) — Fine-grained native build reuse, correctness identity, concurrency, and CI integration.
- [DeepSeek V4.1 Flash 单池历史打包设计](dsv41_packed_history_zh.md) — Experimental placement, per-request quotas and CPU validation.
