# NPU Device Allocators for vllm-ascend

This directory contains pluggable NPU memory allocator backends for vllm-ascend.
Each backend is an `NPUPluggableAllocator` implementation that replaces
PyTorch-NPU's default caching allocator with a custom memory pool.

| File | Backend | Status |
|------|---------|--------|
| `camem.py` / `csrc/camem_allocator.cpp` | CANN built-in CA-mem pool | Upstream |
| `shmem_allocator.py` / `csrc/shmem_allocator.cpp` | SHMEM dynamic pool | This feature |

---

## SHMEM Dynamic Memory Allocator

### Background

[SHMEM](https://gitee.com/ascend/shmem) (Ascend Symmetric Heterogeneous Memory)
is a multi-device memory communication library for Ascend NPUs. Its memory
heap exposes the standard `aclshmem_malloc` / `aclshmem_free` interface
compatible with any NPU device pointer.

The library was extended with a **dynamic expansion** capability
(`shmem_dynamic_mm.cpp`): when the initial pool is exhausted, additional CANN
device memory blocks are allocated transparently — up to 4 GiB per expansion,
with a 1.5× growth factor. This eliminates the need to pre-size the pool for
the entire model and allows vllm to grow its KV-cache on demand.

### Architecture

```
Python layer (vllm-ascend)
│
│  worker.py: load_model()            → weights in SHMEM pool
│  worker.py: initialize_from_config() → KV cache in SHMEM pool
│      ↓
│  ShmemAllocator.use_memory_pool(tag)
│      ↓
│  torch.npu.memory.NPUPluggableAllocator(lib_path, 'my_malloc', 'my_free')
│      ↓
│  torch.npu.memory.MemPool  +  torch.npu.memory.use_mem_pool(pool)
│      │
│      │  every torch.empty() / tensor creation inside the context
│      ↓
C++ layer (shmem_allocator.cpython-*.so)
│
│  my_malloc(size, device, stream)
│      ├─ ensure_shmem_initialized()    ← lazy, idempotent, thread-safe
│      ├─ aclshmem_malloc(size)         ← SHMEM pool (expands automatically)
│      │    └─ on failure: aclrtMalloc  ← ACL fallback
│      └─ record ptr in g_shmem_ptrs
│
│  my_free(ptr, size, device, stream)
│      ├─ ptr in g_shmem_ptrs?
│      │    yes → aclshmem_free(ptr)
│      └─ no  → aclrtFree(ptr)
│
SHMEM library (libshmem.so)
│
│  Dynamic pool manager
│      ├─ initial block  (SHMEM_INITIAL_POOL_SIZE, default 2 GiB)
│      └─ expansion blocks added by aclrtMalloc on demand
```

**Key design decisions**

* **Separate `.so` module** — `camem_allocator.cpp` (compiled into
  `vllm_ascend_C.so`) already defines `my_malloc`/`my_free`. A second
  definition in the same shared object would cause a linker duplicate-symbol
  error, so `shmem_allocator` is built as its own `pybind11_add_module`.

* **Dual-role `.so`** — the same `shmem_allocator.cpython-*.so` is:
  1. Importable as `vllm_ascend.shmem_allocator` (Python management API).
  2. Passed by filesystem path to `NPUPluggableAllocator`, which uses `dlsym`
     to resolve `my_malloc`/`my_free` from the already-loaded shared object.

* **Pointer tracking** — `aclshmem_ptr_valid` does not exist in the public
  SHMEM API. The allocator maintains an `std::unordered_set<void*>` of every
  pointer returned by `aclshmem_malloc` so that `my_free` can route each
  deallocation to the correct backend.

* **No ACL lifecycle management** — `aclInit`, `aclrtSetDevice`, `aclFinalize`
  are called by vllm-ascend before any allocator is used. The SHMEM allocator
  must not duplicate or interfere with these calls.

* **SHMEM vs sleep mode are mutually exclusive** — `CaMemAllocator` (sleep
  mode) supports CPU offloading; `ShmemAllocator` does not. Enabling both
  simultaneously raises a `RuntimeError`.

### Allocation scope in vllm-ascend

When `ENABLE_SHMEM=1`, both major NPU allocation phases are routed through
the SHMEM dynamic pool:

| Phase | Function | Pool tag |
|-------|----------|----------|
| Model weight loading | `worker.load_model()` | `"weights"` |
| KV cache initialisation | `worker.initialize_from_config()` | `"kv_cache"` |

---

### Build

#### 1. Build and install SHMEM (source)

```bash
git clone <shmem-repo-url>
cd shmem
bash scripts/build.sh

# Exports SHMEM_HOME_PATH, e.g. /root/shmem/install
source install/set_env.sh
echo $SHMEM_HOME_PATH
```

#### 2. Build vllm-ascend with SHMEM support

```bash
# SHMEM_HOME_PATH must be set (from step 1)
ENABLE_SHMEM=1 pip install -e .
```

CMake variables (set automatically from env, or pass explicitly):

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_SHMEM` | Enable shmem_allocator module | `OFF` |
| `SHMEM_HOME_PATH` | Parent of the `shmem/` install dir | `/usr/local/Ascend/shmem/latest` |

The build produces `vllm_ascend/shmem_allocator.cpython-*.so` alongside the
existing `vllm_ascend_C.cpython-*.so`.

---

### Usage

#### Option A — automatic (recommended)

Set `ENABLE_SHMEM=1` at runtime. The worker picks it up automatically for
both weight loading and KV-cache initialisation.

```bash
ENABLE_SHMEM=1 python your_vllm_script.py
```

#### Option B — manual via Python API

```python
from vllm_ascend.device_allocator.shmem_allocator import ShmemAllocator

allocator = ShmemAllocator.get_instance()

with allocator.use_memory_pool(tag="kv_cache"):
    kv_cache = torch.empty(shape, dtype=torch.float16, device="npu")
    ...

stats = allocator.get_memory_stats()
if stats:
    total, used, avail = stats
    print(f"SHMEM pool: {used/1e9:.2f} / {total/1e9:.2f} GiB used")
```

---

### Runtime Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_SHMEM` | Enable SHMEM allocator at runtime | `0` |
| `SHMEM_HOME_PATH` | Path set by `source install/set_env.sh` | — |
| `SHMEM_INITIAL_POOL_SIZE` | Initial pool size in bytes | `2147483648` (2 GiB) |

`SHMEM_INITIAL_POOL_SIZE` is read by the C++ layer at first allocation.
The pool grows automatically beyond this size via dynamic expansion —
setting it large enough reduces expansion overhead.

```bash
# Reserve 8 GiB upfront
export SHMEM_INITIAL_POOL_SIZE=$((8 * 1024 * 1024 * 1024))
ENABLE_SHMEM=1 python your_vllm_script.py
```

---

### API Reference

#### Python — `ShmemAllocator`

| Method | Description |
|--------|-------------|
| `ShmemAllocator.get_instance()` | Return singleton instance |
| `.initialize() → bool` | Explicitly initialise the pool (idempotent) |
| `.finalize()` | Release the pool and reset state |
| `.use_memory_pool(tag=None)` | Context manager: route allocations to SHMEM |
| `.get_memory_stats() → (total, used, avail)` | Pool statistics in bytes |
| `.get_current_usage() → int` | Bytes currently allocated |
| `.sleep(offload_tags=None)` | No-op (API compat with CaMemAllocator) |
| `.wake_up(tags=None)` | No-op (API compat with CaMemAllocator) |

#### Module-level flag

```python
from vllm_ascend.device_allocator.shmem_allocator import shmem_available
# True  → extension was compiled and imported successfully
# False → ENABLE_SHMEM=ON was not set at build time (or import failed)
```

#### C++ — NPUPluggableAllocator symbols

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `my_malloc` | `void*(ssize_t size, int device, aclrtStream stream)` | Allocate from SHMEM pool; falls back to `aclrtMalloc` |
| `my_free` | `void(void* ptr, ssize_t size, int device, aclrtStream stream)` | Route to `aclshmem_free` or `aclrtFree` based on origin |

---

### Troubleshooting

**`ImportError: No module named 'vllm_ascend.shmem_allocator'`**
Build was done without `ENABLE_SHMEM=1`. Rebuild as shown above.

**`RuntimeError: SHMEM allocator is not available`**
The extension module was not compiled in. Ensure `ENABLE_SHMEM=1` and
`SHMEM_HOME_PATH` are set, then rebuild.

**`RuntimeError: ENABLE_SHMEM and sleep mode are mutually exclusive`**
`enable_sleep_mode=True` and `ENABLE_SHMEM=1` were set simultaneously.
SHMEM does not support CPU offloading; choose one or the other.

**`RuntimeError: aclshmemx_init_attr failed: <code>`**
Common causes:
- `SHMEM_HOME_PATH` not set or wrong — check `echo $SHMEM_HOME_PATH`.
- Bootstrap plugins (`aclshmem_bootstrap_uid.so`) not found — ensure
  `${SHMEM_HOME_PATH}/shmem/lib/` is on `LD_LIBRARY_PATH` or the RPATH
  embedded at build time is correct.

**`aclshmem_malloc failed … falling back to aclrtMalloc`**
The SHMEM pool failed to expand (device OOM). Reduce
`SHMEM_INITIAL_POOL_SIZE` or the model batch size.
