/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * SHMEM-backed NPUPluggableAllocator for vllm-ascend.
 *
 * This file is compiled into a standalone Python extension module
 * (shmem_allocator.cpython-*.so).  The same .so file serves two roles:
 *
 *   1. Python extension – exposes shmem_init / shmem_finalize /
 *      is_shmem_initialized / get_memory_stats for the Python layer.
 *
 *   2. NPUPluggableAllocator backend – exports my_malloc / my_free with
 *      C linkage so that torch.npu.memory.NPUPluggableAllocator can load
 *      and call them via dlsym.
 *
 * Design notes
 * ============
 * • shmem.h contains C++ declarations (default arguments, constexpr, …)
 *   and must be included **before** any extern "C" block.
 * • ACL is already initialised by vllm-ascend before this allocator is
 *   used.  We must NOT call aclInit / aclrtSetDevice / aclFinalize here.
 * • shmem APIs return int (checked against ACLSHMEM_SUCCESS), not aclError.
 * • aclshmem_ptr_valid does not exist in the public API; we track shmem
 *   pointers in an unordered_set so that my_free can dispatch correctly.
 */

// ── C++ standard headers (must precede extern "C") ──────────────────────────
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <deque>
#include <unordered_set>
#include <vector>

// ── ACL runtime API (needed for aclrtEvent / aclrtStream types) ──────────────
// Included here (outside extern "C") because the helper functions that use
// aclrtEvent, aclrtCreateEventWithFlag, aclrtRecordEvent, etc. are defined
// before the extern "C" block.  acl_rt.h ships its own extern "C" guards so
// it is safe to include from C++ without wrapping.
#include <sys/types.h>
#include "acl/acl.h"

// ── SHMEM public API (C++ header, cannot be inside extern "C") ──────────────
#include "shmem.h"

// ── DeviceStats layout mirror ─────────────────────────────────────────────────
// Mirrors c10_npu::NPUCachingAllocator::DeviceStats from torch_npu 2.8.0.
// Defined here to avoid pulling in the heavy NPUCachingAllocator.h (which
// drags in HCCL, ATen, and other large dependencies incompatible with the
// shmem_allocator build target).
//
// Binary layout must match exactly so the aarch64 struct-return ABI works
// correctly when torch_npu calls our function via the stored std::function.
// StatType::NUM_TYPES == 3 in torch_npu 2.8.0.
namespace shmem_npu_compat {

struct Stat {
    int64_t current   = 0;
    int64_t peak      = 0;
    int64_t allocated = 0;
    int64_t freed     = 0;
};

using StatArray = std::array<Stat, 3>;  // NUM_TYPES = 3

struct DeviceStats {
    StatArray allocation;
    StatArray segment;
    StatArray active;
    StatArray inactive_split;
    StatArray allocated_bytes;
    StatArray reserved_bytes;
    StatArray active_bytes;
    StatArray inactive_split_bytes;
    StatArray requested_bytes;
    int64_t   num_alloc_retries = 0;
    int64_t   num_ooms          = 0;
    Stat      oversize_allocations;
    Stat      oversize_segments;
    int64_t   max_split_size    = 0;
};

// Compile-time ABI sanity checks against torch_npu 2.8.0.
// sizeof(Stat)=32, sizeof(StatArray)=96. Layout (all offsets in bytes):
//   9 × StatArray (0..863), then int64 × 2 (864,872), Stat × 2 (880,912), int64 (944).
static_assert(sizeof(Stat)      == 32,  "Stat size mismatch");
static_assert(sizeof(StatArray) == 96,  "StatArray size mismatch");
static_assert(sizeof(DeviceStats) == 952, "DeviceStats total size mismatch");
static_assert(offsetof(DeviceStats, allocation)            ==   0);
static_assert(offsetof(DeviceStats, segment)               ==  96);
static_assert(offsetof(DeviceStats, active)                == 192);
static_assert(offsetof(DeviceStats, inactive_split)        == 288);
static_assert(offsetof(DeviceStats, allocated_bytes)       == 384);
static_assert(offsetof(DeviceStats, reserved_bytes)        == 480);
static_assert(offsetof(DeviceStats, active_bytes)          == 576);
static_assert(offsetof(DeviceStats, inactive_split_bytes)  == 672);
static_assert(offsetof(DeviceStats, requested_bytes)       == 768);
static_assert(offsetof(DeviceStats, num_alloc_retries)     == 864);
static_assert(offsetof(DeviceStats, num_ooms)              == 872);
static_assert(offsetof(DeviceStats, oversize_allocations)  == 880);
static_assert(offsetof(DeviceStats, oversize_segments)     == 912);
static_assert(offsetof(DeviceStats, max_split_size)        == 944);

} // namespace shmem_npu_compat

// ── Per-allocation byte tracking (outside extern "C") ────────────────────────
// Track the bytes currently held by live my_malloc allocations and the
// lifetime peak so that memory_stats()["allocated_bytes.all.peak"] reports the
// true high-water mark seen during profile_run(), not just the post-cleanup
// current value.
//
// Both SHMEM-pool allocations (aclshmem_malloc) and fallback device allocations
// (aclrtMalloc) are counted here, matching the set of pointers returned to
// PyTorch by my_malloc / taken back by my_free.

static std::atomic<int64_t> g_alloc_cur_bytes{0};   // net live bytes
static std::atomic<int64_t> g_alloc_peak_bytes{0};  // high-water mark

static void alloc_track_add(int64_t size) noexcept
{
    int64_t cur = g_alloc_cur_bytes.fetch_add(size, std::memory_order_relaxed) + size;
    // CAS loop to update peak atomically.
    int64_t pk = g_alloc_peak_bytes.load(std::memory_order_relaxed);
    while (cur > pk &&
           !g_alloc_peak_bytes.compare_exchange_weak(pk, cur,
               std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

static void alloc_track_sub(int64_t size) noexcept
{
    g_alloc_cur_bytes.fetch_sub(size, std::memory_order_relaxed);
}

// ── Event-based deferred freeing ─────────────────────────────────────────────
// Instead of blocking on aclrtSynchronizeStream in every my_free call, we
// record a lightweight ACL event on the stream, push the pending free onto a
// deferred queue, and actually free the memory later once the event signals
// completion.  This is the same pattern used by PyTorch's NPU caching
// allocator and avoids host-device synchronisation on the hot path.

struct DeferredFreeEntry {
    void       *ptr;
    ssize_t     size;
    bool        is_shmem;
    aclrtEvent  event;
};

// All access to g_deferred_frees and g_event_pool is guarded by g_ptrs_mutex
// (declared later in the extern "C" block).  Forward-declare the mutex here so
// the helpers can reference it in comments; actual locking is done by callers.
static std::deque<DeferredFreeEntry>  g_deferred_frees;
static std::vector<aclrtEvent>        g_event_pool;

// NOTE: g_deferred_frees and g_event_pool are guarded by g_ptrs_mutex which
// is declared in the extern "C" block below.  All helper functions below
// require the caller to already hold that lock.

// Acquire an ACL event from the pool, or create a new one.
// Caller MUST hold g_ptrs_mutex.
static aclrtEvent acquire_event()
{
    if (!g_event_pool.empty()) {
        aclrtEvent ev = g_event_pool.back();
        g_event_pool.pop_back();
        return ev;
    }
    aclrtEvent ev = nullptr;
    aclError err = aclrtCreateEventWithFlag(&ev, ACL_EVENT_CAPTURE_STREAM_PROGRESS);
    if (err != ACL_ERROR_NONE) {
        std::cerr << "[shmem_allocator] aclrtCreateEventWithFlag failed: "
                  << err << "\n";
        return nullptr;
    }
    return ev;
}

// Return an ACL event to the pool for reuse.
// Caller MUST hold g_ptrs_mutex.
static void release_event(aclrtEvent ev)
{
    g_event_pool.push_back(ev);
}

// Poll the deferred-free queue and actually free entries whose events have
// completed.  Stops at the first not-ready entry (events on a stream complete
// in FIFO order, so everything behind it is also not ready).
// Caller MUST hold g_ptrs_mutex.
static void process_deferred_frees()
{
    while (!g_deferred_frees.empty()) {
        auto &entry = g_deferred_frees.front();
        aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
        aclError err = aclrtQueryEventStatus(entry.event, &status);
        if (err != ACL_ERROR_NONE ||
            status != ACL_EVENT_RECORDED_STATUS_COMPLETE) {
            break;
        }
        // Event completed – reclaim the memory.
        release_event(entry.event);
        if (entry.is_shmem) {
            aclshmem_free(entry.ptr);
        } else {
            aclrtFree(entry.ptr);
        }
        alloc_track_sub(entry.size);
        g_deferred_frees.pop_front();
    }
}

// Blocking drain: synchronise every outstanding event and free all deferred
// entries.  Used during finalization when we must release everything.
// Caller MUST hold g_ptrs_mutex.
static void drain_deferred_frees()
{
    for (auto &entry : g_deferred_frees) {
        aclrtSynchronizeEvent(entry.event);
        release_event(entry.event);
        if (entry.is_shmem) {
            aclshmem_free(entry.ptr);
        } else {
            aclrtFree(entry.ptr);
        }
        alloc_track_sub(entry.size);
    }
    g_deferred_frees.clear();
}

// Destroy all pooled events.  Called during finalization after draining.
// Caller MUST hold g_ptrs_mutex.
static void destroy_event_pool()
{
    for (auto ev : g_event_pool) {
        aclrtDestroyEvent(ev);
    }
    g_event_pool.clear();
}

// ── Stats callbacks (C++ linkage, outside extern "C") ────────────────────────
// g_shmem_initialized lives inside extern "C"; access it via this accessor.
static bool shmem_is_initialized_flag() noexcept;  // defined after extern "C"

// getDeviceStats: returns SHMEM pool statistics mapped to DeviceStats format.
// The AGGREGATE slot (index 0) is filled; SMALL_POOL/LARGE_POOL remain zero.
//
// allocated_bytes tracks actual tensor bytes handed to PyTorch (current + peak).
// reserved_bytes  tracks the total SHMEM pool capacity as seen by the driver,
//                 which is used by determine_available_memory() to exclude the
//                 pool's internal slack from the non-torch-allocation estimate.
static shmem_npu_compat::DeviceStats shmem_get_device_stats_impl(int /*device*/)
{
    shmem_npu_compat::DeviceStats stats{};
    int64_t cur  = g_alloc_cur_bytes.load(std::memory_order_relaxed);
    int64_t peak = g_alloc_peak_bytes.load(std::memory_order_relaxed);
    // allocated_bytes: tensor bytes actively held by live allocations.
    stats.allocated_bytes[0].current   = cur;
    stats.allocated_bytes[0].peak      = peak;
    stats.allocated_bytes[0].allocated = peak;  // lifetime high-water
    if (shmem_is_initialized_flag()) {
        uint64_t total = 0, used = 0, avail = 0;
        aclshmem_get_memory_stats(&total, &used, &avail);
        // reserved_bytes: total capacity of the SHMEM pool (driver-level view).
        // determine_available_memory() subtracts this from the driver-level
        // total_allocated_bytes so that the pool's unused capacity is not
        // mistakenly treated as non-torch overhead.
        stats.reserved_bytes[0].current    = static_cast<int64_t>(total);
        stats.reserved_bytes[0].peak       = static_cast<int64_t>(total);
        stats.reserved_bytes[0].allocated  = static_cast<int64_t>(total);
    }
    return stats;
}

// resetPeakStats: reset the peak counter to the current live allocation level.
static void shmem_reset_peak_stats_impl(int /*device*/)
{
    int64_t cur = g_alloc_cur_bytes.load(std::memory_order_relaxed);
    g_alloc_peak_bytes.store(cur, std::memory_order_relaxed);
}

// ── Everything else can live inside extern "C" ──────────────────────────────
extern "C" {

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// acl/acl.h and sys/types.h are already included above (outside extern "C").

// ---------------------------------------------------------------------------
// Module-level state

static std::mutex      g_shmem_mutex;
static bool            g_shmem_initialized = false;

// Pointers obtained from aclshmem_malloc are recorded here so that
// my_free can route them back through aclshmem_free instead of aclrtFree.
static std::mutex                  g_ptrs_mutex;
static std::unordered_set<void *>  g_shmem_ptrs;

// ---------------------------------------------------------------------------
// Internal helper: initialise the SHMEM pool (idempotent, thread-safe).
//
// Precondition: ACL must already be initialised (vllm-ascend does this).
// The pool size is read from SHMEM_INITIAL_POOL_SIZE (bytes); default 2 GiB.

static void ensure_shmem_initialized()
{
    std::lock_guard<std::mutex> lock(g_shmem_mutex);
    if (g_shmem_initialized) {
        return;
    }

    // Dynamic expansion must be enabled BEFORE aclshmemx_init_attr.
    aclshmem_enable_dynamic_expansion(true);

    // Determine initial pool size.
    //
    // Default: 2 GiB.  Override via SHMEM_INITIAL_POOL_SIZE (bytes).
    // The value is clamped to ACLSHMEM_MAX_LOCAL_SIZE (40 GiB) which is the hard
    // device limit for symmetric SHMEM memory; the dynamic expansion path (aclrtMalloc)
    // handles growth beyond this, so there is no need to pre-allocate the full HBM.
    //
    // NOTE: ACLSHMEM_MAX_LOCAL_SIZE is a macro defined in shmem_common_types.h;
    //       do NOT redeclare it as a local variable (macro expansion would corrupt syntax).
    static const int64_t kShmemMaxLocalSize =
        static_cast<int64_t>(ACLSHMEM_MAX_LOCAL_SIZE);  // 40 GiB hard cap
    int64_t local_mem_size = 2LL * 1024 * 1024 * 1024;  // 2 GiB default

    const char *env_size = std::getenv("SHMEM_INITIAL_POOL_SIZE");
    if (env_size != nullptr) {
        try {
            int64_t requested = std::stoll(env_size);
            if (requested <= 0) {
                std::cerr << "[shmem_allocator] SHMEM_INITIAL_POOL_SIZE must be > 0, "
                             "using 2 GiB default.\n";
            } else {
                local_mem_size = std::min(requested, kShmemMaxLocalSize);
                if (local_mem_size != requested) {
                    std::cerr << "[shmem_allocator] SHMEM_INITIAL_POOL_SIZE clamped from "
                              << requested / (1024*1024) << " MiB to "
                              << local_mem_size / (1024*1024)
                              << " MiB (ACLSHMEM_MAX_LOCAL_SIZE limit).\n";
                }
            }
        } catch (...) {
            std::cerr << "[shmem_allocator] Invalid SHMEM_INITIAL_POOL_SIZE, "
                         "using 2 GiB default.\n";
        }
    }

    // Bootstrap with UniqueID mode, single PE (n_pes = 1).
    aclshmemx_uniqueid_t  uid{};
    aclshmemx_init_attr_t attr{};

    int ret = aclshmemx_get_uniqueid(&uid);
    if (ret != ACLSHMEM_SUCCESS) {
        throw std::runtime_error(
            "[shmem_allocator] aclshmemx_get_uniqueid failed: " +
            std::to_string(ret));
    }

    ret = aclshmemx_set_attr_uniqueid_args(0, 1, local_mem_size, &uid, &attr);
    if (ret != ACLSHMEM_SUCCESS) {
        throw std::runtime_error(
            "[shmem_allocator] aclshmemx_set_attr_uniqueid_args failed: " +
            std::to_string(ret));
    }

    ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr);
    if (ret != ACLSHMEM_SUCCESS) {
        throw std::runtime_error(
            "[shmem_allocator] aclshmemx_init_attr failed: " +
            std::to_string(ret));
    }

    g_shmem_initialized = true;
    std::cout << "[shmem_allocator] SHMEM pool initialised ("
              << (local_mem_size / 1024 / 1024)
              << " MiB initial, dynamic expansion enabled).\n";
}

// ---------------------------------------------------------------------------
// NPUPluggableAllocator interface
//
// Signature required by torch_npu:
//   void* my_malloc(ssize_t size, int device, aclrtStream stream)
//   void  my_free  (void* ptr, ssize_t size, int device, aclrtStream stream)

__attribute__((visibility("default")))
void *my_malloc(ssize_t size, int device, aclrtStream stream)
{
    // PyTorch issues zero-size allocations for empty tensors; SHMEM does not
    // handle size=0 and aclshmem_malloc(0) returns nullptr anyway.  Return
    // early to avoid log noise and unnecessary aclrtMalloc fallback.
    if (size <= 0) {
        return nullptr;
    }

    // Reclaim memory from completed deferred frees before allocating, so that
    // the SHMEM pool / device allocator can reuse it immediately.
    {
        std::lock_guard<std::mutex> lock(g_ptrs_mutex);
        process_deferred_frees();
    }

    try {
        ensure_shmem_initialized();
    } catch (const std::exception &e) {
        std::cerr << "[shmem_allocator] my_malloc: init failed: " << e.what()
                  << ", falling back to aclrtMalloc\n";
        void *fb = nullptr;
        aclrtMalloc(&fb, static_cast<size_t>(size), ACL_MEM_MALLOC_HUGE_FIRST);
        if (fb != nullptr) {
            alloc_track_add(size);
        }
        return fb;
    }

    void *ptr = aclshmem_malloc(static_cast<size_t>(size));
    if (ptr != nullptr) {
        {
            std::lock_guard<std::mutex> lock(g_ptrs_mutex);
            g_shmem_ptrs.insert(ptr);
        }
        alloc_track_add(size);
        return ptr;
    }

    // SHMEM pool exhausted even after dynamic expansion; fall back.
    std::cerr << "[shmem_allocator] my_malloc: aclshmem_malloc failed for "
                 "size=" << size << ", falling back to aclrtMalloc\n";
    void *fb = nullptr;
    aclrtMalloc(&fb, static_cast<size_t>(size), ACL_MEM_MALLOC_HUGE_FIRST);
    if (fb != nullptr) {
        alloc_track_add(size);
    } else {
        // Both SHMEM and direct aclrtMalloc failed – genuine OOM.
        // Log explicitly so the user gets a clear message instead of a
        // cryptic "Get data ptr failed" / "ERR00001" from the NPU runtime.
        std::cerr << "[shmem_allocator] my_malloc: fallback aclrtMalloc also "
                     "failed for size=" << size
                  << " -- device is out of memory (OOM). "
                     "Consider reducing gpu_memory_utilization or "
                     "max_model_len.\n";
    }
    return fb;
}

__attribute__((visibility("default")))
void my_free(void *ptr, ssize_t size, int device, aclrtStream stream)
{
    if (ptr == nullptr) {
        return;
    }

    // Event-based deferred freeing.
    //
    // The SHMEM pool and aclrtFree have no stream awareness – freed memory is
    // immediately available for reuse.  If an NPU kernel submitted on `stream`
    // is still reading/writing this memory, the next aclshmem_malloc could
    // hand the same region to a new tensor, causing a data race.
    //
    // Instead of blocking the host with aclrtSynchronizeStream (which
    // serialises host/device and severely hurts throughput), we record a
    // lightweight ACL event on the stream and push the pending free onto a
    // deferred queue.  The memory is actually freed later – in my_malloc or
    // my_free – once the event signals that all prior work on the stream has
    // completed.  If event creation or recording fails we fall back to the
    // synchronous path for safety.

    bool is_shmem = false;
    {
        std::lock_guard<std::mutex> lock(g_ptrs_mutex);

        // Opportunistically reclaim completed deferred frees.
        process_deferred_frees();

        auto it = g_shmem_ptrs.find(ptr);
        if (it != g_shmem_ptrs.end()) {
            is_shmem = true;
            g_shmem_ptrs.erase(it);
        }

        if (stream != nullptr) {
            aclrtEvent ev = acquire_event();
            if (ev != nullptr) {
                aclError err = aclrtRecordEvent(ev, stream);
                if (err == ACL_ERROR_NONE) {
                    g_deferred_frees.push_back({ptr, size, is_shmem, ev});
                    return;  // deferred – do not free or track_sub yet
                }
                // Record failed; return event and fall back to sync path.
                release_event(ev);
                std::cerr << "[shmem_allocator] aclrtRecordEvent failed: "
                          << err << ", falling back to sync free\n";
            }
            // Event creation failed; fall back to sync path.
            aclrtSynchronizeStream(stream);
        }
    }

    // Immediate free (stream == nullptr, or event fallback).
    if (is_shmem) {
        aclshmem_free(ptr);
    } else {
        aclrtFree(ptr);
    }
    alloc_track_sub(size);
}

// ---------------------------------------------------------------------------
// Python extension interface

static PyObject *py_shmem_init(PyObject * /*self*/, PyObject * /*args*/)
{
    try {
        ensure_shmem_initialized();
        Py_RETURN_TRUE;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

static PyObject *py_shmem_finalize(PyObject * /*self*/, PyObject * /*args*/)
{
    std::lock_guard<std::mutex> lock(g_shmem_mutex);
    if (g_shmem_initialized) {
        // Drain all deferred frees before tearing down the pool – any
        // shmem pointers still in the deferred queue would become invalid
        // after aclshmem_finalize.
        {
            std::lock_guard<std::mutex> plock(g_ptrs_mutex);
            drain_deferred_frees();
            destroy_event_pool();
        }

        int ret = aclshmem_finalize();
        if (ret != ACLSHMEM_SUCCESS) {
            std::cerr << "[shmem_allocator] aclshmem_finalize returned "
                      << ret << "\n";
        }
        g_shmem_initialized = false;
        {
            std::lock_guard<std::mutex> plock(g_ptrs_mutex);
            g_shmem_ptrs.clear();
        }
        std::cout << "[shmem_allocator] SHMEM pool finalized.\n";
    }
    Py_RETURN_NONE;
}

static PyObject *py_is_shmem_initialized(PyObject * /*self*/,
                                          PyObject * /*args*/)
{
    return PyBool_FromLong(static_cast<long>(g_shmem_initialized));
}

static PyObject *py_get_memory_stats(PyObject * /*self*/, PyObject * /*args*/)
{
    if (!g_shmem_initialized) {
        PyErr_SetString(PyExc_RuntimeError,
                        "SHMEM not initialized; call shmem_init() first");
        return nullptr;
    }
    uint64_t total = 0, used = 0, avail = 0;
    aclshmem_get_memory_stats(&total, &used, &avail);

    PyObject *result = PyTuple_New(3);
    if (!result) {
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, PyLong_FromUnsignedLongLong(total));
    PyTuple_SET_ITEM(result, 1, PyLong_FromUnsignedLongLong(used));
    PyTuple_SET_ITEM(result, 2, PyLong_FromUnsignedLongLong(avail));
    return result;
}

// ---------------------------------------------------------------------------
// Module registration

static PyMethodDef g_module_methods[] = {
    {"shmem_init",
     py_shmem_init,
     METH_NOARGS,
     "Initialize the SHMEM dynamic memory pool."},
    {"shmem_finalize",
     py_shmem_finalize,
     METH_NOARGS,
     "Release the SHMEM dynamic memory pool."},
    {"is_shmem_initialized",
     py_is_shmem_initialized,
     METH_NOARGS,
     "Return True if the SHMEM pool is initialised."},
    {"get_memory_stats",
     py_get_memory_stats,
     METH_NOARGS,
     "Return (total_bytes, used_bytes, available_bytes)."},
    {nullptr, nullptr, 0, nullptr} // sentinel
};

static struct PyModuleDef g_module_def = {
    PyModuleDef_HEAD_INIT,
    "shmem_allocator",
    "SHMEM dynamic memory allocator for NPUPluggableAllocator",
    -1,
    g_module_methods
};

PyMODINIT_FUNC PyInit_shmem_allocator(void)
{
    return PyModule_Create(&g_module_def);
}

} // extern "C"

// ── Post-extern-"C" definitions ───────────────────────────────────────────────

// Accessor for g_shmem_initialized (defined in extern "C" above).
static bool shmem_is_initialized_flag() noexcept
{
    return g_shmem_initialized;
}

// C-linkage address getters: Python uses ctypes to retrieve these and passes
// them to torch_npu's set_get_device_stats_fn / set_reset_peak_status_fn.
// Using uint64_t matches the type expected by Module.cpp's Python bindings.
extern "C" {

__attribute__((visibility("default")))
uint64_t shmem_get_device_stats_fn_addr(void)
{
    return reinterpret_cast<uint64_t>(&shmem_get_device_stats_impl);
}

__attribute__((visibility("default")))
uint64_t shmem_reset_peak_stats_fn_addr(void)
{
    return reinterpret_cast<uint64_t>(&shmem_reset_peak_stats_impl);
}

} // extern "C" (address getters)
