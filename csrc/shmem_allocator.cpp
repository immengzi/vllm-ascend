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
#include <deque>
#include <iostream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ── SHMEM public API (C++ header, cannot be inside extern "C") ──────────────
#include "acl/acl.h"
#include "shmem.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

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

// Compile-time sanity check: 9 StatArrays × 96 bytes + 88 bytes = 952 bytes.
static_assert(sizeof(DeviceStats) == 952,
    "DeviceStats layout mismatch – verify against torch_npu 2.8.0");

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

// ── Module state shared by the Python shim and allocator callbacks ───────────
static std::mutex g_shmem_mutex;
static bool g_shmem_initialized = false;
static std::mutex g_ptrs_mutex;
static std::unordered_set<void *> g_live_ptrs;
static std::unordered_set<void *> g_shmem_ptrs;

// ── Stream-aware deferred freeing ────────────────────────────────────────────
// changeCurrentAllocator bypasses torch_npu's caching allocator. To preserve
// correct stream ordering, we defer the real free until every stream that
// touched the pointer has reached a recorded event.

struct DeferredFreeEntry {
    void *ptr = nullptr;
    ssize_t size = 0;
    bool is_shmem = false;
    std::vector<aclrtEvent> events;
};

static std::deque<DeferredFreeEntry> g_deferred_frees;
static std::vector<aclrtEvent> g_event_pool;
static std::unordered_map<void *, std::unordered_set<uintptr_t>> g_ptr_streams;

static aclrtEvent acquire_event_locked()
{
    if (!g_event_pool.empty()) {
        aclrtEvent ev = g_event_pool.back();
        g_event_pool.pop_back();
        return ev;
    }

    aclrtEvent ev = nullptr;
    const aclError err =
        aclrtCreateEventWithFlag(&ev, ACL_EVENT_CAPTURE_STREAM_PROGRESS);
    if (err != ACL_ERROR_NONE) {
        std::cerr << "[shmem_allocator] aclrtCreateEventWithFlag failed: "
                  << err << "\n";
        return nullptr;
    }
    return ev;
}

static void release_event_locked(aclrtEvent ev)
{
    if (ev != nullptr) {
        g_event_pool.push_back(ev);
    }
}

static void free_now(const DeferredFreeEntry &entry)
{
    if (entry.is_shmem) {
        aclshmem_free(entry.ptr);
    } else {
        aclrtFree(entry.ptr);
    }
    alloc_track_sub(entry.size);
}

static void reclaim_ready_deferred_frees_locked(
    std::vector<DeferredFreeEntry> &ready_entries)
{
    for (auto it = g_deferred_frees.begin(); it != g_deferred_frees.end();) {
        bool all_ready = true;
        for (aclrtEvent ev : it->events) {
            aclrtEventRecordedStatus status =
                ACL_EVENT_RECORDED_STATUS_NOT_READY;
            const aclError err = aclrtQueryEventStatus(ev, &status);
            if (err != ACL_ERROR_NONE ||
                status != ACL_EVENT_RECORDED_STATUS_COMPLETE) {
                all_ready = false;
                break;
            }
        }

        if (!all_ready) {
            ++it;
            continue;
        }

        for (aclrtEvent ev : it->events) {
            release_event_locked(ev);
        }
        ready_entries.push_back(std::move(*it));
        it = g_deferred_frees.erase(it);
    }
}

static void drain_deferred_frees_locked(
    std::vector<DeferredFreeEntry> &entries_to_free)
{
    entries_to_free.clear();
    entries_to_free.insert(entries_to_free.end(),
                           std::make_move_iterator(g_deferred_frees.begin()),
                           std::make_move_iterator(g_deferred_frees.end()));
    g_deferred_frees.clear();
}

static std::vector<aclrtEvent> destroy_event_pool_locked()
{
    std::vector<aclrtEvent> events_to_destroy;
    events_to_destroy.swap(g_event_pool);
    return events_to_destroy;
}

static uintptr_t to_stream_key(aclrtStream stream) noexcept
{
    return reinterpret_cast<uintptr_t>(stream);
}

static void record_stream_impl(void *ptr, c10_npu::NPUStream stream)
{
    if (ptr == nullptr) {
        return;
    }
    const uintptr_t stream_key = to_stream_key(stream.stream());
    if (stream_key == 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_ptrs_mutex);
    if (!g_live_ptrs.count(ptr)) {
        return;
    }
    g_ptr_streams[ptr].insert(stream_key);
}

static void erase_stream_impl(void *ptr, c10_npu::NPUStream stream)
{
    if (ptr == nullptr) {
        return;
    }
    const uintptr_t stream_key = to_stream_key(stream.stream());

    std::lock_guard<std::mutex> lock(g_ptrs_mutex);
    if (!g_live_ptrs.count(ptr)) {
        return;
    }
    auto it = g_ptr_streams.find(ptr);
    if (it == g_ptr_streams.end()) {
        return;
    }
    if (stream_key != 0) {
        it->second.erase(stream_key);
    }
    if (it->second.empty()) {
        g_ptr_streams.erase(it);
    }
}

// ── Stats callbacks (C++ linkage, outside extern "C") ────────────────────────
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
    // Default: 2 MiB (one NPU large page).  Override via SHMEM_INITIAL_POOL_SIZE
    // (bytes).  A minimal initial pool avoids competing with model loading and
    // KV cache allocation for device memory; the dynamic expansion path
    // (expand_pool → aclrtMalloc) handles growth on demand at runtime.
    //
    // The value is clamped to ACLSHMEM_MAX_LOCAL_SIZE (40 GiB) which is the hard
    // device limit for symmetric SHMEM memory.
    //
    // NOTE: ACLSHMEM_MAX_LOCAL_SIZE is a macro defined in shmem_common_types.h;
    //       do NOT redeclare it as a local variable (macro expansion would corrupt syntax).
    static const int64_t kShmemMaxLocalSize =
        static_cast<int64_t>(ACLSHMEM_MAX_LOCAL_SIZE);  // 40 GiB hard cap
    int64_t local_mem_size = 2LL * 1024 * 1024;  // 2 MiB default

    const char *env_size = std::getenv("SHMEM_INITIAL_POOL_SIZE");
    if (env_size != nullptr) {
        try {
            int64_t requested = std::stoll(env_size);
            if (requested <= 0) {
                std::cerr << "[shmem_allocator] SHMEM_INITIAL_POOL_SIZE must be > 0, "
                             "using 2 MiB default.\n";
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
                         "using 2 MiB default.\n";
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

    std::vector<DeferredFreeEntry> ready_entries;
    {
        std::lock_guard<std::mutex> lock(g_ptrs_mutex);
        reclaim_ready_deferred_frees_locked(ready_entries);
    }
    for (const auto &entry : ready_entries) {
        free_now(entry);
    }

    try {
        ensure_shmem_initialized();
    } catch (const std::exception &e) {
        std::cerr << "[shmem_allocator] my_malloc: init failed: " << e.what()
                  << ", falling back to aclrtMalloc\n";
        void *fb = nullptr;
        aclrtMalloc(&fb, static_cast<size_t>(size), ACL_MEM_MALLOC_HUGE_FIRST);
        if (fb != nullptr) {
            std::lock_guard<std::mutex> lock(g_ptrs_mutex);
            g_live_ptrs.insert(fb);
        }
        if (fb != nullptr) {
            alloc_track_add(size);
        }
        return fb;
    }

    void *ptr = aclshmem_malloc(static_cast<size_t>(size));
    if (ptr != nullptr) {
        {
            std::lock_guard<std::mutex> lock(g_ptrs_mutex);
            g_live_ptrs.insert(ptr);
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
        {
            std::lock_guard<std::mutex> lock(g_ptrs_mutex);
            g_live_ptrs.insert(fb);
        }
        alloc_track_add(size);
    }
    return fb;
}

__attribute__((visibility("default")))
void my_free(void *ptr, ssize_t size, int device, aclrtStream stream)
{
    if (ptr == nullptr) {
        return;
    }

    std::vector<DeferredFreeEntry> ready_entries;
    std::vector<uintptr_t> sync_stream_keys;
    std::vector<aclrtEvent> destroy_events;
    bool is_shmem = false;
    bool defer_free = false;
    {
        std::lock_guard<std::mutex> lock(g_ptrs_mutex);
        reclaim_ready_deferred_frees_locked(ready_entries);

        auto it = g_shmem_ptrs.find(ptr);
        if (it != g_shmem_ptrs.end()) {
            is_shmem = true;
            g_shmem_ptrs.erase(it);
        }
        g_live_ptrs.erase(ptr);

        std::unordered_set<uintptr_t> stream_keys;
        if (stream != nullptr) {
            stream_keys.insert(to_stream_key(stream));
        }
        auto streams_it = g_ptr_streams.find(ptr);
        if (streams_it != g_ptr_streams.end()) {
            stream_keys.insert(streams_it->second.begin(),
                               streams_it->second.end());
            g_ptr_streams.erase(streams_it);
        }

        if (!stream_keys.empty()) {
            sync_stream_keys.assign(stream_keys.begin(), stream_keys.end());
            DeferredFreeEntry entry;
            entry.ptr = ptr;
            entry.size = size;
            entry.is_shmem = is_shmem;
            entry.events.reserve(stream_keys.size());

            bool event_record_failed = false;
            for (uintptr_t stream_key : stream_keys) {
                aclrtEvent ev = acquire_event_locked();
                if (ev == nullptr) {
                    event_record_failed = true;
                    break;
                }
                const aclError err =
                    aclrtRecordEvent(ev,
                                     reinterpret_cast<aclrtStream>(stream_key));
                if (err != ACL_ERROR_NONE) {
                    std::cerr << "[shmem_allocator] aclrtRecordEvent failed: "
                              << err
                              << ", falling back to synchronous free\n";
                    destroy_events.push_back(ev);
                    event_record_failed = true;
                    break;
                }
                entry.events.push_back(ev);
            }

            if (!event_record_failed && !entry.events.empty()) {
                g_deferred_frees.push_back(std::move(entry));
                defer_free = true;
            } else {
                for (aclrtEvent ev : entry.events) {
                    destroy_events.push_back(ev);
                }
            }
        }
    }

    for (const auto &entry : ready_entries) {
        free_now(entry);
    }

    if (defer_free) {
        return;
    }

    for (uintptr_t stream_key : sync_stream_keys) {
        aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream_key));
    }
    for (aclrtEvent ev : destroy_events) {
        aclrtDestroyEvent(ev);
    }

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
        std::vector<DeferredFreeEntry> deferred_entries;
        std::vector<aclrtEvent> pooled_events;
        {
            std::lock_guard<std::mutex> plock(g_ptrs_mutex);
            drain_deferred_frees_locked(deferred_entries);
            pooled_events = destroy_event_pool_locked();
            g_ptr_streams.clear();
            g_live_ptrs.clear();
            g_shmem_ptrs.clear();
        }
        for (auto &entry : deferred_entries) {
            for (aclrtEvent ev : entry.events) {
                aclrtSynchronizeEvent(ev);
                aclrtDestroyEvent(ev);
            }
            free_now(entry);
        }
        for (aclrtEvent ev : pooled_events) {
            aclrtDestroyEvent(ev);
        }

        int ret = aclshmem_finalize();
        if (ret != ACLSHMEM_SUCCESS) {
            std::cerr << "[shmem_allocator] aclshmem_finalize returned "
                      << ret << "\n";
        }
        g_shmem_initialized = false;
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

// Accessor for g_shmem_initialized.
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

__attribute__((visibility("default")))
uint64_t shmem_record_stream_fn_addr(void)
{
    return reinterpret_cast<uint64_t>(&record_stream_impl);
}

__attribute__((visibility("default")))
uint64_t shmem_erase_stream_fn_addr(void)
{
    return reinterpret_cast<uint64_t>(&erase_stream_impl);
}

} // extern "C" (address getters)
