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
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>

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

// Compile-time sanity check: 9 StatArrays × 96 bytes + 88 bytes = 952 bytes.
static_assert(sizeof(DeviceStats) == 952,
    "DeviceStats layout mismatch – verify against torch_npu 2.8.0");

} // namespace shmem_npu_compat

// ── Stats callbacks (C++ linkage, outside extern "C") ────────────────────────
// g_shmem_initialized lives inside extern "C"; access it via this accessor.
static bool shmem_is_initialized_flag() noexcept;  // defined after extern "C"

// getDeviceStats: returns SHMEM pool statistics mapped to DeviceStats format.
// The AGGREGATE slot (index 0) is filled; SMALL_POOL/LARGE_POOL remain zero.
static shmem_npu_compat::DeviceStats shmem_get_device_stats_impl(int /*device*/)
{
    shmem_npu_compat::DeviceStats stats{};
    if (shmem_is_initialized_flag()) {
        uint64_t total = 0, used = 0, avail = 0;
        aclshmem_get_memory_stats(&total, &used, &avail);
        // allocated_bytes: how much the allocator client currently holds.
        stats.allocated_bytes[0].current   = static_cast<int64_t>(used);
        stats.allocated_bytes[0].peak      = static_cast<int64_t>(used);
        stats.allocated_bytes[0].allocated = static_cast<int64_t>(used);
        // reserved_bytes: total capacity of the SHMEM pool.
        stats.reserved_bytes[0].current    = static_cast<int64_t>(total);
        stats.reserved_bytes[0].peak       = static_cast<int64_t>(total);
        stats.reserved_bytes[0].allocated  = static_cast<int64_t>(total);
    }
    return stats;
}

// resetPeakStats: SHMEM does not track peak separately; no-op.
static void shmem_reset_peak_stats_impl(int /*device*/) {}

// ── Everything else can live inside extern "C" ──────────────────────────────
extern "C" {

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <sys/types.h>
#include "acl/acl.h"

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
    int64_t local_mem_size = 2LL * 1024 * 1024 * 1024; // 2 GiB
    const char *env_size = std::getenv("SHMEM_INITIAL_POOL_SIZE");
    if (env_size != nullptr) {
        try {
            local_mem_size = std::stoll(env_size);
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
    try {
        ensure_shmem_initialized();
    } catch (const std::exception &e) {
        std::cerr << "[shmem_allocator] my_malloc: init failed: " << e.what()
                  << ", falling back to aclrtMalloc\n";
        void *fb = nullptr;
        aclrtMalloc(&fb, static_cast<size_t>(size), ACL_MEM_MALLOC_HUGE_FIRST);
        return fb;
    }

    void *ptr = aclshmem_malloc(static_cast<size_t>(size));
    if (ptr != nullptr) {
        std::lock_guard<std::mutex> lock(g_ptrs_mutex);
        g_shmem_ptrs.insert(ptr);
        return ptr;
    }

    // SHMEM pool exhausted even after dynamic expansion; fall back.
    std::cerr << "[shmem_allocator] my_malloc: aclshmem_malloc failed for "
                 "size=" << size << ", falling back to aclrtMalloc\n";
    void *fb = nullptr;
    aclrtMalloc(&fb, static_cast<size_t>(size), ACL_MEM_MALLOC_HUGE_FIRST);
    return fb;
}

__attribute__((visibility("default")))
void my_free(void *ptr, ssize_t size, int device, aclrtStream stream)
{
    if (ptr == nullptr) {
        return;
    }

    bool is_shmem = false;
    {
        std::lock_guard<std::mutex> lock(g_ptrs_mutex);
        auto it = g_shmem_ptrs.find(ptr);
        if (it != g_shmem_ptrs.end()) {
            is_shmem = true;
            g_shmem_ptrs.erase(it);
        }
    }

    if (is_shmem) {
        aclshmem_free(ptr);
    } else {
        aclrtFree(ptr);
    }
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
