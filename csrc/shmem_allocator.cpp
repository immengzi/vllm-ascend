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
#include <algorithm>
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
enum class BlockState : uint8_t {
    ACTIVE,
    CACHED_LOCAL,
    CACHED_GLOBAL,
    DEFERRED,
};

struct BlockMeta {
    void *ptr = nullptr;
    size_t alloc_size = 0;
    size_t requested_size = 0;
    uintptr_t alloc_stream_key = 0;
    bool is_shmem = false;
    BlockState state = BlockState::ACTIVE;
    uint64_t generation = 0;
    std::unordered_set<uintptr_t> extra_streams;
};

struct CachedBlockRef {
    void *ptr = nullptr;
    uint64_t generation = 0;
};

struct DeferredFreeEntry {
    void *ptr = nullptr;
    bool is_shmem = false;
    uintptr_t home_stream_key = 0;
    uint64_t generation = 0;
    std::vector<aclrtEvent> events;
};

struct PhysicalBlock {
    void *ptr = nullptr;
    bool is_shmem = false;
};

struct StreamSyncedBlock {
    PhysicalBlock block;
    uintptr_t stream_key = 0;
};

static std::mutex g_shmem_mutex;
static bool g_shmem_initialized = false;
static std::mutex g_ptrs_mutex;
static std::unordered_map<void *, BlockMeta> g_blocks;
static std::unordered_map<uintptr_t,
                          std::unordered_map<size_t, std::vector<CachedBlockRef>>>
    g_stream_local_bins;
static std::unordered_map<size_t, std::vector<CachedBlockRef>> g_global_bins;
static std::unordered_map<uintptr_t, std::deque<DeferredFreeEntry>>
    g_deferred_by_stream;
static std::vector<aclrtEvent> g_event_pool;
static std::atomic<int64_t> g_cached_cur_bytes{0};
static std::atomic<int64_t> g_cached_peak_bytes{0};
static std::atomic<uint64_t> g_next_generation{1};
static int64_t g_cache_limit_bytes = 1024LL * 1024 * 1024;  // 1 GiB
static int g_reclaim_budget = 8;

static void cached_track_add(size_t size) noexcept
{
    const int64_t delta = static_cast<int64_t>(size);
    int64_t cur = g_cached_cur_bytes.fetch_add(delta, std::memory_order_relaxed) + delta;
    int64_t pk = g_cached_peak_bytes.load(std::memory_order_relaxed);
    while (cur > pk &&
           !g_cached_peak_bytes.compare_exchange_weak(pk, cur,
               std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

static void cached_track_sub(size_t size) noexcept
{
    g_cached_cur_bytes.fetch_sub(static_cast<int64_t>(size),
                                 std::memory_order_relaxed);
}

static size_t round_up(size_t size, size_t align)
{
    return ((size + align - 1) / align) * align;
}

static size_t round_size_class(size_t size)
{
    constexpr size_t kMinBlockSize = 512;
    constexpr size_t kSmallThreshold = 1ULL << 20;   // 1 MiB
    constexpr size_t kMediumThreshold = 16ULL << 20; // 16 MiB
    constexpr size_t kMediumAlign = 2ULL << 20;      // 2 MiB
    constexpr size_t kLargeAlign = 16ULL << 20;      // 16 MiB

    if (size <= kMinBlockSize) {
        return kMinBlockSize;
    }
    if (size <= kSmallThreshold) {
        return round_up(size, kMinBlockSize);
    }
    if (size <= kMediumThreshold) {
        return round_up(size, kMediumAlign);
    }
    return round_up(size, kLargeAlign);
}

static int64_t parse_env_i64(const char *name, int64_t default_value,
                             int64_t min_value = 1)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }
    try {
        int64_t value = std::stoll(raw);
        if (value < min_value) {
            return default_value;
        }
        return value;
    } catch (...) {
        return default_value;
    }
}

static void load_runtime_config()
{
    g_cache_limit_bytes =
        parse_env_i64("SHMEM_CACHE_LIMIT_MB", 1024, 1) * 1024LL * 1024LL;
    g_reclaim_budget = static_cast<int>(
        parse_env_i64("SHMEM_RECLAIM_BUDGET", 8, 1));
}

// ── Stream-aware deferred freeing with local/global caches ───────────────────
// changeCurrentAllocator bypasses torch_npu's caching allocator. To recover a
// similar fast path, we keep freed blocks in a stream-local cache and only use
// event-based deferred reclamation for blocks that have actually crossed streams.

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

static void free_storage(const PhysicalBlock &block)
{
    if (block.ptr == nullptr) {
        return;
    }
    if (block.is_shmem) {
        aclshmem_free(block.ptr);
    } else {
        aclrtFree(block.ptr);
    }
}

static void free_blocks(const std::vector<PhysicalBlock> &blocks)
{
    for (const PhysicalBlock &block : blocks) {
        free_storage(block);
    }
}

static void sync_streams_and_free_blocks(
    const std::vector<StreamSyncedBlock> &blocks)
{
    std::unordered_set<uintptr_t> synced_streams;
    synced_streams.reserve(blocks.size());

    for (const StreamSyncedBlock &entry : blocks) {
        if (!synced_streams.insert(entry.stream_key).second) {
            continue;
        }
        const aclError err =
            aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(entry.stream_key));
        if (err != ACL_ERROR_NONE) {
            std::cerr << "[shmem_allocator] aclrtSynchronizeStream failed: "
                      << err << "\n";
        }
    }

    for (const StreamSyncedBlock &entry : blocks) {
        free_storage(entry.block);
    }
}

static void erase_empty_bins_locked()
{
    for (auto it = g_stream_local_bins.begin(); it != g_stream_local_bins.end();) {
        for (auto inner = it->second.begin(); inner != it->second.end();) {
            if (inner->second.empty()) {
                inner = it->second.erase(inner);
            } else {
                ++inner;
            }
        }
        if (it->second.empty()) {
            it = g_stream_local_bins.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = g_global_bins.begin(); it != g_global_bins.end();) {
        if (it->second.empty()) {
            it = g_global_bins.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = g_deferred_by_stream.begin(); it != g_deferred_by_stream.end();) {
        if (it->second.empty()) {
            it = g_deferred_by_stream.erase(it);
        } else {
            ++it;
        }
    }
}

static void prune_empty_local_bin_locked(uintptr_t stream_key, size_t alloc_size)
{
    auto stream_it = g_stream_local_bins.find(stream_key);
    if (stream_it == g_stream_local_bins.end()) {
        return;
    }
    auto bin_it = stream_it->second.find(alloc_size);
    if (bin_it != stream_it->second.end() && bin_it->second.empty()) {
        stream_it->second.erase(bin_it);
    }
    if (stream_it->second.empty()) {
        g_stream_local_bins.erase(stream_it);
    }
}

static void prune_empty_global_bin_locked(size_t alloc_size)
{
    auto bin_it = g_global_bins.find(alloc_size);
    if (bin_it != g_global_bins.end() && bin_it->second.empty()) {
        g_global_bins.erase(bin_it);
    }
}

static void prune_empty_deferred_queue_locked(uintptr_t stream_key)
{
    auto queue_it = g_deferred_by_stream.find(stream_key);
    if (queue_it != g_deferred_by_stream.end() && queue_it->second.empty()) {
        g_deferred_by_stream.erase(queue_it);
    }
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

static uint64_t next_generation() noexcept
{
    return g_next_generation.fetch_add(1, std::memory_order_relaxed);
}

static void mark_block_active_locked(BlockMeta &meta,
                                     uintptr_t stream_key,
                                     size_t requested_size)
{
    cached_track_sub(meta.alloc_size);
    meta.alloc_stream_key = stream_key;
    meta.requested_size = requested_size;
    meta.state = BlockState::ACTIVE;
    meta.generation = next_generation();
    meta.extra_streams.clear();
}

static void cache_block_local_locked(BlockMeta &meta)
{
    meta.state = BlockState::CACHED_LOCAL;
    meta.requested_size = 0;
    meta.extra_streams.clear();
    g_stream_local_bins[meta.alloc_stream_key][meta.alloc_size].push_back(
        {meta.ptr, meta.generation});
    cached_track_add(meta.alloc_size);
}

static void cache_block_global_locked(BlockMeta &meta)
{
    meta.state = BlockState::CACHED_GLOBAL;
    meta.requested_size = 0;
    meta.extra_streams.clear();
    g_global_bins[meta.alloc_size].push_back({meta.ptr, meta.generation});
    cached_track_add(meta.alloc_size);
}

static void *try_pop_local_cached_block_locked(size_t alloc_size,
                                               uintptr_t stream_key,
                                               size_t requested_size)
{
    auto stream_it = g_stream_local_bins.find(stream_key);
    if (stream_it == g_stream_local_bins.end()) {
        return nullptr;
    }
    auto bin_it = stream_it->second.find(alloc_size);
    if (bin_it == stream_it->second.end()) {
        return nullptr;
    }
    auto &bin = bin_it->second;
    while (!bin.empty()) {
        CachedBlockRef ref = bin.back();
        bin.pop_back();
        auto meta_it = g_blocks.find(ref.ptr);
        if (meta_it == g_blocks.end()) {
            continue;
        }
        BlockMeta &meta = meta_it->second;
        if (meta.generation != ref.generation ||
            meta.state != BlockState::CACHED_LOCAL ||
            meta.alloc_size != alloc_size) {
            continue;
        }
        mark_block_active_locked(meta, stream_key, requested_size);
        return meta.ptr;
    }
    prune_empty_local_bin_locked(stream_key, alloc_size);
    return nullptr;
}

static void *try_pop_global_cached_block_locked(size_t alloc_size,
                                                uintptr_t stream_key,
                                                size_t requested_size)
{
    auto bin_it = g_global_bins.find(alloc_size);
    if (bin_it == g_global_bins.end()) {
        return nullptr;
    }
    auto &bin = bin_it->second;
    while (!bin.empty()) {
        CachedBlockRef ref = bin.back();
        bin.pop_back();
        auto meta_it = g_blocks.find(ref.ptr);
        if (meta_it == g_blocks.end()) {
            continue;
        }
        BlockMeta &meta = meta_it->second;
        if (meta.generation != ref.generation ||
            meta.state != BlockState::CACHED_GLOBAL ||
            meta.alloc_size != alloc_size) {
            continue;
        }
        mark_block_active_locked(meta, stream_key, requested_size);
        return meta.ptr;
    }
    prune_empty_global_bin_locked(alloc_size);
    return nullptr;
}

static size_t collect_trimmed_global_blocks_locked(
    size_t target_bytes, std::vector<PhysicalBlock> &blocks_to_free)
{
    if (target_bytes == 0) {
        return 0;
    }
    size_t released_bytes = 0;
    for (auto &bucket : g_global_bins) {
        auto &bin = bucket.second;
        while (!bin.empty() && target_bytes > 0) {
            CachedBlockRef ref = bin.back();
            bin.pop_back();
            auto meta_it = g_blocks.find(ref.ptr);
            if (meta_it == g_blocks.end()) {
                continue;
            }
            BlockMeta meta = meta_it->second;
            if (meta.generation != ref.generation ||
                meta.state != BlockState::CACHED_GLOBAL) {
                continue;
            }
            cached_track_sub(meta.alloc_size);
            blocks_to_free.push_back({meta.ptr, meta.is_shmem});
            target_bytes =
                (target_bytes > meta.alloc_size) ? target_bytes - meta.alloc_size : 0;
            released_bytes += meta.alloc_size;
            g_blocks.erase(meta_it);
        }
    }
    erase_empty_bins_locked();
    return released_bytes;
}

static size_t collect_trimmed_local_blocks_locked(
    size_t target_bytes, std::vector<StreamSyncedBlock> &blocks_to_free)
{
    if (target_bytes == 0) {
        return 0;
    }
    size_t released_bytes = 0;
    for (auto &stream_bins : g_stream_local_bins) {
        for (auto &bucket : stream_bins.second) {
            auto &bin = bucket.second;
            while (!bin.empty() && target_bytes > 0) {
                CachedBlockRef ref = bin.back();
                bin.pop_back();
                auto meta_it = g_blocks.find(ref.ptr);
                if (meta_it == g_blocks.end()) {
                    continue;
                }
                BlockMeta meta = meta_it->second;
                if (meta.generation != ref.generation ||
                    meta.state != BlockState::CACHED_LOCAL) {
                    continue;
                }
                cached_track_sub(meta.alloc_size);
                blocks_to_free.push_back(
                    {{meta.ptr, meta.is_shmem}, meta.alloc_stream_key});
                target_bytes =
                    (target_bytes > meta.alloc_size) ? target_bytes - meta.alloc_size : 0;
                released_bytes += meta.alloc_size;
                g_blocks.erase(meta_it);
            }
        }
    }
    erase_empty_bins_locked();
    return released_bytes;
}

static void enforce_cache_limit_locked(
    std::vector<PhysicalBlock> &blocks_to_free)
{
    int64_t cached = g_cached_cur_bytes.load(std::memory_order_relaxed);
    if (cached <= g_cache_limit_bytes) {
        return;
    }
    collect_trimmed_global_blocks_locked(
        static_cast<size_t>(cached - g_cache_limit_bytes), blocks_to_free);
}

static void reclaim_ready_deferred_queue_locked(
    std::deque<DeferredFreeEntry> &queue, int &budget)
{
    size_t scan_limit = queue.size();
    while (budget > 0 && scan_limit > 0 && !queue.empty()) {
        DeferredFreeEntry entry = std::move(queue.front());
        queue.pop_front();
        bool all_ready = true;
        for (aclrtEvent ev : entry.events) {
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
            queue.push_back(std::move(entry));
            --scan_limit;
            continue;
        }

        for (aclrtEvent ev : entry.events) {
            release_event_locked(ev);
        }
        auto meta_it = g_blocks.find(entry.ptr);
        if (meta_it != g_blocks.end()) {
            BlockMeta &meta = meta_it->second;
            if (meta.generation == entry.generation &&
                meta.state == BlockState::DEFERRED) {
                cache_block_global_locked(meta);
            }
        }
        --budget;
        --scan_limit;
    }
}

static void reclaim_ready_deferred_for_stream_locked(uintptr_t stream_key,
                                                     int budget)
{
    auto it = g_deferred_by_stream.find(stream_key);
    if (it == g_deferred_by_stream.end()) {
        return;
    }
    reclaim_ready_deferred_queue_locked(it->second, budget);
    prune_empty_deferred_queue_locked(stream_key);
}

static void reclaim_ready_deferred_global_locked(int budget)
{
    for (auto &queue : g_deferred_by_stream) {
        if (budget <= 0) {
            break;
        }
        reclaim_ready_deferred_queue_locked(queue.second, budget);
    }
    erase_empty_bins_locked();
}

static void drain_releasable_blocks_locked(
    std::vector<PhysicalBlock> &global_blocks_to_free,
    std::vector<StreamSyncedBlock> &local_blocks_to_free,
    std::vector<DeferredFreeEntry> &deferred_entries)
{
    deferred_entries.clear();
    for (auto &queue : g_deferred_by_stream) {
        while (!queue.second.empty()) {
            deferred_entries.push_back(std::move(queue.second.front()));
            queue.second.pop_front();
        }
    }
    g_deferred_by_stream.clear();

    for (auto it = g_blocks.begin(); it != g_blocks.end();) {
        if (it->second.state == BlockState::CACHED_LOCAL) {
            local_blocks_to_free.push_back(
                {{it->second.ptr, it->second.is_shmem}, it->second.alloc_stream_key});
            it = g_blocks.erase(it);
        } else if (it->second.state == BlockState::CACHED_GLOBAL) {
            global_blocks_to_free.push_back({it->second.ptr, it->second.is_shmem});
            it = g_blocks.erase(it);
        } else if (it->second.state == BlockState::DEFERRED) {
            it = g_blocks.erase(it);
        } else {
            ++it;
        }
    }
    g_stream_local_bins.clear();
    g_global_bins.clear();
    g_cached_cur_bytes.store(0, std::memory_order_relaxed);
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
    auto it = g_blocks.find(ptr);
    if (it == g_blocks.end()) {
        return;
    }
    BlockMeta &meta = it->second;
    if (meta.state != BlockState::ACTIVE ||
        stream_key == meta.alloc_stream_key) {
        return;
    }
    meta.extra_streams.insert(stream_key);
}

static void erase_stream_impl(void *ptr, c10_npu::NPUStream stream)
{
    if (ptr == nullptr) {
        return;
    }
    const uintptr_t stream_key = to_stream_key(stream.stream());

    std::lock_guard<std::mutex> lock(g_ptrs_mutex);
    auto it = g_blocks.find(ptr);
    if (it == g_blocks.end()) {
        return;
    }
    BlockMeta &meta = it->second;
    if (meta.state != BlockState::ACTIVE || stream_key == 0) {
        return;
    }
    meta.extra_streams.erase(stream_key);
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
    int64_t cached = g_cached_cur_bytes.load(std::memory_order_relaxed);
    // allocated_bytes: tensor bytes actively held by live allocations.
    stats.allocated_bytes[0].current   = cur;
    stats.allocated_bytes[0].peak      = peak;
    stats.allocated_bytes[0].allocated = peak;  // lifetime high-water
    stats.active_bytes[0].current = cur + cached;
    stats.active_bytes[0].peak =
        std::max<int64_t>(peak, cur + g_cached_peak_bytes.load(std::memory_order_relaxed));
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

static void shmem_reset_impl(bool /*check_error*/)
{
    std::vector<PhysicalBlock> global_blocks_to_free;
    std::vector<StreamSyncedBlock> local_blocks_to_free;
    std::vector<DeferredFreeEntry> deferred_entries;
    std::vector<aclrtEvent> pooled_events;
    {
        std::lock_guard<std::mutex> lock(g_ptrs_mutex);
        reclaim_ready_deferred_global_locked(std::max(1, g_reclaim_budget * 8));
        drain_releasable_blocks_locked(
            global_blocks_to_free, local_blocks_to_free, deferred_entries);
        pooled_events = destroy_event_pool_locked();
        erase_empty_bins_locked();
    }

    sync_streams_and_free_blocks(local_blocks_to_free);
    for (auto &entry : deferred_entries) {
        for (aclrtEvent ev : entry.events) {
            aclrtSynchronizeEvent(ev);
            aclrtDestroyEvent(ev);
        }
        free_storage({entry.ptr, entry.is_shmem});
    }
    for (aclrtEvent ev : pooled_events) {
        aclrtDestroyEvent(ev);
    }
    free_blocks(global_blocks_to_free);
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

    load_runtime_config();

    // Dynamic expansion must be enabled BEFORE aclshmemx_init_attr.
    aclshmem_enable_dynamic_expansion(true);

    // Determine initial pool size.
    //
    // Default: 1 GiB. Override via SHMEM_INITIAL_POOL_SIZE (bytes).
    // A moderate initial pool reduces dynamic expansion churn on the first
    // model/profile/KV-cache allocations while still leaving room for the
    // rest of the runtime memory budget.
    //
    // The value is clamped to ACLSHMEM_MAX_LOCAL_SIZE (40 GiB) which is the hard
    // device limit for symmetric SHMEM memory.
    //
    // NOTE: ACLSHMEM_MAX_LOCAL_SIZE is a macro defined in shmem_common_types.h;
    //       do NOT redeclare it as a local variable (macro expansion would corrupt syntax).
    static const int64_t kShmemMaxLocalSize =
        static_cast<int64_t>(ACLSHMEM_MAX_LOCAL_SIZE);  // 40 GiB hard cap
    constexpr int64_t kLargePageSize = 2LL * 1024 * 1024;  // 2 MiB
    int64_t local_mem_size = 1LL * 1024 * 1024 * 1024;  // 1 GiB default

    const char *env_size = std::getenv("SHMEM_INITIAL_POOL_SIZE");
    if (env_size != nullptr) {
        try {
            int64_t requested = std::stoll(env_size);
            if (requested <= 0) {
                std::cerr << "[shmem_allocator] SHMEM_INITIAL_POOL_SIZE must be > 0, "
                             "using 1 GiB default.\n";
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
                         "using 1 GiB default.\n";
        }
    }
    local_mem_size =
        std::min(round_up(static_cast<size_t>(local_mem_size),
                          static_cast<size_t>(kLargePageSize)),
                 static_cast<size_t>(kShmemMaxLocalSize));

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

    try {
        ensure_shmem_initialized();
    } catch (const std::exception &e) {
        std::cerr << "[shmem_allocator] my_malloc: init failed: " << e.what()
                  << ", falling back to aclrtMalloc\n";
        void *fb = nullptr;
        aclrtMalloc(&fb, static_cast<size_t>(size), ACL_MEM_MALLOC_HUGE_FIRST);
        if (fb != nullptr) {
            std::lock_guard<std::mutex> lock(g_ptrs_mutex);
            g_blocks[fb] = BlockMeta{
                fb,
                static_cast<size_t>(size),
                static_cast<size_t>(size),
                to_stream_key(stream),
                false,
                BlockState::ACTIVE,
                next_generation(),
                {}
            };
        }
        if (fb != nullptr) {
            alloc_track_add(size);
        }
        return fb;
    }

    const size_t requested_size = static_cast<size_t>(size);
    const size_t alloc_size = round_size_class(requested_size);
    const uintptr_t stream_key = to_stream_key(stream);
    std::vector<PhysicalBlock> global_blocks_to_free;
    std::vector<StreamSyncedBlock> local_blocks_to_free;

    {
        std::lock_guard<std::mutex> lock(g_ptrs_mutex);
        reclaim_ready_deferred_for_stream_locked(stream_key, g_reclaim_budget);
        if (void *cached =
                try_pop_local_cached_block_locked(alloc_size, stream_key, requested_size)) {
            alloc_track_add(size);
            return cached;
        }
        if (void *cached =
                try_pop_global_cached_block_locked(alloc_size, stream_key, requested_size)) {
            alloc_track_add(size);
            return cached;
        }
    }

    void *ptr = aclshmem_malloc(alloc_size);
    if (ptr != nullptr) {
        std::lock_guard<std::mutex> lock(g_ptrs_mutex);
        g_blocks[ptr] = BlockMeta{
            ptr,
            alloc_size,
            requested_size,
            stream_key,
            true,
            BlockState::ACTIVE,
            next_generation(),
            {}
        };
        alloc_track_add(size);
        return ptr;
    }

    {
        std::lock_guard<std::mutex> lock(g_ptrs_mutex);
        reclaim_ready_deferred_global_locked(std::max(1, g_reclaim_budget * 4));
        if (void *cached = try_pop_global_cached_block_locked(
                alloc_size, stream_key, requested_size)) {
            alloc_track_add(size);
            return cached;
        }
        collect_trimmed_global_blocks_locked(alloc_size, global_blocks_to_free);
    }
    free_blocks(global_blocks_to_free);

    ptr = aclshmem_malloc(alloc_size);
    if (ptr != nullptr) {
        std::lock_guard<std::mutex> lock(g_ptrs_mutex);
        g_blocks[ptr] = BlockMeta{
            ptr,
            alloc_size,
            requested_size,
            stream_key,
            true,
            BlockState::ACTIVE,
            next_generation(),
            {}
        };
        alloc_track_add(size);
        return ptr;
    }

    {
        std::lock_guard<std::mutex> lock(g_ptrs_mutex);
        collect_trimmed_local_blocks_locked(alloc_size, local_blocks_to_free);
    }
    sync_streams_and_free_blocks(local_blocks_to_free);

    ptr = aclshmem_malloc(alloc_size);
    if (ptr != nullptr) {
        std::lock_guard<std::mutex> lock(g_ptrs_mutex);
        g_blocks[ptr] = BlockMeta{
            ptr,
            alloc_size,
            requested_size,
            stream_key,
            true,
            BlockState::ACTIVE,
            next_generation(),
            {}
        };
        alloc_track_add(size);
        return ptr;
    }

    // SHMEM pool exhausted even after cache reclaim and dynamic expansion.
    std::cerr << "[shmem_allocator] my_malloc: aclshmem_malloc failed for "
                 "size=" << size << ", falling back to aclrtMalloc\n";
    void *fb = nullptr;
    aclrtMalloc(&fb, requested_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (fb != nullptr) {
        std::lock_guard<std::mutex> lock(g_ptrs_mutex);
        g_blocks[fb] = BlockMeta{
            fb,
            requested_size,
            requested_size,
            stream_key,
            false,
            BlockState::ACTIVE,
            next_generation(),
            {}
        };
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

    std::vector<uintptr_t> sync_stream_keys;
    std::vector<aclrtEvent> destroy_events;
    bool cache_locally = false;
    bool defer_free = false;
    bool cache_globally_after_sync = false;
    uint64_t sync_generation = 0;
    size_t tracked_size = static_cast<size_t>(size);
    std::vector<PhysicalBlock> trimmed_global_blocks;
    {
        std::lock_guard<std::mutex> lock(g_ptrs_mutex);
        reclaim_ready_deferred_for_stream_locked(to_stream_key(stream),
                                                 g_reclaim_budget);

        auto it = g_blocks.find(ptr);
        if (it == g_blocks.end()) {
            std::cerr << "[shmem_allocator] my_free: unknown pointer " << ptr
                      << ", ignoring to avoid mismatched backend free.\n";
            return;
        }
        BlockMeta &meta = it->second;
        if (meta.state != BlockState::ACTIVE) {
            std::cerr << "[shmem_allocator] my_free: pointer " << ptr
                      << " is not active, skipping duplicate free.\n";
            return;
        }
        tracked_size = meta.requested_size > 0 ? meta.requested_size : tracked_size;

        if (meta.extra_streams.empty()) {
            cache_block_local_locked(meta);
            cache_locally = true;
        } else {
            std::unordered_set<uintptr_t> stream_keys(meta.extra_streams.begin(),
                                                      meta.extra_streams.end());
            stream_keys.insert(meta.alloc_stream_key);
            sync_stream_keys.assign(stream_keys.begin(), stream_keys.end());

            DeferredFreeEntry entry;
            entry.ptr = ptr;
            entry.is_shmem = meta.is_shmem;
            entry.home_stream_key = meta.alloc_stream_key;
            entry.generation = meta.generation;
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
                              << ", falling back to synchronized global cache\n";
                    destroy_events.push_back(ev);
                    event_record_failed = true;
                    break;
                }
                entry.events.push_back(ev);
            }

            if (!event_record_failed && !entry.events.empty()) {
                meta.state = BlockState::DEFERRED;
                meta.requested_size = 0;
                g_deferred_by_stream[meta.alloc_stream_key].push_back(std::move(entry));
                defer_free = true;
            } else {
                for (aclrtEvent ev : entry.events) {
                    destroy_events.push_back(ev);
                }
                meta.state = BlockState::DEFERRED;
                meta.requested_size = 0;
                meta.extra_streams.clear();
                cache_globally_after_sync = true;
                sync_generation = meta.generation;
            }
        }
        enforce_cache_limit_locked(trimmed_global_blocks);
    }
    free_blocks(trimmed_global_blocks);

    if (cache_locally || defer_free) {
        alloc_track_sub(static_cast<int64_t>(tracked_size));
        return;
    }

    for (uintptr_t stream_key : sync_stream_keys) {
        aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream_key));
    }
    for (aclrtEvent ev : destroy_events) {
        aclrtDestroyEvent(ev);
    }

    if (cache_globally_after_sync) {
        std::vector<PhysicalBlock> post_sync_trimmed_blocks;
        {
            std::lock_guard<std::mutex> lock(g_ptrs_mutex);
            auto it = g_blocks.find(ptr);
            if (it != g_blocks.end() &&
                it->second.generation == sync_generation &&
                it->second.state == BlockState::DEFERRED) {
                cache_block_global_locked(it->second);
                enforce_cache_limit_locked(post_sync_trimmed_blocks);
            }
        }
        free_blocks(post_sync_trimmed_blocks);
    }

    alloc_track_sub(static_cast<int64_t>(tracked_size));
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
        shmem_reset_impl(false);
        {
            std::lock_guard<std::mutex> plock(g_ptrs_mutex);
            for (const auto &pair : g_blocks) {
                if (pair.second.state == BlockState::ACTIVE) {
                    std::cerr << "[shmem_allocator] finalize called with live block "
                              << pair.first << "; clearing bookkeeping only.\n";
                }
            }
            g_blocks.clear();
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
uint64_t shmem_reset_fn_addr(void)
{
    return reinterpret_cast<uint64_t>(&shmem_reset_impl);
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
