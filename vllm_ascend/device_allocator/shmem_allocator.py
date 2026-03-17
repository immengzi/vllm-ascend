#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SHMEM-based pytorch pluggable allocator with dynamic expansion support.
#
"""
ShmemAllocator – NPUPluggableAllocator backed by the SHMEM dynamic memory pool.

The SHMEM extension module (shmem_allocator.cpython-*.so) serves two roles:

  1. Python extension  – shmem_init / shmem_finalize / get_memory_stats
  2. NPUPluggableAllocator backend – my_malloc / my_free symbols loaded via
     torch.npu.memory.NPUPluggableAllocator(lib_path, 'my_malloc', 'my_free')

This allocator uses ``change_current_allocator`` (not the MemPool API), so
every NPU tensor allocation is routed directly to ``my_malloc`` / ``my_free``
without going through PyTorch's caching layer.  As a result:

* SHMEM sees the **exact tensor size**, not a rounded 20 MiB segment.
* SHMEM's best-fit and coalescing algorithms operate at real allocation
  granularity.
* ``torch.npu.empty_cache()`` is a no-op (no cache exists).

Build requirement
-----------------
vllm-ascend must be compiled with ``ENABLE_SHMEM=ON`` (and ``SHMEM_HOME``
pointing to the installed SHMEM library).  Without it the module is absent and
``shmem_available`` stays False, making the allocator silently unavailable.
"""

from typing import Optional, Tuple

import torch
from vllm.logger import logger


# ---------------------------------------------------------------------------
# Locate the .so that was loaded by the import below.
# We scan /proc/self/maps because the cpython suffix in the filename makes it
# impractical to predict the exact name at build time.

def _find_loaded_library(lib_name: str) -> Optional[str]:
    """
    Return the filesystem path of a loaded shared library whose filename
    contains *lib_name*, or None if not found / not on Linux.
    """
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if lib_name not in line or ".so" not in line:
                    continue
                try:
                    start = line.index("/")
                except ValueError:
                    continue
                path = line[start:].strip()
                filename = path.split("/")[-1]
                # Guard against accidental partial matches (e.g. "libshmem.so"
                # matching a search for "shmem_allocator").
                base = filename.rpartition(".so")[0]
                if base.startswith(lib_name):
                    return path
    except FileNotFoundError:
        pass  # Non-Linux environment
    return None


# ---------------------------------------------------------------------------
# Try to import the native extension.  If the build did not include SHMEM
# support the import will fail and we degrade gracefully.

shmem_available: bool = False
_lib_path: Optional[str] = None

try:
    from vllm_ascend.shmem_allocator import (  # type: ignore[import]
        get_memory_stats,
        is_shmem_initialized,
        shmem_finalize,
        shmem_init,
    )

    # The import already loaded the .so; locate it in the process map.
    _lib_path = _find_loaded_library("shmem_allocator")
    if _lib_path is None:
        raise ImportError(
            "shmem_allocator .so was imported but not found in /proc/self/maps"
        )

    shmem_available = True
    logger.info("SHMEM dynamic allocator loaded from %s", _lib_path)

except ImportError as _e:
    logger.warning(
        "SHMEM allocator not available: %s. "
        "Rebuild vllm-ascend with ENABLE_SHMEM=ON to enable.",
        _e,
    )
    # Keep type-checker happy; these names are never called when
    # shmem_available is False.
    shmem_init = None          # type: ignore[assignment]
    shmem_finalize = None      # type: ignore[assignment]
    is_shmem_initialized = None  # type: ignore[assignment]
    get_memory_stats = None    # type: ignore[assignment]


# ---------------------------------------------------------------------------

class ShmemAllocator:
    """
    Singleton that manages the SHMEM dynamic NPU memory pool for vllm-ascend.

    Usage::

        allocator = ShmemAllocator.get_instance()
        # Install once, before any NPU allocation (e.g. in init_device):
        allocator.install()
        # All subsequent torch.empty() / tensor creations go to SHMEM.

    Why singleton?
    The C extension keeps global state (initialisation flag, pointer set).
    Creating multiple Python-side instances would not create multiple pools
    and would confuse the lifecycle management.

    Why ``change_current_allocator`` instead of MemPool?
    In the MemPool + use_mem_pool pattern, PyTorch's caching allocator
    interposes between tensor requests and the SHMEM backend, requesting memory
    in fixed 20 MiB segments regardless of actual tensor size.  SHMEM's
    fine-grained best-fit and coalescing algorithms therefore have no effect.

    ``change_current_allocator`` replaces the global allocator entirely so
    every ``torch.empty()`` call reaches ``my_malloc`` with the exact tensor
    size, allowing SHMEM to manage memory at real allocation granularity.
    """

    instance: Optional["ShmemAllocator"] = None

    @staticmethod
    def get_instance() -> "ShmemAllocator":
        if ShmemAllocator.instance is None:
            ShmemAllocator.instance = ShmemAllocator()
        return ShmemAllocator.instance

    def __init__(self) -> None:
        # Set _initialized first so __del__ → finalize() never raises
        # AttributeError even if __init__ raises before completing.
        self._initialized: bool = False
        self._installed: bool = False
        self._alloc = None  # strong reference to the installed allocator
        if not shmem_available:
            raise RuntimeError(
                "SHMEM allocator is not available. "
                "Rebuild vllm-ascend with ENABLE_SHMEM=ON."
            )
        import os
        conf = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
        if "expandable_segments:True" in conf:
            raise RuntimeError(
                "expandable_segments:True is not compatible with the SHMEM "
                "memory pool. Please track "
                "https://github.com/pytorch/pytorch/issues/147851 "
                "for the latest updates."
            )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def initialize(self) -> bool:
        """Initialise the SHMEM pool.  Idempotent."""
        if self._initialized:
            return True
        try:
            shmem_init()
            self._initialized = True
            logger.info("SHMEM allocator initialised.")
            return True
        except Exception as e:
            logger.error("Failed to initialise SHMEM allocator: %s", e)
            return False

    def finalize(self) -> None:
        """Release the SHMEM pool.  Safe to call more than once."""
        if self._initialized:
            try:
                shmem_finalize()
                self._initialized = False
                logger.info("SHMEM allocator finalized.")
            except Exception as e:
                logger.error("Failed to finalize SHMEM allocator: %s", e)

    # ------------------------------------------------------------------ #
    # Global allocator installation                                        #
    # ------------------------------------------------------------------ #

    def install(self) -> None:
        """Replace the global NPU allocator with the SHMEM backend.

        Must be called **before** any NPU memory allocation occurs (i.e. before
        ``torch.npu.set_device()`` or any ``torch.empty()`` on NPU).  This is
        a one-time, process-wide change — calling it more than once is a no-op.

        After this call, every ``torch.empty()`` / tensor creation on NPU
        routes directly to ``my_malloc`` / ``my_free`` in the SHMEM allocator
        shared library, bypassing PyTorch's caching layer entirely.
        """
        if self._installed:
            return
        if _lib_path is None:
            raise RuntimeError(
                "shmem_allocator .so path is unknown; cannot install allocator"
            )
        alloc = torch.npu.memory.NPUPluggableAllocator(
            _lib_path, "my_malloc", "my_free"
        )
        torch.npu.memory.change_current_allocator(alloc)
        # Keep a strong reference so the allocator object is not GC'd.
        self._alloc = alloc
        self._installed = True
        logger.info(
            "SHMEM: installed as global NPU allocator (changeCurrentAllocator). "
            "All NPU allocations now go directly to my_malloc/my_free."
        )
        self._register_stats_callbacks(alloc)

    # ------------------------------------------------------------------ #
    # torch_npu memory-stats compatibility                                 #
    # ------------------------------------------------------------------ #

    def _register_stats_callbacks(self, alloc) -> None:
        """Register getDeviceStats / resetPeakStats C++ callbacks.

        NPUPluggableAllocator::getDeviceStats() in torch_npu has a missing
        return statement when no callback is registered.  On aarch64 this
        causes stack-canary corruption ("*** stack smashing detected ***")
        because the function returns a large struct by value via a hidden
        pointer but never writes to it.

        The crash is triggered from *both* the Python path
        (torch_npu.npu.memory_stats()) and directly from within CANN during
        model compilation.  A Python-level monkey-patch cannot intercept the
        C++ call, so we must register a proper C++ function pointer via
        set_get_device_stats_fn / set_reset_peak_status_fn.

        shmem_allocator.cpython-*.so exports two address-getter functions with
        C linkage that return the addresses of the C++ stat callback stubs.
        We load those addresses with ctypes and hand them to the allocator.
        """
        import ctypes

        try:
            so = ctypes.CDLL(_lib_path)

            so.shmem_get_device_stats_fn_addr.restype = ctypes.c_uint64
            so.shmem_reset_peak_stats_fn_addr.restype = ctypes.c_uint64

            get_stats_addr = int(so.shmem_get_device_stats_fn_addr())
            reset_addr = int(so.shmem_reset_peak_stats_fn_addr())

            cpp_alloc = alloc.allocator()
            cpp_alloc.set_get_device_stats_fn(get_stats_addr)
            cpp_alloc.set_reset_peak_status_fn(reset_addr)

            logger.info(
                "SHMEM: registered getDeviceStats / resetPeakStats C++ "
                "callbacks with NPUPluggableAllocator."
            )
        except Exception as e:
            logger.error(
                "SHMEM: failed to register stats callbacks: %s. "
                "Calls to torch_npu.npu.memory_stats() will likely crash.",
                e,
            )

    # ------------------------------------------------------------------ #
    # Memory statistics                                                    #
    # ------------------------------------------------------------------ #

    def get_memory_stats(self) -> Optional[Tuple[int, int, int]]:
        """
        Return ``(total_bytes, used_bytes, available_bytes)`` from the SHMEM
        pool, or *None* on error.
        """
        if not self._initialized and not self.initialize():
            return None
        try:
            return get_memory_stats()
        except Exception as e:
            logger.error("Failed to get SHMEM memory stats: %s", e)
            return None

    def get_current_usage(self) -> int:
        """Return the number of bytes currently allocated from the SHMEM pool."""
        stats = self.get_memory_stats()
        if stats is not None:
            _, used, _ = stats
            return used
        return 0

    # ------------------------------------------------------------------ #
    # Sleep / wake-up stubs (API compatibility with CaMemAllocator)       #
    # ------------------------------------------------------------------ #

    def sleep(self, offload_tags=None) -> None:
        """
        No-op.  SHMEM keeps device memory resident at all times; CPU
        offloading is not supported by this backend.
        """
        logger.debug("ShmemAllocator.sleep() called (no-op for shmem backend)")

    def wake_up(self, tags=None) -> None:
        """
        No-op.  Counterpart to :meth:`sleep`.
        """
        logger.debug(
            "ShmemAllocator.wake_up() called (no-op for shmem backend)"
        )

    # ------------------------------------------------------------------ #

    def __del__(self) -> None:
        self.finalize()


# ---------------------------------------------------------------------------
# Auto-install at module import time.
#
# ``change_current_allocator`` must be called BEFORE the NPU allocator is
# first initialised (i.e. before any NPU memory allocation in this process).
# By the time ``NPUWorker.init_device()`` is reached, several operations in
# ``NPUWorker.__init__()`` — such as ``get_ascend_device_type()``,
# ``_register_atb_extensions()``, and ``super().__init__()`` — have already
# triggered the first NPU allocation and thus initialised the default
# allocator, making a late ``change_current_allocator`` call fail with
# "Can't swap an already initialized allocator".
#
# Installing here — at module import time — guarantees we arrive before any
# of those operations.  The ``install()`` method is idempotent, so the
# ``install()`` call that remains in ``init_device()`` becomes a no-op.

import os as _os

if shmem_available and bool(int(_os.getenv("ENABLE_SHMEM", "0"))):
    try:
        ShmemAllocator.get_instance().install()
    except Exception as _early_install_err:
        logger.warning(
            "SHMEM: early auto-install at import time failed: %s. "
            "Allocator will NOT be active — NPU allocations go to default backend.",
            _early_install_err,
        )
