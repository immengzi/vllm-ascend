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

Build requirement
-----------------
vllm-ascend must be compiled with ``ENABLE_SHMEM=ON`` (and ``SHMEM_HOME``
pointing to the installed SHMEM library).  Without it the module is absent and
``shmem_available`` stays False, making the allocator silently unavailable.
"""

from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple, Union

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
        with allocator.use_memory_pool(tag="kv_cache"):
            # All NPU allocations here go through the SHMEM pool.
            ...

    Why singleton?
    The C extension keeps global state (initialisation flag, pointer set).
    Creating multiple Python-side instances would not create multiple pools
    and would confuse the lifecycle management.
    """

    instance: Optional["ShmemAllocator"] = None
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "ShmemAllocator":
        if ShmemAllocator.instance is None:
            ShmemAllocator.instance = ShmemAllocator()
        return ShmemAllocator.instance

    # Default segment size requested from PyTorch's caching allocator when
    # SHMEM is the backend.  PyTorch normally uses kLargeBuffer = 20 MiB for
    # allocations in the 1–10 MiB range, which means SHMEM only ever sees
    # coarse 20 MiB requests and its fine-grained pool algorithms have no
    # effect at the tensor level.  We reduce this to 2 MiB so SHMEM operates
    # at a granularity closer to individual KV-cache blocks.
    # Users can override via SHMEM_SEGMENT_SIZE_MB (1–512).
    DEFAULT_SEGMENT_SIZE_MB: int = 2

    def __init__(self) -> None:
        # Set _initialized first so __del__ → finalize() never raises
        # AttributeError even if __init__ raises before completing.
        self._initialized: bool = False
        self.current_tag: str = ShmemAllocator.default_tag
        self._pools: Dict[str, Any] = {}
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
        self._configure_segment_size(conf)

    # ------------------------------------------------------------------ #
    # Segment-size configuration                                           #
    # ------------------------------------------------------------------ #

    def _configure_segment_size(self, existing_conf: str) -> None:
        """Inject *segment_size_mb* into PYTORCH_NPU_ALLOC_CONF.

        PyTorch NPU's caching allocator batches tensor allocations into
        fixed-size *segments* before calling the pluggable allocator backend.
        The default large-pool segment is 20 MiB (``kLargeBuffer``), making
        SHMEM's fine-grained algorithms operate at 20 MiB granularity.

        We lower this to ``DEFAULT_SEGMENT_SIZE_MB`` (2 MiB by default) so
        SHMEM sees allocations closer in size to individual KV-cache blocks,
        enabling its best-fit and coalescing logic to be effective.

        Override with the ``SHMEM_SEGMENT_SIZE_MB`` environment variable.
        """
        import os
        try:
            seg_mb = int(os.environ.get(
                "SHMEM_SEGMENT_SIZE_MB", self.DEFAULT_SEGMENT_SIZE_MB))
            seg_mb = max(1, min(seg_mb, 512))
        except ValueError:
            seg_mb = self.DEFAULT_SEGMENT_SIZE_MB

        # Only inject if segment_size_mb is not already set by the user.
        if "segment_size_mb" not in existing_conf:
            new_entry = f"segment_size_mb:{seg_mb}"
            new_conf = (f"{existing_conf},{new_entry}"
                        if existing_conf else new_entry)
            os.environ["PYTORCH_NPU_ALLOC_CONF"] = new_conf
            logger.info(
                "SHMEM: set PYTORCH_NPU_ALLOC_CONF segment_size_mb=%d MiB "
                "(override with SHMEM_SEGMENT_SIZE_MB env var)", seg_mb)
        else:
            logger.info(
                "SHMEM: segment_size_mb already configured in "
                "PYTORCH_NPU_ALLOC_CONF, skipping auto-configuration")

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
                self._pools.clear()
                logger.info("SHMEM allocator finalized.")
            except Exception as e:
                logger.error("Failed to finalize SHMEM allocator: %s", e)

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
    # NPUPluggableAllocator                                                #
    # ------------------------------------------------------------------ #

    def _get_pluggable_allocator(self) -> torch.npu.memory.NPUPluggableAllocator:
        """Build an NPUPluggableAllocator that uses my_malloc / my_free."""
        if _lib_path is None:
            raise RuntimeError(
                "shmem_allocator .so path is unknown; cannot create allocator"
            )
        return torch.npu.memory.NPUPluggableAllocator(
            _lib_path, "my_malloc", "my_free"
        )

    # ------------------------------------------------------------------ #
    # Context manager                                                      #
    # ------------------------------------------------------------------ #

    @contextmanager
    def use_memory_pool(self, tag: Optional[str] = None):
        """
        Context manager: NPU tensor allocations inside the block are served
        by the SHMEM dynamic pool.

        :param tag: Optional label used to identify this allocation group
            (e.g. ``"kv_cache"``, ``"weights"``).  Currently informational
            only; it does not affect allocation routing.

        **Important – pool lifetime**: The ``MemPool`` / ``NPUPluggableAllocator``
        pair is intentionally kept alive in ``self._pools`` *even after the
        context exits*.  Dropping the pool object while live tensors backed by
        it still exist can cause PyTorch to release the underlying SHMEM
        segments, turning those tensors into dangling pointers and triggering
        OOM or corruption during inference.  Callers that genuinely want to
        reclaim all memory for a tag should call :meth:`release_pool` explicitly
        after all tensors allocated under that tag have been freed.
        """
        if tag is None:
            tag = ShmemAllocator.default_tag

        if not self._initialized and not self.initialize():
            raise RuntimeError("SHMEM allocator failed to initialise")

        old_tag = self.current_tag
        self.current_tag = tag

        alloc = self._get_pluggable_allocator()
        pool = torch.npu.memory.MemPool(alloc._allocator)

        # Keep hard references alive **beyond** the context so that tensors
        # allocated here (weights, KV cache) remain valid during inference.
        # This mirrors the approach in camem.py.
        # See https://github.com/pytorch/pytorch/issues/146431.
        self._pools[tag] = (pool, alloc)
        try:
            with torch.npu.memory.use_mem_pool(pool):
                yield
        finally:
            self.current_tag = old_tag
            # Do NOT pop self._pools[tag] here – the pool must outlive
            # the context so that tensors allocated inside remain valid.

    def release_pool(self, tag: Optional[str] = None) -> None:
        """Drop the pool reference for *tag*, allowing GC to reclaim it.

        Only call this after all tensors allocated under *tag* have been
        freed; otherwise those tensors will reference released memory.
        """
        if tag is None:
            tag = ShmemAllocator.default_tag
        self._pools.pop(tag, None)

    # ------------------------------------------------------------------ #
    # Sleep / wake-up stubs (API compatibility with CaMemAllocator)       #
    # ------------------------------------------------------------------ #

    def sleep(
        self,
        offload_tags: Optional[Union[Tuple[str, ...], str]] = None,
    ) -> None:
        """
        No-op.  SHMEM keeps device memory resident at all times; CPU
        offloading is not supported by this backend.
        """
        logger.debug("ShmemAllocator.sleep() called (no-op for shmem backend)")

    def wake_up(self, tags: Optional[list] = None) -> None:  # type: ignore[type-arg]
        """
        No-op.  Counterpart to :meth:`sleep`.
        """
        logger.debug(
            "ShmemAllocator.wake_up() called (no-op for shmem backend)"
        )

    # ------------------------------------------------------------------ #

    def __del__(self) -> None:
        self.finalize()
