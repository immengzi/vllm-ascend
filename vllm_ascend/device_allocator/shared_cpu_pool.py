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
# Shared CPU Memory Pool for Sleep Mode.
# This module provides a process-wide shared memory pool that allows
# multiple NPU workers to share CPU memory for offloaded weights.
# Memory blocks with identical SHA256 hash values are shared.
#

import hashlib
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
from vllm.logger import logger


@dataclass
class SharedMemoryBlock:
    """
    Represents a shared memory block in the CPU memory pool.
    
    Attributes:
        sha256_hash: The SHA256 hash of the memory content
        size: Size in bytes
        cpu_tensor: The shared CPU tensor (pinned memory)
        ref_count: Reference count for memory management
        last_access_time: Last access timestamp for LRU eviction
        npu_ptrs: Set of NPU device pointers that reference this block
    """
    sha256_hash: str
    size: int
    cpu_tensor: torch.Tensor
    ref_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    npu_ptrs: set = field(default_factory=set)


class SharedCPUMemoryPool:
    """
    A process-wide singleton shared CPU memory pool for sleep mode.
    
    This pool allows multiple NPU workers to share CPU memory when offloading
    weights during sleep. Memory blocks with identical SHA256 hash values
    are automatically deduplicated and shared.
    
    Key features:
    1. SHA256-based content deduplication
    2. Reference counting for automatic memory management
    3. LRU eviction policy when memory limit is reached
    4. Thread-safe operations
    
    Usage:
        pool = SharedCPUMemoryPool.get_instance()
        
        # Sleep: offload memory to shared pool
        cpu_tensor = pool.allocate_from_npu(npu_ptr, size_in_bytes)
        
        # Wake up: get shared memory back to NPU
        pool.copy_to_npu(cpu_tensor, npu_ptr, size_in_bytes)
        
        # Release reference when done
        pool.release(sha256_hash, npu_ptr)
    """
    
    _instance: Optional["SharedCPUMemoryPool"] = None
    _instance_lock: threading.Lock = threading.Lock()
    
    # Default memory limit: 256GB (should be enough for most models)
    DEFAULT_MEMORY_LIMIT_BYTES: int = 256 * 1024 * 1024 * 1024
    
    @classmethod
    def get_instance(cls) -> "SharedCPUMemoryPool":
        """Get the singleton instance of SharedCPUMemoryPool."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance. Useful for testing."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.clear()
                cls._instance = None
    
    def __init__(self, memory_limit_bytes: Optional[int] = None):
        """
        Initialize the shared CPU memory pool.
        
        Args:
            memory_limit_bytes: Maximum CPU memory to use for the pool.
                              Defaults to 256GB.
        """
        self.memory_limit_bytes = memory_limit_bytes or self.DEFAULT_MEMORY_LIMIT_BYTES
        
        # Hash -> SharedMemoryBlock mapping for deduplication
        self._hash_to_block: Dict[str, SharedMemoryBlock] = {}
        
        # NPU pointer -> SHA256 hash mapping for tracking
        self._npu_ptr_to_hash: Dict[int, str] = {}
        
        # Statistics
        self._stats = {
            "total_allocated_bytes": 0,
            "total_shared_bytes": 0,  # Saved memory through sharing
            "total_allocations": 0,
            "total_sharing_hits": 0,
            "total_evictions": 0,
        }
        
        # Thread safety
        self._lock: threading.RLock = threading.RLock()
        
        logger.info(
            "SharedCPUMemoryPool initialized with memory limit: %.2f GB",
            self.memory_limit_bytes / (1024 ** 3)
        )
    
    def _compute_sha256(self, npu_ptr: int, size: int) -> str:
        """
        Compute SHA256 hash of NPU memory content.
        
        Note: This copies data to a temporary CPU buffer for hashing.
        For large tensors, consider using a chunk-based approach.
        """
        from acl.rt import memcpy  # type: ignore
        
        # Allocate temporary CPU buffer
        temp_buffer = torch.empty(size, dtype=torch.uint8, device='cpu')
        cpu_ptr = temp_buffer.data_ptr()
        
        # Copy from NPU to CPU
        ACL_MEMCPY_DEVICE_TO_HOST = 2
        dest_max = cpu_ptr + size * 2
        memcpy(cpu_ptr, dest_max, npu_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST)
        
        # Compute hash
        hash_value = hashlib.sha256(temp_buffer.numpy().tobytes()).hexdigest()
        
        # Clean up temporary buffer
        del temp_buffer
        
        return hash_value
    
    def _compute_sha256_from_tensor(self, tensor: torch.Tensor) -> str:
        """Compute SHA256 hash from a CPU tensor."""
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    
    def allocate_from_npu(
        self,
        npu_ptr: int,
        size: int,
        compute_hash: bool = True,
        provided_hash: Optional[str] = None
    ) -> Tuple[torch.Tensor, str]:
        """
        Allocate CPU memory from the shared pool and copy NPU data to it.
        
        If a block with the same SHA256 hash already exists, return the
        shared block and increment its reference count.
        
        Args:
            npu_ptr: NPU device pointer
            size: Size in bytes
            compute_hash: Whether to compute SHA256 hash (default: True)
            provided_hash: Optional pre-computed hash to skip computation
        
        Returns:
            Tuple of (cpu_tensor, sha256_hash)
        """
        from acl.rt import memcpy  # type: ignore
        
        with self._lock:
            # Check if this NPU pointer already has an associated block
            if npu_ptr in self._npu_ptr_to_hash:
                existing_hash = self._npu_ptr_to_hash[npu_ptr]
                block = self._hash_to_block[existing_hash]
                block.ref_count += 1
                block.last_access_time = time.time()
                block.npu_ptrs.add(npu_ptr)
                logger.debug(
                    "Reusing existing shared block for NPU ptr %s, hash: %s...",
                    hex(npu_ptr), existing_hash[:16]
                )
                return block.cpu_tensor, existing_hash
            
            # Compute hash if not provided
            sha256_hash = provided_hash
            if sha256_hash is None and compute_hash:
                sha256_hash = self._compute_sha256(npu_ptr, size)
            elif sha256_hash is None:
                sha256_hash = f"no_hash_{npu_ptr}_{time.time()}"
            
            # Check if we already have a block with this hash
            if sha256_hash in self._hash_to_block:
                block = self._hash_to_block[sha256_hash]
                
                # Verify size matches
                if block.size != size:
                    logger.warning(
                        "Hash collision detected! Same hash but different sizes: "
                        "existing=%d, new=%d. Treating as different blocks.",
                        block.size, size
                    )
                    # Use a unique hash to avoid collision
                    sha256_hash = f"{sha256_hash}_{npu_ptr}_{time.time()}"
                else:
                    # Share existing block
                    block.ref_count += 1
                    block.last_access_time = time.time()
                    block.npu_ptrs.add(npu_ptr)
                    self._npu_ptr_to_hash[npu_ptr] = sha256_hash
                    
                    self._stats["total_sharing_hits"] += 1
                    self._stats["total_shared_bytes"] += size
                    
                    logger.debug(
                        "Sharing block with hash %s... (ref_count=%d, saved %.2f MB)",
                        sha256_hash[:16], block.ref_count, size / (1024 * 1024)
                    )
                    return block.cpu_tensor, sha256_hash
            
            # Need to create a new block
            # Check memory limit and evict if necessary
            self._ensure_memory_available(size)
            
            # Allocate new CPU pinned memory
            cpu_tensor = torch.empty(size, dtype=torch.uint8, device='cpu', pin_memory=True)
            cpu_ptr = cpu_tensor.data_ptr()
            
            # Copy data from NPU to CPU
            ACL_MEMCPY_DEVICE_TO_HOST = 2
            dest_max = cpu_ptr + size * 2
            memcpy(cpu_ptr, dest_max, npu_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST)
            
            # Create shared memory block
            block = SharedMemoryBlock(
                sha256_hash=sha256_hash,
                size=size,
                cpu_tensor=cpu_tensor,
                ref_count=1,
                npu_ptrs={npu_ptr}
            )
            
            self._hash_to_block[sha256_hash] = block
            self._npu_ptr_to_hash[npu_ptr] = sha256_hash
            self._stats["total_allocated_bytes"] += size
            self._stats["total_allocations"] += 1
            
            logger.debug(
                "Created new shared block hash=%s..., size=%.2f MB",
                sha256_hash[:16], size / (1024 * 1024)
            )
            
            return cpu_tensor, sha256_hash
    
    def copy_to_npu(
        self,
        sha256_hash: str,
        npu_ptr: int,
        size: int,
        validate: bool = False
    ) -> None:
        """
        Copy data from shared CPU memory back to NPU.
        
        Args:
            sha256_hash: The SHA256 hash of the memory block
            npu_ptr: Target NPU device pointer
            size: Size in bytes
            validate: If True, verify the hash after copying
        """
        from acl.rt import memcpy  # type: ignore
        
        with self._lock:
            if sha256_hash not in self._hash_to_block:
                raise ValueError(f"Shared memory block with hash {sha256_hash} not found")
            
            block = self._hash_to_block[sha256_hash]
            
            if block.size != size:
                raise ValueError(
                    f"Size mismatch: block size={block.size}, requested={size}"
                )
            
            # Copy from CPU to NPU
            cpu_ptr = block.cpu_tensor.data_ptr()
            ACL_MEMCPY_HOST_TO_DEVICE = 1
            dest_max = npu_ptr + size * 2
            memcpy(npu_ptr, dest_max, cpu_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE)
            
            # Track this NPU pointer
            block.npu_ptrs.add(npu_ptr)
            self._npu_ptr_to_hash[npu_ptr] = sha256_hash
            
            block.last_access_time = time.time()
            
            logger.debug(
                "Copied shared block hash=%s... to NPU ptr %s",
                sha256_hash[:16], hex(npu_ptr)
            )
    
    def release(self, sha256_hash: str, npu_ptr: int) -> bool:
        """
        Release a reference to a shared memory block.
        
        When reference count reaches zero, the block becomes eligible
        for eviction (but is not immediately freed).
        
        Args:
            sha256_hash: The SHA256 hash of the memory block
            npu_ptr: The NPU pointer that was using this block
        
        Returns:
            True if the block was removed (ref_count reached 0)
        """
        with self._lock:
            if sha256_hash not in self._hash_to_block:
                return False
            
            block = self._hash_to_block[sha256_hash]
            
            # Remove NPU pointer tracking
            if npu_ptr in block.npu_ptrs:
                block.npu_ptrs.discard(npu_ptr)
            
            if npu_ptr in self._npu_ptr_to_hash:
                del self._npu_ptr_to_hash[npu_ptr]
            
            # Decrement reference count
            block.ref_count -= 1
            
            logger.debug(
                "Released shared block hash=%s... (ref_count=%d)",
                sha256_hash[:16], block.ref_count
            )
            
            if block.ref_count <= 0:
                # Don't immediately delete - let LRU eviction handle it
                # This allows quick re-allocation if needed
                logger.debug(
                    "Block hash=%s... now eligible for eviction",
                    sha256_hash[:16]
                )
                return True
            
            return False
    
    def _ensure_memory_available(self, required_bytes: int) -> None:
        """
        Ensure enough memory is available by evicting LRU blocks if necessary.
        
        Only blocks with ref_count <= 0 are eligible for eviction.
        """
        current_usage = sum(b.size for b in self._hash_to_block.values())
        available = self.memory_limit_bytes - current_usage
        
        if available >= required_bytes:
            return
        
        # Need to evict some blocks
        bytes_needed = required_bytes - available
        bytes_evicted = 0
        
        # Sort by last access time (LRU) - only consider unreferenced blocks
        evictable_blocks = [
            (block.last_access_time, h, block)
            for h, block in self._hash_to_block.items()
            if block.ref_count <= 0
        ]
        evictable_blocks.sort()  # Oldest first
        
        for _, hash_key, block in evictable_blocks:
            if bytes_evicted >= bytes_needed:
                break
            
            # Remove block
            del self._hash_to_block[hash_key]
            
            # Clean up NPU pointer mappings
            for npu_ptr in list(block.npu_ptrs):
                if npu_ptr in self._npu_ptr_to_hash:
                    del self._npu_ptr_to_hash[npu_ptr]
            
            bytes_evicted += block.size
            self._stats["total_evictions"] += 1
            
            logger.debug(
                "Evicted block hash=%s..., size=%.2f MB",
                hash_key[:16], block.size / (1024 * 1024)
            )
        
        if bytes_evicted < bytes_needed:
            logger.warning(
                "Could not free enough memory. Needed %.2f MB, freed %.2f MB. "
                "Consider increasing memory limit.",
                bytes_needed / (1024 * 1024),
                bytes_evicted / (1024 * 1024)
            )
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get statistics about the shared memory pool."""
        with self._lock:
            stats = self._stats.copy()
            stats["current_blocks"] = len(self._hash_to_block)
            stats["current_bytes"] = sum(b.size for b in self._hash_to_block.values())
            stats["current_unique_npu_ptrs"] = len(self._npu_ptr_to_hash)
            stats["memory_limit_bytes"] = self.memory_limit_bytes
            stats["memory_utilization"] = stats["current_bytes"] / self.memory_limit_bytes
            return stats
    
    def clear(self) -> None:
        """Clear all shared memory blocks. Use with caution!"""
        with self._lock:
            self._hash_to_block.clear()
            self._npu_ptr_to_hash.clear()
            self._stats = {
                "total_allocated_bytes": 0,
                "total_shared_bytes": 0,
                "total_allocations": 0,
                "total_sharing_hits": 0,
                "total_evictions": 0,
            }
            logger.info("SharedCPUMemoryPool cleared")
    
    def get_block_info(self, sha256_hash: str) -> Optional[Dict]:
        """Get information about a specific memory block."""
        with self._lock:
            if sha256_hash not in self._hash_to_block:
                return None
            
            block = self._hash_to_block[sha256_hash]
            return {
                "sha256_hash": block.sha256_hash,
                "size": block.size,
                "ref_count": block.ref_count,
                "last_access_time": block.last_access_time,
                "npu_ptrs": list(block.npu_ptrs),
            }
    
    def log_summary(self) -> None:
        """Log a summary of the shared memory pool status."""
        stats = self.get_stats()
        logger.info(
            "SharedCPUMemoryPool Summary:\n"
            "  Current blocks: %d\n"
            "  Current usage: %.2f GB / %.2f GB (%.1f%%)\n"
            "  Total allocations: %d\n"
            "  Sharing hits: %d (saved %.2f GB)\n"
            "  Evictions: %d",
            stats["current_blocks"],
            stats["current_bytes"] / (1024 ** 3),
            stats["memory_limit_bytes"] / (1024 ** 3),
            stats["memory_utilization"] * 100,
            stats["total_allocations"],
            stats["total_sharing_hits"],
            stats["total_shared_bytes"] / (1024 ** 3),
            stats["total_evictions"]
        )
