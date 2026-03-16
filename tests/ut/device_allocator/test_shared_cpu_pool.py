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
# Unit tests for SharedCPUMemoryPool
#

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.device_allocator.shared_cpu_pool import (
    SharedCPUMemoryPool,
    SharedMemoryBlock,
)


class TestSharedMemoryBlock(unittest.TestCase):
    """Test cases for SharedMemoryBlock dataclass."""
    
    def test_block_creation(self):
        """Test creating a SharedMemoryBlock."""
        tensor = torch.empty(1024, dtype=torch.uint8)
        block = SharedMemoryBlock(
            sha256_hash="abc123",
            size=1024,
            cpu_tensor=tensor,
            ref_count=1
        )
        
        self.assertEqual(block.sha256_hash, "abc123")
        self.assertEqual(block.size, 1024)
        self.assertEqual(block.ref_count, 1)
        self.assertEqual(block.npu_ptrs, set())


class TestSharedCPUMemoryPool(unittest.TestCase):
    """Test cases for SharedCPUMemoryPool singleton."""
    
    def setUp(self):
        """Reset singleton before each test."""
        SharedCPUMemoryPool.reset_instance()
    
    def tearDown(self):
        """Clean up after each test."""
        SharedCPUMemoryPool.reset_instance()
    
    def test_singleton_pattern(self):
        """Test that get_instance returns the same object."""
        pool1 = SharedCPUMemoryPool.get_instance()
        pool2 = SharedCPUMemoryPool.get_instance()
        
        self.assertIs(pool1, pool2)
    
    def test_reset_instance(self):
        """Test that reset_instance creates a new object."""
        pool1 = SharedCPUMemoryPool.get_instance()
        pool1._hash_to_block["test"] = MagicMock()
        
        SharedCPUMemoryPool.reset_instance()
        pool2 = SharedCPUMemoryPool.get_instance()
        
        self.assertIsNot(pool1, pool2)
        self.assertEqual(len(pool2._hash_to_block), 0)
    
    def test_initial_stats(self):
        """Test initial statistics."""
        pool = SharedCPUMemoryPool.get_instance()
        stats = pool.get_stats()
        
        self.assertEqual(stats["current_blocks"], 0)
        self.assertEqual(stats["current_bytes"], 0)
        self.assertEqual(stats["total_allocations"], 0)
        self.assertEqual(stats["total_sharing_hits"], 0)
    
    def test_memory_limit_default(self):
        """Test default memory limit."""
        pool = SharedCPUMemoryPool.get_instance()
        self.assertEqual(pool.memory_limit_bytes, 256 * 1024 * 1024 * 1024)
    
    def test_memory_limit_custom(self):
        """Test custom memory limit."""
        SharedCPUMemoryPool.reset_instance()
        pool = SharedCPUMemoryPool(memory_limit_bytes=1024 * 1024 * 1024)
        
        self.assertEqual(pool.memory_limit_bytes, 1024 * 1024 * 1024)


class TestSharedCPUMemoryPoolAllocation(unittest.TestCase):
    """Test cases for allocation functionality."""
    
    def setUp(self):
        """Reset singleton before each test."""
        SharedCPUMemoryPool.reset_instance()
        self.pool = SharedCPUMemoryPool.get_instance()
    
    def tearDown(self):
        """Clean up after each test."""
        SharedCPUMemoryPool.reset_instance()
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_allocate_with_provided_hash(self, mock_memcpy):
        """Test allocation with pre-computed hash."""
        # Mock the memcpy function
        mock_memcpy.return_value = None
        
        npu_ptr = 0x1000
        size = 1024
        hash_value = "test_hash_12345"
        
        cpu_tensor, returned_hash = self.pool.allocate_from_npu(
            npu_ptr=npu_ptr,
            size=size,
            compute_hash=False,
            provided_hash=hash_value
        )
        
        self.assertEqual(returned_hash, hash_value)
        self.assertEqual(cpu_tensor.shape[0], size)
        self.assertEqual(self.pool._hash_to_block[hash_value].ref_count, 1)
        self.assertIn(npu_ptr, self.pool._npu_ptr_to_hash)
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_allocate_same_hash_sharing(self, mock_memcpy):
        """Test that same hash values share memory."""
        mock_memcpy.return_value = None
        
        hash_value = "shared_hash_12345"
        
        # First allocation
        tensor1, hash1 = self.pool.allocate_from_npu(
            npu_ptr=0x1000,
            size=1024,
            compute_hash=False,
            provided_hash=hash_value
        )
        
        # Second allocation with same hash but different NPU pointer
        tensor2, hash2 = self.pool.allocate_from_npu(
            npu_ptr=0x2000,
            size=1024,
            compute_hash=False,
            provided_hash=hash_value
        )
        
        # Should return the same tensor (shared)
        self.assertIs(hash1, hash2)
        self.assertEqual(self.pool._hash_to_block[hash_value].ref_count, 2)
        self.assertEqual(self.pool._stats["total_sharing_hits"], 1)
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_allocate_different_sizes_collision(self, mock_memcpy):
        """Test hash collision with different sizes."""
        mock_memcpy.return_value = None
        
        hash_value = "collision_hash"
        
        # First allocation
        self.pool.allocate_from_npu(
            npu_ptr=0x1000,
            size=1024,
            compute_hash=False,
            provided_hash=hash_value
        )
        
        # Second allocation with same hash but different size
        tensor2, hash2 = self.pool.allocate_from_npu(
            npu_ptr=0x2000,
            size=2048,
            compute_hash=False,
            provided_hash=hash_value
        )
        
        # Should create a new unique hash
        self.assertNotEqual(hash2, hash_value)
        self.assertTrue(hash2.startswith(hash_value))
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_allocate_same_npu_ptr_reuse(self, mock_memcpy):
        """Test that same NPU pointer reuses existing block."""
        mock_memcpy.return_value = None
        
        npu_ptr = 0x1000
        hash_value = "test_hash"
        
        # First allocation
        tensor1, hash1 = self.pool.allocate_from_npu(
            npu_ptr=npu_ptr,
            size=1024,
            compute_hash=False,
            provided_hash=hash_value
        )
        
        # Second allocation with same NPU pointer
        tensor2, hash2 = self.pool.allocate_from_npu(
            npu_ptr=npu_ptr,
            size=1024,
            compute_hash=False,
            provided_hash="different_hash"
        )
        
        # Should return the existing block
        self.assertEqual(hash1, hash2)
        self.assertIs(tensor1, tensor2)


class TestSharedCPUMemoryPoolRelease(unittest.TestCase):
    """Test cases for release functionality."""
    
    def setUp(self):
        """Reset singleton before each test."""
        SharedCPUMemoryPool.reset_instance()
        self.pool = SharedCPUMemoryPool.get_instance()
    
    def tearDown(self):
        """Clean up after each test."""
        SharedCPUMemoryPool.reset_instance()
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_release_decrements_ref_count(self, mock_memcpy):
        """Test that release decrements reference count."""
        mock_memcpy.return_value = None
        
        hash_value = "test_hash"
        npu_ptr = 0x1000
        
        self.pool.allocate_from_npu(
            npu_ptr=npu_ptr,
            size=1024,
            compute_hash=False,
            provided_hash=hash_value
        )
        
        self.assertEqual(self.pool._hash_to_block[hash_value].ref_count, 1)
        
        self.pool.release(hash_value, npu_ptr)
        
        self.assertEqual(self.pool._hash_to_block[hash_value].ref_count, 0)
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_release_unknown_hash(self, mock_memcpy):
        """Test releasing unknown hash."""
        result = self.pool.release("unknown_hash", 0x1000)
        self.assertFalse(result)
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_release_multiple_refs(self, mock_memcpy):
        """Test release with multiple references."""
        mock_memcpy.return_value = None
        
        hash_value = "shared_hash"
        
        # Allocate twice (same hash, different NPU ptrs)
        self.pool.allocate_from_npu(
            npu_ptr=0x1000,
            size=1024,
            compute_hash=False,
            provided_hash=hash_value
        )
        self.pool.allocate_from_npu(
            npu_ptr=0x2000,
            size=1024,
            compute_hash=False,
            provided_hash=hash_value
        )
        
        self.assertEqual(self.pool._hash_to_block[hash_value].ref_count, 2)
        
        # Release one
        self.pool.release(hash_value, 0x1000)
        self.assertEqual(self.pool._hash_to_block[hash_value].ref_count, 1)
        
        # Block should still exist
        self.assertIn(hash_value, self.pool._hash_to_block)


class TestSharedCPUMemoryPoolEviction(unittest.TestCase):
    """Test cases for memory eviction."""
    
    def setUp(self):
        """Reset singleton with small memory limit."""
        SharedCPUMemoryPool.reset_instance()
        self.pool = SharedCPUMemoryPool(memory_limit_bytes=4096)  # 4KB limit
    
    def tearDown(self):
        """Clean up after each test."""
        SharedCPUMemoryPool.reset_instance()
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_eviction_on_memory_limit(self, mock_memcpy):
        """Test that old blocks are evicted when memory limit is reached."""
        mock_memcpy.return_value = None
        
        # Allocate first block
        self.pool.allocate_from_npu(
            npu_ptr=0x1000,
            size=2048,
            compute_hash=False,
            provided_hash="hash1"
        )
        
        # Release it (make it eligible for eviction)
        self.pool.release("hash1", 0x1000)
        
        # Allocate second block (should trigger eviction)
        self.pool.allocate_from_npu(
            npu_ptr=0x2000,
            size=3072,  # Larger than remaining space
            compute_hash=False,
            provided_hash="hash2"
        )
        
        # First block should be evicted
        self.assertNotIn("hash1", self.pool._hash_to_block)
        self.assertIn("hash2", self.pool._hash_to_block)
        self.assertEqual(self.pool._stats["total_evictions"], 1)
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_no_eviction_for_referenced_blocks(self, mock_memcpy):
        """Test that referenced blocks are not evicted."""
        mock_memcpy.return_value = None
        
        # Allocate and keep reference
        self.pool.allocate_from_npu(
            npu_ptr=0x1000,
            size=2048,
            compute_hash=False,
            provided_hash="hash1"
        )
        # Don't release - ref_count stays at 1
        
        # Try to allocate more (should log warning but not evict referenced block)
        self.pool.allocate_from_npu(
            npu_ptr=0x2000,
            size=3072,
            compute_hash=False,
            provided_hash="hash2"
        )
        
        # Both should exist (even if over limit, referenced blocks stay)
        self.assertIn("hash1", self.pool._hash_to_block)


class TestSharedCPUMemoryPoolThreadSafety(unittest.TestCase):
    """Test cases for thread safety."""
    
    def setUp(self):
        """Reset singleton before each test."""
        SharedCPUMemoryPool.reset_instance()
        self.pool = SharedCPUMemoryPool.get_instance()
    
    def tearDown(self):
        """Clean up after each test."""
        SharedCPUMemoryPool.reset_instance()
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_concurrent_allocations(self, mock_memcpy):
        """Test concurrent allocations from multiple threads."""
        mock_memcpy.return_value = None
        
        num_threads = 10
        allocations_per_thread = 10
        
        def allocate_worker(thread_id):
            for i in range(allocations_per_thread):
                npu_ptr = thread_id * 1000 + i
                self.pool.allocate_from_npu(
                    npu_ptr=npu_ptr,
                    size=1024,
                    compute_hash=False,
                    provided_hash=f"hash_{thread_id}_{i}"
                )
        
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=allocate_worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify all allocations exist
        stats = self.pool.get_stats()
        self.assertEqual(stats["current_blocks"], num_threads * allocations_per_thread)


class TestSharedCPUMemoryPoolStats(unittest.TestCase):
    """Test cases for statistics."""
    
    def setUp(self):
        """Reset singleton before each test."""
        SharedCPUMemoryPool.reset_instance()
        self.pool = SharedCPUMemoryPool.get_instance()
    
    def tearDown(self):
        """Clean up after each test."""
        SharedCPUMemoryPool.reset_instance()
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_stats_after_allocation(self, mock_memcpy):
        """Test statistics after allocation."""
        mock_memcpy.return_value = None
        
        self.pool.allocate_from_npu(
            npu_ptr=0x1000,
            size=1024,
            compute_hash=False,
            provided_hash="hash1"
        )
        
        stats = self.pool.get_stats()
        self.assertEqual(stats["total_allocations"], 1)
        self.assertEqual(stats["current_bytes"], 1024)
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_stats_after_sharing(self, mock_memcpy):
        """Test statistics after memory sharing."""
        mock_memcpy.return_value = None
        
        # Allocate same hash twice
        self.pool.allocate_from_npu(
            npu_ptr=0x1000,
            size=1024,
            compute_hash=False,
            provided_hash="shared_hash"
        )
        self.pool.allocate_from_npu(
            npu_ptr=0x2000,
            size=1024,
            compute_hash=False,
            provided_hash="shared_hash"
        )
        
        stats = self.pool.get_stats()
        self.assertEqual(stats["total_allocations"], 1)
        self.assertEqual(stats["total_sharing_hits"], 1)
        self.assertEqual(stats["total_shared_bytes"], 1024)
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_clear_resets_stats(self, mock_memcpy):
        """Test that clear resets statistics."""
        mock_memcpy.return_value = None
        
        self.pool.allocate_from_npu(
            npu_ptr=0x1000,
            size=1024,
            compute_hash=False,
            provided_hash="hash1"
        )
        
        self.pool.clear()
        
        stats = self.pool.get_stats()
        self.assertEqual(stats["current_blocks"], 0)
        self.assertEqual(stats["total_allocations"], 0)


class TestSharedCPUMemoryPoolBlockInfo(unittest.TestCase):
    """Test cases for block info retrieval."""
    
    def setUp(self):
        """Reset singleton before each test."""
        SharedCPUMemoryPool.reset_instance()
        self.pool = SharedCPUMemoryPool.get_instance()
    
    def tearDown(self):
        """Clean up after each test."""
        SharedCPUMemoryPool.reset_instance()
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_get_block_info_existing(self, mock_memcpy):
        """Test getting info for existing block."""
        mock_memcpy.return_value = None
        
        hash_value = "test_hash"
        npu_ptr = 0x1000
        
        self.pool.allocate_from_npu(
            npu_ptr=npu_ptr,
            size=1024,
            compute_hash=False,
            provided_hash=hash_value
        )
        
        info = self.pool.get_block_info(hash_value)
        
        self.assertIsNotNone(info)
        self.assertEqual(info["sha256_hash"], hash_value)
        self.assertEqual(info["size"], 1024)
        self.assertEqual(info["ref_count"], 1)
        self.assertIn(npu_ptr, info["npu_ptrs"])
    
    def test_get_block_info_nonexistent(self):
        """Test getting info for non-existent block."""
        info = self.pool.get_block_info("nonexistent_hash")
        self.assertIsNone(info)


if __name__ == "__main__":
    unittest.main()
