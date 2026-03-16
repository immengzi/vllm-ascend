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
# SharedCPUMemoryPool 单元测试
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
    """SharedMemoryBlock 数据类的测试用例。"""
    
    def test_block_creation(self):
        """测试创建 SharedMemoryBlock。"""
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
    """SharedCPUMemoryPool 单例的测试用例。"""
    
    def setUp(self):
        """每个测试前重置单例。"""
        SharedCPUMemoryPool.reset_instance()
    
    def tearDown(self):
        """每个测试后清理。"""
        SharedCPUMemoryPool.reset_instance()
    
    def test_singleton_pattern(self):
        """测试 get_instance 返回相同对象。"""
        pool1 = SharedCPUMemoryPool.get_instance()
        pool2 = SharedCPUMemoryPool.get_instance()
        
        self.assertIs(pool1, pool2)
    
    def test_reset_instance(self):
        """测试 reset_instance 创建新对象。"""
        pool1 = SharedCPUMemoryPool.get_instance()
        pool1._hash_to_block["test"] = MagicMock()
        
        SharedCPUMemoryPool.reset_instance()
        pool2 = SharedCPUMemoryPool.get_instance()
        
        self.assertIsNot(pool1, pool2)
        self.assertEqual(len(pool2._hash_to_block), 0)
    
    def test_initial_stats(self):
        """测试初始统计信息。"""
        pool = SharedCPUMemoryPool.get_instance()
        stats = pool.get_stats()
        
        self.assertEqual(stats["current_blocks"], 0)
        self.assertEqual(stats["current_bytes"], 0)
        self.assertEqual(stats["total_allocations"], 0)
        self.assertEqual(stats["total_sharing_hits"], 0)
    
    def test_memory_limit_default(self):
        """测试默认内存限制。"""
        pool = SharedCPUMemoryPool.get_instance()
        self.assertEqual(pool.memory_limit_bytes, 256 * 1024 * 1024 * 1024)
    
    def test_memory_limit_custom(self):
        """测试自定义内存限制。"""
        SharedCPUMemoryPool.reset_instance()
        pool = SharedCPUMemoryPool(memory_limit_bytes=1024 * 1024 * 1024)
        
        self.assertEqual(pool.memory_limit_bytes, 1024 * 1024 * 1024)


class TestSharedCPUMemoryPoolAllocation(unittest.TestCase):
    """分配功能的测试用例。"""
    
    def setUp(self):
        """每个测试前重置单例。"""
        SharedCPUMemoryPool.reset_instance()
        self.pool = SharedCPUMemoryPool.get_instance()
    
    def tearDown(self):
        """每个测试后清理。"""
        SharedCPUMemoryPool.reset_instance()
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_allocate_with_provided_hash(self, mock_memcpy):
        """测试使用预计算哈希分配。"""
        # Mock memcpy 函数
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
        """测试相同哈希值共享内存。"""
        mock_memcpy.return_value = None
        
        hash_value = "shared_hash_12345"
        
        # 第一次分配
        tensor1, hash1 = self.pool.allocate_from_npu(
            npu_ptr=0x1000,
            size=1024,
            compute_hash=False,
            provided_hash=hash_value
        )
        
        # 第二次分配，相同哈希但不同 NPU 指针
        tensor2, hash2 = self.pool.allocate_from_npu(
            npu_ptr=0x2000,
            size=1024,
            compute_hash=False,
            provided_hash=hash_value
        )
        
        # 应返回相同张量（共享）
        self.assertIs(hash1, hash2)
        self.assertEqual(self.pool._hash_to_block[hash_value].ref_count, 2)
        self.assertEqual(self.pool._stats["total_sharing_hits"], 1)
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_allocate_different_sizes_collision(self, mock_memcpy):
        """测试不同大小但哈希碰撞的情况。"""
        mock_memcpy.return_value = None
        
        hash_value = "collision_hash"
        
        # 第一次分配
        self.pool.allocate_from_npu(
            npu_ptr=0x1000,
            size=1024,
            compute_hash=False,
            provided_hash=hash_value
        )
        
        # 第二次分配，相同哈希但不同大小
        tensor2, hash2 = self.pool.allocate_from_npu(
            npu_ptr=0x2000,
            size=2048,
            compute_hash=False,
            provided_hash=hash_value
        )
        
        # 应创建新的唯一哈希
        self.assertNotEqual(hash2, hash_value)
        self.assertTrue(hash2.startswith(hash_value))
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_allocate_same_npu_ptr_reuse(self, mock_memcpy):
        """测试相同 NPU 指针复用现有块。"""
        mock_memcpy.return_value = None
        
        npu_ptr = 0x1000
        hash_value = "test_hash"
        
        # 第一次分配
        tensor1, hash1 = self.pool.allocate_from_npu(
            npu_ptr=npu_ptr,
            size=1024,
            compute_hash=False,
            provided_hash=hash_value
        )
        
        # 第二次分配，相同 NPU 指针
        tensor2, hash2 = self.pool.allocate_from_npu(
            npu_ptr=npu_ptr,
            size=1024,
            compute_hash=False,
            provided_hash="different_hash"
        )
        
        # 应返回现有块
        self.assertEqual(hash1, hash2)
        self.assertIs(tensor1, tensor2)


class TestSharedCPUMemoryPoolRelease(unittest.TestCase):
    """释放功能的测试用例。"""
    
    def setUp(self):
        """每个测试前重置单例。"""
        SharedCPUMemoryPool.reset_instance()
        self.pool = SharedCPUMemoryPool.get_instance()
    
    def tearDown(self):
        """每个测试后清理。"""
        SharedCPUMemoryPool.reset_instance()
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_release_decrements_ref_count(self, mock_memcpy):
        """测试释放减少引用计数。"""
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
        """测试释放未知哈希。"""
        result = self.pool.release("unknown_hash", 0x1000)
        self.assertFalse(result)
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_release_multiple_refs(self, mock_memcpy):
        """测试多引用释放。"""
        mock_memcpy.return_value = None
        
        hash_value = "shared_hash"
        
        # 分配两次（相同哈希，不同 NPU 指针）
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
        
        # 释放一个
        self.pool.release(hash_value, 0x1000)
        self.assertEqual(self.pool._hash_to_block[hash_value].ref_count, 1)
        
        # 块应仍存在
        self.assertIn(hash_value, self.pool._hash_to_block)


class TestSharedCPUMemoryPoolEviction(unittest.TestCase):
    """内存淘汰的测试用例。"""
    
    def setUp(self):
        """使用较小内存限制重置单例。"""
        SharedCPUMemoryPool.reset_instance()
        self.pool = SharedCPUMemoryPool(memory_limit_bytes=4096)  # 4KB 限制
    
    def tearDown(self):
        """每个测试后清理。"""
        SharedCPUMemoryPool.reset_instance()
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_eviction_on_memory_limit(self, mock_memcpy):
        """测试达到内存限制时淘汰旧块。"""
        mock_memcpy.return_value = None
        
        # 分配第一个块
        self.pool.allocate_from_npu(
            npu_ptr=0x1000,
            size=2048,
            compute_hash=False,
            provided_hash="hash1"
        )
        
        # 释放（使其可被淘汰）
        self.pool.release("hash1", 0x1000)
        
        # 分配第二个块（应触发淘汰）
        self.pool.allocate_from_npu(
            npu_ptr=0x2000,
            size=3072,  # 大于剩余空间
            compute_hash=False,
            provided_hash="hash2"
        )
        
        # 第一个块应被淘汰
        self.assertNotIn("hash1", self.pool._hash_to_block)
        self.assertIn("hash2", self.pool._hash_to_block)
        self.assertEqual(self.pool._stats["total_evictions"], 1)
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_no_eviction_for_referenced_blocks(self, mock_memcpy):
        """测试已引用块不被淘汰。"""
        mock_memcpy.return_value = None
        
        # 分配并保留引用
        self.pool.allocate_from_npu(
            npu_ptr=0x1000,
            size=2048,
            compute_hash=False,
            provided_hash="hash1"
        )
        # 不释放 - 引用计数保持为 1
        
        # 尝试分配更多（应记录警告但不淘汰已引用块）
        self.pool.allocate_from_npu(
            npu_ptr=0x2000,
            size=3072,
            compute_hash=False,
            provided_hash="hash2"
        )
        
        # 两者都应存在（即使超过限制，已引用块保留）
        self.assertIn("hash1", self.pool._hash_to_block)


class TestSharedCPUMemoryPoolThreadSafety(unittest.TestCase):
    """线程安全的测试用例。"""
    
    def setUp(self):
        """每个测试前重置单例。"""
        SharedCPUMemoryPool.reset_instance()
        self.pool = SharedCPUMemoryPool.get_instance()
    
    def tearDown(self):
        """每个测试后清理。"""
        SharedCPUMemoryPool.reset_instance()
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_concurrent_allocations(self, mock_memcpy):
        """测试多线程并发分配。"""
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
        
        # 验证所有分配存在
        stats = self.pool.get_stats()
        self.assertEqual(stats["current_blocks"], num_threads * allocations_per_thread)


class TestSharedCPUMemoryPoolStats(unittest.TestCase):
    """统计信息的测试用例。"""
    
    def setUp(self):
        """每个测试前重置单例。"""
        SharedCPUMemoryPool.reset_instance()
        self.pool = SharedCPUMemoryPool.get_instance()
    
    def tearDown(self):
        """每个测试后清理。"""
        SharedCPUMemoryPool.reset_instance()
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_stats_after_allocation(self, mock_memcpy):
        """测试分配后统计。"""
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
        """测试共享后统计。"""
        mock_memcpy.return_value = None
        
        # 相同哈希分配两次
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
        """测试 clear 重置统计。"""
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
    """块信息获取的测试用例。"""
    
    def setUp(self):
        """每个测试前重置单例。"""
        SharedCPUMemoryPool.reset_instance()
        self.pool = SharedCPUMemoryPool.get_instance()
    
    def tearDown(self):
        """每个测试后清理。"""
        SharedCPUMemoryPool.reset_instance()
    
    @patch("vllm_ascend.device_allocator.shared_cpu_pool.memcpy")
    def test_get_block_info_existing(self, mock_memcpy):
        """测试获取现有块信息。"""
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
        """测试获取不存在的块信息。"""
        info = self.pool.get_block_info("nonexistent_hash")
        self.assertIsNone(info)


if __name__ == "__main__":
    unittest.main()
