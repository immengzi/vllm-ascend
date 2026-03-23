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
# 多进程共享 CPU 内存池实现
#
# 本模块提供跨进程共享 CPU 内存的能力，支持：
# 1. POSIX 共享内存（POSIX Shared Memory）
# 2. 内存映射文件（Memory-Mapped Files）
# 3. 基于文件的共享（File-backed Sharing）
#
# 使用场景：
# - 多进程并行推理（每个进程一个 NPU）
# - 多卡训练时的权重共享
# - 模型并行场景下的内存去重
#

from __future__ import annotations

import hashlib
import json
import os
import pickle
import struct
import threading
import time
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from filelock import FileLock, Timeout
from vllm.logger import logger


# 共享内存配置
DEFAULT_SHM_NAME = "vllm_ascend_shared_pool"
DEFAULT_SHM_SIZE = 256 * 1024 * 1024 * 1024  # 256GB
DEFAULT_CACHE_DIR = "/tmp/vllm_ascend_shared_pool"
DEFAULT_LOCK_TIMEOUT = 30  # 秒


@dataclass
class SharedMemoryBlockInfo:
    """
    共享内存块元数据（可序列化）
    
    注意：这是元数据，实际张量数据存储在共享内存中
    """
    sha256_hash: str
    size: int
    shm_name: str  # 共享内存段名称
    ref_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    npu_ptrs: List[int] = field(default_factory=list)  # 使用 list 替代 set 以便序列化
    
    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "sha256_hash": self.sha256_hash,
            "size": self.size,
            "shm_name": self.shm_name,
            "ref_count": self.ref_count,
            "last_access_time": self.last_access_time,
            "npu_ptrs": self.npu_ptrs,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SharedMemoryBlockInfo":
        """从字典反序列化"""
        return cls(**data)


class CrossProcessSharedPool:
    """
    跨进程共享 CPU 内存池
    
    实现方式：
    1. 使用 POSIX 共享内存存储元数据（哈希表）
    2. 使用独立的共享内存段存储实际的张量数据
    3. 使用文件锁实现进程间同步
    
    限制：
    - 只适用于同一主机上的进程
    - 需要 /dev/shm 有足够的空间（Linux）
    - Windows 使用不同的共享内存机制
    
    使用示例：
        # 进程 A
        pool = CrossProcessSharedPool.get_instance()
        cpu_tensor, hash_val = pool.allocate_from_npu(npu_ptr, size)
        
        # 进程 B（同时或之后）
        pool = CrossProcessSharedPool.get_instance()
        cpu_tensor, hash_val = pool.allocate_from_npu(npu_ptr, size)
        # 如果 SHA256 相同，将复用进程 A 创建的共享内存
    """
    
    _instance: Optional["CrossProcessSharedPool"] = None
    _instance_lock: threading.Lock = threading.Lock()
    
    @classmethod
    def get_instance(
        cls,
        shm_name: str = DEFAULT_SHM_NAME,
        max_shm_size: int = DEFAULT_SHM_SIZE,
        cache_dir: str = DEFAULT_CACHE_DIR,
    ) -> "CrossProcessSharedPool":
        """获取跨进程共享池单例"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(
                        shm_name=shm_name,
                        max_shm_size=max_shm_size,
                        cache_dir=cache_dir,
                    )
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（主要用于测试）"""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.cleanup()
                cls._instance = None
    
    def __init__(
        self,
        shm_name: str = DEFAULT_SHM_NAME,
        max_shm_size: int = DEFAULT_SHM_SIZE,
        cache_dir: str = DEFAULT_CACHE_DIR,
    ):
        """
        初始化跨进程共享池
        
        Args:
            shm_name: 共享内存段名称前缀
            max_shm_size: 单个共享内存段的最大大小
            cache_dir: 用于文件锁和元数据备份的目录
        """
        self.shm_name = shm_name
        self.max_shm_size = max_shm_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件锁路径
        self.lock_file = self.cache_dir / "shared_pool.lock"
        self.meta_file = self.cache_dir / "metadata.json"
        
        # 进程内缓存（避免频繁的共享内存访问）
        self._local_cache: Dict[str, shared_memory.SharedMemory] = {}
        self._local_meta_cache: Optional[Dict[str, Any]] = None
        self._local_lock: threading.RLock = threading.RLock()
        
        # 统计信息（进程本地）
        self._stats = {
            "total_allocations": 0,
            "total_sharing_hits": 0,
            "total_shared_bytes": 0,
            "total_cross_process_hits": 0,  # 跨进程命中
        }
        
        logger.info(
            "CrossProcessSharedPool initialized:\n"
            "  shm_name: %s\n"
            "  max_shm_size: %.2f GB\n"
            "  cache_dir: %s",
            shm_name, max_shm_size / (1024 ** 3), cache_dir
        )
    
    def _acquire_lock(self, timeout: float = DEFAULT_LOCK_TIMEOUT) -> FileLock:
        """获取进程间锁"""
        lock = FileLock(self.lock_file, timeout=timeout)
        try:
            lock.acquire()
            return lock
        except Timeout:
            logger.warning("Timeout acquiring shared pool lock")
            raise
    
    def _load_metadata(self) -> Dict[str, SharedMemoryBlockInfo]:
        """从文件加载元数据"""
        if not self.meta_file.exists():
            return {}
        
        try:
            with open(self.meta_file, 'r') as f:
                data = json.load(f)
            return {
                k: SharedMemoryBlockInfo.from_dict(v)
                for k, v in data.items()
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load metadata: %s", e)
            return {}
    
    def _save_metadata(self, metadata: Dict[str, SharedMemoryBlockInfo]) -> None:
        """保存元数据到文件"""
        data = {k: v.to_dict() for k, v in metadata.items()}
        # 使用临时文件 + 原子重命名避免损坏
        temp_file = self.meta_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(data, f)
        temp_file.replace(self.meta_file)
    
    def _get_shm_name(self, hash_value: str) -> str:
        """根据哈希值生成共享内存名称"""
        # POSIX 共享内存名称限制：必须以 / 开头，长度有限制
        # 使用哈希的前 16 个字符作为标识
        return f"/{self.shm_name}_{hash_value[:16]}"
    
    def _create_shared_memory(
        self,
        name: str,
        size: int
    ) -> shared_memory.SharedMemory:
        """创建或打开共享内存段"""
        try:
            # 尝试创建新共享内存
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
            logger.debug("Created new shared memory: %s (size=%d)", name, size)
            return shm
        except FileExistsError:
            # 已存在，打开它
            shm = shared_memory.SharedMemory(name=name)
            logger.debug("Opened existing shared memory: %s", name)
            return shm
    
    def _compute_sha256(self, npu_ptr: int, size: int) -> str:
        """计算 NPU 内存内容的 SHA256 哈希"""
        from acl.rt import memcpy  # type: ignore
        
        # 分配临时 CPU 缓冲区
        temp_buffer = torch.empty(size, dtype=torch.uint8, device='cpu')
        cpu_ptr = temp_buffer.data_ptr()
        
        # 从 NPU 复制到 CPU
        ACL_MEMCPY_DEVICE_TO_HOST = 2
        dest_max = cpu_ptr + size * 2
        memcpy(cpu_ptr, dest_max, npu_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST)
        
        # 计算哈希
        hash_value = hashlib.sha256(temp_buffer.numpy().tobytes()).hexdigest()
        
        # 清理临时缓冲区
        del temp_buffer
        
        return hash_value
    
    def allocate_from_npu(
        self,
        npu_ptr: int,
        size: int,
        compute_hash: bool = True,
        provided_hash: Optional[str] = None,
    ) -> Tuple[torch.Tensor, str]:
        """
        从 NPU 分配共享内存并复制数据
        
        如果其他进程已创建相同哈希的共享内存，将复用它
        
        Returns:
            (cpu_tensor, sha256_hash)
            cpu_tensor 是基于共享内存创建的，可以被多个进程访问
        """
        from acl.rt import memcpy  # type: ignore
        
        # 计算哈希
        sha256_hash = provided_hash
        if sha256_hash is None and compute_hash:
            sha256_hash = self._compute_sha256(npu_ptr, size)
        elif sha256_hash is None:
            sha256_hash = f"no_hash_{npu_ptr}_{time.time()}"
        
        shm_name = self._get_shm_name(sha256_hash)
        
        # 获取进程间锁
        with self._acquire_lock():
            metadata = self._load_metadata()
            
            # 检查是否已存在
            if sha256_hash in metadata:
                block_info = metadata[sha256_hash]
                
                # 验证大小
                if block_info.size != size:
                    logger.warning(
                        "Hash collision: same hash but different sizes: "
                        "existing=%d, new=%d",
                        block_info.size, size
                    )
                    sha256_hash = f"{sha256_hash}_{npu_ptr}_{time.time()}"
                    shm_name = self._get_shm_name(sha256_hash)
                else:
                    # 复用现有共享内存
                    block_info.ref_count += 1
                    block_info.last_access_time = time.time()
                    if npu_ptr not in block_info.npu_ptrs:
                        block_info.npu_ptrs.append(npu_ptr)
                    
                    metadata[sha256_hash] = block_info
                    self._save_metadata(metadata)
                    
                    self._stats["total_sharing_hits"] += 1
                    self._stats["total_cross_process_hits"] += 1
                    self._stats["total_shared_bytes"] += size
                    
                    logger.debug(
                        "Cross-process sharing hit: hash=%s..., ref_count=%d",
                        sha256_hash[:16], block_info.ref_count
                    )
                    
                    # 打开现有共享内存
                    if shm_name not in self._local_cache:
                        self._local_cache[shm_name] = shared_memory.SharedMemory(
                            name=shm_name
                        )
                    shm = self._local_cache[shm_name]
                    
                    # 创建张量视图
                    cpu_tensor = torch.frombuffer(
                        shm.buf[:size],
                        dtype=torch.uint8
                    )
                    
                    return cpu_tensor, sha256_hash
            
            # 创建新的共享内存块
            try:
                shm = self._create_shared_memory(shm_name, size)
                self._local_cache[shm_name] = shm
            except Exception as e:
                logger.error("Failed to create shared memory %s: %s", shm_name, e)
                raise
            
            # 复制数据到共享内存
            cpu_temp = torch.frombuffer(shm.buf[:size], dtype=torch.uint8)
            cpu_ptr = cpu_temp.data_ptr()
            
            ACL_MEMCPY_DEVICE_TO_HOST = 2
            dest_max = cpu_ptr + size * 2
            memcpy(cpu_ptr, dest_max, npu_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST)
            
            # 创建元数据
            block_info = SharedMemoryBlockInfo(
                sha256_hash=sha256_hash,
                size=size,
                shm_name=shm_name,
                ref_count=1,
                npu_ptrs=[npu_ptr],
            )
            
            metadata[sha256_hash] = block_info
            self._save_metadata(metadata)
            
            self._stats["total_allocations"] += 1
            
            logger.debug(
                "Created new cross-process shared block: hash=%s..., size=%d",
                sha256_hash[:16], size
            )
            
            return cpu_temp, sha256_hash
    
    def copy_to_npu(
        self,
        sha256_hash: str,
        npu_ptr: int,
        size: int,
    ) -> None:
        """将数据从共享内存复制回 NPU"""
        from acl.rt import memcpy  # type: ignore
        
        shm_name = self._get_shm_name(sha256_hash)
        
        # 获取或打开共享内存
        with self._local_lock:
            if shm_name not in self._local_cache:
                self._local_cache[shm_name] = shared_memory.SharedMemory(
                    name=shm_name
                )
            shm = self._local_cache[shm_name]
        
        # 复制到 NPU
        cpu_tensor = torch.frombuffer(shm.buf[:size], dtype=torch.uint8)
        cpu_ptr = cpu_tensor.data_ptr()
        
        ACL_MEMCPY_HOST_TO_DEVICE = 1
        dest_max = npu_ptr + size * 2
        memcpy(npu_ptr, dest_max, cpu_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE)
        
        # 更新元数据
        with self._acquire_lock():
            metadata = self._load_metadata()
            if sha256_hash in metadata:
                metadata[sha256_hash].last_access_time = time.time()
                if npu_ptr not in metadata[sha256_hash].npu_ptrs:
                    metadata[sha256_hash].npu_ptrs.append(npu_ptr)
                self._save_metadata(metadata)
    
    def release(self, sha256_hash: str, npu_ptr: int) -> bool:
        """
        释放对共享内存块的引用
        
        当引用计数降到 0 时，会立即清理共享内存资源
        """
        with self._acquire_lock():
            metadata = self._load_metadata()
            
            if sha256_hash not in metadata:
                return False
            
            block_info = metadata[sha256_hash]
            
            # 移除 NPU 指针
            if npu_ptr in block_info.npu_ptrs:
                block_info.npu_ptrs.remove(npu_ptr)
            
            # 减少引用计数
            block_info.ref_count -= 1
            
            should_cleanup = block_info.ref_count <= 0
            
            if should_cleanup:
                # 立即清理共享内存资源
                shm_name = block_info.shm_name
                
                # 关闭本地缓存的引用
                if shm_name in self._local_cache:
                    try:
                        self._local_cache[shm_name].close()
                    except Exception:
                        pass
                    del self._local_cache[shm_name]
                
                # 删除共享内存段
                try:
                    shm = shared_memory.SharedMemory(name=shm_name)
                    shm.close()
                    shm.unlink()
                    logger.debug("Unlinked shared memory: %s", shm_name)
                except FileNotFoundError:
                    pass  # 已被其他进程清理
                except Exception as e:
                    logger.warning("Failed to unlink shared memory %s: %s", shm_name, e)
                
                # 从元数据中移除
                del metadata[sha256_hash]
                
                logger.debug(
                    "Cleaned up block %s... (ref_count=%d)",
                    sha256_hash[:16], block_info.ref_count
                )
            else:
                # 更新元数据
                metadata[sha256_hash] = block_info
            
            self._save_metadata(metadata)
            
            return should_cleanup
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._acquire_lock():
            metadata = self._load_metadata()
            
            stats = self._stats.copy()
            stats["current_blocks"] = len(metadata)
            stats["current_bytes"] = sum(b.size for b in metadata.values())
            stats["total_npu_ptrs"] = sum(
                len(b.npu_ptrs) for b in metadata.values()
            )
            return stats
    
    def cleanup(self, force: bool = False) -> None:
        """
        清理共享内存资源
        
        Args:
            force: 如果为 True，强制清理所有共享内存，即使引用计数 > 0
        """
        with self._acquire_lock():
            metadata = self._load_metadata()
            
            to_remove = []
            for hash_val, block_info in metadata.items():
                if force or block_info.ref_count <= 0:
                    shm_name = block_info.shm_name
                    
                    # 关闭本地缓存的引用
                    if shm_name in self._local_cache:
                        self._local_cache[shm_name].close()
                        del self._local_cache[shm_name]
                    
                    # 尝试删除共享内存
                    try:
                        shm = shared_memory.SharedMemory(name=shm_name)
                        shm.close()
                        shm.unlink()
                        logger.debug("Unlinked shared memory: %s", shm_name)
                    except FileNotFoundError:
                        pass  # 已被其他进程清理
                    
                    to_remove.append(hash_val)
            
            for hash_val in to_remove:
                del metadata[hash_val]
            
            self._save_metadata(metadata)
            
            logger.info("Cleanup complete: removed %d blocks", len(to_remove))
    
    def __del__(self):
        """析构时关闭本地缓存"""
        for shm in self._local_cache.values():
            try:
                shm.close()
            except:
                pass


class HybridSharedPool:
    """
    混合共享池：优先使用进程内共享，必要时使用跨进程共享
    
    策略：
    1. 首先在进程内查找（最快）
    2. 如果未找到，检查跨进程共享池
    3. 如果仍未找到，创建新的共享内存
    
    这提供了性能和资源共享的最佳平衡
    """
    
    def __init__(
        self,
        enable_cross_process: bool = True,
        **cross_process_kwargs
    ):
        """
        初始化混合共享池
        
        Args:
            enable_cross_process: 是否启用跨进程共享
            **cross_process_kwargs: 传递给 CrossProcessSharedPool 的参数
        """
        from vllm_ascend.device_allocator.shared_cpu_pool import (
            SharedCPUMemoryPool,
        )
        
        # 进程内池
        self._local_pool = SharedCPUMemoryPool.get_instance()
        
        # 跨进程池
        self._cross_process_pool: Optional[CrossProcessSharedPool] = None
        if enable_cross_process:
            self._cross_process_pool = CrossProcessSharedPool.get_instance(
                **cross_process_kwargs
            )
        
        self._enable_cross_process = enable_cross_process
    
    def allocate_from_npu(
        self,
        npu_ptr: int,
        size: int,
        compute_hash: bool = True,
        provided_hash: Optional[str] = None,
    ) -> Tuple[torch.Tensor, str]:
        """
        分配共享内存
        
        优先顺序：
        1. 进程内缓存（最快）
        2. 跨进程共享池（与其他进程共享）
        3. 创建新的跨进程共享块
        """
        sha256_hash = provided_hash
        
        if sha256_hash is None and compute_hash:
            # 先在本地计算哈希
            sha256_hash = self._local_pool._compute_sha256(npu_ptr, size)
        elif sha256_hash is None:
            sha256_hash = f"no_hash_{npu_ptr}_{time.time()}"
        
        # 检查进程内缓存（最快路径）
        with self._local_pool._lock:
            if sha256_hash in self._local_pool._hash_to_block:
                block = self._local_pool._hash_to_block[sha256_hash]
                block.ref_count += 1
                block.last_access_time = time.time()
                block.npu_ptrs.add(npu_ptr)
                self._local_pool._npu_ptr_to_hash[npu_ptr] = sha256_hash
                return block.cpu_tensor, sha256_hash
        
        # 如果启用跨进程，检查跨进程池
        if self._enable_cross_process and self._cross_process_pool:
            try:
                # 尝试从跨进程池分配
                cpu_tensor, returned_hash = self._cross_process_pool.allocate_from_npu(
                    npu_ptr=npu_ptr,
                    size=size,
                    compute_hash=False,  # 已计算
                    provided_hash=sha256_hash,
                )
                
                # 同时在本地池注册，加速后续访问
                with self._local_pool._lock:
                    from vllm_ascend.device_allocator.shared_cpu_pool import (
                        SharedMemoryBlock,
                    )
                    block = SharedMemoryBlock(
                        sha256_hash=sha256_hash,
                        size=size,
                        cpu_tensor=cpu_tensor,
                        ref_count=1,
                        npu_ptrs={npu_ptr},
                    )
                    self._local_pool._hash_to_block[sha256_hash] = block
                    self._local_pool._npu_ptr_to_hash[npu_ptr] = sha256_hash
                
                return cpu_tensor, returned_hash
                
            except Exception as e:
                logger.warning("Cross-process allocation failed, using local: %s", e)
        
        # 回退到本地池
        return self._local_pool.allocate_from_npu(
            npu_ptr=npu_ptr,
            size=size,
            compute_hash=False,
            provided_hash=sha256_hash,
        )
    
    def copy_to_npu(self, sha256_hash: str, npu_ptr: int, size: int) -> None:
        """复制数据回 NPU"""
        # 优先使用本地池
        with self._local_pool._lock:
            if sha256_hash in self._local_pool._hash_to_block:
                self._local_pool.copy_to_npu(
                    tags=None  # 使用内部逻辑
                )
                return
        
        # 使用跨进程池
        if self._cross_process_pool:
            self._cross_process_pool.copy_to_npu(sha256_hash, npu_ptr, size)
        else:
            raise ValueError(f"Hash {sha256_hash} not found in any pool")
    
    def release(self, sha256_hash: str, npu_ptr: int) -> bool:
        """释放引用"""
        # 释放本地引用
        with self._local_pool._lock:
            data = self._local_pool._hash_to_block.get(sha256_hash)
            if data:
                if npu_ptr in data.npu_ptrs:
                    data.npu_ptrs.discard(npu_ptr)
                if npu_ptr in self._local_pool._npu_ptr_to_hash:
                    del self._local_pool._npu_ptr_to_hash[npu_ptr]
                data.ref_count -= 1
        
        # 释放跨进程引用
        if self._cross_process_pool:
            return self._cross_process_pool.release(sha256_hash, npu_ptr)
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取合并的统计信息"""
        local_stats = self._local_pool.get_stats()
        
        result = {
            "local": local_stats,
            "cross_process": None,
        }
        
        if self._cross_process_pool:
            result["cross_process"] = self._cross_process_pool.get_stats()
        
        return result


def get_shared_pool(
    cross_process: bool = False,
    **kwargs
) -> Union[SharedCPUMemoryPool, CrossProcessSharedPool, HybridSharedPool]:
    """
    获取共享池实例的工厂函数
    
    Args:
        cross_process: 是否使用跨进程共享
            - False: 仅进程内共享（默认）
            - True: 使用混合模式（进程内 + 跨进程）
        **kwargs: 传递给具体实现的参数
    
    Returns:
        共享池实例
    """
    if cross_process:
        return HybridSharedPool(**kwargs)
    else:
        from vllm_ascend.device_allocator.shared_cpu_pool import (
            SharedCPUMemoryPool,
        )
        return SharedCPUMemoryPool.get_instance()
