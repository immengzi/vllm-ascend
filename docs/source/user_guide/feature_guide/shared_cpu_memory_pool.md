# Sleep Mode 共享 CPU 内存池

## 概述

**共享 CPU 内存池**是 vLLM-Ascend sleep mode 的高级内存管理功能。它支持多 NPU Worker（跨不同进程或设备）通过基于 SHA256 的内容去重来共享卸载的权重内存。

### 主要优势

| 优势 | 说明 |
|------|------|
| **内存去重** | 相同的权重张量共享同一块 CPU 内存，降低整体内存占用 |
| **多 NPU 共享** | 运行相同模型的不同 NPU 实例可共享卸载的权重 |
| **快速 Sleep/Wake** | 通过复用现有共享块避免重复内存分配 |
| **LRU 淘汰** | 可配置限制下的自动内存管理 |
| **引用计数** | 安全的内存生命周期管理，自动清理 |

## 架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            NPU 0                    NPU 1        NPU N      │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │   LLM 实例        │     │   LLM 实例        │     │   LLM 实例        │    │
│  │   (模型权重)      │     │   (模型权重)      │     │   (模型权重)      │    │
│  └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘    │
│           │ Sleep                   │ Sleep                   │ Sleep      │
│           ▼                         ▼                         ▼            │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │ CaMemAllocator   │     │ CaMemAllocator   │     │ CaMemAllocator   │    │
│  │ (使用共享池)      │     │ (使用共享池)      │     │ (使用共享池)      │    │
│  └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘    │
│           │                        │                        │              │
│           │    SHA256 哈希查找     │                        │              │
│           └───────────────────────▶│◀───────────────────────┘              │
│                                    │                                       │
│                                    ▼                                       │
│                    ┌───────────────────────────────┐                       │
│                    │   SharedCPUMemoryPool         │                       │
│                    │   (进程级单例)                 │                       │
│                    │                               │                       │
│                    │  ┌─────────────────────────┐  │                       │
│                    │  │ 哈希表                   │  │                       │
│                    │  │ hash_abc → cpu_tensor_1 │  │                       │
│                    │  │ hash_def → cpu_tensor_2 │  │                       │
│                    │  │ ...                     │  │                       │
│                    │  └─────────────────────────┘  │                       │
│                    │                               │                       │
│                    │  引用计数：3 (共享)            │                       │
│                    └───────────────────────────────┘                       │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 工作原理

### 1. 使用共享池的 Sleep 模式

当调用 `sleep()` 且 `enable_deduplication=True` 时：

1. **哈希计算**：计算每个 NPU 内存块的 SHA256 哈希
2. **去重检查**：共享池检查是否存在相同哈希的块
3. **共享**：如找到，增加引用计数并共享现有 CPU 张量
4. **新分配**：如未找到，分配新的固定 CPU 内存并存入池中

### 2. 从共享池 Wake Up

当调用 `wake_up()` 时：

1. **内存映射**：NPU 虚拟内存重新映射到物理内存
2. **数据恢复**：数据从共享 CPU 张量复制回 NPU
3. **引用追踪**：NPU 指针注册到共享块

### 3. 引用计数与淘汰

- 每个共享块维护引用计数
- 调用 `release()` 时（张量垃圾回收期间），计数递减
- `ref_count <= 0` 的块可被 LRU 淘汰
- 达到内存限制时触发淘汰

## 使用方式

### 基本用法（自动）

默认情况下，使用 sleep mode 时自动启用共享池：

```python
from vllm import LLM

# 自动使用共享 CPU 内存池
llm = LLM("Qwen/Qwen2.5-0.5B-Instruct", enable_sleep_mode=True)

# Sleep - 权重通过 SHA256 去重卸载到共享池
llm.sleep(level=1)

# Wake up - 从共享池恢复权重
llm.wake_up()
```

### 高级配置

```python
from vllm_ascend.device_allocator import SharedCPUMemoryPool

# 配置自定义内存限制（默认：256GB）
import os
os.environ["VLLM_ASCEND_SHARED_POOL_LIMIT_GB"] = "128"

# 获取池实例并检查统计信息
pool = SharedCPUMemoryPool.get_instance()
stats = pool.get_stats()
print(f"共享节省内存: {stats['total_shared_bytes'] / 1e9:.2f} GB")
print(f"共享命中: {stats['total_sharing_hits']}")
```

### 禁用共享池

如需传统行为（不共享）：

```python
from vllm_ascend.device_allocator import CaMemAllocator

# 创建不使用共享池的分配器
allocator = CaMemAllocator(use_shared_cpu_pool=False)
```

### 禁用去重

使用共享池但跳过 SHA256 计算（更快但不去重）：

```python
llm.sleep(level=1, enable_deduplication=False)
```

## 内存节省分析

### 场景：多 NPU 使用相同模型推理

| 配置 | 无共享池 | 有共享池 | 节省 |
|------|----------|----------|------|
| 4 NPU × 7B 模型 (14GB 权重) | 56 GB CPU 内存 | 14 GB CPU 内存 | **75%** |
| 8 NPU × 70B 模型 (140GB 权重) | 1,120 GB CPU 内存 | 140 GB CPU 内存 | **87.5%** |
| 16 NPU × 405B 模型 (810GB 权重) | 12,960 GB CPU 内存 | 810 GB CPU 内存 | **93.75%** |

### 实际案例：RLHF 训练

PPO/GRPO 场景：
- **Actor 模型**（vLLM 推理）和 **Critic 模型**（训练）通常架构相同
- Critic 训练期间，actor 进入 sleep
- 共享池允许 actor 和 critic 权重共享 CPU 内存
- 两模型在同一节点时节省显著

## API 参考

### SharedCPUMemoryPool

```python
class SharedCPUMemoryPool:
    """进程级单例，用于共享 CPU 内存管理。"""
    
    @staticmethod
    def get_instance() -> "SharedCPUMemoryPool":
        """获取单例实例。"""
        
    def allocate_from_npu(
        self,
        npu_ptr: int,
        size: int,
        compute_hash: bool = True,
        provided_hash: Optional[str] = None
    ) -> Tuple[torch.Tensor, str]:
        """分配 CPU 内存并将 NPU 数据复制到其中。"""
        
    def copy_to_npu(
        self,
        sha256_hash: str,
        npu_ptr: int,
        size: int
    ) -> None:
        """将数据从共享池复制回 NPU。"""
        
    def release(self, sha256_hash: str, npu_ptr: int) -> bool:
        """释放对共享内存块的引用。"""
        
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """获取池统计信息。"""
        
    def log_summary(self) -> None:
        """记录池状态摘要。"""
```

### 修改后的 CaMemAllocator

```python
class CaMemAllocator:
    def __init__(self, use_shared_cpu_pool: bool = True):
        """初始化，可选共享池支持。"""
        
    def sleep(
        self,
        offload_tags: Optional[Union[Tuple[str, ...], str]] = None,
        enable_deduplication: bool = True
    ) -> None:
        """Sleep，可选 SHA256 去重。"""
        
    def get_shared_pool_stats(self) -> Optional[Dict[str, Any]]:
        """获取共享 CPU 内存池统计信息。"""
```

## 配置

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `VLLM_ASCEND_SHARED_POOL_LIMIT_GB` | 共享池最大 CPU 内存 (GB) | 256 |

### 编程配置

```python
from vllm_ascend.device_allocator import SharedCPUMemoryPool

# 自定义内存限制
pool = SharedCPUMemoryPool(memory_limit_bytes=512 * 1024 * 1024 * 1024)  # 512GB
```

## 性能考虑

### SHA256 计算开销

- 计算 SHA256 哈希在 `sleep()` 期间增加开销
- 对大模型，与内存复制时间相比通常可忽略
- 如不需要，可通过 `enable_deduplication=False` 禁用

### 内存访问模式

- 固定（页锁定）CPU 内存支持更快的 DMA 传输
- 共享池使用 `pin_memory=True` 实现最佳 NPU 传输性能

### 线程安全

- 所有 SharedCPUMemoryPool 操作线程安全
- 使用 RLock 支持多 NPU Worker 并发访问

## 限制

1. **进程范围**：共享限于单进程。跨进程共享需要额外机制（如 POSIX 共享内存）

2. **哈希碰撞**：SHA256 碰撞理论上可能但极不可能

3. **内存对齐**：内存块必须对齐以实现高效 DMA 传输

4. **回退**：共享池分配失败时自动回退到传统模式

## 调试

### 启用调试日志

```python
import logging
logging.getLogger("vllm_ascend.device_allocator").setLevel(logging.DEBUG)
```

### 检查池统计

```python
from vllm_ascend.device_allocator import SharedCPUMemoryPool

pool = SharedCPUMemoryPool.get_instance()
pool.log_summary()
```

示例输出：
```
SharedCPUMemoryPool Summary:
  Current blocks: 42
  Current usage: 14.23 GB / 256.00 GB (5.6%)
  Total allocations: 42
  Sharing hits: 126 (saved 42.69 GB)
  Evictions: 0
```

## 从传统 Sleep Mode 迁移

共享池**完全向后兼容**。现有代码无需修改继续工作：

```python
# 传统代码（仍可用）
llm = LLM(model_name, enable_sleep_mode=True)
llm.sleep(level=1)
llm.wake_up()

# 自动受益于共享池
```

选择退出：
```python
# 为特定分配器禁用共享池
allocator = CaMemAllocator(use_shared_cpu_pool=False)
```
