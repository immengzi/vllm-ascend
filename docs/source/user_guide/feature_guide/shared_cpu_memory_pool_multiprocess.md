# 跨进程共享 CPU 内存池

## 概述

跨进程共享 CPU 内存池（CrossProcessSharedPool）是 SharedCPUMemoryPool 的多进程扩展，允许**不同进程**中的 NPU Worker 共享卸载的权重内存。

### 与进程内共享池的区别

| 特性 | SharedCPUMemoryPool (进程内) | CrossProcessSharedPool (跨进程) |
|------|------------------------------|----------------------------------|
| **共享范围** | 单个进程内的多个 NPU | 同一主机上的多个进程 |
| **实现机制** | Python 对象 | POSIX 共享内存 + 文件锁 |
| **性能** | 更高（无序列化开销） | 稍低（需要进程同步） |
| **适用场景** | 单进程多线程/多 NPU | 多进程并行推理 |
| **系统要求** | 无特殊要求 | 需要足够的 /dev/shm |

## 工作原理

### 架构

```
进程 A                              进程 B
┌─────────────────────┐            ┌─────────────────────┐
│  LLM Instance       │            │  LLM Instance       │
│  ├─ Weights (NPU 0) │            │  ├─ Weights (NPU 1) │
│  └─ KV Cache        │            │  └─ KV Cache        │
│                     │            │                     │
│  CaMemAllocator     │            │  CaMemAllocator     │
│  ├─ Local Cache     │            │  ├─ Local Cache     │
│  └─ Shared Pool     │◄──────────►│  └─ Shared Pool     │
│                     │            │                     │
│  HybridSharedPool   │            │  HybridSharedPool   │
│  ├─ Local Pool      │            │  ├─ Local Pool      │
│  └─ Cross-Process   │◄──────────►│  └─ Cross-Process   │
└─────────────────────┘            └─────────────────────┘
         │                                  │
         └──────────┬───────────────────────┘
                    ▼
         ┌─────────────────────┐
         │  POSIX Shared Memory│
         │  (/dev/shm)         │
         │                     │
         │  ┌───────────────┐  │
         │  │ Hash Table    │  │
         │  │ Metadata File │  │
         │  └───────────────┘  │
         │                     │
         │  ┌───────────────┐  │
         │  │ Tensor Data 1 │  │
         │  │ Tensor Data 2 │  │
         │  │ ...           │  │
         │  └───────────────┘  │
         └─────────────────────┘
```

### 数据流

#### Sleep（跨进程）

1. **计算 SHA256**：计算 NPU 内存内容的哈希值
2. **获取文件锁**：确保元数据操作的原子性
3. **检查全局元数据**：查看其他进程是否已创建相同哈希的共享内存
4. **复用或创建**：
   - 如存在：打开现有共享内存，增加引用计数
   - 如不存在：创建新的 POSIX 共享内存段
5. **复制数据**：将 NPU 数据复制到共享内存
6. **释放锁**：允许其他进程访问

#### Wake Up（跨进程）

1. **查找共享内存**：根据哈希值定位共享内存段
2. **映射到进程**：打开共享内存（如尚未打开）
3. **复制回 NPU**：将数据从共享内存复制到 NPU
4. **更新访问时间**：用于 LRU 淘汰

## 系统要求

### Linux 环境

```bash
# 1. 检查 /dev/shm 大小
df -h /dev/shm

# 2. 如需要，增加 /dev/shm 大小
# 临时增加（重启失效）
sudo mount -o remount,size=256G /dev/shm

# 永久增加（编辑 /etc/fstab）
echo "tmpfs /dev/shm tmpfs defaults,size=256G 0 0" | sudo tee -a /etc/fstab
```

### 必要的 Python 包

```bash
pip install filelock  # 用于进程间锁
```

### NPU 要求

- 所有进程必须在**同一主机**上
- 每个进程需要独立的 NPU 设备
- 建议设置 `ASCEND_RT_VISIBLE_DEVICES` 隔离设备

## 使用方式

### 基本用法

```python
from vllm import LLM
from vllm_ascend.device_allocator import CrossProcessSharedPool

# 每个进程中初始化共享池
pool = CrossProcessSharedPool.get_instance()

# 正常使用 LLM（会自动使用跨进程共享池）
llm = LLM("Qwen/Qwen2.5-0.5B-Instruct", enable_sleep_mode=True)

# Sleep - 权重卸载到跨进程共享池
llm.sleep(level=1)

# Wake Up - 从共享池恢复（可能复用其他进程的共享内存）
llm.wake_up()
```

### 多进程并行推理

```python
import multiprocessing
from vllm import LLM, SamplingParams
from vllm_ascend.device_allocator import CrossProcessSharedPool

def worker(worker_id: int, device_id: int):
    """工作进程"""
    # 设置当前进程可见的 NPU
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
    
    # 初始化共享池（每个进程都需要调用）
    pool = CrossProcessSharedPool.get_instance()
    
    # 加载模型
    llm = LLM("model_name", enable_sleep_mode=True)
    
    # 推理
    output = llm.generate("Hello")
    
    # Sleep - 与其他进程共享权重
    llm.sleep(level=1)
    
    # 查看统计
    stats = pool.get_stats()
    print(f"Worker {worker_id}: 共享池块数: {stats['current_blocks']}")
    print(f"Worker {worker_id}: 跨进程命中: {stats['total_cross_process_hits']}")

# 启动多个进程
processes = []
for i in range(4):  # 4 个进程，分别使用 NPU 0-3
    p = multiprocessing.Process(target=worker, args=(i, i))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
```

### 混合模式（推荐）

混合模式结合进程内缓存和跨进程共享，提供最佳性能：

```python
from vllm_ascend.device_allocator import HybridSharedPool

# 创建混合共享池
pool = HybridSharedPool(
    enable_cross_process=True,
    shm_name="vllm_shared_pool",
    max_shm_size=256 * 1024 * 1024 * 1024,  # 256GB
)

# 分配内存（自动选择最优路径）
cpu_tensor, hash_val = pool.allocate_from_npu(npu_ptr, size)
# 1. 首先检查进程内缓存（最快）
# 2. 然后检查跨进程共享池
# 3. 最后创建新的共享内存
```

## 配置选项

### CrossProcessSharedPool 参数

```python
CrossProcessSharedPool.get_instance(
    shm_name="vllm_ascend_shared_pool",  # 共享内存名称前缀
    max_shm_size=256 * 1024**3,          # 单个共享内存段最大大小
    cache_dir="/tmp/vllm_ascend_shared_pool",  # 元数据和锁文件目录
)
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `VLLM_ASCEND_CROSS_PROCESS_SHM_NAME` | 共享内存名称前缀 | `vllm_ascend_shared_pool` |
| `VLLM_ASCEND_CROSS_PROCESS_SHM_SIZE` | 最大共享内存大小 (GB) | `256` |
| `VLLM_ASCEND_CROSS_PROCESS_CACHE_DIR` | 缓存目录 | `/tmp/vllm_ascend_shared_pool` |

## 内存节省分析

### 场景：4 进程并行推理 7B 模型

```
模型权重: ~14 GB

无共享池:
  进程 0: 14 GB (NPU 0)
  进程 1: 14 GB (NPU 1)
  进程 2: 14 GB (NPU 2)
  进程 3: 14 GB (NPU 3)
  总计: 56 GB CPU 内存

使用 CrossProcessSharedPool:
  进程 0: 14 GB (创建共享内存)
  进程 1: 0 GB (复用进程 0 的共享内存)
  进程 2: 0 GB (复用进程 0 的共享内存)
  进程 3: 0 GB (复用进程 0 的共享内存)
  总计: 14 GB CPU 内存 (/dev/shm 中)
  
节省: 42 GB (75%)
```

### 性能对比

| 指标 | 进程内共享池 | 跨进程共享池 | 无共享池 |
|------|-------------|-------------|----------|
| **Sleep 延迟** | 低 | 中（文件锁开销） | 低 |
| **Wake Up 延迟** | 低 | 中（共享内存映射） | 低 |
| **内存占用** | 中 | 低（多进程共享） | 高 |
| **跨进程共享** | ❌ | ✅ | ❌ |

## 限制和注意事项

### 1. 同一主机限制

```
CrossProcessSharedPool 只能在同一主机的进程间共享。
不同主机之间无法直接共享内存。

解决方案:
- 使用分布式存储（如 NFS）+ 文件级共享
- 使用 RDMA 网络共享内存
```

### 2. /dev/shm 空间

```python
# 检查 /dev/shm 可用空间
import shutil
shm_stats = shutil.disk_usage("/dev/shm")
print(f"/dev/shm 可用: {shm_stats.free / 1e9:.2f} GB")

# 如果空间不足，可以：
# 1. 增加 /dev/shm 大小
# 2. 使用文件支持的共享内存（见下方高级配置）
```

### 3. 进程崩溃处理

```python
# 如果进程崩溃，引用计数可能不准确
# 需要定期清理或重启后清理

pool = CrossProcessSharedPool.get_instance()

# 清理可回收的内存（ref_count <= 0）
pool.cleanup()

# 强制清理所有内存（谨慎使用）
pool.cleanup(force=True)
```

### 4. 权限问题

```bash
# 确保所有进程对缓存目录有读写权限
chmod 777 /tmp/vllm_ascend_shared_pool

# 确保 /dev/shm 有正确权限
ls -ld /dev/shm
# 应该显示: drwxrwxrwt
```

## 故障排除

### 问题 1: /dev/shm 空间不足

```
错误: OSError: [Errno 28] No space left on device

解决方案:
1. sudo mount -o remount,size=512G /dev/shm
2. 减少模型大小或 batch size
3. 使用文件支持的共享内存（修改 cache_dir 到磁盘）
```

### 问题 2: 文件锁超时

```
错误: filelock.Timeout

解决方案:
1. 检查是否有进程死锁
2. 增加超时时间:
   pool._acquire_lock(timeout=60)
3. 手动清理锁文件:
   rm /tmp/vllm_ascend_shared_pool/shared_pool.lock
```

### 问题 3: 共享内存段残留

```bash
# 列出所有 vllm 相关的共享内存
ls -la /dev/shm/ | grep vllm

# 手动清理（谨慎！）
rm /dev/shm/vllm_ascend_shared_pool_*
```

### 问题 4: 权限被拒绝

```
错误: PermissionError: [Errno 13] Permission denied

解决方案:
1. 检查 cache_dir 权限
2. 确保所有进程使用相同的用户/组
3. 设置 umask: os.umask(0o000)
```

## 最佳实践

### 1. 使用 HybridSharedPool

```python
from vllm_ascend.device_allocator import HybridSharedPool

# 混合模式自动平衡性能和共享能力
pool = HybridSharedPool(enable_cross_process=True)
```

### 2. 设置合适的环境变量

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_ASCEND_CROSS_PROCESS_SHM_SIZE=512  # 512GB
```

### 3. 监控共享效果

```python
# 定期打印统计信息
pool = CrossProcessSharedPool.get_instance()
stats = pool.get_stats()

print(f"跨进程命中: {stats['total_cross_process_hits']}")
print(f"共享节省内存: {stats['total_shared_bytes'] / 1e9:.2f} GB")
```

### 4. 优雅退出

```python
import atexit

def cleanup():
    """进程退出时清理"""
    pool = CrossProcessSharedPool.get_instance()
    pool.cleanup()  # 只清理可回收的内存

atexit.register(cleanup)
```

## 示例：完整的 4 卡并行推理

```python
import multiprocessing
import os
from vllm import LLM, SamplingParams
from vllm_ascend.device_allocator import CrossProcessSharedPool

def inference_worker(
    rank: int,
    world_size: int,
    model_name: str,
    prompts: list,
    output_queue: multiprocessing.Queue
):
    """推理工作进程"""
    # 设置 NPU 设备
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(rank)
    
    # 初始化共享池
    pool = CrossProcessSharedPool.get_instance(
        shm_name="vllm_multiprocess_demo",
        max_shm_size=128 * 1024 * 1024 * 1024,  # 128GB
    )
    
    # 加载模型
    llm = LLM(
        model=model_name,
        enable_sleep_mode=True,
        tensor_parallel_size=1,
    )
    
    # 执行推理
    sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
    outputs = llm.generate(prompts, sampling_params)
    
    # Sleep - 与其他进程共享权重
    llm.sleep(level=1)
    
    # 获取统计
    stats = pool.get_stats()
    
    output_queue.put({
        "rank": rank,
        "outputs": [o.outputs[0].text for o in outputs],
        "shared_blocks": stats["current_blocks"],
        "cross_process_hits": stats["total_cross_process_hits"],
    })

def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    world_size = 4
    prompts_per_worker = 10
    
    # 创建提示
    all_prompts = [f"Prompt {i}: 请介绍自己" for i in range(world_size * prompts_per_worker)]
    
    # 创建输出队列
    output_queue = multiprocessing.Queue()
    
    # 启动工作进程
    processes = []
    for rank in range(world_size):
        start_idx = rank * prompts_per_worker
        end_idx = start_idx + prompts_per_worker
        worker_prompts = all_prompts[start_idx:end_idx]
        
        p = multiprocessing.Process(
            target=inference_worker,
            args=(rank, world_size, model_name, worker_prompts, output_queue)
        )
        p.start()
        processes.append(p)
    
    # 收集结果
    results = []
    for _ in range(world_size):
        results.append(output_queue.get())
    
    # 等待完成
    for p in processes:
        p.join()
    
    # 分析结果
    print("\n=== 结果汇总 ===")
    for r in sorted(results, key=lambda x: x["rank"]):
        print(f"\nRank {r['rank']}:")
        print(f"  生成数量: {len(r['outputs'])}")
        print(f"  共享块数: {r['shared_blocks']}")
        print(f"  跨进程命中: {r['cross_process_hits']}")
    
    # 计算节省
    total_hits = sum(r["cross_process_hits"] for r in results)
    print(f"\n总跨进程共享命中: {total_hits}")
    print("跨进程共享有效！" if total_hits > 0 else "跨进程共享未触发")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
```

## 参考

- [进程内共享池](./shared_cpu_memory_pool.md)
- [Sleep Mode](./sleep_mode.md)
- [Python multiprocessing.shared_memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html)
- [POSIX Shared Memory](https://man7.org/linux/man-pages/man7/shm_overview.7.html)
