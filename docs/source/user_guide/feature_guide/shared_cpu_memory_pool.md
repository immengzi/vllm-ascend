# Shared CPU Memory Pool for Sleep Mode

## Overview

The **Shared CPU Memory Pool** is an advanced memory management feature for vLLM-Ascend's sleep mode. It enables multiple NPU workers (across different processes or devices) to share offloaded weight memory through SHA256-based content deduplication.

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Memory Deduplication** | Identical weight tensors share the same CPU memory, reducing overall memory footprint |
| **Multi-NPU Sharing** | Different NPU instances running the same model can share offloaded weights |
| **Fast Sleep/Wake** | Avoids redundant memory allocation by reusing existing shared blocks |
| **LRU Eviction** | Automatic memory management with configurable limits |
| **Reference Counting** | Safe memory lifecycle management with automatic cleanup |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            NPU 0                    NPU 1        NPU N      │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │   LLM Instance   │     │   LLM Instance   │     │   LLM Instance   │    │
│  │   (Model Weights)│     │   (Model Weights)│     │   (Model Weights)│    │
│  └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘    │
│           │ Sleep                     │ Sleep                    │ Sleep   │
│           ▼                           ▼                        ▼           │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │ CaMemAllocator   │     │ CaMemAllocator   │     │ CaMemAllocator   │    │
│  │ (with shared pool)│     │ (with shared pool)│    │ (with shared pool)│   │
│  └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘    │
│           │                        │                        │              │
│           │    SHA256 Hash Lookup  │                        │              │
│           └───────────────────────▶│◀───────────────────────┘              │
│                                    │                                       │
│                                    ▼                                       │
│                    ┌───────────────────────────────┐                       │
│                    │   SharedCPUMemoryPool         │                       │
│                    │   (Process-wide Singleton)    │                       │
│                    │                               │                       │
│                    │  ┌─────────────────────────┐  │                       │
│                    │  │ Hash Table              │  │                       │
│                    │  │ hash_abc → cpu_tensor_1 │  │                       │
│                    │  │ hash_def → cpu_tensor_2 │  │                       │
│                    │  │ ...                     │  │                       │
│                    │  └─────────────────────────┘  │                       │
│                    │                               │                       │
│                    │  Reference Count: 3 (shared)  │                       │
│                    └───────────────────────────────┘                       │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## How It Works

### 1. Sleep Mode with Shared Pool

When `sleep()` is called with `enable_deduplication=True`:

1. **Hash Computation**: Each NPU memory block's SHA256 hash is computed
2. **Deduplication Check**: The shared pool checks if a block with the same hash exists
3. **Sharing**: If found, increment reference count and share the existing CPU tensor
4. **New Allocation**: If not found, allocate new pinned CPU memory and store in pool

### 2. Wake Up from Shared Pool

When `wake_up()` is called:

1. **Memory Mapping**: NPU virtual memory is remapped to physical memory
2. **Data Restoration**: Data is copied from the shared CPU tensor back to NPU
3. **Reference Tracking**: The NPU pointer is registered with the shared block

### 3. Reference Counting & Eviction

- Each shared block maintains a reference count
- When `release()` is called (during tensor garbage collection), the count decrements
- Blocks with `ref_count <= 0` are eligible for LRU eviction
- Eviction occurs when memory limit is reached

## Usage

### Basic Usage (Automatic)

By default, the shared pool is automatically enabled when using sleep mode:

```python
from vllm import LLM

# Shared pool is automatically used
llm = LLM("Qwen/Qwen2.5-0.5B-Instruct", enable_sleep_mode=True)

# Sleep - weights are offloaded to shared pool with SHA256 deduplication
llm.sleep(level=1)

# Wake up - weights are restored from shared pool
llm.wake_up()
```

### Advanced Configuration

```python
from vllm_ascend.device_allocator import SharedCPUMemoryPool

# Configure custom memory limit (default: 256GB)
import os
os.environ["VLLM_ASCEND_SHARED_POOL_LIMIT_GB"] = "128"

# Get pool instance and check statistics
pool = SharedCPUMemoryPool.get_instance()
stats = pool.get_stats()
print(f"Memory saved through sharing: {stats['total_shared_bytes'] / 1e9:.2f} GB")
print(f"Sharing hits: {stats['total_sharing_hits']}")
```

### Disabling Shared Pool

If you need the legacy behavior (no sharing):

```python
from vllm_ascend.device_allocator import CaMemAllocator

# Create allocator without shared pool
allocator = CaMemAllocator(use_shared_cpu_pool=False)
```

### Sleep without Deduplication

To use shared pool but skip SHA256 computation (faster but no sharing):

```python
llm.sleep(level=1, enable_deduplication=False)
```

## Memory Savings Analysis

### Scenario: Multi-NPU Inference with Same Model

| Configuration | Without Shared Pool | With Shared Pool | Savings |
|--------------|---------------------|------------------|---------|
| 4 NPUs × 7B Model (14GB weights) | 56 GB CPU memory | 14 GB CPU memory | **75%** |
| 8 NPUs × 70B Model (140GB weights) | 1,120 GB CPU memory | 140 GB CPU memory | **87.5%** |
| 16 NPUs × 405B Model (810GB weights) | 12,960 GB CPU memory | 810 GB CPU memory | **93.75%** |

### Real-World Example: RLHF Training

In RLHF with PPO/GRPO:
- **Actor model** (vLLM inference) and **Critic model** (training) often share architecture
- During critic training, actor is put to sleep
- Shared pool allows actor and critic weights to share CPU memory
- Significant savings when both models are on the same node

## API Reference

### SharedCPUMemoryPool

```python
class SharedCPUMemoryPool:
    """Process-wide singleton for shared CPU memory management."""
    
    @staticmethod
    def get_instance() -> "SharedCPUMemoryPool":
        """Get the singleton instance."""
        
    def allocate_from_npu(
        self,
        npu_ptr: int,
        size: int,
        compute_hash: bool = True,
        provided_hash: Optional[str] = None
    ) -> Tuple[torch.Tensor, str]:
        """Allocate CPU memory and copy NPU data to it."""
        
    def copy_to_npu(
        self,
        sha256_hash: str,
        npu_ptr: int,
        size: int
    ) -> None:
        """Copy data from shared pool back to NPU."""
        
    def release(self, sha256_hash: str, npu_ptr: int) -> bool:
        """Release a reference to a shared memory block."""
        
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get pool statistics."""
        
    def log_summary(self) -> None:
        """Log a summary of pool status."""
```

### Modified CaMemAllocator

```python
class CaMemAllocator:
    def __init__(self, use_shared_cpu_pool: bool = True):
        """Initialize with optional shared pool support."""
        
    def sleep(
        self,
        offload_tags: Optional[Union[Tuple[str, ...], str]] = None,
        enable_deduplication: bool = True
    ) -> None:
        """Sleep with optional SHA256 deduplication."""
        
    def get_shared_pool_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics from the shared CPU memory pool."""
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_ASCEND_SHARED_POOL_LIMIT_GB` | Maximum CPU memory for shared pool (GB) | 256 |

### Programmatic Configuration

```python
from vllm_ascend.device_allocator import SharedCPUMemoryPool

# Custom memory limit
pool = SharedCPUMemoryPool(memory_limit_bytes=512 * 1024 * 1024 * 1024)  # 512GB
```

## Performance Considerations

### SHA256 Computation Overhead

- Computing SHA256 hashes adds overhead during `sleep()`
- For large models, this is typically negligible compared to memory copy time
- Can be disabled with `enable_deduplication=False` if not needed

### Memory Access Patterns

- Pinned (page-locked) CPU memory enables faster DMA transfers
- Shared pool uses `pin_memory=True` for optimal NPU transfer performance

### Thread Safety

- All SharedCPUMemoryPool operations are thread-safe
- Uses RLock for concurrent access from multiple NPU workers

## Limitations

1. **Process Scope**: Sharing is limited to a single process. Cross-process sharing requires additional mechanisms (e.g., POSIX shared memory)

2. **Hash Collisions**: SHA256 collisions are theoretically possible but extremely unlikely

3. **Memory Alignment**: Memory blocks must be aligned for efficient DMA transfers

4. **Fallback**: If shared pool allocation fails, automatically falls back to legacy mode

## Debugging

### Enable Debug Logging

```python
import logging
logging.getLogger("vllm_ascend.device_allocator").setLevel(logging.DEBUG)
```

### Check Pool Statistics

```python
from vllm_ascend.device_allocator import SharedCPUMemoryPool

pool = SharedCPUMemoryPool.get_instance()
pool.log_summary()
```

Example output:
```
SharedCPUMemoryPool Summary:
  Current blocks: 42
  Current usage: 14.23 GB / 256.00 GB (5.6%)
  Total allocations: 42
  Sharing hits: 126 (saved 42.69 GB)
  Evictions: 0
```

## Migration from Legacy Sleep Mode

The shared pool is **fully backward compatible**. Existing code continues to work without changes:

```python
# Legacy code (still works)
llm = LLM(model_name, enable_sleep_mode=True)
llm.sleep(level=1)
llm.wake_up()

# Automatically benefits from shared pool
```

To opt-out:
```python
# Disable shared pool for specific allocator
allocator = CaMemAllocator(use_shared_cpu_pool=False)
```
