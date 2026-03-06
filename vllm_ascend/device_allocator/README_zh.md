# vllm-ascend NPU 设备内存分配器

本目录包含 vllm-ascend 的可插拔 NPU 内存分配器后端。每个后端都实现了
`NPUPluggableAllocator` 接口，用自定义内存池替换 PyTorch-NPU 默认的缓存分配器。

| 文件 | 后端 | 状态 |
|------|------|------|
| `camem.py` / `csrc/camem_allocator.cpp` | CANN 内置 CA-mem 池 | 上游已有 |
| `shmem_allocator.py` / `csrc/shmem_allocator.cpp` | SHMEM 动态内存池 | 本特性 |

---

## SHMEM 动态内存分配器

### 背景

[SHMEM](https://gitee.com/ascend/shmem)（Ascend Symmetric Heterogeneous Memory）
是面向昇腾 NPU 的多设备内存通信库，其内存堆通过标准的
`aclshmem_malloc` / `aclshmem_free` 接口对外暴露，与任意 NPU 设备指针兼容。

在原有实现基础上，我们通过 `shmem_dynamic_mm.cpp` 增加了**动态扩容**能力：
当初始内存池耗尽时，系统会透明地调用 `aclrtMalloc` 申请新的设备内存块并追加到池中，
单次扩容上限 4 GiB，扩容系数 1.5×。这消除了为整个模型预先分配固定大小内存池的需要，
让 vllm 能够按需扩展 KV Cache。

### 整体架构

```
Python 层（vllm-ascend）
│
│  worker.py: init_device()
│      └─ ShmemAllocator.install()           ← 一次性全局替换
│              └─ change_current_allocator(NPUPluggableAllocator)
│
│  worker.py: load_model()             → 权重走 SHMEM
│  worker.py: initialize_from_config() → KV Cache 走 SHMEM
│      │
│      │  所有 torch.empty() / NPU 张量创建均走此路径
│      ↓
C++ 层（shmem_allocator.cpython-*.so）
│
│  my_malloc(size, device, stream)        ← 收到精确张量大小
│      ├─ ensure_shmem_initialized()    ← 懒初始化，幂等，线程安全
│      ├─ aclshmem_malloc(size)         ← SHMEM 池（自动动态扩容）
│      │    └─ 失败时: aclrtMalloc      ← ACL 兜底
│      └─ 将指针记录到 g_shmem_ptrs
│
│  my_free(ptr, size, device, stream)
│      ├─ ptr 在 g_shmem_ptrs 中?
│      │    是 → aclshmem_free(ptr)
│      └─ 否 → aclrtFree(ptr)
│
SHMEM 库（libshmem.so）
│
│  动态内存池管理器
│      ├─ 初始块（SHMEM_INITIAL_POOL_SIZE，默认 2 GiB）
│      └─ 按需通过 aclrtMalloc 追加扩容块
```

**关键设计决策**

* **独立 `.so` 模块** — `camem_allocator.cpp`（编译进 `vllm_ascend_C.so`）已经定义了
  `my_malloc`/`my_free`。若将 shmem 代码合入同一 `.so` 会产生链接器重复符号错误，
  因此 `shmem_allocator` 单独作为一个 `pybind11_add_module` 构建。

* **`.so` 双重角色** — 同一个 `shmem_allocator.cpython-*.so` 文件同时承担：
  1. Python 扩展模块（可通过 `import vllm_ascend.shmem_allocator` 访问管理接口）
  2. `NPUPluggableAllocator` 后端（通过文件路径加载，用 `dlsym` 解析 `my_malloc`/`my_free`）

* **指针追踪** — SHMEM 公开 API 中不存在 `aclshmem_ptr_valid`。分配器通过
  `std::unordered_set<void*>` 记录所有来自 `aclshmem_malloc` 的指针，使 `my_free`
  能够正确路由到对应的释放函数。

* **不重复 ACL 生命周期** — `aclInit`、`aclrtSetDevice`、`aclFinalize` 由
  vllm-ascend 负责调用，SHMEM 分配器不干预这些操作。

* **SHMEM 与 sleep mode 互斥** — `CaMemAllocator`（sleep mode）支持 CPU 卸载，
  `ShmemAllocator` 不支持。同时开启两者会抛出 `RuntimeError`。

### vllm-ascend 中的分配范围

当 `ENABLE_SHMEM=1` 时，`ShmemAllocator.install()` 在 `worker.init_device()` 中、
任何 NPU 内存分配发生之前被调用，将全局 NPU 分配器替换为 SHMEM 后端。
此后进程内**所有** NPU 张量分配——模型权重、KV Cache、激活值、临时 buffer——
均直接走 SHMEM 的 `my_malloc` / `my_free`。

| 阶段 | 函数 | 处理方式 |
|------|------|----------|
| 全局分配器安装 | `worker.init_device()` | `ShmemAllocator.install()` |
| 模型权重加载 | `worker.load_model()` | 全局 SHMEM 分配器 |
| KV Cache 初始化 | `worker.initialize_from_config()` | 全局 SHMEM 分配器 |

---

### 构建

#### 第一步：从源码构建并安装 SHMEM

```bash
git clone <shmem 仓库地址>
cd shmem
bash scripts/build.sh

# 导出 SHMEM_HOME_PATH 环境变量，例如 /root/shmem/install
source install/set_env.sh
echo $SHMEM_HOME_PATH
```

#### 第二步：构建支持 SHMEM 的 vllm-ascend

```bash
# 必须先设置 SHMEM_HOME_PATH（第一步已完成）
ENABLE_SHMEM=1 pip install -e .
```

CMake 变量（由环境变量自动传入，也可手动指定）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ENABLE_SHMEM` | 是否构建 shmem_allocator 模块 | `OFF` |
| `SHMEM_HOME_PATH` | `shmem/` 安装目录的父目录 | `/usr/local/Ascend/shmem/latest` |

构建完成后，`vllm_ascend/shmem_allocator.cpython-*.so` 会与
`vllm_ascend_C.cpython-*.so` 并列生成。

---

### 使用方式

#### 方式 A — 自动接入（推荐）

运行时设置 `ENABLE_SHMEM=1`，worker 会自动将权重加载和 KV Cache 初始化
都路由到 SHMEM 动态池。

```bash
ENABLE_SHMEM=1 python your_vllm_script.py
```

#### 方式 B — 通过 Python API 手动接入

```python
from vllm_ascend.device_allocator.shmem_allocator import ShmemAllocator

allocator = ShmemAllocator.get_instance()

# 在任何 NPU 分配之前安装全局分配器
allocator.install()

# 此后所有 NPU 分配均走 SHMEM
kv_cache = torch.empty(shape, dtype=torch.float16, device="npu")

stats = allocator.get_memory_stats()
if stats:
    total, used, avail = stats
    print(f"SHMEM 池: {used/1e9:.2f} / {total/1e9:.2f} GiB 已用")
```

---

### 运行时环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ENABLE_SHMEM` | 运行时启用 SHMEM 分配器 | `0` |
| `SHMEM_HOME_PATH` | 由 `source install/set_env.sh` 导出的路径 | — |
| `SHMEM_INITIAL_POOL_SIZE` | 初始内存池大小（字节） | `2147483648`（2 GiB） |

`SHMEM_INITIAL_POOL_SIZE` 在 C++ 层首次分配时读取。池可自动扩容至该值以上——
将其设置得足够大可减少扩容开销。

```bash
# 预留 8 GiB 初始池
export SHMEM_INITIAL_POOL_SIZE=$((8 * 1024 * 1024 * 1024))
ENABLE_SHMEM=1 python your_vllm_script.py
```

---

### API 参考

#### Python — `ShmemAllocator`

| 方法 | 说明 |
|------|------|
| `ShmemAllocator.get_instance()` | 获取单例实例 |
| `.install()` | 将全局 NPU 分配器替换为 SHMEM（在任何 NPU 分配之前调用，一次性生效） |
| `.initialize() → bool` | 显式初始化 SHMEM 池（幂等；`my_malloc` 首次调用时也会懒初始化） |
| `.finalize()` | 释放内存池并重置状态 |
| `.get_memory_stats() → (total, used, avail)` | 内存池统计（字节） |
| `.get_current_usage() → int` | 当前已分配字节数 |
| `.sleep(offload_tags=None)` | 空操作（兼容 CaMemAllocator 接口） |
| `.wake_up(tags=None)` | 空操作（兼容 CaMemAllocator 接口） |

#### 模块级标志

```python
from vllm_ascend.device_allocator.shmem_allocator import shmem_available
# True  → 扩展模块已编译并成功导入
# False → 构建时未设置 ENABLE_SHMEM=ON（或导入失败）
```

#### C++ — NPUPluggableAllocator 导出符号

| 符号 | 签名 | 说明 |
|------|------|------|
| `my_malloc` | `void*(ssize_t size, int device, aclrtStream stream)` | 从 SHMEM 池分配；失败时回退到 `aclrtMalloc` |
| `my_free` | `void(void* ptr, ssize_t size, int device, aclrtStream stream)` | 根据来源路由到 `aclshmem_free` 或 `aclrtFree` |

---

### 内存分配粒度说明

#### 精确张量大小分配

`changeCurrentAllocator` 方式完全绕过 PyTorch 缓存分配器，每次 `torch.empty()`
都以**精确的张量大小**调用 `my_malloc`：

```
torch.empty(512 KB)
  → change_current_allocator 路径
  → my_malloc(512 KB)    ← SHMEM 收到精确大小
  → aclshmem_malloc(512 KB) 或 dynamic_memory_manager 子分配
```

SHMEM 的最优适配选择（best-fit）和指针合并逻辑因此可以在真实分配粒度上生效，
不再受 PyTorch 缓存分配器固定 20 MiB 段大小的制约。

#### 动态扩容

扩容机制不变：内存池耗尽时，`aclshmem_malloc` 透明扩充
（扩容系数 1.5×，每次最多 4 GiB）。

#### 内存统计

由于没有缓存层，每次 `del tensor` / 引用归零后会立即调用 `my_free`，内存即刻
归还 SHMEM。SHMEM 统计始终准确反映实际活跃分配。
`torch.npu.empty_cache()` 在此模式下为空操作。

---

### 常见问题排查

**`ImportError: No module named 'vllm_ascend.shmem_allocator'`**
构建时未设置 `ENABLE_SHMEM=1`，请按上述步骤重新构建。

**`RuntimeError: SHMEM allocator is not available`**
扩展模块未编译进来。确保 `ENABLE_SHMEM=1` 和 `SHMEM_HOME_PATH` 均已设置，
然后重新构建。

**`RuntimeError: ENABLE_SHMEM and sleep mode are mutually exclusive`**
同时开启了 `enable_sleep_mode=True` 和 `ENABLE_SHMEM=1`。SHMEM 不支持
CPU 卸载，二者只能选其一。

**`RuntimeError: aclshmemx_init_attr failed: <错误码>`**
常见原因：
- `SHMEM_HOME_PATH` 未设置或路径有误，检查 `echo $SHMEM_HOME_PATH`。
- 引导插件 `aclshmem_bootstrap_uid.so` 找不到：确保
  `${SHMEM_HOME_PATH}/shmem/lib/` 在 `LD_LIBRARY_PATH` 中，
  或构建时嵌入的 RPATH 正确。

**`aclshmem_malloc failed … falling back to aclrtMalloc`**
SHMEM 池动态扩容失败（设备显存不足）。请减小 `SHMEM_INITIAL_POOL_SIZE`
或降低模型批次大小。

**`RuntimeError: expandable_segments:True is not compatible with the SHMEM memory pool`**
环境变量中设置了 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`。
PyTorch 的 expandable_segments 与可插拔内存池不兼容
（参见 [pytorch#147851](https://github.com/pytorch/pytorch/issues/147851)）。
请从 `PYTORCH_NPU_ALLOC_CONF` 中删除或取消 `expandable_segments:True`。
