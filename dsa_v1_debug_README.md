# DSA v1 Debug Patch 使用说明

## 问题背景

- **错误**: NPU error 507011 (D-cache bus error / OOB access)
- **场景**: DeepSeek-V4 模型运行 batch prefill 模式
- **错误位置**: `vllm_ascend/attention/dsa_v1.py` 中的 `_get_padded_compressed_position` 函数

## 补丁说明

此补丁在以下位置添加调试代码：

1. `build_prefill_metadata` 函数入口：打印元数据信息
2. `_get_padded_compressed_position` 函数：打印输入张量信息和计算步骤

## 应用补丁

在远程机器上执行：

```bash
cd /vllm-workspace/vllm-ascend

# 应用补丁
patch -p1 < /path/to/dsa_v1_debug.patch

# 或者手动编辑
vim /vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v1.py
```

## 运行调试

```bash
# 启用调试
export DEBUG_DSA_COMPRESS=1
export DEBUG_BATCH_PREFILL=1

# 运行测试
bash examples/disaggregated_prefill_v1/run_laps_sharegpt_benchmark_deepseek_v4_single_node_prefill.sh
```

## 预期输出

调试信息会显示：
- `build_prefill_metadata` 入口时的元数据状态
- `_get_padded_compressed_position` 每次调用的详细信息
- NPU stream 同步状态
- prefill_input_positions 的形状、数值范围、内容

## 关键检查点

1. **Stream 同步状态**: 如果在函数入口处同步失败，说明之前的 CUDA graph 执行已出错
2. **positions 数值**: 检查是否有异常值（负数、超大值）
3. **mask 计算结果**: 检查 mask 的数量和分布是否符合预期

## 远程仓库参考

- 仓库: https://github.com/GDzhu01/vllm-ascend-deepseekv4
- 分支: v4_v0.18.0_0412
- 文件: vllm_ascend/attention/dsa_v1.py
