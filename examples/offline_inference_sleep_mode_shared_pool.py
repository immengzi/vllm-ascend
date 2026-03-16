#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#
# 示例：使用共享 CPU 内存池的 Sleep Mode
#
# 本示例演示共享 CPU 内存池如何使多个 LLM 实例通过
# SHA256 去重共享卸载的权重内存。
#

import os
import time

import torch
from vllm import LLM, SamplingParams
from vllm.utils.mem_constants import GiB_bytes

from vllm_ascend.device_allocator import SharedCPUMemoryPool

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ASCEND_ENABLE_NZ"] = "0"


def print_memory_stats(label: str):
    """打印当前 NPU 内存使用情况。"""
    free, total = torch.npu.mem_get_info()
    used = total - free
    print(f"[{label}] NPU 内存: 已用={used / GiB_bytes:.2f} GB, "
          f"空闲={free / GiB_bytes:.2f} GB, 总计={total / GiB_bytes:.2f} GB")


def print_shared_pool_stats(label: str):
    """打印共享 CPU 内存池统计信息。"""
    pool = SharedCPUMemoryPool.get_instance()
    stats = pool.get_stats()
    print(f"[{label}] 共享池统计:")
    print(f"  - 当前块数: {stats['current_blocks']}")
    print(f"  - 当前使用: {stats['current_bytes'] / GiB_bytes:.2f} GB")
    print(f"  - 内存利用率: {stats['memory_utilization'] * 100:.1f}%")
    print(f"  - 总共享命中: {stats['total_sharing_hits']}")
    print(f"  - 总节省内存: {stats['total_shared_bytes'] / GiB_bytes:.2f} GB")


def demo_basic_sleep_wake():
    """演示 1：使用共享池的基本 sleep/wake。"""
    print("\n" + "=" * 70)
    print("演示 1：使用共享 CPU 内存池的基本 Sleep/Wake")
    print("=" * 70 + "\n")
    
    prompt = "How are you?"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print_memory_stats("模型加载前")
    
    # 初始化 LLM，启用 sleep mode
    # 默认使用共享 CPU 内存池
    llm = LLM(model_name, enable_sleep_mode=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    
    print_memory_stats("模型加载后")
    
    # 第一次推理
    print("\n--- 第一次推理 ---")
    output1 = llm.generate(prompt, sampling_params)
    print(f"输出: {output1[0].outputs[0].text}")
    
    # Sleep - 权重卸载到共享 CPU 内存池
    print("\n--- Sleep (Level 1) ---")
    llm.sleep(level=1)
    print_memory_stats("Sleep 后")
    print_shared_pool_stats("Sleep 后")
    
    # Wake up
    print("\n--- Wake Up ---")
    llm.wake_up()
    print_memory_stats("Wake Up 后")
    
    # 第二次推理
    print("\n--- 第二次推理 ---")
    output2 = llm.generate(prompt, sampling_params)
    print(f"输出: {output2[0].outputs[0].text}")
    
    # 验证输出一致
    assert output1[0].outputs[0].text == output2[0].outputs[0].text
    print("\n✓ 输出一致！")
    
    print("\n" + "=" * 70)


def demo_multiple_models_sharing():
    """演示 2：多模型通过共享池共享内存。"""
    print("\n" + "=" * 70)
    print("演示 2：多模型共享内存（模拟）")
    print("=" * 70 + "\n")
    
    # 注意：在真实的多进程/NPU 场景中，
    # 共享池会自动去重相同的权重。
    # 这里演示单进程内的池行为。
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print("加载第一个模型实例...")
    llm1 = LLM(model_name, enable_sleep_mode=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=5)
    
    output1 = llm1.generate("Hello", sampling_params)
    print(f"模型 1 输出: {output1[0].outputs[0].text}")
    
    # Sleep 第一个模型
    print("\nSleep 第一个模型...")
    llm1.sleep(level=1)
    print_shared_pool_stats("模型 1 Sleep 后")
    
    # 在多进程场景中，加载第二个相同架构的模型会共享已卸载的权重。
    # 本演示仅展示池状态。
    
    print("\n--- 模拟第二个模型使用相同权重 ---")
    print("真实多进程场景中，第二个模型会：")
    print("1. 在 Sleep 期间计算权重 SHA256")
    print("2. 在共享池中找到匹配的块")
    print("3. 复用现有 CPU 内存而非分配新内存")
    
    print("\n" + "=" * 70)


def demo_memory_savings():
    """演示 3：展示去重带来的内存节省。"""
    print("\n" + "=" * 70)
    print("演示 3：内存节省分析")
    print("=" * 70 + "\n")
    
    pool = SharedCPUMemoryPool.get_instance()
    
    print("场景：4 个 NPU 运行相同 7B 模型（每个 14GB 权重）")
    print("-" * 50)
    print("不使用共享池：")
    print("  需要 CPU 内存：4 × 14GB = 56 GB")
    print("")
    print("使用共享池（SHA256 去重）：")
    print("  需要 CPU 内存：14 GB（所有 NPU 共享）")
    print("  节省：42 GB（减少 75%）")
    print("-" * 50)
    
    print("\n当前池统计：")
    print_shared_pool_stats("当前")
    
    print("\n" + "=" * 70)


def demo_level2_sleep():
    """演示 4：Level 2 sleep 与共享池。"""
    print("\n" + "=" * 70)
    print("演示 4：Level 2 Sleep（权重更新场景）")
    print("=" * 70 + "\n")
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    llm = LLM(model_name, enable_sleep_mode=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=5)
    
    output1 = llm.generate("Test", sampling_params)
    print(f"Sleep 前: {output1[0].outputs[0].text}")
    
    # Level 2 sleep - 丢弃权重
    print("\n--- Level 2 Sleep ---")
    llm.sleep(level=2)
    print_memory_stats("Level 2 Sleep 后")
    print_shared_pool_stats("Level 2 Sleep 后")
    
    # 仅唤醒权重
    print("\n--- Wake Up（仅权重）---")
    llm.wake_up(tags=["weights"])
    print_memory_stats("Wake Up（权重）后")
    
    # 模拟权重更新
    print("\n--- 模拟权重更新 ---")
    print("此时您可以更新模型权重。")
    print("更新完成后，唤醒 KV cache：")
    
    llm.wake_up(tags=["kv_cache"])
    print_memory_stats("完全 Wake Up 后")
    
    print("\n" + "=" * 70)


def main():
    """运行所有演示。"""
    print("\n" + "#" * 70)
    print("# vLLM-Ascend Sleep Mode 共享 CPU 内存池演示")
    print("#" * 70)
    
    # 检查 NPU 是否可用
    if not torch.npu.is_available():
        print("错误：NPU 不可用。本演示需要昇腾 NPU。")
        return
    
    print("\n初始化共享 CPU 内存池...")
    pool = SharedCPUMemoryPool.get_instance()
    print(f"内存限制：{pool.memory_limit_bytes / GiB_bytes:.2f} GB")
    
    try:
        # 运行演示
        demo_basic_sleep_wake()
        demo_multiple_models_sharing()
        demo_memory_savings()
        demo_level2_sleep()
        
        print("\n" + "#" * 70)
        print("# 所有演示成功完成！")
        print("#" * 70 + "\n")
        
    except Exception as e:
        print(f"\n演示期间出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 打印最终统计
        print("\n最终共享池统计：")
        pool.log_summary()


if __name__ == "__main__":
    main()
