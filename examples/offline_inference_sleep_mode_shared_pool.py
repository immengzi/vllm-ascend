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
# Example: Using Shared CPU Memory Pool with Sleep Mode
#
# This example demonstrates how the shared CPU memory pool enables
# multiple LLM instances to share offloaded weight memory through
# SHA256-based deduplication.
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
    """Print current NPU memory usage."""
    free, total = torch.npu.mem_get_info()
    used = total - free
    print(f"[{label}] NPU Memory: Used={used / GiB_bytes:.2f} GB, "
          f"Free={free / GiB_bytes:.2f} GB, Total={total / GiB_bytes:.2f} GB")


def print_shared_pool_stats(label: str):
    """Print shared CPU memory pool statistics."""
    pool = SharedCPUMemoryPool.get_instance()
    stats = pool.get_stats()
    print(f"[{label}] Shared Pool Stats:")
    print(f"  - Current blocks: {stats['current_blocks']}")
    print(f"  - Current usage: {stats['current_bytes'] / GiB_bytes:.2f} GB")
    print(f"  - Memory utilization: {stats['memory_utilization'] * 100:.1f}%")
    print(f"  - Total sharing hits: {stats['total_sharing_hits']}")
    print(f"  - Total shared bytes (saved): {stats['total_shared_bytes'] / GiB_bytes:.2f} GB")


def demo_basic_sleep_wake():
    """Demo 1: Basic sleep/wake with shared pool."""
    print("\n" + "=" * 70)
    print("Demo 1: Basic Sleep/Wake with Shared CPU Memory Pool")
    print("=" * 70 + "\n")
    
    prompt = "How are you?"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print_memory_stats("Before Model Load")
    
    # Initialize LLM with sleep mode enabled
    # By default, it uses the shared CPU memory pool
    llm = LLM(model_name, enable_sleep_mode=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    
    print_memory_stats("After Model Load")
    
    # First inference
    print("\n--- First inference ---")
    output1 = llm.generate(prompt, sampling_params)
    print(f"Output: {output1[0].outputs[0].text}")
    
    # Sleep - weights are offloaded to shared CPU memory pool
    print("\n--- Sleep (Level 1) ---")
    llm.sleep(level=1)
    print_memory_stats("After Sleep")
    print_shared_pool_stats("After Sleep")
    
    # Wake up
    print("\n--- Wake Up ---")
    llm.wake_up()
    print_memory_stats("After Wake Up")
    
    # Second inference
    print("\n--- Second inference ---")
    output2 = llm.generate(prompt, sampling_params)
    print(f"Output: {output2[0].outputs[0].text}")
    
    # Verify outputs match
    assert output1[0].outputs[0].text == output2[0].outputs[0].text
    print("\n✓ Outputs match!")
    
    print("\n" + "=" * 70)


def demo_multiple_models_sharing():
    """Demo 2: Multiple models sharing memory through the pool."""
    print("\n" + "=" * 70)
    print("Demo 2: Multiple Models Sharing Memory (Simulated)")
    print("=" * 70 + "\n")
    
    # Note: In a real scenario with multiple processes/NPUs,
    # the shared pool would automatically deduplicate identical weights.
    # Here we demonstrate the pool behavior within a single process.
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print("Loading first model instance...")
    llm1 = LLM(model_name, enable_sleep_mode=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=5)
    
    output1 = llm1.generate("Hello", sampling_params)
    print(f"Model 1 output: {output1[0].outputs[0].text}")
    
    # Sleep first model
    print("\nSleeping first model...")
    llm1.sleep(level=1)
    print_shared_pool_stats("After Model 1 Sleep")
    
    # In a multi-process scenario, loading a second model with the same
    # architecture would share the offloaded weights.
    # For this demo, we just show the pool state.
    
    print("\n--- Simulating second model with same weights ---")
    print("In a real multi-process setup, the second model would:")
    print("1. Compute SHA256 of its weights during sleep")
    print("2. Find matching blocks in the shared pool")
    print("3. Reuse existing CPU memory instead of allocating new")
    
    print("\n" + "=" * 70)


def demo_memory_savings():
    """Demo 3: Show memory savings with deduplication."""
    print("\n" + "=" * 70)
    print("Demo 3: Memory Savings Analysis")
    print("=" * 70 + "\n")
    
    pool = SharedCPUMemoryPool.get_instance()
    
    print("Scenario: 4 NPUs running the same 7B model (14GB weights each)")
    print("-" * 50)
    print("Without Shared Pool:")
    print("  CPU Memory needed: 4 × 14GB = 56 GB")
    print("")
    print("With Shared Pool (SHA256 deduplication):")
    print("  CPU Memory needed: 14 GB (shared across all NPUs)")
    print("  Savings: 42 GB (75% reduction)")
    print("-" * 50)
    
    print("\nCurrent pool statistics:")
    print_shared_pool_stats("Current")
    
    print("\n" + "=" * 70)


def demo_level2_sleep():
    """Demo 4: Level 2 sleep with shared pool."""
    print("\n" + "=" * 70)
    print("Demo 4: Level 2 Sleep (Weight Update Scenario)")
    print("=" * 70 + "\n")
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    llm = LLM(model_name, enable_sleep_mode=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=5)
    
    output1 = llm.generate("Test", sampling_params)
    print(f"Before sleep: {output1[0].outputs[0].text}")
    
    # Level 2 sleep - discard weights
    print("\n--- Level 2 Sleep ---")
    llm.sleep(level=2)
    print_memory_stats("After Level 2 Sleep")
    print_shared_pool_stats("After Level 2 Sleep")
    
    # Wake up only weights
    print("\n--- Wake Up (Weights Only) ---")
    llm.wake_up(tags=["weights"])
    print_memory_stats("After Wake Up (Weights)")
    
    # Simulate weight update
    print("\n--- Simulating Weight Update ---")
    print("At this point, you would update the model weights.")
    print("After update, wake up KV cache:")
    
    llm.wake_up(tags=["kv_cache"])
    print_memory_stats("After Full Wake Up")
    
    print("\n" + "=" * 70)


def main():
    """Run all demos."""
    print("\n" + "#" * 70)
    print("# Shared CPU Memory Pool Demo for vLLM-Ascend Sleep Mode")
    print("#" * 70)
    
    # Check if NPU is available
    if not torch.npu.is_available():
        print("ERROR: NPU is not available. This demo requires Ascend NPU.")
        return
    
    print("\nInitializing Shared CPU Memory Pool...")
    pool = SharedCPUMemoryPool.get_instance()
    print(f"Memory limit: {pool.memory_limit_bytes / GiB_bytes:.2f} GB")
    
    try:
        # Run demos
        demo_basic_sleep_wake()
        demo_multiple_models_sharing()
        demo_memory_savings()
        demo_level2_sleep()
        
        print("\n" + "#" * 70)
        print("# All demos completed successfully!")
        print("#" * 70 + "\n")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Print final statistics
        print("\nFinal Shared Pool Statistics:")
        pool.log_summary()


if __name__ == "__main__":
    main()
