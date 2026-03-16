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
# 示例：跨进程共享 CPU 内存池的 Sleep Mode
#
# 本示例演示如何使用 CrossProcessSharedPool 在多个进程中
# 共享卸载的权重内存。适用于多卡推理场景。
#
# 注意：需要足够的 /dev/shm 空间（建议模型权重的 2 倍）
#

import multiprocessing
import os
import time

import torch
from vllm.utils.mem_constants import GiB_bytes

# 设置多进程启动方式
multiprocessing.set_start_method("spawn", force=True)

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ASCEND_ENABLE_NZ"] = "0"


def worker_process(
    worker_id: int,
    model_name: str,
    prompt: str,
    barrier: multiprocessing.Barrier,
    results_queue: multiprocessing.Queue,
):
    """
    工作进程：加载模型、执行推理、进入 sleep
    
    多个工作进程可以通过 CrossProcessSharedPool 共享卸载的权重
    """
    print(f"[Worker {worker_id}] 启动")
    
    # 延迟导入，避免在子进程中重复初始化
    from vllm import LLM, SamplingParams
    from vllm_ascend.device_allocator.shared_cpu_pool_multiprocess import (
        CrossProcessSharedPool,
    )
    
    try:
        # 初始化跨进程共享池
        pool = CrossProcessSharedPool.get_instance()
        print(f"[Worker {worker_id}] 共享池初始化完成")
        
        # 等待所有工作进程就绪
        barrier.wait()
        
        # 加载模型
        print(f"[Worker {worker_id}] 加载模型...")
        llm = LLM(
            model_name,
            enable_sleep_mode=True,
            tensor_parallel_size=1,  # 每个进程一个 NPU
        )
        sampling_params = SamplingParams(temperature=0, max_tokens=10)
        
        print(f"[Worker {worker_id}] 模型加载完成")
        
        # 同步点：所有进程加载完成
        barrier.wait()
        
        # 第一次推理
        print(f"[Worker {worker_id}] 执行第一次推理...")
        output1 = llm.generate(prompt, sampling_params)
        print(f"[Worker {worker_id}] 输出: {output1[0].outputs[0].text}")
        
        # 同步点
        barrier.wait()
        
        # Sleep - 权重卸载到共享池
        print(f"[Worker {worker_id}] 进入 Sleep...")
        sleep_start = time.time()
        llm.sleep(level=1)
        sleep_time = time.time() - sleep_start
        
        # 获取统计
        stats = pool.get_stats()
        results_queue.put({
            "worker_id": worker_id,
            "sleep_time": sleep_time,
            "stats": stats,
        })
        
        print(f"[Worker {worker_id}] Sleep 完成，耗时: {sleep_time:.2f}s")
        print(f"[Worker {worker_id}] 共享池块数: {stats['current_blocks']}")
        
        # 同步点
        barrier.wait()
        
        # Wake up
        print(f"[Worker {worker_id}] 执行 Wake Up...")
        wake_start = time.time()
        llm.wake_up()
        wake_time = time.time() - wake_start
        
        print(f"[Worker {worker_id}] Wake Up 完成，耗时: {wake_time:.2f}s")
        
        # 第二次推理
        print(f"[Worker {worker_id}] 执行第二次推理...")
        output2 = llm.generate(prompt, sampling_params)
        print(f"[Worker {worker_id}] 输出: {output2[0].outputs[0].text}")
        
        # 验证一致性
        if output1[0].outputs[0].text == output2[0].outputs[0].text:
            print(f"[Worker {worker_id}] ✓ 输出一致！")
        else:
            print(f"[Worker {worker_id}] ✗ 输出不一致！")
        
        results_queue.put({
            "worker_id": worker_id,
            "wake_time": wake_time,
            "output_match": output1[0].outputs[0].text == output2[0].outputs[0].text,
        })
        
    except Exception as e:
        print(f"[Worker {worker_id}] 错误: {e}")
        import traceback
        traceback.print_exc()
        results_queue.put({"worker_id": worker_id, "error": str(e)})


def demo_multiprocess_sharing():
    """
    演示：多进程共享权重内存
    
    场景：2 个进程分别使用 NPU 0 和 NPU 1 运行相同模型
    预期：第二个进程 sleep 时应该复用第一个进程的共享内存
    """
    print("\n" + "=" * 70)
    print("演示：跨进程共享权重内存")
    print("=" * 70 + "\n")
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    num_workers = 2
    
    # 检查可用 NPU
    if not torch.npu.is_available():
        print("错误：NPU 不可用")
        return
    
    available_npus = torch.npu.device_count()
    if available_npus < num_workers:
        print(f"警告：需要 {num_workers} 个 NPU，但只有 {available_npus} 个可用")
        print(f"将使用 {available_npus} 个 worker")
        num_workers = available_npus
    
    print(f"启动 {num_workers} 个工作进程...")
    print(f"模型: {model_name}")
    print(f"预期内存节省: {(num_workers - 1) * 100 / num_workers:.1f}%\n")
    
    # 创建同步屏障
    barrier = multiprocessing.Barrier(num_workers)
    
    # 创建结果队列
    results_queue = multiprocessing.Queue()
    
    # 创建工作进程
    processes = []
    for i in range(num_workers):
        # 为每个进程设置不同的可见 NPU
        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(i)
        
        p = multiprocessing.Process(
            target=worker_process,
            args=(
                i,
                model_name,
                "Hello, how are you?",
                barrier,
                results_queue,
            ),
        )
        p.start()
        processes.append(p)
    
    # 收集结果
    results = []
    for _ in range(num_workers * 2):  # 每个进程 2 个结果
        try:
            result = results_queue.get(timeout=120)
            results.append(result)
        except:
            break
    
    # 等待所有进程完成
    for p in processes:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
    
    # 分析结果
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    
    sleep_times = []
    wake_times = []
    
    for r in results:
        if "error" in r:
            print(f"Worker {r['worker_id']}: 错误 - {r['error']}")
        elif "sleep_time" in r:
            sleep_times.append((r['worker_id'], r['sleep_time']))
            print(f"Worker {r['worker_id']}: Sleep 耗时 {r['sleep_time']:.2f}s")
            if 'stats' in r:
                stats = r['stats']
                print(f"  - 共享池块数: {stats['current_blocks']}")
                print(f"  - 跨进程命中: {stats.get('total_cross_process_hits', 0)}")
        elif "wake_time" in r:
            wake_times.append((r['worker_id'], r['wake_time']))
            print(f"Worker {r['worker_id']}: Wake Up 耗时 {r['wake_time']:.2f}s")
            if r.get('output_match'):
                print(f"  - 输出验证: ✓ 通过")
    
    # 分析共享效果
    if len(sleep_times) >= 2:
        print("\n共享效果分析：")
        first_sleep = min(sleep_times, key=lambda x: x[0])[1]
        second_sleep = max(sleep_times, key=lambda x: x[0])[1]
        if second_sleep < first_sleep * 0.8:
            print(f"✓ 第二个进程 Sleep 更快 ({second_sleep:.2f}s vs {first_sleep:.2f}s)")
            print("  说明复用了第一个进程的共享内存")
        else:
            print(f"  Sleep 时间相近，可能未触发共享")


def demo_memory_savings_calculation():
    """
    计算多进程共享的内存节省
    """
    print("\n" + "=" * 70)
    print("内存节省计算")
    print("=" * 70 + "\n")
    
    scenarios = [
        (2, "7B", 14),    # 2 NPUs, 7B model, ~14GB weights
        (4, "7B", 14),    # 4 NPUs, 7B model
        (8, "70B", 140),  # 8 NPUs, 70B model
        (8, "405B", 810), # 8 NPUs, 405B model
    ]
    
    print(f"{'配置':<20} {'无共享':<15} {'有共享':<15} {'节省':<15}")
    print("-" * 70)
    
    for num_npus, model_size, weight_gb in scenarios:
        without_pool = num_npus * weight_gb
        with_pool = weight_gb  # Shared
        saved = without_pool - with_pool
        saved_pct = saved / without_pool * 100
        
        config = f"{num_npus} NPU × {model_size}"
        print(f"{config:<20} {without_pool:<15}GB {with_pool:<15}GB {saved_pct:.1f}%")
    
    print("\n注意：实际节省取决于模型架构和权重分布")
    print("      CrossProcessSharedPool 需要足够的 /dev/shm 空间")


def check_system_requirements():
    """检查系统要求"""
    print("\n" + "=" * 70)
    print("系统要求检查")
    print("=" * 70 + "\n")
    
    # 检查 /dev/shm
    shm_path = "/dev/shm"
    if os.path.exists(shm_path):
        import shutil
        shm_stats = shutil.disk_usage(shm_path)
        shm_gb = shm_stats.total / (1024 ** 3)
        shm_free_gb = shm_stats.free / (1024 ** 3)
        
        print(f"✓ /dev/shm 存在")
        print(f"  总大小: {shm_gb:.2f} GB")
        print(f"  可用空间: {shm_free_gb:.2f} GB")
        
        if shm_gb < 50:
            print(f"  ⚠ 警告: /dev/shm 较小，可能需要增加")
            print(f"    建议: sudo mount -o remount,size=256G /dev/shm")
    else:
        print(f"✗ /dev/shm 不存在（非 Linux 系统？）")
    
    # 检查 NPU
    if torch.npu.is_available():
        npu_count = torch.npu.device_count()
        print(f"\n✓ NPU 可用，数量: {npu_count}")
        for i in range(npu_count):
            name = torch.npu.get_device_name(i)
            print(f"  NPU {i}: {name}")
    else:
        print(f"\n✗ NPU 不可用")
    
    # 检查 Python 版本
    import sys
    print(f"\nPython 版本: {sys.version}")
    
    # 检查必要的包
    try:
        import filelock
        print(f"✓ filelock 已安装")
    except ImportError:
        print(f"✗ filelock 未安装，请运行: pip install filelock")


def main():
    """主函数"""
    print("\n" + "#" * 70)
    print("# vLLM-Ascend 跨进程共享 CPU 内存池演示")
    print("#" * 70)
    
    # 检查系统
    check_system_requirements()
    
    # 显示内存节省计算
    demo_memory_savings_calculation()
    
    # 运行多进程演示
    try:
        demo_multiprocess_sharing()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "#" * 70)
    print("# 演示完成")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
