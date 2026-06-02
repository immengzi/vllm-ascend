#!/usr/bin/env python3
"""
自动为 dsa_v1.py 添加调试代码

使用方法:
    python3 apply_dsa_debug.py
"""

import sys

def main():
    file_path = "/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v1.py"
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        print("请确保在远程机器上运行此脚本")
        sys.exit(1)
    
    # 检查是否已经添加过调试代码
    if any("DEBUG_DSA_COMPRESS" in line for line in lines):
        print("调试代码已存在，无需重复添加")
        return
    
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)
        
        # 在 build_prefill_metadata 函数开始处添加调试代码
        if i == 535 and "def build_prefill_metadata(" in line:
            indent = "    "
            debug_code = f"""{indent}# DEBUG: Check metadata at function entry
{indent}import os
{indent}if os.getenv("DEBUG_DSA_COMPRESS") == "1":
{indent}    print(f"[DEBUG_DSA] build_prefill_metadata ENTRY:")
{indent}    print(f"  self.num_actual_tokens: {{self.num_actual_tokens}}")
{indent}    print(f"  self.num_prefill_tokens: {{self.num_prefill_tokens}}")
{indent}    print(f"  self.num_prefills: {{self.num_prefills}}")
{indent}    print(f"  self.num_decodes: {{self.num_decodes}}")
{indent}    print(f"  self.layer_compressor_ratio: {{self.layer_compressor_ratio}}")
{indent}    print(f"  self.compressor_ratio: {{self.compressor_ratio}}")
{indent}    try:
{indent}        import torch_npu
{indent}        torch_npu.npu.synchronize()
{indent}        print(f"  Stream sync: OK at entry")
{indent}    except Exception as e:
{indent}        print(f"  Stream sync FAILED at entry: {{e}}")

"""
            new_lines.append(debug_code)
        
        # 在 _get_padded_compressed_position 函数开始处添加调试代码
        elif i == 582 and "def _get_padded_compressed_position(prefill_input_positions," in line:
            indent = "        "
            debug_code = f"""{indent}# DEBUG: Check input before processing
{indent}import os
{indent}if os.getenv("DEBUG_DSA_COMPRESS") == "1":
{indent}    print(f"[DEBUG_DSA] _get_padded_compressed_position:")
{indent}    print(f"  compress_ratio: {{compress_ratio}}")
{indent}    print(f"  prefill_input_positions shape: {{prefill_input_positions.shape}}")
{indent}    print(f"  prefill_input_positions dtype: {{prefill_input_positions.dtype}}")
{indent}    print(f"  prefill_input_positions device: {{prefill_input_positions.device}}")
{indent}    try:
{indent}        print(f"  prefill_input_positions min/max: {{prefill_input_positions.min().item()}}/{{prefill_input_positions.max().item()}}")
{indent}        print(f"  prefill_input_positions first 20: {{prefill_input_positions[:20].tolist()}}")
{indent}        print(f"  prefill_input_positions last 20: {{prefill_input_positions[-20:].tolist()}}")
{indent}    except Exception as e:
{indent}        print(f"  ERROR accessing positions: {{e}}")
{indent}    try:
{indent}        import torch_npu
{indent}        torch_npu.npu.synchronize()
{indent}        print(f"  Stream sync: OK before mask calculation")
{indent}    except Exception as e:
{indent}        print(f"  Stream sync FAILED before mask: {{e}})

"""
            new_lines.append(debug_code)
        
        # 在 mask 计算后添加调试代码
        elif "mask = ((prefill_input_positions + 1) % compress_ratio) == 0" in line:
            indent = "        "
            debug_code = f"""{indent}if os.getenv("DEBUG_DSA_COMPRESS") == "1":
{indent}    try:
{indent}        print(f"  mask calculated, sum={{mask.sum().item()}}, shape={{mask.shape}}")
{indent}        print(f"  mask first 20: {{mask[:20].tolist()}}")
{indent}    except Exception as e:
{indent}        print(f"  ERROR accessing mask: {{e}}")

"""
            new_lines.append(debug_code)
        
        # 在 input_positions = prefill_input_positions[mask] 后添加调试代码
        elif "input_positions = prefill_input_positions[mask]" in line and "compress_ratio" not in line:
            indent = "        "
            debug_code = f"""{indent}if os.getenv("DEBUG_DSA_COMPRESS") == "1":
{indent}    try:
{indent}        print(f"  input_positions after mask: shape={{input_positions.shape}}")
{indent}        print(f"  input_positions first 20: {{input_positions[:20].tolist()}}")
{indent}    except Exception as e:
{indent}        print(f"  ERROR accessing input_positions: {{e}}")

"""
            new_lines.append(debug_code)
        
        # 在 pad_positions 返回前添加调试代码
        elif "pad_positions = F.pad(input_positions" in line and "value=0.0)" in line:
            indent = "        "
            debug_code = f"""{indent}if os.getenv("DEBUG_DSA_COMPRESS") == "1":
{indent}    print(f"  Returning: shape={{pad_positions.shape}}, pad_right={{pad_right}}")

"""
            new_lines.append(debug_code)
        
        i += 1
    
    # 写回文件
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    
    print("调试代码已成功添加到 dsa_v1.py")
    print("使用方法:")
    print("  export DEBUG_DSA_COMPRESS=1")
    print("  bash examples/disaggregated_prefill_v1/run_laps_sharegpt_benchmark_deepseek_v4_single_node_prefill.sh")

if __name__ == "__main__":
    main()
