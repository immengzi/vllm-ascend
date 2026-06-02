#!/usr/bin/env python3
"""
为 dsa_v1.py 添加调试代码 - V2 (更简单可靠的版本)
"""

import sys

def main():
    file_path = "/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v1.py"
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        sys.exit(1)
    
    # 检查是否已经添加过调试代码
    if any("DEBUG_DSA_COMPRESS" in line for line in lines):
        print("调试代码已存在，无需重复添加")
        print("如需移除调试代码，请执行: git restore vllm_ascend/attention/dsa_v1.py")
        return
    
    # 检查并添加 import os
    has_import_os = any("import os" in line for line in lines[:50])
    if not has_import_os:
        # 在 import math 后添加
        new_lines = []
        for line in lines:
            new_lines.append(line)
            if line.strip() == "import math":
                new_lines.append("import os\n")
        lines = new_lines
        print("已添加 import os")
    
    # 添加调试代码
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 1. 在 build_prefill_metadata 函数开始处添加（第540行附近，query_start_loc = ...）
        if i == 539 and "query_start_loc = common_attn_metadata.query_start_loc" in line:
            indent = "        "
            new_lines.append(line)
            new_lines.append(f"\n{indent}# DEBUG: Check metadata at function entry\n")
            new_lines.append(f"{indent}if __debug__ and os.getenv(\"DEBUG_DSA_COMPRESS\") == \"1\":\n")
            new_lines.append(f"{indent}    print(f\"[DEBUG_DSA] build_prefill_metadata ENTRY:\")\n")
            new_lines.append(f"{indent}    print(f\"  self.num_actual_tokens: {{self.num_actual_tokens}}\")\n")
            new_lines.append(f"{indent}    print(f\"  self.num_prefill_tokens: {{self.num_prefill_tokens}}\")\n")
            new_lines.append(f"{indent}    print(f\"  self.num_prefills: {{self.num_prefills}}\")\n")
            new_lines.append(f"{indent}    print(f\"  self.num_decodes: {{self.num_decodes}}\")\n")
            new_lines.append(f"{indent}    print(f\"  self.layer_compressor_ratio: {{self.layer_compressor_ratio}}\")\n")
            new_lines.append(f"{indent}    print(f\"  self.compressor_ratio: {{self.compressor_ratio}}\")\n")
            new_lines.append(f"{indent}    try:\n")
            new_lines.append(f"{indent}        import torch_npu\n")
            new_lines.append(f"{indent}        torch_npu.npu.synchronize()\n")
            new_lines.append(f"{indent}        print(f\"  Stream sync: OK at entry\")\n")
            new_lines.append(f"{indent}    except Exception as e:\n")
            new_lines.append(f"{indent}        print(f\"  Stream sync FAILED at entry: {{e}}\")\n")
            i += 1
            continue
        
        # 2. 在 _get_padded_compressed_position 函数开始处添加（第586行，prefill版本）
        if i == 586 and "def _get_padded_compressed_position(prefill_input_positions," in line:
            indent = "            "
            new_lines.append(line)
            new_lines.append(f"\n{indent}# DEBUG: Check input before processing\n")
            new_lines.append(f"{indent}if __debug__ and os.getenv(\"DEBUG_DSA_COMPRESS\") == \"1\":\n")
            new_lines.append(f"{indent}    print(f\"[DEBUG_DSA] _get_padded_compressed_position (prefill):\")\n")
            new_lines.append(f"{indent}    print(f\"  compress_ratio: {{compress_ratio}}\")\n")
            new_lines.append(f"{indent}    print(f\"  prefill_input_positions shape: {{prefill_input_positions.shape}}\")\n")
            new_lines.append(f"{indent}    print(f\"  prefill_input_positions dtype: {{prefill_input_positions.dtype}}\")\n")
            new_lines.append(f"{indent}    print(f\"  prefill_input_positions device: {{prefill_input_positions.device}}\")\n")
            new_lines.append(f"{indent}    try:\n")
            new_lines.append(f"{indent}        min_val = prefill_input_positions.min().item()\n")
            new_lines.append(f"{indent}        max_val = prefill_input_positions.max().item()\n")
            new_lines.append(f"{indent}        print(f\"  prefill_input_positions min/max: {{min_val}}/{{max_val}}\")\n")
            new_lines.append(f"{indent}        if prefill_input_positions.shape[0] >= 20:\n")
            new_lines.append(f"{indent}            print(f\"  prefill_input_positions first 20: {{prefill_input_positions[:20].tolist()}}\")\n")
            new_lines.append(f"{indent}            print(f\"  prefill_input_positions last 20: {{prefill_input_positions[-20:].tolist()}}\")\n")
            new_lines.append(f"{indent}    except Exception as e:\n")
            new_lines.append(f"{indent}        print(f\"  ERROR accessing positions: {{e}}\")\n")
            new_lines.append(f"{indent}    try:\n")
            new_lines.append(f"{indent}        import torch_npu\n")
            new_lines.append(f"{indent}        torch_npu.npu.synchronize()\n")
            new_lines.append(f"{indent}        print(f\"  Stream sync: OK before mask calculation\")\n")
            new_lines.append(f"{indent}    except Exception as e:\n")
            new_lines.append(f"{indent}        print(f\"  Stream sync FAILED before mask: {{e}}\")\n")
            i += 1
            continue
        
        # 3. 在 mask 计算后添加调试代码
        if "mask = ((prefill_input_positions + 1) % compress_ratio) == 0" in line:
            new_lines.append(line)
            indent = "            "
            new_lines.append(f"\n{indent}if __debug__ and os.getenv(\"DEBUG_DSA_COMPRESS\") == \"1\":\n")
            new_lines.append(f"{indent}    try:\n")
            new_lines.append(f"{indent}        mask_sum = mask.sum().item()\n")
            new_lines.append(f"{indent}        print(f\"  mask calculated, sum={{mask_sum}}, shape={{mask.shape}}\")\n")
            new_lines.append(f"{indent}        if mask.shape[0] >= 20:\n")
            new_lines.append(f"{indent}            print(f\"  mask first 20: {{mask[:20].tolist()}}\")\n")
            new_lines.append(f"{indent}    except Exception as e:\n")
            new_lines.append(f"{indent}        print(f\"  ERROR accessing mask: {{e}}\")\n")
            i += 1
            continue
        
        # 4. 在 input_positions = prefill_input_positions[mask] 后添加
        if "input_positions = prefill_input_positions[mask]" in line and i < 650:
            new_lines.append(line)
            indent = "            "
            new_lines.append(f"\n{indent}if __debug__ and os.getenv(\"DEBUG_DSA_COMPRESS\") == \"1\":\n")
            new_lines.append(f"{indent}    try:\n")
            new_lines.append(f"{indent}        print(f\"  input_positions after mask: shape={{input_positions.shape}}\")\n")
            new_lines.append(f"{indent}        if input_positions.shape[0] >= 20:\n")
            new_lines.append(f"{indent}            print(f\"  input_positions first 20: {{input_positions[:20].tolist()}}\")\n")
            new_lines.append(f"{indent}    except Exception as e:\n")
            new_lines.append(f"{indent}        print(f\"  ERROR accessing input_positions: {{e}}\")\n")
            i += 1
            continue
        
        # 5. 在 pad_positions 返回前添加
        if "pad_positions = F.pad(input_positions" in line and "value=0.0)" in line and i < 650:
            new_lines.append(line)
            indent = "            "
            new_lines.append(f"\n{indent}if __debug__ and os.getenv(\"DEBUG_DSA_COMPRESS\") == \"1\":\n")
            new_lines.append(f"{indent}    print(f\"  Returning: shape={{pad_positions.shape}}, pad_right={{pad_right}}\")\n")
            i += 1
            continue
        
        new_lines.append(line)
        i += 1
    
    # 写回文件
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    
    print("\n调试代码添加完成！")
    print("\n使用方法:")
    print("  export DEBUG_DSA_COMPRESS=1")
    print("  bash examples/disaggregated_prefill_v1/run_laps_sharegpt_benchmark_deepseek_v4_single_node_prefill.sh")
    print("\n移除调试代码:")
    print("  git restore vllm_ascend/attention/dsa_v1.py")

if __name__ == "__main__":
    main()
