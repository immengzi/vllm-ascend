#!/usr/bin/env python3
"""
为 dsa_v1.py 添加调试代码 - 修复版

基于远程仓库 GDzhu01/vllm-ascend-deepseekv4 分支 v4_v0.18.0_0412
"""

import sys
import re

def main():
    file_path = "/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v1.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        print("请确保在远程机器上运行此脚本")
        sys.exit(1)
    
    # 检查是否已经添加过调试代码
    if "DEBUG_DSA_COMPRESS" in content:
        print("调试代码已存在，无需重复添加")
        print("如需移除调试代码，请执行: git restore vllm_ascend/attention/dsa_v1.py")
        return
    
    # 在文件开头添加 import os（如果没有）
    if "import os" not in content[:1000]:
        # 在第一个 import 语句后添加
        content = re.sub(
            r'(import math\n)',
            r'\1import os\n',
            content,
            count=1
        )
        print("已添加 import os")
    
    # 在 build_prefill_metadata 函数开始处添加调试代码
    # 找到 "def build_prefill_metadata(" 后的第一个具体行
    pattern1 = r'(    def build_prefill_metadata\(\s*self,.*?\):.*?common_attn_metadata.*?\):.*?)'
    replacement1 = r'''\1
        # DEBUG: Check metadata at function entry
        if __debug__ and os.getenv("DEBUG_DSA_COMPRESS") == "1":
            print(f"[DEBUG_DSA] build_prefill_metadata ENTRY:")
            print(f"  self.num_actual_tokens: {self.num_actual_tokens}")
            print(f"  self.num_prefill_tokens: {self.num_prefill_tokens}")
            print(f"  self.num_prefills: {self.num_prefills}")
            print(f"  self.num_decodes: {self.num_decodes}")
            print(f"  self.layer_compressor_ratio: {self.layer_compressor_ratio}")
            print(f"  self.compressor_ratio: {self.compressor_ratio}")
            try:
                import torch_npu
                torch_npu.npu.synchronize()
                print(f"  Stream sync: OK at entry")
            except Exception as e:
                print(f"  Stream sync FAILED at entry: {e}")
'''
    content = re.sub(pattern1, replacement1, content, flags=re.DOTALL)
    print("已添加 build_prefill_metadata 入口调试代码")
    
    # 在 _get_padded_compressed_position (prefill 版本) 添加调试代码
    # 这是 build_prefill_metadata 内部的函数
    pattern2 = r'(        def _get_padded_compressed_position\(prefill_input_positions,\s+compress_ratio\):)'
    replacement2 = r'''\1
            # DEBUG: Check input before processing
            if __debug__ and os.getenv("DEBUG_DSA_COMPRESS") == "1":
                print(f"[DEBUG_DSA] _get_padded_compressed_position (prefill):")
                print(f"  compress_ratio: {compress_ratio}")
                print(f"  prefill_input_positions shape: {prefill_input_positions.shape}")
                print(f"  prefill_input_positions dtype: {prefill_input_positions.dtype}")
                print(f"  prefill_input_positions device: {prefill_input_positions.device}")
                try:
                    min_val = prefill_input_positions.min().item()
                    max_val = prefill_input_positions.max().item()
                    print(f"  prefill_input_positions min/max: {min_val}/{max_val}")
                    if prefill_input_positions.shape[0] >= 20:
                        print(f"  prefill_input_positions first 20: {prefill_input_positions[:20].tolist()}")
                        print(f"  prefill_input_positions last 20: {prefill_input_positions[-20:].tolist()}")
                except Exception as e:
                    print(f"  ERROR accessing positions: {e}")
                try:
                    import torch_npu
                    torch_npu.npu.synchronize()
                    print(f"  Stream sync: OK before mask calculation")
                except Exception as e:
                    print(f"  Stream sync FAILED before mask: {e}")
'''
    content = re.sub(pattern2, replacement2, content)
    print("已添加 _get_padded_compressed_position 调试代码")
    
    # 在 mask 计算后添加调试代码（prefill 版本）
    pattern3 = r'(            mask = \(\(prefill_input_positions \+ 1\) % compress_ratio\) == 0)'
    replacement3 = r'''\1
            
            if __debug__ and os.getenv("DEBUG_DSA_COMPRESS") == "1":
                try:
                    mask_sum = mask.sum().item()
                    print(f"  mask calculated, sum={mask_sum}, shape={mask.shape}")
                    if mask.shape[0] >= 20:
                        print(f"  mask first 20: {mask[:20].tolist()}")
                except Exception as e:
                    print(f"  ERROR accessing mask: {e}")'''
    content = re.sub(pattern3, replacement3, content)
    print("已添加 mask 调试代码")
    
    # 在 input_positions = prefill_input_positions[mask] 后添加调试代码
    pattern4 = r'(            input_positions = prefill_input_positions\[mask\]\n)(            input_positions = \(input_positions \+ 1\) - compress_ratio)'
    replacement4 = r'''\1
            if __debug__ and os.getenv("DEBUG_DSA_COMPRESS") == "1":
                try:
                    print(f"  input_positions after mask: shape={input_positions.shape}")
                    if input_positions.shape[0] >= 20:
                        print(f"  input_positions first 20: {input_positions[:20].tolist()}")
                except Exception as e:
                    print(f"  ERROR accessing input_positions: {e}")
        \2'''
    content = re.sub(pattern4, replacement4)
    print("已添加 mask 后调试代码")
    
    # 在 pad_positions 返回前添加调试代码
    pattern5 = r'(            pad_positions = F\.pad\(input_positions, \(0, pad_right\), value=0\.0)\n)(            return pad_positions)'
    replacement5 = r'''\1
            if __debug__ and os.getenv("DEBUG_DSA_COMPRESS") == "1":
                print(f"  Returning: shape={pad_positions.shape}, pad_right={pad_right}")
        \2'''
    content = re.sub(pattern5, replacement5)
    print("已添加返回前调试代码")
    
    # 写回文件
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("\n调试代码添加完成！")
    print("\n使用方法:")
    print("  export DEBUG_DSA_COMPRESS=1")
    print("  bash examples/disaggregated_prefill_v1/run_laps_sharegpt_benchmark_deepseek_v4_single_node_prefill.sh")
    print("\n移除调试代码:")
    print("  git restore vllm_ascend/attention/dsa_v1.py")

if __name__ == "__main__":
    main()
