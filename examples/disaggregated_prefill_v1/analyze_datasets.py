#!/usr/bin/env python3
"""分析 ShareGPT 和 Claude trace 数据集分布"""
import json
import sys
from pathlib import Path
from collections import Counter
from typing import List, Dict
from transformers import AutoTokenizer


def analyze_sharegpt(filepath: str, tokenizer, sample_size: int = None) -> Dict:
    """分析 ShareGPT 数据集"""
    print(f"\n{'='*60}")
    print(f"分析 ShareGPT: {filepath}")
    print(f"{'='*60}")

    with open(filepath) as f:
        data = json.load(f)

    total_convos = len(data)
    if sample_size and sample_size < total_convos:
        print(f"采样分析: {sample_size} / {total_convos} 个对话")
        import random
        data = random.sample(data, sample_size)
        total_convos = sample_size

    turn_counts = []
    input_lengths = []
    output_lengths = []

    for idx, item in enumerate(data):
        if (idx + 1) % 1000 == 0:
            print(f"进度: {idx + 1}/{total_convos} ({(idx+1)/total_convos*100:.1f}%)")

        conv = item.get('conversations', [])
        # 计算轮数（user 消息数）
        turns = len([m for m in conv if m.get('from') == 'human'])
        turn_counts.append(turns)

        # 累积计算每轮的输入输出长度
        cum_input_text = []
        for i, msg in enumerate(conv):
            content = msg.get('value', '')
            if msg.get('from') == 'human':
                cum_input_text.append(content)
                # 当前轮的 input = 累积的历史 + 当前问题
                input_text = '\n'.join(cum_input_text)
                input_tokens = len(tokenizer(input_text, add_special_tokens=False).input_ids)
                input_lengths.append(input_tokens)
            elif msg.get('from') == 'gpt':
                output_tokens = len(tokenizer(content, add_special_tokens=False).input_ids)
                output_lengths.append(output_tokens)

    return {
        'dataset': 'ShareGPT',
        'total_conversations': total_convos,
        'turn_distribution': {
            'min': min(turn_counts),
            'max': max(turn_counts),
            'avg': sum(turn_counts) / len(turn_counts),
            'median': sorted(turn_counts)[len(turn_counts)//2],
        },
        'input_tokens': {
            'min': min(input_lengths) if input_lengths else 0,
            'max': max(input_lengths) if input_lengths else 0,
            'avg': sum(input_lengths) / len(input_lengths) if input_lengths else 0,
            'p50': sorted(input_lengths)[len(input_lengths)//2] if input_lengths else 0,
            'p90': sorted(input_lengths)[int(len(input_lengths)*0.9)] if input_lengths else 0,
            'p99': sorted(input_lengths)[int(len(input_lengths)*0.99)] if input_lengths else 0,
        },
        'output_tokens': {
            'min': min(output_lengths) if output_lengths else 0,
            'max': max(output_lengths) if output_lengths else 0,
            'avg': sum(output_lengths) / len(output_lengths) if output_lengths else 0,
            'p50': sorted(output_lengths)[len(output_lengths)//2] if output_lengths else 0,
            'p90': sorted(output_lengths)[int(len(output_lengths)*0.9)] if output_lengths else 0,
            'p99': sorted(output_lengths)[int(len(output_lengths)*0.99)] if output_lengths else 0,
        },
        'total_turns': sum(turn_counts),
    }


def analyze_claude_trace_dir(dirpath: str, tokenizer) -> Dict:
    """分析 Claude trace 目录（所有 json 文件）"""
    print(f"\n{'='*60}")
    print(f"分析 Claude trace 目录: {dirpath}")
    print(f"{'='*60}")

    import glob
    json_files = glob.glob(str(Path(dirpath) / "*.json"))
    print(f"找到 {len(json_files)} 个 JSON 文件")

    # 每个文件是一个 session
    sessions = {}
    all_reqs = []

    for filepath in json_files:
        try:
            with open(filepath) as f:
                data = json.load(f)
            reqs = data.get('reqs', [])
            # 用文件名（不含扩展名）作为 session_id
            session_id = Path(filepath).stem
            sessions[session_id] = reqs
            all_reqs.extend(reqs)
        except Exception as e:
            print(f"警告: 跳过 {filepath}: {e}")

    total_requests = len(all_reqs)

    # 按 timestamp 排序每个 session 的请求
    for session_id in sessions:
        sessions[session_id].sort(key=lambda x: x.get('timestamp', 0))

    turn_counts = [len(reqs) for reqs in sessions.values()]
    input_lengths = []
    output_lengths = []

    for idx, req in enumerate(all_reqs):
        if (idx + 1) % 1000 == 0:
            print(f"进度: {idx + 1}/{total_requests} ({(idx+1)/total_requests*100:.1f}%)")

        request_data = req.get('request', {})
        messages = request_data.get('messages', [])
        response = req.get('response', {})

        # 计算 input tokens（messages 内容）
        input_text = ''
        for msg in messages:
            content = msg.get('content', '')
            if isinstance(content, str):
                input_text += content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        input_text += part.get('text', '')

        input_tokens = len(tokenizer(input_text, add_special_tokens=False).input_ids)
        input_lengths.append(input_tokens)

        # 计算 output tokens（从 response body）
        if response:
            output_content = ''
            try:
                body = getattr(response, 'body', None) or response
                if hasattr(body, 'choices'):
                    choices = body.choices
                    if choices and hasattr(choices[0], 'message'):
                        output_content = choices[0].message.content or ''
                elif isinstance(response, dict):
                    # 尝试从 dict 中提取
                    choices = response.get('choices', [])
                    if choices and isinstance(choices[0], dict):
                        msg = choices[0].get('message', {})
                        if isinstance(msg, dict):
                            output_content = msg.get('content', '')

                if output_content:
                    output_tokens = len(tokenizer(output_content, add_special_tokens=False).input_ids)
                    output_lengths.append(output_tokens)
            except Exception as e:
                # 解析失败，跳过这个 output
                pass

    return {
        'dataset': 'Claude trace',
        'total_sessions': len(sessions),
        'total_requests': total_requests,
        'turn_distribution': {
            'min': min(turn_counts) if turn_counts else 0,
            'max': max(turn_counts) if turn_counts else 0,
            'avg': sum(turn_counts) / len(turn_counts) if turn_counts else 0,
            'median': sorted(turn_counts)[len(turn_counts)//2] if turn_counts else 0,
        },
        'input_tokens': {
            'min': min(input_lengths) if input_lengths else 0,
            'max': max(input_lengths) if input_lengths else 0,
            'avg': sum(input_lengths) / len(input_lengths) if input_lengths else 0,
            'p50': sorted(input_lengths)[len(input_lengths)//2] if input_lengths else 0,
            'p90': sorted(input_lengths)[int(len(input_lengths)*0.9)] if input_lengths else 0,
            'p99': sorted(input_lengths)[int(len(input_lengths)*0.99)] if input_lengths else 0,
        },
        'output_tokens': {
            'min': min(output_lengths) if output_lengths else 0,
            'max': max(output_lengths) if output_lengths else 0,
            'avg': sum(output_lengths) / len(output_lengths) if output_lengths else 0,
            'p50': sorted(output_lengths)[len(output_lengths)//2] if output_lengths else 0,
            'p90': sorted(output_lengths)[int(len(output_lengths)*0.9)] if output_lengths else 0,
            'p99': sorted(output_lengths)[int(len(output_lengths)*0.99)] if output_lengths else 0,
        },
        'total_turns': sum(turn_counts),
    }


def print_stats(stats: Dict):
    """打印统计结果"""
    print(f"\n数据集: {stats['dataset']}")
    print(f"{'-'*50}")

    if stats.get('total_conversations'):
        print(f"对话数: {stats['total_conversations']:,}")
    if stats.get('total_sessions'):
        print(f"Session 数: {stats['total_sessions']:,}")
    if stats.get('total_requests'):
        print(f"总请求数: {stats['total_requests']:,}")
    print(f"总轮数: {stats.get('total_turns', 0):,}")

    turns = stats.get('turn_distribution', {})
    print(f"\n对话轮数分布:")
    print(f"  最小: {turns.get('min', 0)}")
    print(f"  最大: {turns.get('max', 0)}")
    print(f"  平均: {turns.get('avg', 0):.1f}")
    print(f"  中位数: {turns.get('median', 0)}")

    inp = stats.get('input_tokens', {})
    print(f"\n输入 tokens 分布:")
    print(f"  最小: {inp.get('min', 0):.0f}")
    print(f"  最大: {inp.get('max', 0):.0f}")
    print(f"  平均: {inp.get('avg', 0):.0f}")
    print(f"  P50: {inp.get('p50', 0):.0f}")
    print(f"  P90: {inp.get('p90', 0):.0f}")
    print(f"  P99: {inp.get('p99', 0):.0f}")

    out = stats.get('output_tokens', {})
    if out.get('avg'):
        print(f"\n输出 tokens 分布:")
        print(f"  最小: {out.get('min', 0):.0f}")
        print(f"  最大: {out.get('max', 0):.0f}")
        print(f"  平均: {out.get('avg', 0):.0f}")
        print(f"  P50: {out.get('p50', 0):.0f}")
        print(f"  P90: {out.get('p90', 0):.0f}")
        print(f"  P99: {out.get('p99', 0):.0f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='分析数据集分布')
    parser.add_argument('--sample', type=int, default=None,
                       help='ShareGPT 采样数量（默认分析全部）')
    parser.add_argument('--skip-sharegpt', action='store_true',
                       help='跳过 ShareGPT 分析，只分析 Claude trace')
    args = parser.parse_args()

    MODEL_PATH = '/workspace/models/DeepSeek-V4-Flash-w8a8-mtp'

    print(f"加载 tokenizer: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"Tokenizer loaded: {tokenizer.name_or_path}")

    claude_dir = '/vllm-workspace/vllm_bench_claude/logs'
    stats2 = analyze_claude_trace_dir(claude_dir, tokenizer)
    print_stats(stats2)

    if not args.skip_sharegpt:
        sharegpt_file = '/home/lzm/ShareGPT_V3_unfiltered_cleaned_split.json'
        stats1 = analyze_sharegpt(sharegpt_file, tokenizer, sample_size=args.sample)
        print_stats(stats1)
    else:
        print("\n跳过 ShareGPT 分析 (--skip-sharegpt)")
        # 使用之前的硬编码结果
        stats1 = {
            'dataset': 'ShareGPT',
            'total_conversations': 94145,
            'turn_distribution': {'avg': 3.5},
            'input_tokens': {'avg': 186, 'p50': 72, 'p90': 472, 'p99': 1540},
            'output_tokens': {'avg': 294, 'p50': 265, 'p90': 593, 'p99': 819},
            'total_turns': 333941,
        }

    print(f"\n{'='*60}")
    print("对比总结")
    print(f"{'='*60}")
    print(f"{'指标':<25} {'ShareGPT':>15} {'Claude trace':>15}")
    print(f"{'-'*55}")
    print(f"{'对话/Session数':<25} {stats1['total_conversations']:>15,} {stats2['total_sessions']:>15,}")
    print(f"{'总轮数':<25} {stats1['total_turns']:>15,} {stats2['total_turns']:>15,}")
    print(f"{'平均轮数/对话':<25} {stats1['turn_distribution']['avg']:>15.1f} {stats2['turn_distribution']['avg']:>15.1f}")
    print(f"{'输入 tokens (平均)':<25} {stats1['input_tokens']['avg']:>15.0f} {stats2['input_tokens']['avg']:>15.0f}")
    print(f"{'输入 tokens (P90)':<25} {stats1['input_tokens']['p90']:>15.0f} {stats2['input_tokens']['p90']:>15.0f}")
    print(f"{'输出 tokens (平均)':<25} {stats1['output_tokens']['avg']:>15.0f} {stats2['output_tokens']['avg']:>15.0f}")
