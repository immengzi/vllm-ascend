#!/usr/bin/env python3
"""
分析 vllm_bench_claude logs 目录下所有 JSON 文件的数据分布
生成 prompt_tokens / completion_tokens / total_tokens 的统计信息和分布图
"""

import json
import os
import sys
import math
import argparse
from collections import Counter

LOG_DIR = "logs"
DEFAULT_FIRST_N = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="分析 vllm_bench_claude logs 目录下 JSON 请求分布。",
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        default=LOG_DIR,
        help=f"日志目录，默认 {LOG_DIR}",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=DEFAULT_FIRST_N,
        help="每个文件只分析按时间顺序的前 N 条请求；传 0 表示全量。默认 4。",
    )
    return parser.parse_args()


def load_reqs(log_dir, first_n):
    """加载每个 JSON 文件中的 reqs，可选仅取按时间顺序前 N 条。"""
    all_reqs = []
    file_stats = []
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(log_dir, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
            reqs = data.get("reqs", [])
            if not isinstance(reqs, list):
                reqs = []
            reqs = sorted(reqs, key=lambda x: x.get("timestamp", ""))
            selected = reqs[:first_n] if first_n > 0 else reqs
            all_reqs.extend(selected)
            file_stats.append(
                {
                    "file": fname,
                    "total_reqs": len(reqs),
                    "selected_reqs": len(selected),
                }
            )
        except Exception as e:
            print(f"[WARN] 跳过 {fname}: {e}")
    return all_reqs, file_stats

def extract_token_info(reqs):
    """从每条 req 中提取 token 信息"""
    records = []
    for r in reqs:
        resp = r.get("response", {})
        usage = resp.get("usage", None)
        if usage is None:
            continue
        # usage 可能是 dict 或对象的 str 表示，尝试两种方式
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
        elif hasattr(usage, "prompt_tokens"):
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
        else:
            # 尝试从字符串解析
            s = str(usage)
            import re
            m_p = re.search(r"prompt_tokens=(\d+)", s)
            m_c = re.search(r"completion_tokens=(\d+)", s)
            m_t = re.search(r"total_tokens=(\d+)", s)
            prompt_tokens = int(m_p.group(1)) if m_p else 0
            completion_tokens = int(m_c.group(1)) if m_c else 0
            total_tokens = int(m_t.group(1)) if m_t else 0

        # 消息轮数
        msgs = r.get("request", {}).get("messages", [])
        num_turns = len(msgs)

        # 是否有 tools
        has_tools = bool(r.get("request", {}).get("tools"))

        # model
        model = r.get("request", {}).get("model", "unknown")

        records.append({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "num_turns": num_turns,
            "has_tools": has_tools,
            "model": model,
        })
    return records

def percentile(sorted_vals, p):
    """计算百分位数"""
    if not sorted_vals:
        return 0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[f]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])

def stats_summary(values, label):
    """打印统计摘要"""
    if not values:
        print(f"  {label}: 无数据")
        return
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    median = percentile(s, 50)
    p5 = percentile(s, 5)
    p25 = percentile(s, 25)
    p75 = percentile(s, 75)
    p90 = percentile(s, 90)
    p95 = percentile(s, 95)
    p99 = percentile(s, 99)
    variance = sum((x - mean) ** 2 for x in s) / n
    std = math.sqrt(variance)

    print(f"\n  {'='*55}")
    print(f"  {label} (n={n})")
    print(f"  {'='*55}")
    print(f"  Min:    {s[0]:>10,.0f}")
    print(f"  P5:     {p5:>10,.0f}")
    print(f"  P25:    {p25:>10,.0f}")
    print(f"  Median: {median:>10,.0f}")
    print(f"  Mean:   {mean:>10,.1f}")
    print(f"  P75:    {p75:>10,.0f}")
    print(f"  P90:    {p90:>10,.0f}")
    print(f"  P95:    {p95:>10,.0f}")
    print(f"  P99:    {p99:>10,.0f}")
    print(f"  Max:    {s[-1]:>10,.0f}")
    print(f"  Std:    {std:>10,.1f}")

def ascii_histogram(values, label, bins=20, width=50):
    """打印 ASCII 直方图"""
    if not values:
        return
    mn, mx = min(values), max(values)
    if mn == mx:
        print(f"\n  {label}: 所有值相同 = {mn}")
        return

    bin_width = (mx - mn) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - mn) / bin_width), bins - 1)
        counts[idx] += 1

    max_count = max(counts)
    print(f"\n  {label} 分布直方图")
    print(f"  {'─'*70}")
    for i in range(bins):
        lo = mn + i * bin_width
        hi = lo + bin_width
        bar_len = int(counts[i] / max_count * width) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"  {lo:>8,.0f}-{hi:>8,.0f} | {bar} {counts[i]}")
    print(f"  {'─'*70}")

def bucket_distribution(values, label, buckets):
    """按自定义区间统计分布"""
    if not values:
        return
    counts = Counter()
    for v in values:
        for i, (lo, hi) in enumerate(buckets):
            if lo <= v < hi:
                counts[f"{lo:,}-{hi:,}"] = counts.get(f"{lo:,}-{hi:,}", 0) + 1
                break
        else:
            counts[f">={buckets[-1][1]:,}"] = counts.get(f">={buckets[-1][1]:,}", 0) + 1

    total = len(values)
    print(f"\n  {label} 区间分布")
    print(f"  {'─'*55}")
    for lo, hi in buckets:
        key = f"{lo:,}-{hi:,}"
        c = counts.get(key, 0)
        pct = c / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {key:>15s} : {c:>5d} ({pct:5.1f}%) {bar}")
    key = f">={buckets[-1][1]:,}"
    c = counts.get(key, 0)
    pct = c / total * 100
    bar = "█" * int(pct / 2)
    print(f"  {key:>15s} : {c:>5d} ({pct:5.1f}%) {bar}")
    print(f"  {'─'*55}")


def main():
    args = parse_args()
    log_dir = args.log_dir
    first_n = args.first_n

    print(f"📁 扫描目录: {log_dir}")
    reqs, file_stats = load_reqs(log_dir, first_n)
    print(f"📄 文件数: {len(file_stats)}")
    if first_n > 0:
        print(f"📊 模式: 每文件前 {first_n} 条请求")
    else:
        print("📊 模式: 全量请求")
    print(f"📊 纳入分析的请求数: {len(reqs)}")
    print(f"📊 原始总请求数: {sum(item['total_reqs'] for item in file_stats)}")
    print(f"📊 截取后总请求数: {sum(item['selected_reqs'] for item in file_stats)}")

    records = extract_token_info(reqs)
    print(f"✅ 有效 usage 记录数: {len(records)}")

    if not records:
        print("没有可分析的数据")
        return

    # ── 基本统计 ──
    prompt_tokens = [r["prompt_tokens"] for r in records]
    completion_tokens = [r["completion_tokens"] for r in records]
    total_tokens = [r["total_tokens"] for r in records]
    num_turns = [r["num_turns"] for r in records]

    print("\n" + "=" * 60)
    title = f" 每文件前 {first_n} 条请求的 TOKEN 分布统计" if first_n > 0 else " TOKEN 分布统计"
    print(title)
    print("=" * 60)

    stats_summary(prompt_tokens, "Prompt Tokens")
    stats_summary(completion_tokens, "Completion Tokens")
    stats_summary(total_tokens, "Total Tokens")
    stats_summary(num_turns, "消息轮数 (messages count)")

    # ── 直方图 ──
    print("\n" + "=" * 60)
    print(" 直方图")
    print("=" * 60)

    ascii_histogram(prompt_tokens, "Prompt Tokens", bins=15)
    ascii_histogram(completion_tokens, "Completion Tokens", bins=15)

    # ── 自定义区间分布 ──
    print("\n" + "=" * 60)
    print(" 区间分布")
    print("=" * 60)

    prompt_buckets = [
        (0, 1000), (1000, 2000), (2000, 4000), (4000, 8000),
        (8000, 16000), (16000, 32000), (32000, 64000), (64000, 128000),
    ]
    completion_buckets = [
        (0, 50), (50, 100), (100, 200), (200, 500),
        (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000),
    ]

    bucket_distribution(prompt_tokens, "Prompt Tokens", prompt_buckets)
    bucket_distribution(completion_tokens, "Completion Tokens", completion_buckets)

    # ── Model 分布 ──
    print("\n" + "=" * 60)
    print(" Model 分布")
    print("=" * 60)
    model_counts = Counter(r["model"] for r in records)
    for model, cnt in model_counts.most_common():
        print(f"  {model:>30s} : {cnt:>5d} ({cnt/len(records)*100:.1f}%)")

    # ── Tool 使用 ──
    tool_count = sum(1 for r in records if r["has_tools"])
    print(f"\n  带 tools 的请求: {tool_count}/{len(records)} ({tool_count/len(records)*100:.1f}%)")

    # ── 每个文件的请求数分布 ──
    print("\n" + "=" * 60)
    print(" 每文件请求数")
    print("=" * 60)
    file_total_req_counts = [item["total_reqs"] for item in file_stats]
    file_selected_req_counts = [item["selected_reqs"] for item in file_stats]
    stats_summary(file_total_req_counts, "每文件原始请求数")
    stats_summary(file_selected_req_counts, "每文件纳入分析请求数")

    print("\n✅ 分析完成")


if __name__ == "__main__":
    main()
