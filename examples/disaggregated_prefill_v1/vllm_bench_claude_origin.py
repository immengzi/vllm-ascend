#!/usr/bin/env python3
"""
vLLM Benchmark using Claude-captured requests

直接使用 litellm_stats 捕获的原始请求数据测试 vLLM 性能。
支持完整的 OpenAI 请求格式：messages、tools、system prompt 等。
使用流式请求获取详细指标：TTFT、TPOT、ITL（对齐 vLLM 官方 benchmark）

用法:
    # 1. 先启动 vLLM 服务
    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8077

    # 2. 测试目录下所有 JSON 文件（多线程，每个文件一个线程）
    python vllm_bench_claude.py -e http://127.0.0.1:8077 --model glm4.7 --log-dir logs/ -m qwen3-next

    # 3. 测试单个 JSON 文件
    python vllm_bench_claude.py -e http://127.0.0.1:8077 --model glm4.7 -f logs/20260301.json

参数说明:
    -e, --endpoint     vLLM 服务地址 (默认: http://localhost:8000)
    --model            目标模型名 (必填)
    --log-dir          litellm_stats 日志目录 (默认: logs)
    -f, --file         单个 JSON 文件测试
    -m, --filter-model 按原始模型名过滤 (子串匹配)
    -v, --verbose      流式输出模型响应
    --max-tokens       最大输出 token 数 (默认: 32768)
    --timeout          请求超时秒数 (默认: 120)
    --output-dir       结果保存目录 (默认: benchmark_results)

指标说明:
    - TTFT (Time To First Token): 首个 token 延迟
    - TPOT (Time Per Output Token): 每个输出 token 的平均时间
    - ITL (Inter-Token Latency): token 间延迟
    - E2EL (End-to-End Latency): 端到端延迟
"""

import json
import argparse
import time
import requests
import statistics
import signal
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# 全局变量用于信号处理
_interrupted = False


def percentile(data: List[float], p: float) -> float:
    """计算百分位数"""
    if not data:
        return 0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


class BenchmarkRunner:
    def __init__(
        self,
        endpoint: str,
        model: str,
        log_dir: str = "logs",
        output_dir: str = "benchmark_results",
        max_tokens: int = 32768,
        timeout: int = 120,
        max_workers: int = 1,
        verbose: bool = False,
        model_filter: str = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_workers = max_workers
        self.verbose = verbose
        self.model_filter = model_filter
        self.results = []
        self.lock = threading.Lock()
        self.interrupted = False
        self.max_filename_len = 0

    def load_all_requests(self, single_file: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        从 logs 目录或单个文件加载所有原始请求

        Returns:
            Dict[str, List]: 按文件名分组的请求列表
        """
        requests_by_file = {}

        if single_file:
            files_to_process = [single_file]
        else:
            files_to_process = list(self.log_dir.rglob("*.json"))

        for json_file in files_to_process:
            print(f"Loading: {json_file}")

            with open(json_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"  Warning: Invalid JSON, skipping")
                    continue

            reqs = data.get("reqs", [])
            if not reqs:
                continue

            # 按时间排序
            reqs.sort(key=lambda x: x.get("timestamp", ""))

            filename = json_file.stem
            file_requests = []

            for idx, req in enumerate(reqs):
                request_data = req.get("request", {})
                messages = request_data.get("messages", [])

                if not messages:
                    continue

                # 获取原始请求的模型名称
                original_model = request_data.get("model", "unknown")

                # 如果指定了模型过滤器，检查是否匹配
                if self.model_filter and self.model_filter.lower() not in original_model.lower():
                    continue

                # 构建完整的请求（保留原始格式）
                request_payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "stream": True,  # 使用流式请求获取详细指标
                    "stream_options": {"include_usage": True},  # 确保返回usage信息
                }

                # 保留其他参数
                for key in ["temperature", "top_p", "tools", "tool_choice",
                           "frequency_penalty", "presence_penalty", "seed"]:
                    if key in request_data:
                        request_payload[key] = request_data[key]

                # 计算上下文信息
                context_turns = sum(1 for m in messages if m.get("role") == "user")
                context_chars = sum(
                    len(m.get("content", "")) if isinstance(m.get("content"), str) else 0
                    for m in messages
                )

                file_requests.append({
                    "id": f"{filename}_req_{idx + 1}",
                    "source_file": str(json_file),
                    "original_model": original_model,
                    "payload": request_payload,
                    "context_turns": context_turns,
                    "context_chars": context_chars,
                    "num_messages": len(messages),
                    "has_tools": "tools" in request_data,
                })

            if file_requests:
                requests_by_file[filename] = file_requests
                # 记录最长文件名
                if len(filename) > self.max_filename_len:
                    self.max_filename_len = len(filename)

        total_requests = sum(len(reqs) for reqs in requests_by_file.values())
        print(f"\nLoaded {len(requests_by_file)} files, {total_requests} total requests")
        return requests_by_file

    def process_file(self, filename: str, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理单个文件的所有请求（串行）"""
        results = []
        for i, req in enumerate(requests, 1):
            if self.interrupted:
                break
            result = self.send_request_stream(req)
            results.append(result)
            self._print_progress(result, f"{filename}[{i}/{len(requests)}]")
        return results

    def send_request_stream(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """发送流式请求并记录详细指标"""
        req_id = req["id"]
        payload = req["payload"]
        original_model = req["original_model"]

        start_time = time.perf_counter()  # 使用 perf_counter 获得更精确的时间
        first_token_time = None  # 第一个有 content 的 token 时间 (对齐 vLLM)
        most_recent_token_time = None  # 上一个有 content 的 token 时间 (用于计算 ITL)
        token_times = []  # 每个 chunk 到达的时间
        output_tokens = 0
        input_tokens = 0
        response_text = ""
        interrupted = False

        # verbose 模式下打印请求头
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[{req_id}] Response:")
            print("-" * 60, end="", flush=True)

        try:
            with requests.post(
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
                stream=True,
            ) as response:

                if response.status_code != 200:
                    end_time = time.perf_counter()
                    if self.verbose:
                        print(f"\n[ERROR] HTTP {response.status_code}")
                    return {
                        "id": req_id,
                        "source_file": req["source_file"],
                        "original_model": original_model,
                        "context_turns": req["context_turns"],
                        "context_chars": req["context_chars"],
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text[:200]}",
                    }

                # 解析 SSE 流，支持中断
                for line in response.iter_lines():
                    # 检查是否被中断
                    if self.interrupted:
                        interrupted = True
                        break

                    if not line:
                        continue

                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])

                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")

                                # 只统计有 content 的 token (对齐 vLLM)
                                if content is not None:
                                    timestamp = time.perf_counter()
                                    token_times.append(timestamp)

                                    # 第一个有 content 的 token: 计算 TTFT
                                    if first_token_time is None:
                                        first_token_time = timestamp
                                    else:
                                        # 后续 token: 计算 ITL
                                        pass  # ITL 在后面统一计算

                                    response_text += content

                                    # verbose 模式下流式输出
                                    if self.verbose:
                                        print(content, end="", flush=True)

                                    most_recent_token_time = timestamp

                            # 获取 usage（在最后一个 chunk 中）
                            if "usage" in chunk:
                                usage = chunk["usage"]
                                input_tokens = usage.get("prompt_tokens", 0)
                                output_tokens = usage.get("completion_tokens", 0)
                        except (json.JSONDecodeError, KeyError, IndexError) as e:
                            continue

                if self.verbose:
                    print("\n" + "-" * 60)

            end_time = time.perf_counter()
            total_latency = end_time - start_time

            # 计算指标 (对齐 vLLM 的计算方式)
            # TTFT: 第一个有 content 的 token 到达时间
            ttft_ms = None
            if first_token_time is not None:
                ttft_ms = (first_token_time - start_time) * 1000

            tpot_ms = None
            itl_ms = None
            itl_list = []

            if token_times:
                # 如果没有从 usage 获取到 output_tokens，用 token_times 估算
                if output_tokens == 0:
                    output_tokens = len(token_times)

                # ITL: token 间延迟（从第二个 token 开始计算）
                if len(token_times) > 1:
                    for i in range(1, len(token_times)):
                        itl_list.append((token_times[i] - token_times[i-1]) * 1000)
                    itl_ms = statistics.mean(itl_list) if itl_list else None

                # TPOT: (总延迟 - TTFT) / (输出 token 数 - 1) (对齐 vLLM)
                # 注意：只统计有 content 的 token，所以使用 len(token_times)
                if output_tokens > 1 and ttft_ms is not None:
                    # 使用 vLLM 的公式: (总延迟 - TTFT) / (output_tokens - 1)
                    latency_minus_ttft = total_latency - (ttft_ms / 1000)
                    tpot_ms = latency_minus_ttft / (output_tokens - 1) * 1000

            return {
                "id": req_id,
                "source_file": req["source_file"],
                "original_model": original_model,
                "context_turns": req["context_turns"],
                "context_chars": req["context_chars"],
                "num_messages": req["num_messages"],
                "has_tools": req["has_tools"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "latency_seconds": round(total_latency, 3),
                "tokens_per_second": round(output_tokens / total_latency, 2) if total_latency > 0 and output_tokens > 0 else 0,
                "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
                "tpot_ms": round(tpot_ms, 2) if tpot_ms is not None else None,
                "itl_ms": round(itl_ms, 2) if itl_ms is not None else None,
                "itl_list": [round(x, 2) for x in itl_list] if itl_list else [],
                "success": True,
                "error": "Interrupted by user" if interrupted else None,
            }

        except requests.exceptions.Timeout:
            return {
                "id": req_id,
                "source_file": req["source_file"],
                "original_model": original_model,
                "context_turns": req["context_turns"],
                "success": False,
                "error": "Request timeout",
            }
        except Exception as e:
            return {
                "id": req_id,
                "source_file": req["source_file"],
                "original_model": original_model,
                "context_turns": req["context_turns"],
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}",
            }

    def run_benchmark(self, requests_by_file: Dict[str, List[Dict[str, Any]]]):
        """
        运行 benchmark

        每个文件一个线程，文件内请求串行执行
        """
        total_requests = sum(len(reqs) for reqs in requests_by_file.values())
        num_files = len(requests_by_file)

        print(f"\nRunning benchmark...")
        print(f"Files: {num_files}, Total requests: {total_requests}")
        print(f"Endpoint: {self.endpoint}")
        print(f"Model: {self.model}")
        print(f"Threads: {num_files} (1 per file)")
        print("Press Ctrl+C to stop and show current results")
        print("-" * 80)

        self.results = []

        if num_files == 1:
            # 单文件：直接串行执行，不启动线程
            filename = list(requests_by_file.keys())[0]
            requests = requests_by_file[filename]
            for i, req in enumerate(requests, 1):
                if self.interrupted:
                    print(f"\n\nInterrupted!")
                    break
                result = self.send_request_stream(req)
                self.results.append(result)
                self._print_progress(result, f"{filename}[{i}/{len(requests)}]")
        else:
            # 多文件：每个文件一个线程
            from concurrent.futures import FIRST_COMPLETED, wait

            with ThreadPoolExecutor(max_workers=num_files) as executor:
                futures = {}
                for filename, requests in requests_by_file.items():
                    future = executor.submit(self.process_file, filename, requests)
                    futures[future] = filename

                # 收集已完成的结果，支持中断
                while futures:
                    if self.interrupted:
                        # 中断时，等待所有正在运行的任务完成（最多等待当前请求）
                        # 并收集已完成的结果
                        print(f"\n\nInterrupted! Waiting for in-progress requests to finish...")
                        # 设置超时等待，让正在进行的请求有机会完成
                        done, not_done = wait(futures.keys(), timeout=5)
                        for future in done:
                            filename = futures[future]
                            try:
                                file_results = future.result()
                                self.results.extend(file_results)
                            except Exception as e:
                                print(f"Error processing {filename}: {e}")
                        # 取消未完成的任务
                        for future in not_done:
                            future.cancel()
                        break

                    # 等待任意一个 future 完成
                    done, _ = wait(futures.keys(), timeout=0.1, return_when=FIRST_COMPLETED)

                    for future in done:
                        filename = futures.pop(future)
                        try:
                            file_results = future.result()
                            self.results.extend(file_results)
                        except Exception as e:
                            print(f"Error processing {filename}: {e}")

        return self.results

    def set_interrupted(self):
        """设置中断标志"""
        self.interrupted = True

    def _print_progress(self, result: Dict, progress_str: str):
        """打印进度"""
        status = "✓" if result["success"] else "✗"

        if result["success"]:
            orig_model = result.get("original_model", "unknown")
            ttft = f"{result['ttft_ms']:.1f}" if result.get("ttft_ms") else "-"
            tpot = f"{result['tpot_ms']:.1f}" if result.get("tpot_ms") else "-"
            itl = f"{result['itl_ms']:.1f}" if result.get("itl_ms") else "-"
            tps = result['tokens_per_second']

            # 单行格式 - 使用动态文件名对齐
            print(
                f"[{progress_str:<{self.max_filename_len + 10}}] {status} "
                f"in={result['input_tokens']:>5} "
                f"out={result['output_tokens']:>4} "
                f"duration={result['latency_seconds']:>5.2f}s "
                f"TTFT={ttft:>7}ms "
                f"TPOT={tpot:>7}ms "
                f"ITL={itl:>7}ms "
                f"tok/s={tps:>5.1f}"
            )
        else:
            print(f"[{progress_str:<20}] {status} ERROR: {result['error'][:50]}")

    def print_summary(self):
        """打印统计摘要 (完全复刻 vLLM benchmark 输出格式)"""
        if not self.results:
            print("No results to summarize")
            return

        success_results = [r for r in self.results if r.get("success")]
        failed_results = [r for r in self.results if not r.get("success")]

        # 计算总耗时（使用成功请求的时间范围）
        if success_results:
            total_duration = sum(r["latency_seconds"] for r in success_results)
        else:
            total_duration = 0

        # 提取指标数据
        e2el_list = [r["latency_seconds"] * 1000 for r in success_results]  # 转换为 ms
        input_tokens = [r["input_tokens"] for r in success_results]
        output_tokens = [r["output_tokens"] for r in success_results]

        ttft_list = [r["ttft_ms"] for r in success_results if r.get("ttft_ms") is not None]
        tpot_list = [r["tpot_ms"] for r in success_results if r.get("tpot_ms") is not None]
        itl_list = [r["itl_ms"] for r in success_results if r.get("itl_ms") is not None]

        # 计算吞吐量指标
        total_input = sum(input_tokens)
        total_output = sum(output_tokens)
        request_throughput = len(success_results) / total_duration if total_duration > 0 else 0
        output_throughput = total_output / total_duration if total_duration > 0 else 0
        total_token_throughput = (total_input + total_output) / total_duration if total_duration > 0 else 0

        # vLLM 默认百分位数
        percentiles = [25, 50, 75, 90, 95, 99]

        def calc_metric_stats(data: List[float]) -> dict:
            """计算指标统计值"""
            if not data:
                return {"mean": 0, "median": 0, "std": 0, "percentiles": []}
            return {
                "mean": statistics.mean(data),
                "median": statistics.median(data),
                "std": statistics.stdev(data) if len(data) > 1 else 0,
                "percentiles": [(p, percentile(data, p)) for p in percentiles],
            }

        # 计算各指标统计值
        ttft_stats = calc_metric_stats(ttft_list)
        tpot_stats = calc_metric_stats(tpot_list)
        itl_stats = calc_metric_stats(itl_list)
        e2el_stats = calc_metric_stats(e2el_list)

        # ========== 完全复刻 vLLM 输出格式 ==========
        print()
        print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
        print("{:<40} {:<10}".format("Successful requests:", len(success_results)))
        print("{:<40} {:<10}".format("Failed requests:", len(failed_results)))
        print("{:<40} {:<10.2f}".format("Benchmark duration (s):", total_duration))
        print("{:<40} {:<10}".format("Total input tokens:", total_input))
        print("{:<40} {:<10}".format("Total generated tokens:", total_output))
        print(
            "{:<40} {:<10.2f}".format(
                "Request throughput (req/s):", request_throughput
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Output token throughput (tok/s):", output_throughput
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Total token throughput (tok/s):", total_token_throughput
            )
        )

        # 定义打印单个指标的函数 (简化版：Mean, Median, P99)
        def print_one_metric(
            metric_name: str,      # e.g., "TTFT"
            metric_header: str,    # e.g., "Time to First Token"
            stats: dict,
        ):
            print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
            print(
                "{:<40} {:<10.2f}".format(
                    f"Mean {metric_name} (ms):", stats["mean"]
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    f"Median {metric_name} (ms):", stats["median"]
                )
            )
            # 只打印 P99
            for p, value in stats["percentiles"]:
                if p == 99:
                    p_word = str(int(p)) if int(p) == p else str(p)
                    print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))

        # 打印各指标 (完全复刻 vLLM 顺序)
        print_one_metric("TTFT", "Time to First Token", ttft_stats)
        print_one_metric("TPOT", "Time per Output Token (excl. 1st token)", tpot_stats)
        print_one_metric("ITL", "Inter-token Latency", itl_stats)
        print_one_metric("E2EL", "End-to-end Latency", e2el_stats)

        # 额外信息（vLLM 没有但对我们有用）
        print()
        print("{s:{c}^{n}}".format(s=" Additional Statistics ", n=50, c="-"))
        print("{:<40} {:<10.1f}".format("Input tokens (avg):", statistics.mean(input_tokens) if input_tokens else 0))
        print("{:<40} {:<10}".format("Input tokens (min):", min(input_tokens) if input_tokens else 0))
        print("{:<40} {:<10}".format("Input tokens (max):", max(input_tokens) if input_tokens else 0))
        print("{:<40} {:<10.1f}".format("Output tokens (avg):", statistics.mean(output_tokens) if output_tokens else 0))
        print("{:<40} {:<10}".format("Output tokens (min):", min(output_tokens) if output_tokens else 0))
        print("{:<40} {:<10}".format("Output tokens (max):", max(output_tokens) if output_tokens else 0))

        # 按上下文轮数分组统计
        print()
        print("{s:{c}^{n}}".format(s=" By Context Turns ", n=50, c="-"))
        turns_groups = {}
        for r in success_results:
            t = r["context_turns"]
            if t not in turns_groups:
                turns_groups[t] = {"ttft": [], "latency": [], "input": []}
            if r.get("ttft_ms") is not None:
                turns_groups[t]["ttft"].append(r["ttft_ms"])
            turns_groups[t]["latency"].append(r["latency_seconds"])
            turns_groups[t]["input"].append(r["input_tokens"])

        for t in sorted(turns_groups.keys()):
            g = turns_groups[t]
            count = len(g["latency"])
            avg_lat = statistics.mean(g["latency"])
            avg_ttft = statistics.mean(g["ttft"]) if g["ttft"] else 0
            print(f"Turn {t}: {count} requests, latency={avg_lat:.2f}s, TTFT={avg_ttft:.1f}ms")

        # 按原始模型分组
        print()
        print("{s:{c}^{n}}".format(s=" By Original Model ", n=50, c="-"))
        model_groups = {}
        for r in success_results:
            m = r.get("original_model", "unknown")
            if m not in model_groups:
                model_groups[m] = {"count": 0, "latency": [], "ttft": []}
            model_groups[m]["count"] += 1
            model_groups[m]["latency"].append(r["latency_seconds"])
            if r.get("ttft_ms") is not None:
                model_groups[m]["ttft"].append(r["ttft_ms"])

        for m in sorted(model_groups.keys()):
            g = model_groups[m]
            avg_lat = statistics.mean(g["latency"])
            avg_ttft = statistics.mean(g["ttft"]) if g["ttft"] else 0
            # 截断过长的模型名
            display_model = m[:35] + ".." if len(m) > 35 else m
            print(f"{display_model}: {g['count']} requests, latency={avg_lat:.2f}s, TTFT={avg_ttft:.1f}ms")

    def save_results(self):
        """保存结果到文件 (对齐 vLLM benchmark 输出格式)"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"benchmark_{timestamp}.json"

        success_results = [r for r in self.results if r.get("success")]
        failed_results = [r for r in self.results if not r.get("success")]

        # 计算汇总指标 (对齐 vLLM)
        if success_results:
            total_duration = sum(r["latency_seconds"] for r in success_results)
            total_input = sum(r["input_tokens"] for r in success_results)
            total_output = sum(r["output_tokens"] for r in success_results)

            ttft_list = [r["ttft_ms"] for r in success_results if r.get("ttft_ms") is not None]
            tpot_list = [r["tpot_ms"] for r in success_results if r.get("tpot_ms") is not None]
            itl_list = [r["itl_ms"] for r in success_results if r.get("itl_ms") is not None]
            e2el_list = [r["latency_seconds"] for r in success_results]

            # 计算统计指标
            def calc_stats(data):
                if not data:
                    return {"mean": None, "std": None, "median": None, "p99": None}
                return {
                    "mean": round(statistics.mean(data), 2),
                    "std": round(statistics.stdev(data), 2) if len(data) > 1 else 0,
                    "median": round(statistics.median(data), 2),
                    "p99": round(percentile(data, 99), 2),
                }

            summary = {
                "duration": round(total_duration, 2),
                "completed": len(success_results),
                "failed": len(failed_results),
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "request_throughput": round(len(success_results) / total_duration, 2) if total_duration > 0 else 0,
                "output_throughput": round(total_output / total_duration, 2) if total_duration > 0 else 0,
                "total_token_throughput": round((total_input + total_output) / total_duration, 2) if total_duration > 0 else 0,
                "ttft_ms": calc_stats(ttft_list),
                "tpot_ms": calc_stats(tpot_list),
                "itl_ms": calc_stats(itl_list),
                "e2el_s": calc_stats(e2el_list),
            }
        else:
            summary = {
                "duration": 0,
                "completed": 0,
                "failed": len(failed_results),
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "request_throughput": 0,
                "output_throughput": 0,
                "total_token_throughput": 0,
                "ttft_ms": {},
                "tpot_ms": {},
                "itl_ms": {},
                "e2el_s": {},
            }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "endpoint": self.endpoint,
                    "model": self.model,
                    **summary,
                    "results": self.results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM using raw request data from litellm_stats"
    )
    parser.add_argument(
        "--endpoint", "-e",
        type=str,
        default="http://localhost:8000",
        help="vLLM endpoint (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Target model name to send to vLLM server",
    )
    parser.add_argument(
        "-m", "--filter-model",
        type=str,
        default=None,
        help="Filter requests by original model name (substring match)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory containing litellm_stats logs (default: logs)",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Single JSON file to benchmark (instead of scanning log-dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results (default: benchmark_results)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Max output tokens (default: 32768)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Stream model output to terminal during benchmark",
    )

    args = parser.parse_args()

    # 检查是使用单个文件还是目录
    if args.file:
        single_file = Path(args.file)
        if not single_file.exists():
            print(f"Error: File not found: {single_file}")
            return
        log_dir = single_file.parent
        use_single_file = single_file
    else:
        log_dir = Path(args.log_dir)
        if not log_dir.exists():
            print(f"Error: Log directory not found: {log_dir}")
            print("Please run litellm_stats to capture some requests first.")
            return
        use_single_file = None

    # 运行 benchmark
    runner = BenchmarkRunner(
        endpoint=args.endpoint,
        model=args.model,
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_workers=1,  # 保留参数兼容
        verbose=args.verbose,
        model_filter=args.filter_model,
    )

    # 设置信号处理器
    def signal_handler(signum, frame):
        print("\n\nReceived Ctrl+C, stopping benchmark...")
        runner.set_interrupted()

    signal.signal(signal.SIGINT, signal_handler)

    if args.filter_model:
        print(f"Filter: only testing requests with model containing '{args.filter_model}'")

    requests_by_file = runner.load_all_requests(single_file=use_single_file)
    if not requests_by_file:
        print("No requests found")
        return

    runner.run_benchmark(requests_by_file)
    runner.print_summary()
    runner.save_results()


if __name__ == "__main__":
    main()