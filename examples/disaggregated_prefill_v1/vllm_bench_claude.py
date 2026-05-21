#!/usr/bin/env python3
"""vLLM benchmark for litellm_stats Claude traces."""

import argparse
import json
import statistics
import signal
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import threading

import requests
from transformers import AutoTokenizer


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
        session_count: int = 0,
        summary_label: str = "",
        tokenizer_path: Optional[str] = None,
        max_model_len: Optional[int] = None,
        over_limit_policy: str = "skip",
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
        self.session_count = session_count
        self.summary_label = summary_label
        self.tokenizer_path = tokenizer_path
        self.max_model_len = max_model_len
        self.over_limit_policy = over_limit_policy
        self.results = []
        self.lock = threading.Lock()
        self.interrupted = False
        self.max_filename_len = 0
        self._benchmark_start_time = None
        self._benchmark_end_time = None
        self.tokenizer = None
        self.skipped_over_limit = 0
        self.over_limit_requests: List[Dict[str, Any]] = []
        if self.tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True,
            )

    @staticmethod
    def _extract_request_body(req: Dict[str, Any]) -> Dict[str, Any]:
        body = req.get("request", req)
        return body if isinstance(body, dict) else {}

    @staticmethod
    def _is_stream_event_payload(chunk: Dict[str, Any]) -> bool:
        choices = chunk.get("choices") or []
        if not choices:
            return False
        delta = choices[0].get("delta") or {}
        if delta.get("content") is not None:
            return True
        if delta.get("tool_calls") is not None:
            return True
        if choices[0].get("finish_reason") is not None:
            return True
        return False

    @staticmethod
    def _summarize_file_entry(entry: Dict[str, Any]) -> str:
        return f"{entry['file']} ({entry['reason']})"

    def _build_request_payload(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        request_payload = {
            "model": self.model,
            "messages": request_data.get("messages", []),
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        for key in [
            "tools",
            "tool_choice",
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "response_format",
            "parallel_tool_calls",
            "extra_body",
        ]:
            if key in request_data:
                if key == "extra_body" and isinstance(request_data[key], dict):
                    request_payload.update(request_data[key])
                else:
                    request_payload[key] = request_data[key]

        return request_payload

    def _estimate_prompt_tokens(
        self,
        request_data: Dict[str, Any],
        request_payload: Dict[str, Any],
    ) -> tuple[Optional[int], bool, Optional[str]]:
        if self.tokenizer is None:
            return None, False, "tokenizer_unavailable"

        messages = request_payload.get("messages") or request_data.get("messages") or []
        if not messages:
            return None, False, "missing_messages"

        tokenize_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        tool_aware = False
        estimate_reason = None

        try:
            if request_payload.get("tools") is not None:
                tokenize_kwargs["tools"] = request_payload["tools"]
                tool_aware = True
            token_ids = self.tokenizer.apply_chat_template(messages, **tokenize_kwargs)
            return len(token_ids), tool_aware, None
        except Exception as exc:
            return None, tool_aware, f"tokenize_failed:{type(exc).__name__}"

    def _build_over_limit_result(
        self,
        filename: str,
        request_index: int,
        source_file: str,
        original_model: str,
        context_turns: int,
        context_chars: int,
        num_messages: int,
        has_tools: bool,
        estimated_prompt_tokens: int,
        tool_aware_estimate: bool,
        estimate_reason: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "id": f"{filename}_req_{request_index + 1}",
            "source_file": source_file,
            "original_model": original_model,
            "context_turns": context_turns,
            "context_chars": context_chars,
            "num_messages": num_messages,
            "has_tools": has_tools,
            "success": False,
            "skipped": True,
            "skip_reason": "over_limit",
            "error": None,
            "request_index_in_session": request_index + 1,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "tool_aware_estimate": tool_aware_estimate,
            "estimate_reason": estimate_reason,
            "max_model_len": self.max_model_len,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0,
            "tokens_per_second": 0,
            "ttft_ms": None,
            "tpot_ms": None,
            "itl_ms": None,
            "itl_list": [],
            "request_start_time": None,
            "request_end_time": None,
            "first_stream_event_type": None,
        }

    def _count_valid_sessions(self, files_to_process: List[Path]) -> tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, str]]]:
        requests_by_file: Dict[str, List[Dict[str, Any]]] = {}
        skipped_files: List[Dict[str, str]] = []

        for json_file in files_to_process:
            print(f"Loading: {json_file}")
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                skipped_files.append({"file": str(json_file), "reason": "invalid_json"})
                continue

            reqs = data.get("reqs", [])
            if not isinstance(reqs, list) or not reqs:
                skipped_files.append({"file": str(json_file), "reason": "empty_or_missing_reqs"})
                continue

            reqs.sort(key=lambda x: x.get("timestamp", ""))
            filename = json_file.stem
            file_requests: List[Dict[str, Any]] = []

            for idx, req in enumerate(reqs):
                request_data = self._extract_request_body(req)
                messages = request_data.get("messages", [])
                if not messages:
                    continue

                original_model = request_data.get("model", "unknown")
                if self.model_filter and self.model_filter.lower() not in str(original_model).lower():
                    continue

                request_payload = self._build_request_payload(request_data)
                context_turns = sum(1 for m in messages if m.get("role") == "user")
                context_chars = sum(
                    len(m.get("content", "")) if isinstance(m.get("content"), str) else 0
                    for m in messages
                )
                has_tools = bool(request_data.get("tools"))

                if self.max_model_len is not None and self.tokenizer is not None:
                    estimated_prompt_tokens, tool_aware_estimate, estimate_reason = self._estimate_prompt_tokens(
                        request_data,
                        request_payload,
                    )
                    if estimated_prompt_tokens is not None and estimated_prompt_tokens + self.max_tokens > self.max_model_len:
                        over_limit_result = self._build_over_limit_result(
                            filename=filename,
                            request_index=idx,
                            source_file=str(json_file),
                            original_model=original_model,
                            context_turns=context_turns,
                            context_chars=context_chars,
                            num_messages=len(messages),
                            has_tools=has_tools,
                            estimated_prompt_tokens=estimated_prompt_tokens,
                            tool_aware_estimate=tool_aware_estimate,
                            estimate_reason=estimate_reason,
                        )
                        self.over_limit_requests.append(over_limit_result)
                        self.skipped_over_limit += 1
                        if self.over_limit_policy == "fail":
                            raise ValueError(
                                f"Over-limit request detected in {json_file} index={idx + 1}, "
                                f"estimated_prompt_tokens={estimated_prompt_tokens}, max_model_len={self.max_model_len}"
                            )
                        if self.over_limit_policy == "skip":
                            continue

                file_requests.append({
                    "id": f"{filename}_req_{idx + 1}",
                    "source_file": str(json_file),
                    "original_model": original_model,
                    "payload": request_payload,
                    "context_turns": context_turns,
                    "context_chars": context_chars,
                    "num_messages": len(messages),
                    "has_tools": has_tools,
                    "request_index_in_session": idx + 1,
                })

            if file_requests:
                requests_by_file[filename] = file_requests
                self.max_filename_len = max(self.max_filename_len, len(filename))
            else:
                skipped_files.append({"file": str(json_file), "reason": "no_valid_messages"})

        return requests_by_file, skipped_files

    def load_all_requests(self, single_file: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        从 logs 目录或单个文件加载所有原始请求

        Returns:
            Dict[str, List]: 按文件名分组的请求列表
        """
        if single_file:
            files_to_process = [single_file]
        else:
            files_to_process = sorted(self.log_dir.rglob("*.json"))
        requests_by_file, skipped_files = self._count_valid_sessions(files_to_process)
        total_requests = sum(len(reqs) for reqs in requests_by_file.values())
        print(f"\nLoaded {len(requests_by_file)} files, {total_requests} total requests")
        print(f"Skipped over-limit requests during precheck: {self.skipped_over_limit}")
        if skipped_files:
            print("Skipped files:")
            for entry in skipped_files:
                print(f"  - {self._summarize_file_entry(entry)}")

        if self.session_count and len(requests_by_file) != self.session_count:
            raise ValueError(
                f"Expected exactly {self.session_count} valid session files, got {len(requests_by_file)}"
            )
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

        start_time = time.perf_counter()
        request_start_time = time.time()
        with self.lock:
            if self._benchmark_start_time is None:
                self._benchmark_start_time = start_time
            else:
                self._benchmark_start_time = min(self._benchmark_start_time, start_time)
        first_stream_time = None
        first_stream_event_type = "unknown"
        token_times = []
        stream_event_times = []
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
                        "request_index_in_session": req.get("request_index_in_session"),
                        "success": False,
                        "skipped": False,
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
                                delta = choices[0].get("delta", {}) or {}
                                timestamp = time.perf_counter()
                                has_content = delta.get("content") is not None
                                has_tool_calls = delta.get("tool_calls") is not None

                                if has_content or has_tool_calls or choices[0].get("finish_reason") is not None:
                                    stream_event_times.append(timestamp)
                                    if first_stream_time is None:
                                        first_stream_time = timestamp
                                        if has_content:
                                            first_stream_event_type = "content"
                                        elif has_tool_calls:
                                            first_stream_event_type = "tool_calls"

                                content = delta.get("content")
                                if content is not None:
                                    token_times.append(timestamp)
                                    response_text += content
                                    if self.verbose:
                                        print(content, end="", flush=True)

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
            request_end_time = time.time()
            with self.lock:
                self._benchmark_end_time = max(self._benchmark_end_time or end_time, end_time)
            total_latency = end_time - start_time

            # 计算指标 (对齐 vLLM 的计算方式)
            # TTFT: 第一个有效流式增量到达时间
            ttft_ms = None
            if first_stream_time is not None:
                ttft_ms = (first_stream_time - start_time) * 1000

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

                # TPOT: (总延迟 - TTFT) / (输出 token 数 - 1)
                if output_tokens > 1 and ttft_ms is not None:
                    latency_minus_ttft = total_latency - (ttft_ms / 1000)
                    tpot_ms = latency_minus_ttft / (output_tokens - 1) * 1000
            elif stream_event_times and output_tokens == 0 and first_stream_event_type == "tool_calls":
                output_tokens = len(stream_event_times)
                if len(stream_event_times) > 1:
                    for i in range(1, len(stream_event_times)):
                        itl_list.append((stream_event_times[i] - stream_event_times[i - 1]) * 1000)
                    itl_ms = statistics.mean(itl_list) if itl_list else None
                if output_tokens > 1 and ttft_ms is not None:
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
                "request_index_in_session": req.get("request_index_in_session"),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "latency_seconds": round(total_latency, 3),
                "tokens_per_second": round(output_tokens / total_latency, 2) if total_latency > 0 and output_tokens > 0 else 0,
                "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
                "tpot_ms": round(tpot_ms, 2) if tpot_ms is not None else None,
                "itl_ms": round(itl_ms, 2) if itl_ms is not None else None,
                "itl_list": [round(x, 2) for x in itl_list] if itl_list else [],
                "request_start_time": request_start_time,
                "request_end_time": request_end_time,
                "first_stream_event_type": first_stream_event_type,
                "success": True,
                "skipped": False,
                "error": "Interrupted by user" if interrupted else None,
            }

        except requests.exceptions.Timeout:
            return {
                "id": req_id,
                "source_file": req["source_file"],
                "original_model": original_model,
                "context_turns": req["context_turns"],
                "success": False,
                "skipped": False,
                "error": "Request timeout",
                "request_index_in_session": req.get("request_index_in_session"),
            }
        except Exception as e:
            return {
                "id": req_id,
                "source_file": req["source_file"],
                "original_model": original_model,
                "context_turns": req["context_turns"],
                "success": False,
                "skipped": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "request_index_in_session": req.get("request_index_in_session"),
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

        self.results = list(self.over_limit_requests)

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
        if result.get("skipped"):
            est = result.get("estimated_prompt_tokens", "?")
            print(f"[{progress_str:<20}] - SKIP over_limit prompt_tokens={est} max_model_len={result.get('max_model_len')}")
            return

        status = "✓" if result["success"] else "✗"

        if result["success"]:
            orig_model = result.get("original_model", "unknown")
            ttft = f"{result['ttft_ms']:.1f}" if result.get("ttft_ms") is not None else "-"
            tpot = f"{result['tpot_ms']:.1f}" if result.get("tpot_ms") is not None else "-"
            itl = f"{result['itl_ms']:.1f}" if result.get("itl_ms") is not None else "-"
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
        failed_results = [r for r in self.results if not r.get("success") and not r.get("skipped")]
        skipped_results = [r for r in self.results if r.get("skipped")]

        if self._benchmark_start_time is not None and self._benchmark_end_time is not None:
            total_duration = max(self._benchmark_end_time - self._benchmark_start_time, 0)
        elif success_results:
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
        print("{:<40} {:<10}".format("Skipped over-limit requests:", len(skipped_results)))
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
        failed_results = [r for r in self.results if not r.get("success") and not r.get("skipped")]
        skipped_results = [r for r in self.results if r.get("skipped")]

        # 计算汇总指标
        if success_results:
            if self._benchmark_start_time is not None and self._benchmark_end_time is not None:
                total_duration = max(self._benchmark_end_time - self._benchmark_start_time, 0)
            else:
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
                "skipped_over_limit": len(skipped_results),
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
                "skipped_over_limit": len(skipped_results),
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
                    "tokenizer_path": self.tokenizer_path,
                    "max_model_len": self.max_model_len,
                    **summary,
                    "results": self.results,
                    "summary_label": self.summary_label,
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
    parser.add_argument(
        "--session-count",
        type=int,
        default=None,
        help="Optional exact number of valid session files to require",
    )
    parser.add_argument(
        "--force-max-tokens",
        type=int,
        default=1,
        help="Force max_tokens for replayed requests (default: 1)",
    )
    parser.add_argument(
        "--summary-label",
        type=str,
        default="",
        help="Optional label used in terminal/output files",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer path for prompt length precheck",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model length for precheck",
    )
    parser.add_argument(
        "--over-limit-policy",
        type=str,
        choices=["skip", "fail"],
        default="skip",
        help="What to do when precheck finds an over-limit request",
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
        max_tokens=args.force_max_tokens,
        timeout=args.timeout,
        max_workers=1,  # 保留参数兼容
        verbose=args.verbose,
        model_filter=args.filter_model,
        session_count=args.session_count or 0,
        summary_label=args.summary_label,
        tokenizer_path=args.tokenizer,
        max_model_len=args.max_model_len,
        over_limit_policy=args.over_limit_policy,
    )

    # 设置信号处理器
    def signal_handler(signum, frame):
        print("\n\nReceived Ctrl+C, stopping benchmark...")
        runner.set_interrupted()

    signal.signal(signal.SIGINT, signal_handler)

    if args.filter_model:
        print(f"Filter: only testing requests with model containing '{args.filter_model}'")
    if args.summary_label:
        print(f"Summary label: {args.summary_label}")

    requests_by_file = runner.load_all_requests(single_file=use_single_file)
    if not requests_by_file:
        print("No requests found")
        return

    runner.run_benchmark(requests_by_file)
    runner.print_summary()
    runner.save_results()


if __name__ == "__main__":
    main()
