"""Official benchmark wrapper; CLI compatibility, telemetry and failure gates."""

import argparse
import json
import math
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

BASE_CASES = [
    {"name": "short16", "input": 2048, "output": 256, "concurrency": 16, "count": 64, "kind": "latency"},
    {"name": "long8", "input": 100000, "output": 512, "concurrency": 8, "count": 8, "kind": "latency"},
    {"name": "long16", "input": 100000, "output": 512, "concurrency": 16, "count": 16, "kind": "latency"},
    {"name": "pressure120", "input": 100000, "output": 512, "concurrency": 120, "count": 120, "kind": "pressure"},
]


def cases(config, suite):
    if suite == "core":
        return [BASE_CASES[0], BASE_CASES[2], BASE_CASES[3]]
    if suite == "diagnostic":
        return BASE_CASES[:2]
    raise ValueError(suite)


def command(config, case, seed, output_dir, filename):
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--host",
        "127.0.0.1",
        "--port",
        str(config["port"]),
        "--endpoint",
        "/v1/completions",
        "--model",
        "/model",
        "--served-model-name",
        "dsv41",
        "--tokenizer",
        "/model",
        "--dataset-name",
        "random",
        "--random-input-len",
        str(case["input"]),
        "--random-output-len",
        str(case["output"]),
        "--random-range-ratio",
        "0",
        "--random-prefix-len",
        "0",
        "--num-prompts",
        str(case["count"]),
        "--max-concurrency",
        str(case["concurrency"]),
        "--request-rate",
        str(case.get("rate", 1000)),
        "--seed",
        str(seed),
        "--skip-chat-template",
        "--temperature",
        "0",
        "--ignore-eos",
        "--save-result",
        "--save-detailed",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,95,99,100",
        "--result-dir",
        str(output_dir),
        "--result-filename",
        filename,
        "--disable-tqdm",
    ]
    if config.get("slo_ms"):
        cmd += ["--goodput", *[f"{k}:{v}" for k, v in config["slo_ms"].items()]]
    return cmd


def fetch(base, path):
    with urllib.request.urlopen(base + path, timeout=5) as response:
        return response.read().decode()


def metric_totals(text):
    out = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name in (
                "vllm:num_preemptions_total",
                "vllm:prefix_cache_hits_total",
                "vllm:num_requests_running",
                "vllm:num_requests_waiting",
            ):
                value = float(line.rsplit(" ", 1)[1])
                out[name] = out.get(name, 0) + value
    return out


def overlap_peak(data):
    starts, firsts, latencies = data.get("start_times", []), data.get("ttfts", []), data.get("latencies", [])
    if not starts or not (len(starts) == len(firsts) == len(latencies)):
        return None
    events = [(s + t, 1) for s, t in zip(starts, firsts)] + [(s + t, -1) for s, t in zip(starts, latencies)]
    peak = active = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    return peak


def cpu_sample(previous):
    # Host-level diagnostics only. These samples do not identify a client bottleneck.
    try:
        counters = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
        total, idle_ticks = sum(counters), counters[3] + counters[4]
        busy = None
        if previous and total > previous[0]:
            busy = 100 * (1 - (idle_ticks - previous[1]) / (total - previous[0]))
        return {"host_cpu_busy_pct": busy, "host_load1": os.getloadavg()[0]}, (total, idle_ticks)
    except (OSError, IndexError, ValueError):
        return {}, previous


def idle(base):
    # Require published gauges to drain, avoiding overlap across test cases.
    for _ in range(90):
        values = metric_totals(fetch(base, "/metrics"))
        names = ("vllm:num_requests_running", "vllm:num_requests_waiting")
        assert all(n in values for n in names), "Missing server gauges; do not assume idle"
        if all(values[n] == 0 for n in names):
            return
        time.sleep(2)
    raise RuntimeError("Server did not drain")


def run_one(config, case, seed, directory, filename, deadline):
    base = f"http://127.0.0.1:{config['port']}"
    idle(base)
    path = directory / filename
    assert not path.exists(), f"Refusing result overwrite: {path}"
    before = fetch(base, "/metrics")
    stop = threading.Event()

    def watch():
        previous = None
        with path.with_suffix(".metrics.jsonl").open("x") as stream:
            while not stop.is_set():
                try:
                    host, previous = cpu_sample(previous)
                    stream.write(json.dumps({"time": time.time(), "text": fetch(base, "/metrics"), **host}) + "\n")
                    stream.flush()
                except Exception as error:
                    stream.write(json.dumps({"error": repr(error)}) + "\n")
                stop.wait(1)

    observer = threading.Thread(target=watch, daemon=True)
    observer.start()
    cmd = command(config, case, seed, directory, filename)
    path.with_suffix(".command.json").write_text(json.dumps(cmd, indent=2))
    try:
        # Whole-run deadline reserves time for shutdown; a per-request timeout
        # alone cannot bound a sequence of queued batches.
        with path.with_suffix(".log").open("x") as log:
            subprocess.run(
                cmd, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=max(1, deadline - time.time() - 90)
            )
    finally:
        stop.set()
        observer.join(timeout=7)
    data = json.loads(path.read_text())
    assert data.get("completed") == case["count"] and data.get("failed", 0) == 0, "Benchmark failures"
    assert len(data.get("ttfts", [])) == case["count"] and data.get("output_lens"), "Missing detailed records"
    assert all(
        isinstance(data.get(k), (int, float)) and math.isfinite(data[k]) and data[k] > 0
        for k in ("output_throughput", "mean_ttft_ms", "mean_tpot_ms", "p95_tpot_ms", "p99_e2el_ms")
    ), "Missing or invalid performance metrics"
    # Save actual sampled lengths; never presume random token->text->token is exact.
    assert all(n == case["output"] for n in data["output_lens"]), "Unexpected output length"
    idle(base)
    time.sleep(6)  # allow published counters to catch up; excluded from benchmark duration
    after = fetch(base, "/metrics")
    a, b = metric_totals(before), metric_totals(after)
    names = ("vllm:num_preemptions_total", "vllm:prefix_cache_hits_total")
    assert all(n in a and n in b for n in names), "Missing counters; review telemetry version"
    deltas = {n: b[n] - a[n] for n in names}
    assert deltas["vllm:prefix_cache_hits_total"] == 0, "Cold-prefix protocol violated"
    peak = overlap_peak(data)
    samples = [json.loads(line) for line in path.with_suffix(".metrics.jsonl").read_text().splitlines()]
    observed = [metric_totals(s["text"]) for s in samples if "text" in s]
    assert observed, "No runtime telemetry captured"
    if case["kind"] == "latency":
        assert deltas["vllm:num_preemptions_total"] == 0, "Latency case preempted: cannot isolate overhead"
        queued = any(v.get("vllm:num_requests_waiting", 0) > 0 for v in observed)
        # Full observable overlap rules out a whole-completion-wave capacity
        # bottleneck for this batch. It does NOT prove zero admission waiting.
        eligible = peak is not None and peak >= case["concurrency"]
        path.with_suffix(".eligibility.json").write_text(
            json.dumps(
                {
                    "waiting_observed": queued,
                    "overlap_peak": peak,
                    "full_overlap_verified": eligible,
                    "review": "Initial prefill waiting is normal. Missing full overlap needs queue/KV timeline review; "
                    "this is not a kernel-only measurement.",
                },
                indent=2,
            )
        )
    path.with_suffix(".checks.json").write_text(
        json.dumps(
            {
                "passed": True,
                "counter_deltas": deltas,
                "case": case,
                "seed": seed,
                "overlap_peak": peak,
                "peak_server_waiting": max(v.get("vllm:num_requests_waiting", 0) for v in observed),
                "peak_server_running": max(v.get("vllm:num_requests_running", 0) for v in observed),
                "sampled_host_cpu_busy_peak_pct": max(
                    (s["host_cpu_busy_pct"] for s in samples if s.get("host_cpu_busy_pct") is not None), default=None
                ),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["core", "diagnostic"], default="core")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path("/results/config.json").read_text())
    directory = Path(os.environ["MENGZI_RESULTS_DIR"]) / args.suite
    directory.mkdir(exist_ok=True)
    from datetime import datetime

    deadline = datetime.fromisoformat(config["deadline"]).timestamp()
    jobs = []
    for case in cases(config, args.suite):
        warm = {**case, "count": case["concurrency"]}
        warm.pop("rate", None)
        jobs.append((warm, 20261000, f"{case['name']}-warmup.json"))
        n = config["pressure_repeats"] if case["kind"] == "pressure" else config["repeats"]
        jobs.extend((case, r, f"{case['name']}-r{r}.json") for r in range(n))
    if args.plan_only:
        print(json.dumps([command(config, c, s, directory, f) for c, s, f in jobs], indent=2))
        return
    help_text = subprocess.check_output(["vllm", "bench", "serve", "--help=all"], text=True)
    for flag in ("--save-detailed", "--random-range-ratio", "--served-model-name", "--metric-percentiles"):
        assert flag in help_text, f"Target CLI lacks required option: {flag}"
    for case, seed, filename in jobs:
        assert time.time() + 120 < deadline, "Too close to deadline"
        run_one(config, case, seed, directory, filename, deadline)
    (directory / "DONE").write_text(str(time.time()))


if __name__ == "__main__":
    main()
