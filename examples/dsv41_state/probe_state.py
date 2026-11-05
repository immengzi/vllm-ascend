"""New-design requests and observed service metrics; never reruns union."""

import argparse
import asyncio
import hashlib
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import aiohttp
from bench import metric_totals
from tokenizers import Tokenizer

ROOT = Path("/results/state")
CONFIG = json.loads(Path("/results/config.json").read_text())
DEADLINE = datetime.fromisoformat(CONFIG["deadline"]).timestamp() - 180


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


async def batch(length, output, count, tag):
    folder = ROOT / "probes" / tag
    folder.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_file("/model/tokenizer.json")
    body = tokenizer.encode(
        "A different city and its history are discussed in this paragraph. ", add_special_tokens=False
    ).ids
    payloads = []
    for i in range(count):
        prefix = tokenizer.encode(
            f"layout-validation-{length}-{i}-" + hashlib.sha256(str(i).encode()).hexdigest() * 12,
            add_special_tokens=False,
        ).ids
        tokens = (prefix + body * (length // len(body) + 1))[:length]
        payloads.append(
            (
                hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                json.dumps(
                    {
                        "model": "dsv41",
                        "prompt": tokens,
                        "max_tokens": output,
                        "temperature": 0,
                        "seed": 20261022,
                        "ignore_eos": True,
                        "stream": True,
                        "return_token_ids": True,
                        "stream_options": {"include_usage": True},
                    }
                ).encode(),
            )
        )
    timeout = min(CONFIG["request_timeout_s"], max(1, DEADLINE - time.time() - 10))
    base = f"http://127.0.0.1:{CONFIG['port']}"
    rows = []
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout), read_bufsize=4 * 1024 * 1024
    ) as session:

        async def metrics(label):
            async with session.get(base + "/metrics") as response:
                response.raise_for_status()
                body = await response.text()
                (folder / f"metrics-{label}.txt").write_text(body)
                return metric_totals(body)

        before = await metrics("before")

        async def request(i):
            item = {
                "index": i,
                "dp": i % 3,
                "prompt_hash": payloads[i][0],
                "ids": [],
                "start": time.perf_counter(),
                "first": None,
                "last": None,
                "sse_token_times": [],
                "completed": False,
            }
            with (folder / f"request-{i}.jsonl").open("x", buffering=1) as raw:
                try:
                    async with session.post(
                        base + "/v1/completions",
                        data=payloads[i][1],
                        headers={"Content-Type": "application/json", "X-data-parallel-rank": str(i % 3)},
                    ) as response:
                        response.raise_for_status()
                        usage = None
                        async for line in response.content:
                            if not line.strip().startswith(b"data: ") or line.strip() == b"data: [DONE]":
                                continue
                            obj = json.loads(line.strip()[6:])
                            stamp = time.perf_counter()
                            raw.write(json.dumps({"time": stamp, "data": obj}) + "\n")
                            if "error" in obj:
                                raise RuntimeError(obj["error"])
                            for choice in obj.get("choices", []):
                                ids = choice.get("token_ids") or []
                                if ids:
                                    item["ids"].extend(ids)
                                    item["first"] = item["first"] or stamp
                                    item["last"] = stamp
                                    item["sse_token_times"].append(stamp)
                            usage = obj.get("usage") or usage
                        assert len(item["ids"]) == output, (i, len(item["ids"]), output)
                        assert usage and usage["prompt_tokens"] == length
                        item["usage"] = usage
                        item["completed"] = True
                except Exception as error:
                    item["error"] = repr(error)
                finally:
                    item["end"] = time.perf_counter()
                    save(folder / f"request-{i}.json", item)
                    rows.append(item)

        await asyncio.gather(*(request(i) for i in range(count)))
        after = await metrics("after")
    completed = [r for r in rows if r["completed"]]
    duration = max(r["end"] for r in rows) - min(r["start"] for r in rows)
    firsts = [(r["first"], 1) for r in completed] + [(r["end"], -1) for r in completed]
    active = peak = 0
    for _, delta in sorted(firsts):
        active += delta
        peak = max(peak, active)
    assert "vllm:num_preemptions_total" in before and "vllm:num_preemptions_total" in after
    preempt = after["vllm:num_preemptions_total"] - before["vllm:num_preemptions_total"]
    ttft = [(r["first"] - r["start"]) * 1000 for r in completed]
    tpot = [(r["last"] - r["first"]) / max(1, len(r["ids"]) - 1) * 1000 for r in completed]
    result = {
        "tag": tag,
        "input_tokens": length,
        "output_tokens": output,
        "count": count,
        "completed": len(completed),
        "failed": count - len(completed),
        "duration_s": duration,
        "output_tokens_per_s": sum(len(r["ids"]) for r in completed) / duration,
        "mean_ttft_ms": sum(ttft) / len(ttft) if ttft else None,
        "mean_tpot_ms": sum(tpot) / len(tpot) if tpot else None,
        "preemptions": preempt,
        "client_observed_generating_overlap": peak,
        "server_counter_totals_before": before,
        "server_counter_totals_after": after,
        "capacity_point_passed": len(completed) == count and peak == count and preempt == 0,
        "execution": json.loads((ROOT / "state-service-status.json").read_text()).get("prototype_execution", "unknown"),
        "service_command_sha256": hashlib.sha256((ROOT / "state-serve-command.json").read_bytes()).hexdigest(),
        "comparison_to_union": "not_performed",
    }
    save(folder / "summary.json", result)
    print(json.dumps(result), flush=True)
    if len(completed) != count:
        raise RuntimeError(f"{tag}: incomplete requests; raw SSE and partial outputs retained")
    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["smoke", "capacity", "bench"], default="smoke")
    args = parser.parse_args()
    result = {"phase": args.phase, "state": "running", "cases": [], "warmup_cases": [], "new_design_only": True}
    target = ROOT / f"state-{args.phase}.json"
    try:
        while time.time() < DEADLINE - 300:
            state = json.loads((ROOT / "state-service-status.json").read_text())
            assert state["state"] != "exited", state
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{CONFIG['port']}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            save(target, result)
            await asyncio.sleep(10)
        else:
            raise TimeoutError("No complete service readiness before resource cutoff")
        workers = list(ROOT.glob("state-worker-*.json"))
        assert len(workers) == 12, ("Need all 12 actual worker layouts", len(workers))
        if args.phase == "smoke":
            shapes = [(2048, 64, 6), (16384, 128, 6), (100000, 128, 6), (131072, 256, 6)]
        elif args.phase == "capacity":
            shapes = [(131072, 512, c) for c in (18, 24, 30)]
        else:
            shapes = [(2048, 256, 16)] * 3 + [(100000, 512, 16)] * 3
            result["observational_repeats_per_shape"] = 3
        for i, (length, output, count) in enumerate(shapes):
            if time.time() > DEADLINE - 300:
                raise TimeoutError("Stop admitting new test cases before resource cutoff")
            if args.phase == "bench" and i in (0, 3):
                warm = await batch(length, 64, count, f"warmup-bench-{i}-{length}-c{count}")
                result["warmup_cases"].append(warm)
                save(target, result)
            case = await batch(length, output, count, f"{args.phase}-{i}-{length}-c{count}")
            result["cases"].append(case)
            save(target, result)
            if args.phase == "capacity" and not case["capacity_point_passed"]:
                break
        result["state"] = "completed"
    except Exception as error:
        result["state"] = "failed_or_incomplete"
        result["error"] = repr(error)
        raise
    finally:
        save(target, result)


if __name__ == "__main__":
    asyncio.run(main())
