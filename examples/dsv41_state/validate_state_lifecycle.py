"""New-design lifecycle and graph-bucket checks without old token-fork analysis."""

import asyncio
import json
import time
from urllib.request import urlopen

import aiohttp
from bench import metric_totals
from probe_state import CONFIG, DEADLINE, ROOT, batch, save
from tokenizers import Tokenizer


async def wait_idle():
    until = min(DEADLINE - 30, time.time() + 180)
    while time.time() < until:
        with urlopen(f"http://127.0.0.1:{CONFIG['port']}/metrics", timeout=5) as response:
            text = response.read().decode()
        totals = metric_totals(text)
        names = ("vllm:num_requests_running", "vllm:num_requests_waiting")
        if all(n in totals and totals[n] == 0 for n in names):
            return totals
        await asyncio.sleep(2)
    raise TimeoutError("Owned requests did not drain after cancellation")


async def cancel_requests():
    folder = ROOT / "lifecycle-cancel"
    folder.mkdir(exist_ok=False)
    tokenizer = Tokenizer.from_file("/model/tokenizer.json")
    unit = tokenizer.encode("Cancel this request after decode starts. ", add_special_tokens=False).ids
    prompts = [
        (tokenizer.encode(f"cancel-{i}-", add_special_tokens=False).ids + unit * 20000)[:131072] for i in range(3)
    ]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), read_bufsize=4 * 1024 * 1024) as session:

        async def run(i):
            ids = []
            started = time.time()
            seen = False
            body = {
                "model": "dsv41",
                "prompt": prompts[i],
                "max_tokens": 1024,
                "temperature": 0,
                "ignore_eos": True,
                "seed": 20261027,
                "stream": True,
                "return_token_ids": True,
            }
            with (folder / f"{i}.jsonl").open("x", buffering=1) as raw:
                async with session.post(
                    f"http://127.0.0.1:{CONFIG['port']}/v1/completions",
                    json=body,
                    headers={"X-data-parallel-rank": str(i)},
                ) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        if not line.strip().startswith(b"data: ") or line.strip() == b"data: [DONE]":
                            continue
                        obj = json.loads(line.strip()[6:])
                        raw.write(json.dumps({"time": time.time(), "data": obj}) + "\n")
                        assert "error" not in obj, obj
                        for c in obj.get("choices", []):
                            ids.extend(c.get("token_ids") or [])
                        if len(ids) >= 8:
                            seen = True
                            break
            assert seen
            result = {
                "dp": i,
                "intentional_client_close": True,
                "output_tokens_at_close": len(ids),
                "start": started,
                "end": time.time(),
            }
            save(folder / f"{i}.json", result)
            return result

        rows = await asyncio.gather(*(run(i) for i in range(3)))
    await asyncio.sleep(2)
    drained = await wait_idle()
    return {"requests": rows, "drained_metrics": drained}


async def main():
    target = ROOT / "state-lifecycle-tests.json"
    status = {"state": "waiting", "cases": [], "old_token_forks": "not_compared"}
    if target.exists():
        previous = json.loads(target.read_text())
        if (
            previous.get("state") == "failed_or_incomplete"
            and len(previous.get("cases", [])) == 4
            and "LineTooLong" in previous.get("error", "")
        ):
            status = previous
            status.pop("error", None)
            status.pop("passed", None)
    prior_cases = min(4, len(status["cases"]))
    try:
        while time.time() < DEADLINE - 300:
            p = ROOT / "state-service-status.json"
            if p.exists():
                service = json.loads(p.read_text())
                assert service["state"] != "exited", service
                try:
                    with urlopen(f"http://127.0.0.1:{CONFIG['port']}/health", timeout=2) as res:
                        if res.status == 200:
                            break
                except OSError:
                    pass
            save(target, status)
            await asyncio.sleep(5)
        else:
            raise TimeoutError("No ready service")
        status["state"] = "running"
        save(target, status)
        for i, (length, output, count) in enumerate([(4096, 64, 3), (4096, 64, 6), (32768, 64, 12), (2048, 64, 16)]):
            if i < prior_cases:
                continue
            # This exclusive fresh service has no earlier requests; initial
            # gauges may be unpublished until its first request completes.
            if i:
                await wait_idle()
            result = await batch(length, output, count, f"lifecycle-{i}-{length}-c{count}")
            assert result["preemptions"] == 0
            status["cases"].append(result)
            save(target, status)
        await wait_idle()
        status["cancellation"] = await cancel_requests()
        save(target, status)
        for i, (length, output, count) in enumerate([(131072, 128, 6), (4096, 64, 6)]):
            result = await batch(length, output, count, f"lifecycle-reuse-{i}-{length}-c{count}")
            assert result["preemptions"] == 0
            status["cases"].append(result)
            save(target, status)
        status["final_idle"] = await wait_idle()
        status.update(state="completed", passed=True)
    except Exception as error:
        status.update(state="failed_or_incomplete", passed=False, error=repr(error))
        raise
    finally:
        save(target, status)


asyncio.run(main())
