"""Small MRV2 API comparison, restricted to the November 18 reservation."""

import argparse
from contextlib import nullcontext
import ast
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

from reference_contract import check_hybrid_reference

RESERVATION_START = datetime.fromisoformat("2026-11-18T04:00:00+08:00")
CLIENT_CUTOFF = datetime.fromisoformat("2026-11-18T08:45:00+08:00")
PROMPT_SOURCE = re.compile(r"^vllm:prompt_tokens_by_source_total\{([^{}]*)\}\s+(\S+)")
SOURCE_LABEL = re.compile(r'(?:^|,)source="([^"]+)"')
TRANSFER_READ = re.compile(
    r"DSV4\.1 state READ complete: request=\S+ transfer=(\S+) boundary=(\d+) rank=(\d+) ranges=(\d+) bytes=(\d+)"
)
NATIVE_READ = re.compile(
    r"KV cache transfer for requests (\[.*?\]) took \S+ ms \((\d+) ranges, (\d+) bytes\).*?"
    r"local_device_id (\d+) remote_session_id"
)
SOURCES = ("local_compute", "local_cache_hit", "external_kv_transfer")


def configure_window(path, endpoints):
    """Explicit per-booking input; never automatically extend the old window."""
    global RESERVATION_START, CLIENT_CUTOFF
    if path is None:return
    document=json.loads(path.read_text())
    start=datetime.fromisoformat(document['start']);cutoff=datetime.fromisoformat(document['client_cutoff'])
    end=datetime.fromisoformat(document['end'])
    if document.get('confirmed') is not True or any(t.tzinfo is None for t in (start,cutoff,end)) or not start<cutoff<=end:
        raise ValueError('Invalid explicitly confirmed reservation window')
    allowed={value.rstrip('/') for value in document.get('endpoints',[])}
    if not endpoints or any(value.rstrip('/') not in allowed for value in endpoints):
        raise ValueError('Endpoint outside this reservation configuration')
    RESERVATION_START=start;CLIENT_CUTOFF=cutoff


def timeout_for_request() -> float:
    now = datetime.now(timezone.utc)
    if not RESERVATION_START <= now < CLIENT_CUTOFF:
        raise RuntimeError(
            "HTTP requests require the November 18 04:00–08:45 Beijing window"
        )
    return min(120.0, (CLIENT_CUTOFF - now).total_seconds())


def fetch(base, path, body=None, dp=None):
    headers = {"Content-Type": "application/json"}
    if api_key := os.environ.get("DSV41_VALIDATION_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    if dp is not None:
        headers["X-data-parallel-rank"] = str(dp)
    request = Request(
        base.rstrip("/") + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers=headers,
    )
    with urlopen(request, timeout=timeout_for_request()) as response:
        return response.read().decode()


def sources_from_metrics(raw):
    values = dict.fromkeys(SOURCES, 0)
    seen = set()
    for line in raw.splitlines():
        if not (metric := PROMPT_SOURCE.match(line)):
            continue
        source = SOURCE_LABEL.search(metric[1])
        if source is None or source[1] not in values:
            continue
        value = float(metric[2])
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError("Invalid prompt source counter")
        values[source[1]] += int(value)
        seen.add(source[1])
    if seen != set(SOURCES):
        raise ValueError(f"Missing prompt source metrics: {set(SOURCES) - seen}")
    return values


def wait_for_source_delta(base, before, length, *, remote=True):
    expected = {
        "local_compute": 1 if remote else length,
        "local_cache_hit": 0,
        "external_kv_transfer": length - 1 if remote else 0,
    }
    for _ in range(10):
        after = sources_from_metrics(fetch(base, "/metrics"))
        delta = {key: after[key] - before[key] for key in SOURCES}
        if delta == expected:
            return delta
        if any(not 0 <= delta[key] <= expected[key] for key in SOURCES):
            break
        time.sleep(0.5)
    raise ValueError(
        f"Prompt source counter mismatch: observed {delta}, expected {expected}"
    )


def check_read_log(raw, rows, tp, *, producer_tp=None):
    by_generation = {}
    for line in raw.splitlines():
        found = TRANSFER_READ.search(line)
        if found is None:
            continue
        transfer, boundary, rank, ranges, size = found.groups()
        source = re.search(r"\bsource_rank=(\d+)\b", line)
        by_generation.setdefault(transfer, []).append(
            (*map(int, (boundary, rank, ranges, size)), int(source[1]) if source else None)
        )
    for row in rows:
        generation = row["transfer_id"]
        records = by_generation.get(generation, [])
        if sorted(rank for _, rank, _, _, _ in records) != list(range(tp)) or any(
            boundary != len(row["prompt"]) - 1 or ranges <= 0 or size <= 0
            or (producer_tp is not None and source != rank * (producer_tp // tp))
            for boundary, rank, ranges, size, source in records
        ):
            raise ValueError(
                f"Missing or invalid READ evidence for transfer {generation}: {records}"
            )
    return len(rows)


def check_native_read_log(raw, rows, tp):
    parsed = [(ast.literal_eval(ids), int(ranges), int(size), int(rank))
              for ids, ranges, size, rank in NATIVE_READ.findall(raw)]
    for row in rows:
        matched = {rank for ids, ranges, size, rank in parsed if ranges > 0 and size > 0
                   and any(re.search(re.escape(row["request_id"]) + r"(?![0-9])", request_id) for request_id in ids)}
        if matched != set(range(tp)):
            raise ValueError(f'Native READ ranks for {row["request_id"]}: {sorted(matched)}')
    return len(rows)


def completion(base, model, prompt, count, dp, kv=None, *, request_id=None):
    payload = dict(
        model=model,
        prompt=prompt,
        max_tokens=count,
        min_tokens=0,
        temperature=0,
        seed=0,
        ignore_eos=True,
        return_token_ids=True,
        logprobs=5,
    )
    if kv is not None:
        payload["kv_transfer_params"] = kv
    if request_id is not None:
        payload["request_id"] = request_id
    result = json.loads(fetch(base, "/v1/completions", payload, dp=dp))
    (choice,) = result["choices"]
    ids = choice.get("token_ids")
    if not isinstance(ids, list) or len(ids) != count:
        raise ValueError(f"Expected {count} generated IDs, got {ids}")
    return result, ids


def make_prompts(base, model, lengths, ranks, text):
    result = json.loads(
        fetch(
            base,
            "/tokenize",
            {"model": model, "prompt": text, "add_special_tokens": False},
        )
    )
    pieces = result["tokens"]
    if not pieces:
        raise ValueError("Tokenizer produced no prompt tokens")
    return [
        {
            "dp": rank,
            "prompt": (pieces * ((length + len(pieces) - 1) // len(pieces)))[:length],
            "output_count": 130 if length == 130 else 8,
        }
        for rank in ranks
        for length in lengths
    ]


def checked_result(base, model, row, kv=None, *, request_id=None):
    result, ids = completion(
        base, model, row["prompt"], row["output_count"], row["dp"], kv, request_id=request_id
    )
    if result["usage"]["prompt_tokens"] != len(row["prompt"]):
        raise ValueError("D or local model did not process the original prompt length")
    return result, ids


def live_topology(base, expected_tp):
    info=json.loads(fetch(base,'/server_info?config_format=json'))
    config=info.get('vllm_config')
    if not isinstance(config,dict):raise ValueError('Need structured live server configuration')
    parallel=config.get('parallel_config',{})
    if type(parallel.get('tensor_parallel_size')) is not int or parallel['tensor_parallel_size']!=expected_tp:
        raise ValueError('Live TP differs from explicit test topology')
    return dict(tp=parallel['tensor_parallel_size'],dp=parallel.get('data_parallel_size'))


def check_reference_deployment(reference,live):
    context=reference.get('runtime_context',{})
    for role in ('producer','consumer'):
        expected=context.get(role,{})
        if expected.get('TP')!=live[role]['tp'] or expected.get('DP')!=live[role]['dp']:
            raise ValueError('R reference DP/TP differs from the running P/D deployment')


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def run(args):
    if args.mode == "audit-read":
        raw = args.log.read_text(errors="replace")
        audit = json.loads(args.reference.read_text())
        if audit.get("mode") != "pd" or not (audit.get("execution_passed") or audit.get("passed")):
            raise ValueError("READ audit requires a completed P/D client run")
        if audit.get("variant") == "control":
            count = check_native_read_log(raw, audit["rows"], args.tp)
        else:
            count = check_read_log(raw, audit["rows"], args.tp, producer_tp=args.producer_tp)
        print(f"{count} requests have complete per-TP READ evidence")
        return

    producer_dp_size = getattr(args, "producer_dp_size", None)
    if producer_dp_size is not None and producer_dp_size < 1:
        raise ValueError("--producer-dp-size requires a positive count")
    endpoints=[args.endpoint]+([args.producer] if args.mode=='pd' else [])
    configure_window(getattr(args,'window_file',None),endpoints)
    timeout_for_request()
    if args.out.exists():
        raise ValueError(f"Refusing to overwrite {args.out}")
    if args.mode == "reference":
        cases = make_prompts(
            args.endpoint, args.model, args.lengths, args.ranks, args.text
        )
    else:
        reference = json.loads(args.reference.read_text())
        if (
            not reference.get("passed")
            or reference.get("mode") != "reference"
            or reference["model"] != args.model
        ):
            raise ValueError("Need a complete local reference for the same model")
        if args.mode == "pd":
            check_hybrid_reference(reference, args.producer_tp, args.consumer_tp,
                                   args.runtime_identity, args.reference.parent)
        cases = reference["rows"]

    pending = {"mode": args.mode, "model": args.model, "passed": False, "rows": [],
               "variant": getattr(args, "variant", "state")}
    if args.mode=='pd':
        pending['live_topology']=dict(producer=live_topology(args.producer,args.producer_tp),
                                     consumer=live_topology(args.endpoint,args.consumer_tp))
        if args.producer_tp!=args.consumer_tp:
            check_reference_deployment(reference,pending['live_topology'])
    if args.mode == 'reference':
        pending['reference_kind']='full_local_prefill_diagnostic'
    elif args.mode == 'pd':
        pending.update(reference_kind=reference.get('reference_kind','legacy_same_tp'),
                       producer_tp=args.producer_tp,consumer_tp=args.consumer_tp,
                       runtime_identity=args.runtime_identity)
    save(args.out, pending)
    for case in cases:
        row = {key: case[key] for key in ("dp", "prompt", "output_count")}
        if args.mode == "reference":
            first, expected = checked_result(args.endpoint, args.model, row)
            result, ids = checked_result(args.endpoint, args.model, row)
            row["first_output"] = first
        else:
            expected = case["token_ids"]
            if args.mode == "pd":
                producer_dp = row["dp"]
                if producer_dp_size is not None:
                    producer_dp %= producer_dp_size
                    row["producer_dp"] = producer_dp
                observer_key=uuid4().hex
                observer_config=getattr(args,'observer_controls',None)
                if observer_config:
                    from observer_controls import armed
                    context=armed(observer_config,observer_key,row['prompt'],args.runtime_identity)
                    row['observer_key']=observer_key
                else:context=nullcontext()
                with context:
                    p, _ = completion(
                        args.producer,
                        args.model,
                        row["prompt"],
                        1,
                        producer_dp,
                        {"do_remote_decode": True, "do_remote_prefill": False},
                        request_id=f"{observer_key}-p",
                    )
                    params = p["kv_transfer_params"]
                    snapshot = params.get("dsv41_state_snapshot")
                    limit = len(row["prompt"]) - 1
                    if params.get("remote_num_prompt_tokens") != limit:
                        raise ValueError(f"Unexpected transferred token count: {params}")
                    if args.variant == "state":
                        if not snapshot or snapshot["boundary"] != limit or snapshot["tp_size"] != args.producer_tp:
                            raise ValueError(f"Unexpected snapshot boundary or TP size: {snapshot}")
                        row["transfer_id"] = snapshot["transfer_id"]
                    elif snapshot is not None:
                        raise ValueError("Native control unexpectedly returned state metadata")
                    row["producer"] = p
                    row["request_id"] = f"{observer_key}-d"
                    before = sources_from_metrics(fetch(args.endpoint, "/metrics"))
                    result, ids = checked_result(
                        args.endpoint, args.model, row, params, request_id=row["request_id"]
                    )
                    row["decode_prompt_sources"] = wait_for_source_delta(
                        args.endpoint, before, len(row["prompt"])
                    )
            else:
                result, ids = checked_result(args.endpoint, args.model, row)

        row.update(output=result, token_ids=ids, ids_match=ids == expected)
        pending["rows"].append(row)
        save(args.out, pending)
        if ids != expected:
            raise ValueError(
                f"Output IDs diverged on DP {row['dp']} at input length {len(row['prompt'])}"
            )
    pending['execution_passed'] = True
    pending['passed'] = args.mode!='pd'
    if not pending['passed']:
        pending['pending_gates']=['per_rank_READ','source_loaded_state','raw_continuation_vs_R','lifecycle']
    save(args.out, pending)


def parser():
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="mode", required=True)
    for mode in ("reference", "compare", "pd"):
        sub = commands.add_parser(mode)
        sub.add_argument("--endpoint", required=True)
        sub.add_argument("--model", default="dsv41")
        sub.add_argument("--out", required=True, type=Path)
        sub.add_argument("--window-file",type=Path,
                         help="Explicit confirmed booking and allowed endpoints; omitted retains the expired original window")
        if mode == "reference":
            sub.add_argument(
                "--lengths", nargs="+", type=int, default=[128, 129, 130, 257]
            )
            sub.add_argument("--ranks", nargs="+", type=int, default=[0])
            sub.add_argument(
                "--text",
                default="The cache stores history and request state independently. ",
            )
        else:
            sub.add_argument("--reference", required=True, type=Path)
        if mode == "pd":
            sub.add_argument("--producer", required=True)
            sub.add_argument("--observer-controls",type=Path,help="Explicit observer roots/SSH configuration for raw source and D state capture")
            sub.add_argument("--variant", choices=("state", "control"), default="state")
            sub.add_argument("--producer-tp", "--tp", type=int, default=4)
            sub.add_argument("--consumer-tp", type=int, required=True,
                             help="Explicit Decode TP size; do not infer equal TP")
            sub.add_argument("--runtime-identity",
                             help="Frozen model/runtime/topology identity for an independent hybrid reference")
            sub.add_argument("--producer-dp-size", type=int,
                             help="Route Decode DP ranks over this many Prefill ranks")

    audit = commands.add_parser("audit-read")
    audit.add_argument("--reference", required=True, type=Path)
    audit.add_argument("--log", required=True, type=Path)
    audit.add_argument("--tp", type=int, default=4)
    audit.add_argument("--producer-tp", type=int)
    return top


if __name__ == "__main__":
    run(parser().parse_args())
