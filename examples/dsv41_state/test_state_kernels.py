"""Device copy tests with an independent absolute-position CPU oracle."""

import json
import time
from pathlib import Path

import torch

from vllm_ascend.models.deepseek_v41.state_kernels import assemble, commit


def run_case(computed, lengths):
    count = len(lengths)
    ids = torch.tensor([2 * i + 1 for i in range(count)], dtype=torch.int32)
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    ends = torch.tensor([a + b for a, b in zip(computed, lengths)], dtype=torch.int32)
    qstarts = torch.tensor(starts, dtype=torch.int32)
    bases = []
    capacity = 0
    for c, n in zip(computed, lengths):
        bases.append(capacity)
        capacity += ((min(c, 127) + n + 127) // 128) * 128
    base = torch.tensor(bases, dtype=torch.int32)
    torch.manual_seed(947 + sum(computed) + sum(lengths))
    before = torch.randn(2 * count + 2, 128, 512).to(torch.bfloat16)
    current = torch.randn(starts[-1], 512).to(torch.bfloat16)
    expected_ring = before.clone()
    expected_output = torch.full((capacity, 512), -13, dtype=torch.bfloat16)
    for req, (c, n) in enumerate(zip(computed, lengths)):
        old = min(c, 127)
        state = int(ids[req])
        positions = torch.arange(c - old, c) % 128
        expected_output[bases[req] : bases[req] + old] = before[state, positions]
        expected_output[bases[req] + old : bases[req] + old + n] = current[starts[req] : starts[req + 1]]
        tail = min(n, 128)
        for j in range(tail):
            absolute = c + n - tail + j
            expected_ring[state, absolute % 128] = current[starts[req + 1] - tail + j]
    ring = before.npu()
    values = current.npu()
    output = torch.full((capacity, 512), -13, dtype=torch.bfloat16, device="npu")
    device_ids, device_q, device_seq, device_base = (x.npu() for x in (ids, qstarts, ends, base))
    assemble(ring, values, device_ids, device_q, device_seq, device_base, output, max(lengths))
    assert torch.equal(output.cpu(), expected_output), ("assemble", computed, lengths)
    assert torch.equal(ring.cpu(), before), "Assembly must not overwrite the old window"
    commit(ring, values, device_ids, device_q, device_seq)
    assert torch.equal(ring.cpu(), expected_ring), ("commit", computed, lengths)
    return {"computed": computed, "query_lengths": lengths, "staging_bytes": output.numel() * 2, "passed": True}


if __name__ == "__main__":
    assert torch.npu.device_count() == 12
    torch.npu.set_device(0)
    cases = [
        ([0], [1]),
        ([0], [8192]),
        ([127], [1]),
        ([128], [1]),
        ([129], [2]),
        ([255], [257]),
        ([131071], [1]),
        ([131072], [8192]),
        ([0, 127, 128, 100000], [512, 1, 129, 7]),
        ([131072, 32768, 0], [1, 1, 2048]),
    ]
    start = time.time()
    results = [run_case(*case) for case in cases]
    value = {
        "kind": "actual_NPU_copy_kernel_tests",
        "physical_device": 0,
        "cases": results,
        "elapsed_s": time.time() - start,
        "passed": True,
        "scope": "assemble, old-window immutability, commit and isolation; not full-model correctness",
    }
    target = Path("/results/state/state-copy-tests.json")
    target.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value))
