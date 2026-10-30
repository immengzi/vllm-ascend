"""Actual graph replay with changing state IDs, sequence lengths and inputs."""

import json
from pathlib import Path

import torch

from vllm_ascend.models.deepseek_v41.state_kernels import commit
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

bootstrap_custom_op_env()
enable_custom_op()
torch.npu.set_device(0)
ring = torch.zeros((3, 128, 1, 512), dtype=torch.bfloat16, device="npu")
current = torch.zeros((1, 512), dtype=torch.bfloat16, device="npu")
q = torch.zeros((1, 16, 512), dtype=torch.bfloat16, device="npu")
ids = torch.ones(1, dtype=torch.int32, device="npu")
starts = torch.tensor([0, 1], dtype=torch.int32, device="npu")
ends = torch.tensor([128], dtype=torch.int32, device="npu")
lengths = ends.clone()
sinks = torch.zeros(16, dtype=torch.float32, device="npu")


def metadata():
    return torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        16,
        1,
        512,
        cu_seqlens_q=starts,
        seqused_ori_kv=lengths,
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_ori_kv=128,
        max_seqlen_cmp_kv=0,
        ori_topk=0,
        cmp_topk=0,
        cmp_ratio=0,
        ori_mask_mode=4,
        cmp_mask_mode=0,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=False,
    )


meta = metadata()


def operation():
    commit(ring, current, ids, starts, ends)
    return torch.ops._C_ascend.npu_sparse_flash_mla(
        q,
        ori_kv=ring,
        ori_block_table=ids.view(1, 1),
        cu_seqlens_q=starts,
        seqused_ori_kv=lengths,
        sinks=sinks,
        metadata=meta,
        softmax_scale=512**-0.5,
        cmp_ratio=0,
        ori_mask_mode=4,
        cmp_mask_mode=0,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        topk_value_mode=1,
        return_softmax_lse=False,
    )[0]


for _ in range(3):
    operation()
torch.npu.synchronize()
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    output = operation()
torch.npu.synchronize()
results = []
for index, total in enumerate([1, 2, 127, 128, 129, 256, 301, 100001] * 2):
    torch.manual_seed(991 + index)
    slot = 1 + index % 2
    size = min(total, 128)
    values = (torch.randn(size, 512) * 0.1).to(torch.bfloat16)
    qcpu = (torch.randn(16, 512) * 0.1).to(torch.bfloat16)
    host = torch.full((3, 128, 1, 512), 42, dtype=torch.bfloat16)
    host[slot].zero_()
    for i in range(size - 1):
        host[slot, (total - size + i) % 128, 0] = values[i]
    ring.copy_(host)
    current.copy_(values[-1:].contiguous())
    q.copy_(qcpu.unsqueeze(0))
    ids.fill_(slot)
    ends.fill_(total)
    lengths.fill_(size)
    meta.copy_(metadata())
    graph.replay()
    actual = output.cpu()[0].float()
    scores = qcpu.float() @ values.float().T * (512**-0.5)
    weights = torch.softmax(torch.cat([scores, torch.zeros(16, 1)], 1), 1)[:, :-1]
    expected = weights @ values.float()
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.003)
    after = ring.cpu()
    assert torch.equal(after[1 if slot == 2 else 2], host[1 if slot == 2 else 2])
    assert torch.equal(after[slot, (total - 1) % 128, 0], values[-1])
    item = {
        "total_tokens": total,
        "state_id": slot,
        "passed": True,
        "CPU_oracle_max_abs_difference": (actual - expected).abs().max().item(),
    }
    results.append(item)
    print(json.dumps(item), flush=True)
    Path("/results/state/state-graph-replay-tests.json").write_text(
        json.dumps({"cases": results, "complete": False}, indent=2)
    )
Path("/results/state/state-graph-replay-tests.json").write_text(
    json.dumps(
        {
            "cases": results,
            "complete": True,
            "passed": True,
            "scope": "NPU graph component replay and state isolation; not full-model graph validation",
        },
        indent=2,
    )
)
