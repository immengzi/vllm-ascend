"""CP local-query windows assembled from replicated global projections."""

import json
from dataclasses import replace
from pathlib import Path

import torch

from vllm_ascend.models.deepseek_v41.state_kernels import assemble_cp
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op


def attention(q, ori, ori_len, cmp, cmp_len, indices, ratio, qlen, ori_table=None):
    device = q.device
    starts = torch.tensor([0, qlen], dtype=torch.int32, device=device)
    orig_lens = torch.tensor([ori_len], dtype=torch.int32, device=device)
    cmp_lens = torch.tensor([cmp_len], dtype=torch.int32, device=device) if ratio else None
    cmp_rem = (
        torch.tensor(
            [int((ori_len if ratio == 0 else attention.original_length) % ratio)], dtype=torch.int32, device=device
        )
        if ratio
        else None
    )
    topk = 512 if ratio else 0
    metadata = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        q.shape[1],
        1,
        512,
        cu_seqlens_q=starts,
        seqused_ori_kv=orig_lens,
        seqused_cmp_kv=cmp_lens,
        cmp_residual_kv=cmp_rem,
        batch_size=1,
        max_seqlen_q=qlen,
        max_seqlen_ori_kv=ori_len,
        max_seqlen_cmp_kv=cmp_len if ratio else 0,
        ori_topk=0,
        cmp_topk=topk,
        cmp_ratio=ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3 if ratio else 0,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=bool(ratio),
    )
    table = (
        ori_table if ori_table is not None else torch.arange(ori.shape[0], dtype=torch.int32, device=device).view(1, -1)
    )
    ctable = torch.arange(cmp.shape[0], dtype=torch.int32, device=device).view(1, -1) if ratio else None
    output, _ = torch.ops._C_ascend.npu_sparse_flash_mla(
        q,
        ori_kv=ori,
        cmp_kv=cmp if ratio else None,
        ori_block_table=table,
        cmp_block_table=ctable,
        ori_sparse_indices=None,
        cmp_sparse_indices=indices if ratio else None,
        cu_seqlens_q=starts,
        seqused_ori_kv=orig_lens,
        seqused_cmp_kv=cmp_lens,
        cmp_residual_kv=cmp_rem,
        sinks=torch.zeros(q.shape[1], dtype=torch.float32, device=device),
        metadata=metadata,
        softmax_scale=512**-0.5,
        cmp_ratio=ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3 if ratio else 0,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        topk_value_mode=1,
        return_softmax_lse=False,
    )
    return output


bootstrap_custom_op_env()
enable_custom_op()
torch.npu.set_device(0)
from vllm_ascend.attention.dsa_v41 import AscendDSAV41Metadata  # noqa: E402 - custom ops must be bootstrapped first

sample = AscendDSAV41Metadata(
    block_table=torch.zeros((1, 1)),
    query_start_loc=torch.zeros(2),
    seq_lens=torch.zeros(1),
    slot_mapping=torch.zeros(1),
    compress_ratio=0,
    storage_block_size=128,
    is_compressor_state=False,
    request_state={"state_ids": "contract-preserved"},
)
assert replace(sample, cp_token_range=(0, 1, 1, 1)).request_state is sample.request_state
results = []
for computed, global_q, offset, local_q in [
    (0, 129, 64, 32),
    (127, 8192, 2048, 1024),
    (100000, 512, 256, 129),
    (300, 17, 16, 1),
    (100000, 1, 0, 1),
]:
    total = computed + global_q
    local_end = computed + offset + local_q
    old = min(computed + offset, 127)
    stage_len = old + local_q
    torch.manual_seed(831 + computed + global_q)
    cpu = (torch.randn(((total + 127) // 128) * 128, 512) * 0.1).to(torch.bfloat16)
    native = cpu.npu().view(-1, 128, 1, 512)
    current = cpu[computed:total].contiguous().npu()
    host = torch.zeros((3, 128, 1, 512), dtype=torch.bfloat16)
    for absolute in range(max(0, computed - 128), computed):
        host[2, absolute % 128, 0] = cpu[absolute]
    ring = host.npu()
    stage = torch.full((((stage_len + 127) // 128), 128, 1, 512), 42, dtype=torch.bfloat16, device="npu")
    ids = torch.tensor([2], dtype=torch.int32, device="npu")
    lstarts = torch.tensor([0, local_q], dtype=torch.int32, device="npu")
    lends = torch.tensor([local_end], dtype=torch.int32, device="npu")
    gstarts = torch.tensor([0, global_q], dtype=torch.int32, device="npu")
    gends = torch.tensor([total], dtype=torch.int32, device="npu")
    base = torch.zeros(1, dtype=torch.int32, device="npu")
    assemble_cp(ring, current, ids, lstarts, lends, gstarts, gends, base, stage, local_q)
    torch.testing.assert_close(
        stage.flatten(0, 2)[:stage_len].cpu(), cpu[computed + offset - old : local_end], rtol=0, atol=0
    )
    assert torch.equal(ring.cpu(), host)
    q = (torch.randn(local_q, 64, 512) * 0.1).to(torch.bfloat16).npu()
    for ratio in (0, 1, 2):
        c_len = local_end // ratio if ratio else 0
        block = 128 // max(1, ratio)
        cmp = (
            (torch.randn(max(block, ((c_len + block - 1) // block) * block), 512) * 0.1)
            .to(torch.bfloat16)
            .npu()
            .view(-1, block, 1, 512)
        )
        idx = torch.full((local_q, 1, 512), -1, dtype=torch.int32)
        if ratio:
            for i in range(local_q):
                visible = (computed + offset + i + 1) // ratio
                selected = sorted(set(range(min(32, visible))) | set(range(max(0, visible - 32), visible)))
                idx[i, 0, : len(selected)] = torch.tensor(selected, dtype=torch.int32)
        idx = idx.npu()
        attention.original_length = local_end
        expected = attention(q, native, local_end, cmp, c_len, idx, ratio, local_q).cpu()
        actual = attention(q, stage, stage_len, cmp, c_len, idx, ratio, local_q).cpu()
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
        result = {
            "computed": computed,
            "global_query": global_q,
            "local_offset": offset,
            "local_query": local_q,
            "ratio": ratio,
            "heads": 64,
            "passed": True,
            "bitwise_equal": bool(torch.equal(actual, expected)),
            "max_abs_difference": (actual.float() - expected.float()).abs().max().item(),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        Path("/results/state/state-cp-attention-tests.json").write_text(
            json.dumps({"cases": results, "complete": False}, indent=2)
        )
# A CP rank without query must not write staging or alter the request ring.
zero = torch.zeros(1, dtype=torch.int32, device="npu")
starts0 = torch.zeros(2, dtype=torch.int32, device="npu")
before = stage.clone()
assemble_cp(ring, current, ids, starts0, zero, gstarts, gends, base, stage, 0)
assert torch.equal(stage.cpu(), before.cpu()) and torch.equal(ring.cpu(), host)
# Mixed batch: one request has no local query, another has one.
windows = (torch.randn(2, 128, 512) * 0.1).to(torch.bfloat16)
host = torch.zeros((3, 128, 1, 512), dtype=torch.bfloat16)
for j in range(2):
    for k in range(127):
        host[j + 1, (100001 - 128 + k) % 128, 0] = windows[j, k]
ring = host.npu()
current = windows[:, -1].contiguous().npu()
ids = torch.tensor([1, 2], dtype=torch.int32, device="npu")
lq = torch.tensor([0, 0, 1], dtype=torch.int32, device="npu")
ls = torch.tensor([0, 100001], dtype=torch.int32, device="npu")
gq = torch.tensor([0, 1, 2], dtype=torch.int32, device="npu")
gs = torch.tensor([100001, 100001], dtype=torch.int32, device="npu")
bases = torch.zeros(2, dtype=torch.int32, device="npu")
stage = torch.full((1, 128, 1, 512), 42, dtype=torch.bfloat16, device="npu")
assemble_cp(ring, current, ids, lq, ls, gq, gs, bases, stage, 1)
torch.testing.assert_close(stage.view(128, 512).cpu(), windows[1], rtol=0, atol=0)
assert torch.equal(ring.cpu(), host)
Path("/results/state/state-cp-attention-tests.json").write_text(
    json.dumps(
        {
            "cases": results,
            "complete": True,
            "passed": True,
            "empty_query_isolation_passed": True,
            "dataclass_replace_preserves_state": True,
            "mixed_empty_query_isolation_passed": True,
        },
        indent=2,
    )
)
