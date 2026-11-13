# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the runner's actual DSV4.1 view-construction branch on CPU."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.core.kv_cache_interface import (
    AscendKVCacheConfig,
    AscendMLAAttentionSpec,
    CachePoolPlan,
    RequestStatePlan,
    get_cache_group_capacity,
    get_request_state_plan,
    get_storage_block_size,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("runner", ["v1", "v2"])
@pytest.mark.parametrize("state", [False, True])
def test_indexer_views_keep_native_offset_and_independent_state_backing(runner, state):
    path = ROOT / ("vllm_ascend/worker/model_runner_v1.py" if runner == "v1" else "vllm_ascend/worker/v2/attn_utils.py")
    tree = ast.parse(path.read_text())
    reshape_name = "_reshape_kv_cache_tensors" if runner == "v1" else "_reshape_kv_cache_v2"
    reshape = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == reshape_name)
    branch = next(
        n
        for n in ast.walk(reshape)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "layer_name in layer_tuple_strides"
    )
    # V2 delegates to the same layout primitive. Test that call's arguments
    # against the production V1 primitive, with real torch CPU storage.
    v1 = ast.parse((ROOT / "vllm_ascend/worker/model_runner_v1.py").read_text())
    adjust = next(n for n in ast.walk(v1) if isinstance(n, ast.FunctionDef) and n.name == "_adjust_kv_layout")
    index = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8,
        tokens_per_state=2,
        storage_block_size=64,
        scale_dim=1,
        scale_dtype=torch.float16,
        model_version="deepseek_v41",
    )
    name = "m.indexer.k_cache"
    main_bytes = 65536
    stride = 8320 if state else main_bytes + 8320
    config = KVCacheConfig(num_blocks=3, kv_cache_tensors=[], kv_cache_groups=[])
    if state:
        config = AscendKVCacheConfig(
            num_blocks=3,
            kv_cache_tensors=[],
            kv_cache_groups=[],
            request_state_plan=RequestStatePlan((CachePoolPlan("history", 3, (0,), "token_block"),), 0, 0),
        )
    raw = torch.zeros(3 * stride, dtype=torch.int8)
    backend = SimpleNamespace(get_kv_cache_shape=lambda *shape: shape)
    scope = dict(
        torch=torch,
        get_dtype_size=lambda dtype: dtype.itemsize,
        AscendMLAAttentionSpec=AscendMLAAttentionSpec,
        get_cache_group_capacity=get_cache_group_capacity,
        get_request_state_plan=get_request_state_plan,
        get_storage_block_size=get_storage_block_size,
        group=SimpleNamespace(kv_cache_group_id=0, backend=backend),
        attn_backend=backend,
        layer_name=name,
        layer_tuple_strides={name: stride},
        kv_cache_config=config,
        current_kv_cache_spec=index,
        kv_cache_spec=index,
        layer_kv_cache_spec={"m.long_kv_cache": SimpleNamespace(unpadded_page_size_bytes=main_bytes)},
        kv_cache_raw_tensors={name: raw},
        kv_caches={},
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, adjust], type_ignores=[])), str(path), "exec"), scope
    )
    adjust_fn = lambda *args, **kwargs: scope["_adjust_kv_layout"](None, *args, **kwargs)
    scope["self"] = SimpleNamespace(_adjust_kv_layout=adjust_fn)
    scope["_adjust_dsv4_kv_layout"] = adjust_fn
    loop = ast.parse("for _ in (None,):\n    pass").body[0]
    loop.body = [branch]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), str(path), "exec"), scope)
    key, scale = scope["kv_caches"][name]
    assert key.shape == (3, 64, 1, 128)
    assert scale.shape == (3, 64, 1, 1)
    assert key.stride()[0] == stride
    assert scale.stride()[0] == stride // 2
    offset = 0 if state else main_bytes
    assert key.storage_offset() == offset
    assert scale.storage_offset() * 2 == offset + 8192
    assert key.untyped_storage().data_ptr() == scale.untyped_storage().data_ptr() == raw.data_ptr()
    key[2].fill_(7)
    assert torch.all(raw[offset + 2 * stride : offset + 2 * stride + 8192] == 7)
    assert torch.all(scale == 0)
