# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of staging ownership, CP metadata copies and execution ordering."""

import ast
import dataclasses
import gc
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendKVCacheConfig, CachePoolPlan, RequestStatePlan
from vllm_ascend.models.deepseek_v41 import request_state as execution
from vllm_ascend.models.deepseek_v41 import window_state as window
from vllm_ascend.models.deepseek_v41.cache_config import RequestWindowSpec


def config(capacity=256):
    spec = RequestWindowSpec(
        block_size=128, num_kv_heads=1, head_size=512, dtype=torch.bfloat16, sliding_window=128, cache_dtype_str="auto"
    )
    group = KVCacheGroupSpec(["a", "b"], UniformTypeKVCacheSpecs.from_specs({"a": spec, "b": spec}))
    return AscendKVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[group],
        request_state_plan=RequestStatePlan(
            (CachePoolPlan("window", 3, (0,), "request_slot"),), capacity, capacity * 1024
        ),
    )


def layers(dtype=torch.bfloat16):
    return {name: SimpleNamespace(kv_cache=[torch.empty((3, 128, 1, 512), dtype=dtype)]) for name in ("a", "b")}


def test_layers_share_one_workspace_but_runners_do_not():
    first, second = layers(), layers()
    owner = window.bind_window_staging_workspace(config(), first)
    other = window.bind_window_staging_workspace(config(), second)
    assert first["a"].request_state_workspace is first["b"].request_state_workspace is owner
    assert second["a"].request_state_workspace is other
    assert owner.tensor.data_ptr() != other.tensor.data_ptr()
    assert owner.tensor.numel() * owner.tensor.element_size() == 256 * 1024
    assert window.bind_window_staging_workspace(config(), first, owner) is owner


@pytest.mark.parametrize("change", ["capacity", "dtype", "shape"])
def test_workspace_never_resizes_or_rebinds_on_layout_change(change):
    bound = layers()
    owner = window.bind_window_staging_workspace(config(), bound)
    cfg = config(512 if change == "capacity" else 256)
    if change == "dtype":
        for layer in bound.values():
            layer.kv_cache[0] = layer.kv_cache[0].float()
    if change == "shape":
        bound["b"].kv_cache[0] = torch.empty((3, 128, 1, 256))
    pointer = owner.tensor.data_ptr()
    with pytest.raises(ValueError):
        window.bind_window_staging_workspace(cfg, bound, owner)
    assert all(layer.request_state_workspace is owner for layer in bound.values())
    assert owner.tensor.data_ptr() == pointer


def test_native_config_allocates_nothing():
    with patch.object(torch, "empty", side_effect=AssertionError("unexpected allocation")):
        assert window.bind_window_staging_workspace(SimpleNamespace(), {}) is None


def metadata_types():
    # Load the actual dataclass fields without importing device backends.
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/dsa_v41.py"
    wanted = {"AscendDSAV41Metadata", "DeepseekV41LayerMetadata"}
    nodes = [
        node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef) and node.name in wanted
    ]
    for node in nodes:
        node.bases = []
    scope = {"dataclass": dataclasses.dataclass}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(source), "exec"),
        scope,
    )
    return scope["AscendDSAV41Metadata"], scope["DeepseekV41LayerMetadata"]


def metadata():
    cache_type, layer_type = metadata_types()
    ids = torch.tensor([2], dtype=torch.int32)
    state = window.RequestWindowMetadata(
        ids, torch.tensor([0]), torch.tensor([128]), torch.tensor([[0]]), 128, 128, 256
    )
    swa = cache_type(
        block_table=ids[:, None],
        query_start_loc=torch.tensor([0, 1]),
        seq_lens=torch.tensor([129]),
        slot_mapping=torch.tensor([-1]),
        compress_ratio=0,
        storage_block_size=128,
        is_compressor_state=False,
        num_reqs=1,
        max_query_len=1,
        num_actual_tokens=1,
        request_state=state,
    )
    return layer_type(attention=None, swa=swa, compressor=None, indexer=None)


def test_cp_replace_keeps_typed_state_and_global_owner():
    value = metadata()
    local = dataclasses.replace(value.swa, global_metadata=value.swa, num_actual_tokens=0, max_query_len=0)
    assert local.request_state is value.swa.request_state
    assert execution.global_swa(dataclasses.replace(value, swa=local)) is value.swa


@pytest.mark.parametrize("fail", [False, True])
def test_attention_restores_ring_and_commits_only_after_success(fail):
    value = metadata()
    layer = SimpleNamespace(kv_cache=["ring"])
    attn = SimpleNamespace(dsa_attn=SimpleNamespace(swa_cache_layer=layer))
    calls = []

    def attention(*args, **kwargs):
        assert layer.kv_cache == ["staging"]
        replacement = args[2]
        assert replacement.swa.block_table is value.swa.request_state.block_table
        assert replacement.swa.seq_lens is value.swa.request_state.lengths
        assert value.swa.seq_lens.tolist() == [129]
        calls.append("attention")
        if fail:
            raise RuntimeError("operator failed")
        return "output"

    kernel = SimpleNamespace(commit=lambda *args: calls.append("commit"))
    with (
        patch.dict("sys.modules", {"vllm_ascend.models.deepseek_v41.state_kernels": kernel}),
        patch.object(execution, "_stage", return_value=("ring", "current", "staging")),
        patch.object(execution, "_release_current_kv", side_effect=lambda *args, **kwargs: calls.append("release")),
    ):
        impl = SimpleNamespace(_forward_attention_impl=attention)
        if fail:
            with pytest.raises(RuntimeError, match="operator failed"):
                execution.forward_request_window(impl, attn, None, value, None)
        else:
            assert execution.forward_request_window(impl, attn, None, value, None) == "output"
    assert layer.kv_cache == ["ring"]
    assert calls == (["attention"] if fail else ["attention", "commit", "release"])


@pytest.mark.parametrize("cp", [False, True])
def test_stage_reuses_workspace_and_preserves_local_global_coordinates(cp):
    value = metadata()
    bound = layers()
    workspace = window.bind_window_staging_workspace(config(), bound)
    global_metadata = dataclasses.replace(
        value.swa, query_start_loc=torch.tensor([0, 3]), seq_lens=torch.tensor([131]), num_actual_tokens=3
    )
    if cp:
        value = dataclasses.replace(value, swa=dataclasses.replace(value.swa, global_metadata=global_metadata))
    current = torch.zeros((3 if cp else 1, 512), dtype=torch.bfloat16)
    attn = SimpleNamespace(_request_state_current_kv=current, dsa_attn=SimpleNamespace(swa_cache_layer=bound["a"]))
    kernel = SimpleNamespace(assemble=Mock(), assemble_cp=Mock())
    with (
        patch.dict("sys.modules", {"vllm_ascend.models.deepseek_v41.state_kernels": kernel}),
        patch.object(torch, "empty", side_effect=AssertionError("hot path allocation")),
    ):
        ring, retained, staging = execution._stage(attn, value)
    assert ring is bound["a"].kv_cache[0]
    assert retained is current and staging is workspace.tensor
    called = kernel.assemble_cp if cp else kernel.assemble
    idle = kernel.assemble if cp else kernel.assemble_cp
    called.assert_called_once()
    idle.assert_not_called()
    args = called.call_args.args
    assert args[2] is value.swa.request_state.state_ids
    assert args[3] is value.swa.query_start_loc
    assert args[4] is value.swa.seq_lens
    if cp:
        assert args[5] is global_metadata.query_start_loc
        assert args[6] is global_metadata.seq_lens


def test_workspace_lifetime_ends_with_its_owners():
    bound = layers()
    workspace = window.bind_window_staging_workspace(config(), bound)
    reference = weakref.ref(workspace)
    del workspace
    gc.collect()
    assert reference() is not None  # Layer references keep captures' storage alive.
    del bound
    gc.collect()
    assert reference() is None  # No module-global device dictionary retains it.


def attention_methods():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/dsa_v41.py"
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "AscendDSAV41Impl"
    )
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_write_swa_cache", "_update_caches", "_forward_attention"}
    ]
    scope = {
        "defer_window_write": Mock(),
        "commit_request_window": Mock(),
        "forward_request_window": Mock(return_value="staged"),
        "scatter_cache_sk": Mock(),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), "exec"), scope)
    return scope, scope["AscendDSAV41Impl"]


def test_explicit_attention_entries_keep_native_and_window_paths_separate():
    scope, cls = attention_methods()
    value = metadata()
    attn = SimpleNamespace(dsa_attn=SimpleNamespace(swa_cache_layer=SimpleNamespace(kv_cache=["ring"])))
    cls._write_swa_cache(attn, value.swa, "current")
    scope["defer_window_write"].assert_called_once_with(attn, value.swa, "current")
    scope["scatter_cache_sk"].assert_not_called()
    native = dataclasses.replace(value, swa=dataclasses.replace(value.swa, request_state=None))
    cls._write_swa_cache(attn, native.swa, "native-current")
    scope["scatter_cache_sk"].assert_called_once_with("ring", native.swa.slot_mapping, "native-current")
    impl = cls()
    impl._forward_attention_impl = Mock(return_value="native")
    assert impl._forward_attention(attn, None, value, None, source_cache="source") == "staged"
    impl._forward_attention_impl.assert_not_called()
    assert impl._forward_attention(attn, None, native, None) == "native"
    impl._forward_attention_impl.assert_called_once()
    scope["forward_request_window"].assert_called_once_with(impl, attn, None, value, None, source_cache="source")


@pytest.mark.parametrize("tokens", [0, 3])
def test_cache_only_entry_commits_after_projection_and_compression(tokens):
    scope, cls = attention_methods()
    calls = []
    impl = cls()
    impl.role = SimpleNamespace(is_kv_source=True)
    impl._project_kv = lambda *args: calls.append("project") or "current"
    impl._write_compressed_source = lambda *args: calls.append("compress")
    scope["defer_window_write"].side_effect = lambda *args: calls.append("defer")
    scope["commit_request_window"].side_effect = lambda *args: calls.append("commit")
    value = SimpleNamespace(
        swa=SimpleNamespace(request_state=object()), positions=torch.arange(tokens), rope=lambda *args: (None, None)
    )
    attn = SimpleNamespace(rotary_emb=SimpleNamespace(layername="layer"))
    impl._update_caches(attn, torch.empty((tokens, 512)), value)
    assert calls == (["project", "defer", "compress", "commit"] if tokens else [])
