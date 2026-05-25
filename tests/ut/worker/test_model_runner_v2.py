from unittest.mock import MagicMock

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.worker.v2.aclgraph_utils import (
    LAPSPrefillGraphStats,
    ModelAclGraphManager,
    PrefillGraphKey,
)
from vllm_ascend.worker.v2.block_table import AscendBlockTables
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _make_manager(max_num_seqs=8, capture_sizes=None):
    if capture_sizes is None:
        capture_sizes = [4, 8]

    vllm_config = MagicMock()
    vllm_config.scheduler_config.max_num_seqs = max_num_seqs
    vllm_config.parallel_config.data_parallel_size = 1
    vllm_config.compilation_config.cudagraph_capture_sizes = capture_sizes
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL

    model_runner = MagicMock()
    model_runner.speculative_config = None
    model_runner.model_config.is_encoder_decoder = False
    model_runner.use_dcp = False

    return ModelAclGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL,
        decode_query_len=1,
        model_runner=model_runner,
    )


def test_laps_prefill_capture_descs_are_installed(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.aclgraph_utils.envs.VLLM_ASCEND_LAPS_SCHEDULING",
        True,
    )
    manager = _make_manager(max_num_seqs=8, capture_sizes=[4])

    assert PrefillGraphKey(num_reqs=1, num_tokens=4, max_query_len=4) in manager.laps_prefill_descs
    assert PrefillGraphKey(num_reqs=2, num_tokens=4, max_query_len=4) in manager.laps_prefill_descs


def test_dispatch_laps_prefill_can_select_padded_target_shape(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.aclgraph_utils.envs.VLLM_ASCEND_LAPS_SCHEDULING",
        True,
    )
    manager = _make_manager(max_num_seqs=8, capture_sizes=[8])
    desc = manager.laps_prefill_descs[PrefillGraphKey(num_reqs=2, num_tokens=8, max_query_len=8)]

    assert manager.dispatch_laps_prefill(2, 6, 4) is None
    manager.graphs[desc] = MagicMock()
    assert manager.dispatch_laps_prefill(2, 6, 4) == desc
    assert manager.dispatch_laps_prefill(3, 6, 4) is None
    assert manager.dispatch_laps_prefill(2, 6, 9) is None
    stats = manager.get_laps_prefill_graph_stats()
    assert stats.candidates == 4
    assert stats.hits == 1
    assert stats.shape_overflow_misses >= 2


def test_dispatch_falls_back_to_none_when_laps_prefill_hint_misses(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.aclgraph_utils.envs.VLLM_ASCEND_LAPS_SCHEDULING",
        True,
    )
    manager = _make_manager(max_num_seqs=8, capture_sizes=[4])
    manager.set_next_laps_prefill_request(2, 4, 2)
    desc = manager.dispatch(2, 4, None)
    assert desc == BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=4,
        num_reqs=2,
    )
    assert manager.get_laps_prefill_graph_stats().fallback_to_none_misses == 1


def test_dispatch_marks_unsupported_mode_miss(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.aclgraph_utils.envs.VLLM_ASCEND_LAPS_SCHEDULING",
        False,
    )
    manager = _make_manager(max_num_seqs=8, capture_sizes=[4])
    manager.set_next_laps_prefill_request(2, 4, 2)
    desc = manager.dispatch(2, 4, None)
    assert desc == BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=4,
        num_reqs=2,
    )
    assert manager.get_laps_prefill_graph_stats().unsupported_mode_misses == 1


def test_model_runner_execute_model_sets_and_clears_laps_hint(monkeypatch):
    scheduler_output = MagicMock()
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.scheduled_encoder_inputs = {}
    scheduler_output.num_scheduled_tokens = {"a": 2, "b": 2}
    scheduler_output.total_num_scheduled_tokens = 4

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.cudagraph_manager = MagicMock()
    runner.cudagraph_manager.supports_laps_prefill_graph.return_value = True
    runner.cudagraph_manager.get_laps_prefill_graph_stats.return_value = LAPSPrefillGraphStats()
    runner.supports_mm_inputs = False
    runner.input_batch = None
    runner._run_laps_prefill_timing = MagicMock()
    runner._should_hint_laps_prefill_graph = NPUModelRunner._should_hint_laps_prefill_graph.__get__(runner, NPUModelRunner)

    captured = {}

    def fake_super_execute_model(self, scheduler_output_arg, intermediate_tensors, dummy_run, skip_attn_for_dummy_run):
        captured["hint"] = runner.cudagraph_manager.set_next_laps_prefill_request.call_args.args
        return "ok"

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.model_runner.GPUModelRunner.execute_model",
        fake_super_execute_model,
    )

    result = NPUModelRunner.execute_model(runner, scheduler_output)

    assert result == "ok"
    assert captured["hint"] == (2, 4, 2)
    runner.cudagraph_manager.clear_next_laps_prefill_request.assert_called_once()
    runner._run_laps_prefill_timing.assert_not_called()


def test_prepare_attn_replay_asserts_persistent_block_tables(monkeypatch):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.kv_cache_config = MagicMock()
    runner.block_tables = AscendBlockTables.__new__(AscendBlockTables)
    runner.block_tables.input_block_tables = [
        torch.zeros((4, 2), dtype=torch.int32),
    ]
    runner.cudagraph_manager = MagicMock()
    state = MagicMock()
    state.slot_mappings = torch.zeros((1, 8), dtype=torch.int32)
    runner.cudagraph_manager._get_or_create_laps_prefill_state.return_value = state
    runner.cudagraph_manager.prepare_laps_prefill_replay_slot_mappings.return_value = {
        "layer0": state.slot_mappings[0]
    }
    runner.cudagraph_manager._validate_laps_replay_abi = MagicMock()

    input_batch = MagicMock()
    input_batch.replay_num_reqs = 4
    input_batch.replay_num_tokens = 8
    input_batch.replay_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=8,
        num_reqs=4,
    )

    persistent_view = runner.block_tables.input_block_tables[0][:4]
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.model_runner.GPUModelRunner.prepare_attn",
        lambda self, input_batch_arg: ((persistent_view,), torch.zeros((1, 8), dtype=torch.int32)),
    )

    block_tables, slot_mappings = NPUModelRunner.prepare_attn(runner, input_batch)

    assert block_tables[0].data_ptr() == runner.block_tables.input_block_tables[0].data_ptr()
    assert block_tables[0].shape[0] == 4
    assert slot_mappings.data_ptr() == state.slot_mappings.data_ptr()
    runner.cudagraph_manager._validate_laps_replay_abi.assert_called_once()


def test_prepare_attn_replay_rejects_non_persistent_block_tables(monkeypatch):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.kv_cache_config = MagicMock()
    runner.block_tables = AscendBlockTables.__new__(AscendBlockTables)
    runner.block_tables.input_block_tables = [
        torch.zeros((4, 2), dtype=torch.int32),
    ]
    runner.cudagraph_manager = MagicMock()

    input_batch = MagicMock()
    input_batch.replay_num_reqs = 4
    input_batch.replay_num_tokens = 8
    input_batch.replay_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=8,
        num_reqs=4,
    )

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.model_runner.GPUModelRunner.prepare_attn",
        lambda self, input_batch_arg: ((torch.zeros((4, 2), dtype=torch.int32),), torch.zeros((1, 8), dtype=torch.int32)),
    )

    try:
        NPUModelRunner.prepare_attn(runner, input_batch)
    except AssertionError as exc:
        assert "persistent input_block_tables" in str(exc)
    else:
        raise AssertionError("expected replay block_tables address assertion to fire")


def test_prepare_attn_replay_rejects_wrong_padded_block_table_shape(monkeypatch):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.kv_cache_config = MagicMock()
    runner.block_tables = AscendBlockTables.__new__(AscendBlockTables)
    persistent = torch.zeros((4, 2), dtype=torch.int32)
    runner.block_tables.input_block_tables = [persistent]
    runner.cudagraph_manager = MagicMock()

    input_batch = MagicMock()
    input_batch.replay_num_reqs = 4
    input_batch.replay_num_tokens = 8
    input_batch.replay_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=8,
        num_reqs=4,
    )

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.model_runner.GPUModelRunner.prepare_attn",
        lambda self, input_batch_arg: ((persistent[:2],), torch.zeros((1, 8), dtype=torch.int32)),
    )

    try:
        NPUModelRunner.prepare_attn(runner, input_batch)
    except AssertionError as exc:
        assert "rows must match the padded target request shape" in str(exc)
    else:
        raise AssertionError("expected replay block_tables shape assertion to fire")


def test_record_laps_prefill_execution_accumulates_stats(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.aclgraph_utils.envs.VLLM_ASCEND_LAPS_SCHEDULING",
        True,
    )
    manager = _make_manager(max_num_seqs=8, capture_sizes=[4])
    manager.record_laps_prefill_execution(replay=True, num_tokens=8, elapsed_us=100)
    manager.record_laps_prefill_execution(replay=False, num_tokens=4, elapsed_us=50)
    assert manager.get_laps_prefill_graph_stats() == LAPSPrefillGraphStats(
        replay_tokens=8,
        eager_tokens=4,
        replay_us=100,
        eager_us=50,
    )
