from unittest.mock import MagicMock

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager, PrefillGraphKey
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

    assert PrefillGraphKey(num_reqs=1, num_tokens=4) in manager.laps_prefill_descs
    assert PrefillGraphKey(num_reqs=2, num_tokens=4) in manager.laps_prefill_descs


def test_dispatch_laps_prefill_requires_exact_graph_hit(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.aclgraph_utils.envs.VLLM_ASCEND_LAPS_SCHEDULING",
        True,
    )
    manager = _make_manager(max_num_seqs=8, capture_sizes=[4])
    desc = manager.laps_prefill_descs[PrefillGraphKey(num_reqs=2, num_tokens=4)]

    assert manager.dispatch_laps_prefill(2, 4, 2) is None
    manager.graphs[desc] = MagicMock()
    assert manager.dispatch_laps_prefill(2, 4, 2) == desc
    assert manager.dispatch_laps_prefill(1, 4, 4) is None


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


def test_model_runner_execute_model_sets_and_clears_laps_hint(monkeypatch):
    scheduler_output = MagicMock()
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.scheduled_encoder_inputs = {}
    scheduler_output.num_scheduled_tokens = {"a": 2, "b": 2}
    scheduler_output.total_num_scheduled_tokens = 4

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.cudagraph_manager = MagicMock()
    runner.cudagraph_manager.supports_laps_prefill_graph.return_value = True
    runner.supports_mm_inputs = False
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
