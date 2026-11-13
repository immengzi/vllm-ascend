# SPDX-License-Identifier: Apache-2.0
"""Actual receiver methods with a pending-DMA SDK double; no NPU imports.

The SDK double can return before its pending write is released. The contract
is no receive completion, no retry/clearing, retained cache/engine references,
and a fatal model-thread error. This does not prove device teardown fences.
"""

import ast
import queue
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

SOURCE = Path(__file__).resolve().parents[5] / ("vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py")


def actual_classes(env):
    tree = ast.parse(SOURCE.read_text())
    methods = {
        "MooncakePullRecvingThread": {
            "run",
            "_get_remote_endpoint",
            "_handle_state_requests",
            "_mark_request_failed",
            "_quarantine_state_dma",
            "raise_if_unfenced_state_dma",
            "get_and_clear_finished_requests",
            "get_and_clear_invalid_block_ids",
            "add_requests",
            "shutdown",
        },
        "MooncakePullConnectorWorker": {"start_load_kv"},
    }
    selected = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name == "UnfencedStateTransferError":
            selected.append(node)
        elif node.name in methods:
            node.bases = []
            node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods[node.name]]
            selected.append(node)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), env)
    return env


class UnfencedStateDMATests(unittest.TestCase):
    def receiver(self, result=0, reject=False):
        snapshot = SimpleNamespace(
            boundary=128,
            original_prompt_tokens=129,
            source_request_id="P/r",
            transfer_id="generation",
            to_wire=lambda: {},
        )
        socket = SimpleNamespace(setsockopt=lambda *args: None)
        env = actual_classes(
            dict(
                queue=queue,
                logger=Mock(),
                torch=SimpleNamespace(npu=SimpleNamespace(set_device=Mock())),
                KVConnectorTransferResults=lambda: SimpleNamespace(finished_recving=set(), failed_recving=set()),
                StateSnapshot=SimpleNamespace(from_wire=lambda wire: snapshot),
                transfer_addresses=lambda *args: ([100], [200], [4]),
                make_zmq_path=lambda *args: "peer",
                zmq_ctx=lambda *args: nullcontext(socket),
                zmq=SimpleNamespace(REQ=1, SNDTIMEO=2, RCVTIMEO=3),
                ensure_zmq_send=Mock(),
                ensure_zmq_recv=lambda *args: b"REJECT" if reject else b"ACK",
                STATE_ACQUIRE_MSG=b"acquire",
                time=SimpleNamespace(perf_counter=lambda: 0),
            )
        )
        obj = env["MooncakePullRecvingThread"]()
        obj._shutdown_lock = threading.Lock()
        obj._closing = False
        obj._unfenced_state_requests = {}
        obj.finished_requests, obj.request_queue = queue.SimpleQueue(), queue.Queue()
        obj.invalid_block_ids, obj.invalid_block_ids_lock = set(), threading.Lock()
        obj.can_report_invalid_block_ids = False
        obj.device, obj.ready_event = "cpu-double", threading.Event()
        obj.tp_rank, obj.tp_size = 0, 4
        sdk = Mock()
        sdk.batch_transfer_sync_read.side_effect = result if isinstance(result, BaseException) else None
        sdk.batch_transfer_sync_read.return_value = result
        obj.engine, obj.state_caches = sdk, {"retained_tensor_double": object()}
        obj.encoder = SimpleNamespace(encode=lambda x: x)
        obj.local_metadata = SimpleNamespace(state_layout={})
        remote = SimpleNamespace(
            metadata_by_pcp_rank={
                0: SimpleNamespace(metadata_by_tp_rank={0: SimpleNamespace(local_ip="P", te_rpc_port=20015)})
            }
        )
        peer = SimpleNamespace(
            pp_size=1, pcp_size=1, dcp_size=1, tp_size=4, use_kv_pp=False, metadata_by_pp_rank={0: remote}
        )
        obj._handle_requests = lambda engine, host, port, reqs: obj._handle_state_requests(peer, host, port, reqs)
        meta = SimpleNamespace(
            state_snapshot={},
            state_consumer="D/r",
            num_computed_tokens=0,
            num_external_tokens=128,
            local_num_prompt_tokens=129,
            remote_num_prompt_tokens=128,
            remote_request_id="P/r",
            local_full_block_ids=([1], [2], [3]),
            remote_block_ids=([4], [5], [6]),
            remote_host="P",
            remote_port=24096,
            local_block_ids=([1], [2], [3]),
        )
        return obj, meta, env

    def run_batch(self, obj, requests):
        obj.request_queue.put(("P-engine", requests))
        obj.request_queue.put(None)
        obj.run()

    def test_success_still_publishes_ordinary_completion(self):
        obj, meta, _ = self.receiver()
        self.run_batch(obj, {"r": meta})
        result = obj.get_and_clear_finished_requests()
        self.assertEqual(result.finished_recving, {"r"})
        self.assertEqual(result.failed_recving, set())
        self.assertEqual(obj._unfenced_state_requests, {})

    def test_pre_sdk_rejection_is_a_fenced_failure(self):
        obj, meta, _ = self.receiver(reject=True)
        self.run_batch(obj, {"r": meta})
        result = obj.get_and_clear_finished_requests()
        self.assertEqual(result.finished_recving, {"r"})
        self.assertEqual(result.failed_recving, {"r"})
        obj.engine.batch_transfer_sync_read.assert_not_called()

    def test_sdk_timeout_with_pending_write_never_publishes_completion(self):
        obj, meta, env = self.receiver(-1)
        pending_write = threading.Event()
        touched = []

        def late_write():
            pending_write.wait()
            touched.append(obj.state_caches["retained_tensor_double"])

        writer = threading.Thread(target=late_write)
        writer.start()
        try:
            self.run_batch(obj, {"r": meta})
            self.assertTrue(obj.finished_requests.empty(), "A failure completion would release destinations and source")
            self.assertIs(obj._unfenced_state_requests["r"], meta)
            with self.assertRaisesRegex(env["UnfencedStateTransferError"], "terminal DMA fence"):
                obj.get_and_clear_finished_requests()
            self.assertFalse(touched)
        finally:
            pending_write.set()
            writer.join(2)
        self.assertEqual(touched, [obj.state_caches["retained_tensor_double"]])

    def test_sdk_exception_and_unexpected_positive_status_are_also_unfenced(self):
        for result in (RuntimeError("SDK exception"), 1):
            with self.subTest(result=result):
                obj, meta, env = self.receiver(result)
                self.run_batch(obj, {"r": meta})
                self.assertTrue(obj.finished_requests.empty())
                with self.assertRaises(env["UnfencedStateTransferError"]):
                    obj.get_and_clear_invalid_block_ids()

    def test_first_unfenced_read_stops_later_requests_and_retains_the_batch(self):
        obj, meta, _ = self.receiver(-1)
        self.run_batch(obj, {"first": meta, "second": meta})
        self.assertEqual(obj.engine.batch_transfer_sync_read.call_count, 1)
        self.assertEqual(set(obj._unfenced_state_requests), {"first", "second"})
        self.assertTrue(obj.finished_requests.empty())
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            obj.add_requests("P", {"retry": meta}, state_destination_ready=True)

    def test_joined_receiver_is_not_mistaken_for_sdk_drain(self):
        obj, meta, env = self.receiver(-1)
        self.run_batch(obj, {"r": meta})
        sdk, caches = obj.engine, obj.state_caches
        obj.join, obj.is_alive, obj.executor = Mock(), Mock(return_value=False), Mock()
        with self.assertRaises(env["UnfencedStateTransferError"]):
            obj.shutdown()
        self.assertIs(obj.engine, sdk)
        self.assertIs(obj.state_caches, caches)
        obj.executor.shutdown.assert_not_called()

    def test_model_thread_raises_before_even_an_empty_next_batch(self):
        obj, meta, env = self.receiver(-1)
        self.run_batch(obj, {"r": meta})
        worker = env["MooncakePullConnectorWorker"]()
        worker.kv_transfer_config = SimpleNamespace(is_kv_consumer=True)
        worker._recving_thread = obj
        with self.assertRaises(env["UnfencedStateTransferError"]):
            worker.start_load_kv(SimpleNamespace(requests={}))

    def test_native_failure_keeps_existing_failure_completion(self):
        obj, meta, _ = self.receiver()
        meta.state_snapshot = None
        obj._handle_requests = lambda *args: {"r"}
        self.run_batch(obj, {"r": meta})
        result = obj.get_and_clear_finished_requests()
        self.assertEqual(result.finished_recving, {"r"})
        self.assertEqual(result.failed_recving, {"r"})


if __name__ == "__main__":
    unittest.main()
