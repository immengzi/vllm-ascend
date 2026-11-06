# SPDX-License-Identifier: Apache-2.0
"""Execute actual worker methods with CPU events, queues and SDK doubles.

These tests check publication ordering and failure contracts, not NPU stream
semantics or a resolution of the full-model timeout.
"""

import ast
import importlib.util
import queue
import sys
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[5]
DIRECTORY = ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p"
WORKER = DIRECTORY / "mooncake/pull_worker.py"
SPEC = importlib.util.spec_from_file_location("destination_ready_protocol", DIRECTORY / "dsv41_state.py")
protocol = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = protocol
SPEC.loader.exec_module(protocol)


def methods(class_name, names, env):
    tree = ast.parse(WORKER.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(WORKER), "exec"), env)
    return env[class_name]


class PublicationTests(unittest.TestCase):
    def make_worker(self, fail=False):
        trace = []

        class Event:
            def record(self, stream):
                self.stream = stream
                trace.append("record")

            def synchronize(self):
                trace.append("ready")
                if fail:
                    raise RuntimeError("event failed")

        env = {
            "torch": SimpleNamespace(npu=SimpleNamespace(Event=Event, current_stream=lambda: "model-stream")),
            "zero_state_destination": lambda *args: trace.append("zero"),
        }
        cls = methods("MooncakePullConnectorWorker", {"start_load_kv"}, env)
        worker = cls()
        worker.kv_transfer_config = SimpleNamespace(is_kv_consumer=True)
        receiver = SimpleNamespace(state_caches={}, local_metadata=SimpleNamespace(state_layout={}))
        receiver.add_requests = Mock(side_effect=lambda *args, **kwargs: trace.append("queue"))
        worker._recving_thread = receiver
        return worker, trace

    def metadata(self, state=True):
        return SimpleNamespace(
            requests={
                "r": SimpleNamespace(
                    remote_engine_id="P", state_snapshot={} if state else None, local_full_block_ids=([1], [2], [3])
                )
            }
        )

    def test_reset_event_completion_precedes_queue_publication(self):
        worker, trace = self.make_worker()
        worker.start_load_kv(self.metadata())
        self.assertEqual(trace, ["zero", "record", "ready", "queue"])
        self.assertEqual(worker._recving_thread.add_requests.call_args.kwargs, {"state_destination_ready": True})

    def test_failed_preparation_never_publishes_dma_work(self):
        worker, trace = self.make_worker(fail=True)
        with self.assertRaisesRegex(RuntimeError, "event failed"):
            worker.start_load_kv(self.metadata())
        worker._recving_thread.add_requests.assert_not_called()

    def test_native_path_never_resets_or_waits(self):
        worker, trace = self.make_worker()
        worker.start_load_kv(self.metadata(state=False))
        self.assertEqual(trace, ["queue"])
        self.assertEqual(worker._recving_thread.add_requests.call_args.kwargs, {})

    def test_empty_batch_does_not_touch_device(self):
        worker, trace = self.make_worker()
        worker.start_load_kv(SimpleNamespace(requests={}))
        self.assertEqual(trace, [])

    def test_state_queue_rejects_unprepared_destination(self):
        cls = methods("MooncakePullRecvingThread", {"add_requests"}, {})
        receiver = cls()
        receiver.request_queue = queue.Queue()
        receiver._shutdown_lock = threading.Lock()
        receiver._closing = False
        requests = self.metadata().requests
        with self.assertRaisesRegex(RuntimeError, "destinations must be ready"):
            receiver.add_requests("P", requests)
        self.assertTrue(receiver.request_queue.empty())
        receiver.add_requests("P", requests, state_destination_ready=True)
        self.assertEqual(receiver.request_queue.get_nowait(), ("P", requests))

    def test_shutdown_still_rejects_prepared_requests(self):
        cls = methods("MooncakePullRecvingThread", {"add_requests"}, {})
        receiver = cls()
        receiver.request_queue = queue.Queue()
        receiver._shutdown_lock = threading.Lock()
        receiver._closing = True
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            receiver.add_requests("P", self.metadata().requests, state_destination_ready=True)
        self.assertTrue(receiver.request_queue.empty())

    def test_receiver_read_never_submits_device_reset_or_stream_sync(self):
        trace = []
        snapshot = SimpleNamespace(
            boundary=128,
            original_prompt_tokens=129,
            source_request_id="source",
            transfer_id="generation",
            to_wire=lambda: {},
        )
        socket = SimpleNamespace(setsockopt=lambda *args: None)
        sdk = Mock()
        sdk.batch_transfer_sync_read.side_effect = lambda *args: trace.append("READ") or 0
        env = {
            "StateSnapshot": SimpleNamespace(from_wire=lambda wire: snapshot),
            "transfer_addresses": lambda *args: ([100], [200], [4]),
            "make_zmq_path": lambda *args: "peer",
            "zmq_ctx": lambda *args: nullcontext(socket),
            "zmq": SimpleNamespace(REQ=1, SNDTIMEO=2, RCVTIMEO=3),
            "ensure_zmq_send": lambda *args: None,
            "ensure_zmq_recv": lambda *args: b"ACK",
            "STATE_ACQUIRE_MSG": b"acquire",
            "time": SimpleNamespace(perf_counter=lambda: 1),
            "logger": Mock(),
            "zero_state_destination": Mock(side_effect=AssertionError("receiver reset")),
            "torch": SimpleNamespace(
                npu=SimpleNamespace(current_stream=Mock(side_effect=AssertionError("receiver stream")))
            ),
        }
        cls = methods("MooncakePullRecvingThread", {"_handle_state_requests"}, env)
        receiver = cls()
        receiver.tp_rank, receiver.tp_size = 0, 4
        receiver.engine, receiver.state_caches = sdk, {}
        receiver.encoder = SimpleNamespace(encode=lambda x: x)
        receiver.local_metadata = SimpleNamespace(state_layout={})
        remote = SimpleNamespace(
            metadata_by_pcp_rank={
                0: SimpleNamespace(metadata_by_tp_rank={0: SimpleNamespace(local_ip="peer", te_rpc_port=1)})
            }
        )
        peer = SimpleNamespace(
            pp_size=1, pcp_size=1, dcp_size=1, tp_size=4, use_kv_pp=False, metadata_by_pp_rank={0: remote}
        )
        meta = SimpleNamespace(
            state_snapshot={},
            state_consumer="D/r",
            num_computed_tokens=0,
            num_external_tokens=128,
            local_num_prompt_tokens=129,
            remote_num_prompt_tokens=128,
            remote_request_id="source",
            local_full_block_ids=([1], [2], [3]),
            remote_block_ids=([4], [5], [6]),
        )
        self.assertEqual(receiver._handle_state_requests(peer, "peer", 1, {"r": meta}), set())
        self.assertEqual(trace, ["READ"])


class ResetValidationTests(unittest.TestCase):
    def setUp(self):
        trace = self.trace = []

        class Tensor:
            def __init__(self, name):
                self.name = name

            def __getitem__(self, slot):
                return SimpleNamespace(zero_=lambda: trace.append((self.name, slot)))

        self.caches = {"h": Tensor("history"), "w": Tensor("window"), "c": [Tensor("compressor")]}
        self.planes = [
            SimpleNamespace(kind=k, name=n, group=g, capacity=4, cache_index=0)
            for k, n, g in [("history", "h", 0), ("window", "w", 1), ("compressor", "c", 2)]
        ]
        tree = ast.parse((DIRECTORY / "mooncake/dsv41.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "zero_state_destination")
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn],
            type_ignores=[],
        )
        env = {"decode_planes": lambda layout: self.planes, "validate_ids": protocol.validate_ids}
        exec(compile(ast.fix_missing_locations(module), "zero-state", "exec"), env)
        self.reset = env["zero_state_destination"]

    def test_valid_state_slots_only_history_is_untouched(self):
        self.reset(self.caches, ([1, 2], [2], [3]), {})
        self.assertEqual(self.trace, [("window", 2), ("compressor", 3)])

    def test_invalid_later_plane_does_not_partially_zero_earlier_plane(self):
        for bad in [[], [0], [-1], [4], [1, 2]]:
            with self.subTest(ids=bad), self.assertRaises(ValueError):
                self.reset(self.caches, ([1], [2], bad), {})
            self.assertEqual(self.trace, [])


if __name__ == "__main__":
    unittest.main()
