# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Hermetic tests of the actual completion RPC methods without NPU imports."""

import ast
import types
import unittest
from pathlib import Path
from unittest.mock import Mock


class StopLoop(Exception):
    pass


class TestTerminalStateDone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[5]
        source = root / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_scheduler.py"
        tree = ast.parse(source.read_text())
        owner = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "MooncakeSchedulerRecvingThread")
        methods = [x for x in owner.body if isinstance(x, ast.FunctionDef) and x.name in ("_send_done_recving", "run")]
        module = ast.Module(body=methods, type_ignores=[])
        cls.namespace = {
            "make_zmq_path": lambda _kind, host, port: f"tcp://{host}:{port}",
            "STATE_DONE_MSG": b"state_done",
            "DONE_RECVING_MSG": b"legacy_done",
            "ACK_MSG": b"ACK",
        }
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), cls.namespace)

    def receiver(self, response=b"ACK", error=None):
        sock = Mock()
        result = types.SimpleNamespace(
            encoder=types.SimpleNamespace(encode=lambda value: value),
            _get_remote_socket=Mock(return_value=sock),
            _return_remote_socket=Mock(),
            ready_event=Mock(),
            request_queue=Mock(),
        )
        self.namespace["logger"] = Mock()
        self.namespace["ensure_zmq_send"] = Mock()
        self.namespace["ensure_zmq_recv"] = Mock(return_value=response, side_effect=error)
        result._send_done_recving = types.MethodType(self.namespace["_send_done_recving"], result)
        return result, sock

    def test_explicit_state_rejection_is_terminal_and_socket_is_reused(self):
        receiver, sock = self.receiver(b"REJECT")
        receiver._send_done_recving("host", 24096, "request", "generation", "consumer")
        receiver._return_remote_socket.assert_called_once_with("tcp://host:24096", sock)
        sock.close.assert_not_called()
        self.namespace["logger"].warning.assert_called_once()
        self.assertEqual(self.namespace["ensure_zmq_send"].call_args.args[1], (b"state_done", "generation", "consumer"))

    def test_state_ack_keeps_normal_success_behavior(self):
        receiver, sock = self.receiver()
        receiver._send_done_recving("host", 24096, "request", "generation", "consumer")
        receiver._return_remote_socket.assert_called_once_with("tcp://host:24096", sock)
        self.namespace["logger"].warning.assert_not_called()

    def test_legacy_reject_and_unexpected_state_response_still_raise(self):
        for transfer_id, response in (("", b"REJECT"), ("generation", b"bad")):
            receiver, sock = self.receiver(response)
            with self.assertRaises(RuntimeError):
                receiver._send_done_recving("host", 24096, "request", transfer_id, "consumer")
            sock.close.assert_called_once_with(linger=0)
            receiver._return_remote_socket.assert_not_called()

    def test_transport_exception_still_retries_in_worker_loop(self):
        receiver, sock = self.receiver(error=OSError("transient transport failure"))
        request = ("host", 24096, "request", "generation", "consumer")
        receiver.request_queue.get.side_effect = [request, StopLoop()]
        with self.assertRaises(StopLoop):
            self.namespace["run"](receiver)
        receiver.request_queue.put.assert_called_once_with(request)
        receiver.request_queue.task_done.assert_called_once()
        sock.close.assert_called_once_with(linger=0)

    def test_explicit_state_rejection_is_not_requeued_by_worker_loop(self):
        receiver, _ = self.receiver(b"REJECT")
        receiver.request_queue.get.side_effect = [
            ("host", 24096, "request", "generation", "consumer"),
            StopLoop(),
        ]
        with self.assertRaises(StopLoop):
            self.namespace["run"](receiver)
        receiver.request_queue.put.assert_not_called()
        receiver.request_queue.task_done.assert_called_once()


if __name__ == "__main__":
    unittest.main()
