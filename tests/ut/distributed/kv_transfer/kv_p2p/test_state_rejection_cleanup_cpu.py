# SPDX-License-Identifier: Apache-2.0
"""Actual scheduler methods: post-admission failure versus frontend cleanup."""

import ast
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class RejectionCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[5] / (
            "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_scheduler.py"
        )
        tree = ast.parse(source.read_text())
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MooncakePullConnectorScheduler")
        owner.bases = []
        owner.body = [
            n
            for n in owner.body
            if isinstance(n, ast.FunctionDef) and n.name in ("request_finished", "update_connector_output")
        ]
        cls.env = {
            "logger": Mock(),
            "RequestStatus": SimpleNamespace(FINISHED_LENGTH_CAPPED="done"),
            "StateSnapshot": SimpleNamespace(from_wire=lambda wire: SimpleNamespace(**wire)),
            "STATE_RECEIVE_COMPLETION_HISTORY_SIZE": 2,
        }
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), owner],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), cls.env)
        cls.scheduler_type = cls.env["MooncakePullConnectorScheduler"]

    def scheduler(self):
        obj = self.scheduler_type()
        obj.state_mode = True
        obj.engine_id = "D-dp2"
        obj._recving_thread = SimpleNamespace(add_request=Mock())
        obj._sending_thread = None
        obj.kv_transfer_config = SimpleNamespace(kv_load_failure_policy="fail")
        obj._reqs_recv_info = {}
        obj._state_receive_completions = OrderedDict()
        return obj

    def request(self, generation="generation", host="P", port=24098, synthetic=True):
        return SimpleNamespace(
            request_id="external-http-id",
            status="aborted",
            abort_immediately=synthetic,
            kv_transfer_params=dict(
                do_remote_prefill=True,
                remote_host=host,
                remote_port=port,
                remote_request_id="P-source",
                dsv41_state_snapshot=dict(transfer_id=generation),
            ),
        )

    def finished_worker(self, obj, generation="generation", host="P", port=24098):
        remote = (host, port, "P-source", generation, "D-dp2/internal-engine-id")
        obj._reqs_recv_info["internal-engine-id"] = remote
        output = SimpleNamespace(
            finished_recving={"internal-engine-id"}, failed_recving={"internal-engine-id"}, finished_sending=None
        )
        obj.update_connector_output(output)
        return remote

    def test_frontend_cleanup_after_worker_failure_keeps_only_aggregate_owner(self):
        obj = self.scheduler()
        remote = self.finished_worker(obj)
        obj.request_finished(self.request(), ([],))
        obj._recving_thread.add_request.assert_called_once_with(*remote)

    def test_pre_admission_rejection_without_worker_completion_still_notifies(self):
        obj = self.scheduler()
        obj.request_finished(self.request(), ([],))
        obj._recving_thread.add_request.assert_called_once_with(
            "P", 24098, "P-source", "generation", "D-dp2/external-http-id"
        )

    def test_synthetic_cleanup_does_not_drop_a_different_generation_or_peer(self):
        for kwargs in (dict(generation="new"), dict(host="another-P"), dict(port=24099)):
            obj = self.scheduler()
            self.finished_worker(obj)
            obj._recving_thread.add_request.reset_mock()
            obj.request_finished(self.request(**kwargs), ([],))
            obj._recving_thread.add_request.assert_called_once()

    def test_no_completion_is_invented_while_dma_still_pending(self):
        obj = self.scheduler()
        obj._reqs_recv_info["internal-engine-id"] = ("P", 24098, "P-source", "generation", "owner")
        obj.request_finished(self.request(), ([],))
        self.assertIn("internal-engine-id", obj._reqs_recv_info)
        self.assertEqual(obj._state_receive_completions, {})

    def test_normal_request_is_not_treated_as_frontend_synthetic_cleanup(self):
        obj = self.scheduler()
        self.finished_worker(obj)
        obj._recving_thread.add_request.reset_mock()
        obj.request_finished(self.request(synthetic=False), ([],))
        obj._recving_thread.add_request.assert_called_once()

    def test_completion_history_is_bounded_and_eviction_does_not_fake_ack(self):
        obj = self.scheduler()
        for generation in ("old", "middle", "new"):
            self.finished_worker(obj, generation)
        self.assertEqual(list(obj._state_receive_completions), [("P", 24098, "middle"), ("P", 24098, "new")])
        obj._recving_thread.add_request.reset_mock()
        obj.request_finished(self.request(generation="old"), ([],))
        obj._recving_thread.add_request.assert_called_once()

    def test_native_worker_completion_does_not_touch_state_history(self):
        obj = self.scheduler()
        obj.state_mode = False
        obj._reqs_recv_info["legacy"] = ("P", 24098, "P-source")
        obj.update_connector_output(
            SimpleNamespace(finished_recving={"legacy"}, failed_recving=set(), finished_sending=None)
        )
        obj._recving_thread.add_request.assert_called_once_with("P", 24098, "P-source")
        self.assertEqual(obj._state_receive_completions, {})


if __name__ == "__main__":
    unittest.main()
