# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU contract tests; no vLLM/torch/NPU installation required.

Run this file directly to avoid the repository-wide UT hardware mocks. Runtime
scheduler methods below are compiled from the actual implementation, not copies.
"""

import ast
import copy
import importlib.util
import json
import math
import queue
import struct
import sys
import threading
import unittest
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[5]
MODULE = ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p/dsv41_state.py"
SPEC = importlib.util.spec_from_file_location("dsv41_state_cpu_test", MODULE)
assert SPEC is not None and SPEC.loader is not None
protocol = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = protocol
SPEC.loader.exec_module(protocol)


def compile_methods(path, class_name, names, env):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    code = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[]
    )
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), env)
    return env[class_name]


def snapshot(boundary):
    return protocol.StateSnapshot.create("generation-1", "producer-request", "weights-v1", boundary, boundary + 1, 4)


def layout(history_capacity=8877, state_capacity=49):
    # Two precisions in the sparse index; row sizes kept small for CPU oracles.
    specs = [
        ("m.long_kv_cache", 0, "history", 0, history_capacity, 64, 8, 544, 2, "bfloat16"),
        ("m.indexer.k_cache", 0, "history", 0, history_capacity, 64, 2, 144, 2, "int8"),
        ("m.indexer.k_cache", 1, "history", 0, history_capacity, 64, 2, 144, 2, "float16"),
        ("m.swa_cache", 0, "window", 1, state_capacity, 128, 8, 1024, 1, "bfloat16"),
        ("m.compressor.state_cache", 0, "compressor", 2, state_capacity, 32, 8, 256, 1, "float32"),
    ]
    return {
        "version": 1,
        "model_id": "weights-v1",
        "tp_size": 4,
        "fingerprint": "format-a",
        "planes": [asdict(protocol.CachePlane(*spec)) for spec in specs],
    }


def runtime_sender():
    path = ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_scheduler.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MooncakeSchedulerSendingThread")
    names = {
        "add_state_lease",
        "_state_control",
        "_retire_state_lease",
        "_handle_finished_request",
        "_mark_finished_locked",
        "_retrieve_expired_requests_locked",
    }
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    code = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[]
    )
    env = {
        "SourceLease": protocol.SourceLease,
        "StateSnapshot": protocol.StateSnapshot,
        "STATE_ACQUIRE_MSG": protocol.STATE_ACQUIRE_MSG,
        "STATE_DONE_MSG": protocol.STATE_DONE_MSG,
        "ACK_MSG": b"ACK",
        "envs": SimpleNamespace(VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=10),
        "time": SimpleNamespace(time=lambda: 1),
        "logger": Mock(),
    }
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), env)
    sender = env[cls.name]()
    sender.state_lock = threading.Lock()
    sender.state_leases = {}
    sender.completed_state_leases = OrderedDict()
    sender.quarantined_state_leases = set()
    sender.delayed_free_requests = OrderedDict()
    sender.finished_request_ids = set()
    sender.early_finished_requests = set()
    sender.finished_requests = queue.SimpleQueue()
    return sender, env


class ProtocolTests(unittest.TestCase):
    def test_connector_shutdown_keeps_timed_out_receiver_and_rejects_new_loads(self):
        folder = ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake"
        worker_cls = compile_methods(
            folder / "pull_worker.py", "MooncakePullConnectorWorker", ["shutdown", "start_load_kv"], {}
        )
        worker = worker_cls()
        receiver = Mock()
        receiver.shutdown.side_effect = TimeoutError("in-flight DMA")
        worker._recving_thread = receiver
        facade_cls = compile_methods(folder / "connector.py", "MooncakeBaseConnector", ["shutdown"], {})
        facade = facade_cls()
        facade.connector_worker = worker
        with self.assertRaises(TimeoutError):
            facade.shutdown()
        self.assertIs(worker._recving_thread, receiver)
        with self.assertRaises(RuntimeError):
            worker.start_load_kv(SimpleNamespace(requests={}))
        receiver.shutdown.side_effect = None
        facade.shutdown()
        self.assertIsNone(worker._recving_thread)
        facade.shutdown()
        facade.connector_worker = None
        facade.shutdown()

    def test_actual_receive_shutdown_drains_inflight_before_releasing_references(self):
        path = ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py"
        env = {"threading": threading, "torch": SimpleNamespace(npu=SimpleNamespace(set_device=Mock()))}
        cls = compile_methods(path, "MooncakePullRecvingThread", ["add_requests", "run", "shutdown"], env)

        # Run the actual queue loop, replacing only DMA/device access.
        class Receiver(cls, threading.Thread):
            pass

        receiver = Receiver.__new__(Receiver)
        threading.Thread.__init__(receiver, daemon=True)
        receiver.device = 0
        receiver.ready_event = threading.Event()
        receiver.request_queue = queue.Queue()
        receiver.finished_requests = queue.SimpleQueue()
        receiver._shutdown_lock = threading.Lock()
        receiver._closing = False
        receiver.engine = object()
        receiver.executor = Mock()
        receiver.state_caches = {"registered": object()}
        receiver.can_report_invalid_block_ids = False
        receiver._get_remote_endpoint = Mock(return_value=("source", 1))
        started, release = threading.Event(), threading.Event()

        def transfer(*args):
            started.set()
            self.assertTrue(release.wait(2))
            return set()

        receiver._handle_requests = transfer
        receiver.start()
        try:
            self.assertTrue(receiver.ready_event.wait(1))
            req = SimpleNamespace(remote_host="source", remote_port=1)
            receiver.add_requests("peer", {"r": req})
            self.assertTrue(started.wait(1))
            with self.assertRaises(TimeoutError):
                receiver.shutdown(timeout=0.01)
            self.assertIsNotNone(receiver.engine)
            self.assertIsNotNone(receiver.state_caches)
            receiver.executor.shutdown.assert_not_called()
            with self.assertRaises(RuntimeError):
                receiver.add_requests("peer", {"late": req})
        finally:
            release.set()
            receiver.shutdown(timeout=1)
        self.assertFalse(receiver.is_alive())
        self.assertEqual(receiver.finished_requests.get_nowait(), ("r", False))
        self.assertEqual(receiver.request_queue.unfinished_tasks, 0)
        receiver.executor.shutdown.assert_called_with(wait=True)
        self.assertIsNone(receiver.engine)
        self.assertIsNone(receiver.state_caches)
        receiver.shutdown(timeout=1)
        self.assertEqual(receiver.request_queue.unfinished_tasks, 0)

    def test_native_index_offset_and_state_capacity_are_separate(self):
        path = ROOT / "vllm_ascend/models/deepseek_v41/request_state.py"
        tree = ast.parse(path.read_text())
        methods = [
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("cache_rows", "index_cache_offset")
        ]
        env = {}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), env)
        module = "vllm_ascend.distributed.kv_transfer.kv_p2p.dsv41_state"
        source = SimpleNamespace(unpadded_page_size_bytes=65536)
        native = SimpleNamespace(num_blocks=8877, kv_cache_groups=[SimpleNamespace(layer_names=["native-kv"])])
        state = SimpleNamespace(
            num_blocks=8877,
            kv_cache_groups=[
                SimpleNamespace(layer_names=[n]) for n in ("m.long_kv_cache", "m.swa_cache", "m.compressor.state_cache")
            ],
            kv_cache_tensors=[SimpleNamespace(layers=["m.swa_cache"], size=17 * 131072, block_stride=131072)],
        )
        with patch.dict(sys.modules, {module: protocol}):
            self.assertEqual(env["cache_rows"](native, "m.swa_cache"), 8877)
            self.assertEqual(env["index_cache_offset"](native, source), 65536)
            self.assertEqual(env["cache_rows"](state, "m.swa_cache"), 17)
            self.assertEqual(env["index_cache_offset"](state, source), 0)

    def test_feature_scope_rejects_unsupported_transfer_combinations(self):
        extra = {"dsv41_state_pd": True, "state_model_id": "weights-v1"}
        cfg = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_connector="MooncakePullConnector",
                kv_load_failure_policy="fail",
                get_from_extra_config=lambda k, default: extra.get(k, default),
            ),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
            ),
            speculative_config=None,
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            model_config=SimpleNamespace(enable_sleep_mode=False),
        )
        self.assertEqual(protocol.validate_pd_config(cfg), "weights-v1")
        for obj, field, value in [
            (cfg.kv_transfer_config, "kv_connector", "AscendStoreConnector"),
            (cfg.kv_transfer_config, "kv_load_failure_policy", "recompute"),
            (cfg.parallel_config, "decode_context_parallel_size", 2),
            (cfg.parallel_config, "prefill_context_parallel_size", 2),
            (cfg.parallel_config, "pipeline_parallel_size", 2),
            (cfg.cache_config, "enable_prefix_caching", True),
            (cfg, "speculative_config", object()),
            (cfg.model_config, "enable_sleep_mode", True),
        ]:
            original = getattr(obj, field)
            setattr(obj, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                protocol.validate_pd_config(cfg)
            setattr(obj, field, original)

    def test_actual_scheduler_last_prompt_token_and_cancel_before_admission(self):
        directory = ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake"
        env = {
            "StateSnapshot": protocol.StateSnapshot,
            "logger": Mock(),
            "RequestStatus": SimpleNamespace(FINISHED_LENGTH_CAPPED="length"),
            "uuid": SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="source-generation")),
            "time": SimpleNamespace(time=lambda: 1),
        }
        base = compile_methods(
            directory / "base_scheduler.py",
            "MooncakeBaseConnectorScheduler",
            {"on_new_request", "_truncate_request_for_prefill", "_state_prefill_token_count"},
            env,
        )
        pull = compile_methods(
            directory / "pull_scheduler.py",
            "MooncakePullConnectorScheduler",
            {"request_finished", "get_num_new_matched_tokens"},
            env,
        )
        producer = pull()
        producer.on_new_request = base.on_new_request.__get__(producer)
        producer._truncate_request_for_prefill = base._truncate_request_for_prefill.__get__(producer)
        producer.state_mode, producer.state_model_id, producer.tp_size = True, "weights-v1", 4
        producer.need_truncate = True
        producer.engine_id, producer.side_channel_host, producer.side_channel_port = "P", "host", 9999
        producer._get_transfer_block_ids = lambda ids, n: ids
        producer._reqs_need_send = {}
        producer._sending_thread, _ = runtime_sender()
        for size in (2, 130):
            params = {"do_remote_decode": True}
            request = SimpleNamespace(
                kv_transfer_params=params,
                num_prompt_tokens=size,
                prompt_token_ids=list(range(size)),
                _all_token_ids=list(range(size)),
                prompt_embeds=None,
                request_id=f"request-{size}",
                num_computed_tokens=size - 1,
                num_in_flight_tokens=0,
                status="length",
                output_token_ids=[999],
            )
            producer.on_new_request(request)
            producer.on_new_request(request)  # preemption/requeue must not truncate twice
            self.assertEqual(request.prompt_token_ids, list(range(size - 1)))
            env["uuid"].uuid4 = lambda size=size: SimpleNamespace(hex=f"source-generation-{size}")
            delay, response = producer.request_finished(request, ([1, 2], [3], [4]))
            self.assertTrue(delay)
            handoff = protocol.StateSnapshot.from_wire(response["dsv41_state_snapshot"])
            self.assertEqual((handoff.boundary, handoff.original_prompt_tokens), (size - 1, size))
            decoder = pull()
            decoder.state_mode, decoder.state_model_id, decoder.tp_size = True, "weights-v1", 4
            decoder.need_truncate, decoder.engine_id = True, "D"
            decoder._state_prefill_token_count = base._state_prefill_token_count.__get__(decoder)
            decoder._recving_thread = Mock()
            decode_request = SimpleNamespace(
                kv_transfer_params=response,
                prompt_token_ids=list(range(size)),
                request_id=f"D-{size}",
                status="cancelled",
            )
            self.assertEqual(decoder.get_num_new_matched_tokens(decode_request, 0), (size - 1, True))
            self.assertEqual(decode_request.prompt_token_ids[-1], size - 1)
            self.assertNotIn(999, decode_request.prompt_token_ids)
            self.assertEqual(decoder.request_finished(decode_request, ([], [], [])), (False, None))
            decoder._recving_thread.add_request.assert_called_once_with(
                "host", 9999, request.request_id, handoff.transfer_id, f"D/{decode_request.request_id}"
            )

    def test_actual_admission_wrapper_checks_both_state_pools(self):
        path = ROOT / "vllm_ascend/models/deepseek_v41/request_state.py"
        tree = ast.parse(path.read_text())
        wrapper = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "allocate_slots")

        class Fixed:
            def __init__(self, free):
                self.req_to_blocks = defaultdict(list)
                self.block_pool = SimpleNamespace(get_num_free_blocks=lambda: free)

        original = Mock(return_value="allocated")
        env = {"FixedStateManager": Fixed, "original_allocate": original}
        module = ast.Module(body=[wrapper], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
        name = "vllm_ascend.distributed.kv_transfer.kv_p2p.dsv41_state"
        request = SimpleNamespace(request_id="r")
        with patch.dict(sys.modules, {name: protocol}):
            for free in [(0, 1), (1, 0), (0, 0)]:
                managers = [Fixed(n) for n in free]
                manager = SimpleNamespace(coordinator=SimpleNamespace(single_type_managers=managers))
                self.assertIsNone(env["allocate_slots"](manager, request, 1))
                original.assert_not_called()
            managers = [Fixed(1), Fixed(1)]
            manager = SimpleNamespace(coordinator=SimpleNamespace(single_type_managers=managers))
            self.assertEqual(env["allocate_slots"](manager, request, 1), "allocated")
            self.assertEqual(original.call_count, 1)
            # Already resident slots remain usable even when no spare slots exist.
            managers = [Fixed(0), Fixed(0)]
            for state in managers:
                state.req_to_blocks["r"] = [9]
            manager.coordinator.single_type_managers = managers
            self.assertEqual(env["allocate_slots"](manager, request, 1), "allocated")

    def test_actual_registration_and_read_path_with_mock_transport(self):
        directory = ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake"
        protocol_name = "vllm_ascend.distributed.kv_transfer.kv_p2p.dsv41_state"
        adapter_name = "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.dsv41"
        adapter_spec = importlib.util.spec_from_file_location(adapter_name, directory / "dsv41.py")
        adapter = importlib.util.module_from_spec(adapter_spec)
        metadata_tree = ast.parse((directory / "metadata.py").read_text())
        metadata_class = next(
            n for n in metadata_tree.body if isinstance(n, ast.ClassDef) and n.name == "MooncakeTransferMetadata"
        )
        metadata_env = {"dataclass": dataclass, "KVConnectorHandshakeMetadata": object}
        metadata_ast = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), metadata_class],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(metadata_ast), "metadata.py", "exec"), metadata_env)
        with patch.dict(sys.modules, {protocol_name: protocol, adapter_name: adapter}):
            adapter_spec.loader.exec_module(adapter)
            memory = {}

            class Tensor:
                def __init__(self, plane, base):
                    self.itemsize = {"bfloat16": 2, "float16": 2, "float32": 4, "int8": 1}[plane.dtype]
                    self.shape = (plane.capacity, plane.rows, 1, plane.row_bytes // self.itemsize)
                    self.dtype = plane.dtype
                    self.base = base
                    self.strides = (
                        plane.block_stride // self.itemsize,
                        plane.row_bytes // self.itemsize,
                        plane.row_bytes // self.itemsize,
                        1,
                    )
                    memory[base] = bytearray(plane.capacity * plane.block_stride)

                def stride(self, dim=None):
                    return self.strides if dim is None else self.strides[dim]

                def element_size(self):
                    return self.itemsize

                def data_ptr(self):
                    return self.base

                def __getitem__(self, slot):
                    start = slot * self.strides[0] * self.itemsize
                    size = math.prod(self.shape[1:]) * self.itemsize

                    def zero():
                        memory[self.base][start : start + size] = bytes(size)

                    return SimpleNamespace(zero_=zero)

            def config():
                extra = {"dsv41_state_pd": True, "state_model_id": "weights-v1"}
                return SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(
                        kv_connector="MooncakePullConnector",
                        kv_load_failure_policy="fail",
                        get_from_extra_config=lambda k, d: extra.get(k, d),
                    ),
                    parallel_config=SimpleNamespace(
                        pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
                    ),
                    speculative_config=None,
                    cache_config=SimpleNamespace(enable_prefix_caching=False),
                    model_config=SimpleNamespace(
                        hf_config=SimpleNamespace(to_dict=lambda: {"layers": 40}),
                        dtype="bfloat16",
                        quantization="ascend",
                    ),
                )

            registry = Mock()
            env = {
                "logger": Mock(),
                "as_kv_cache_tensors": lambda v: tuple(v) if isinstance(v, tuple) else (v,),
                "get_kv_cache_tensor_layers": lambda t: t.layers,
                "collect_configured_register_regions": lambda cfg, views: SimpleNamespace(
                    ptrs=[t.data_ptr() for v in views.values() for t in (v if isinstance(v, tuple) else (v,))],
                    lengths=[1],
                ),
                "validate_register_region_count": Mock(),
                "global_te": registry,
                "math": math,
                "MooncakeTransferMetadata": metadata_env["MooncakeTransferMetadata"],
                "AscendSFAIndexerCacheSpec": type("UnusedIndexerSpec", (), {}),
            }
            cls = compile_methods(
                directory / "base_worker.py", "MooncakeBaseConnectorWorker", {"register_kv_caches"}, env
            )

            def register(history, states, address_start):
                obj = cls()
                planes = protocol.decode_planes(layout(history, states))
                groups = [
                    SimpleNamespace(
                        layer_names=[n for n in dict.fromkeys(p.name for p in planes) if protocol.cache_kind(n) == kind]
                    )
                    for kind in ("history", "window", "compressor")
                ]
                tensors, caches = [], {}
                specs = []
                for index, plane in enumerate(planes):
                    tensor = Tensor(plane, address_start + index * 10000000)
                    caches.setdefault(plane.name, []).append(tensor)
                for name, values in caches.items():
                    plane = next(p for p in planes if p.name == name)
                    tensors.append(
                        SimpleNamespace(
                            layers=[name], size=plane.capacity * plane.block_stride, block_stride=plane.block_stride
                        )
                    )
                    specs.append(SimpleNamespace(block_size=128, tokens_per_state=plane.tokens_per_row))
                obj.kv_cache_config = SimpleNamespace(
                    num_blocks=history, kv_cache_groups=groups, kv_cache_tensors=tensors
                )
                obj._build_kv_cache_spec_mappings = lambda: None
                obj.layer_name_to_group_index = {p.name: p.group for p in planes}
                obj.layer_name_to_spec_index = {n: i for i, n in enumerate(caches)}
                obj.kv_cache_specs = specs
                obj.vllm_config = config()
                obj.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1), enable_dsa_cp=True)
                obj.pcp_rank, obj.tp_size, obj.tp_rank = 0, 4, 0
                obj.engine_id, obj.te_rpc_port, obj.block_size = "engine-" + str(address_start), 9000, 128
                obj.side_channel_host, obj.handshake_port = "test-host", 9100
                obj.register_kv_caches({n: tuple(v) if len(v) > 1 else v[0] for n, v in caches.items()})
                return obj

            producer, decoder = register(8877, 49, 100000000), register(4096, 17, 200000000)
            self.assertTrue(all(scales == [1] * len(scales) for scales in producer.transfer_metadata.block_size_scales))
            self.assertTrue(all(scales == [1] * len(scales) for scales in decoder.transfer_metadata.block_size_scales))
            self.assertEqual(registry.register_buffer.call_count, 2)
            source_meta = producer.transfer_metadata
            for base, values in memory.items():
                values[:] = bytes([65 if base < 200000000 else 37]) * len(values)
            pp = SimpleNamespace(
                state_layout=source_meta.state_layout,
                layer_names=source_meta.layer_names,
                metadata_by_pcp_rank={0: SimpleNamespace(metadata_by_tp_rank={0: source_meta})},
            )
            remote = SimpleNamespace(
                pp_size=1, pcp_size=1, dcp_size=1, tp_size=4, use_kv_pp=False, metadata_by_pp_rank={0: pp}
            )
            state = snapshot(129)
            p_ids, d_ids = ([1, 2], [3], [4]), ([8, 9], [7], [5])
            sender, _ = runtime_sender()
            sender.add_state_lease(state, 0, p_ids)
            meta = SimpleNamespace(
                state_snapshot=state.to_wire(),
                state_consumer="D/r",
                num_computed_tokens=0,
                num_external_tokens=129,
                local_num_prompt_tokens=130,
                remote_num_prompt_tokens=129,
                remote_request_id=state.source_request_id,
                remote_block_ids=p_ids,
                local_full_block_ids=d_ids,
            )
            submitted = []

            def resolve(address, length):
                for base, data in memory.items():
                    if base <= address and address + length <= base + len(data):
                        return data, address - base
                raise AssertionError("Out-of-bounds DMA address")

            def read(session, local_destinations, remote_sources, sizes):
                # Mooncake READ follows (session, local, peer, lengths), not
                # memcpy's (source, destination) convention. Disjoint address
                # spaces make an accidental reversal fail registration checks.
                self.assertTrue(all(address >= 200000000 for address in local_destinations))
                self.assertTrue(all(address < 200000000 for address in remote_sources))
                submitted.append((local_destinations, remote_sources, sizes))
                for src, dst, size in zip(remote_sources, local_destinations, sizes):
                    source, lo = resolve(src, size)
                    destination, hi = resolve(dst, size)
                    destination[hi : hi + size] = source[lo : lo + size]
                return 0

            class Socket:
                reply = None

                def setsockopt(self, *args):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

            stream = Mock()
            wenv = {
                "StateSnapshot": protocol.StateSnapshot,
                "STATE_ACQUIRE_MSG": protocol.STATE_ACQUIRE_MSG,
                "transfer_addresses": adapter.transfer_addresses,
                "zero_state_destination": adapter.zero_state_destination,
                "make_zmq_path": lambda *a: "endpoint",
                "zmq_ctx": lambda *a: Socket(),
                "zmq": SimpleNamespace(REQ=1, SNDTIMEO=2, RCVTIMEO=3),
                "ensure_zmq_send": lambda sock, message, path: setattr(sock, "reply", sender._state_control(message)),
                "ensure_zmq_recv": lambda sock, path: sock.reply,
                "torch": SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: stream)),
                "time": SimpleNamespace(perf_counter=lambda: 1),
                "logger": Mock(),
            }
            receiver = compile_methods(
                directory / "pull_worker.py", "MooncakePullRecvingThread", {"_handle_state_requests"}, wenv
            )()
            receiver.local_metadata, receiver.tp_size, receiver.tp_rank = decoder.transfer_metadata, 4, 0
            receiver.encoder = SimpleNamespace(encode=lambda x: x)
            receiver.engine = SimpleNamespace(batch_transfer_sync_read=read)
            receiver.state_caches = decoder.kv_caches
            self.assertEqual(receiver._handle_state_requests(remote, "test-host", 9000, {"r": meta}), set())
            self.assertEqual(len(submitted), 1)
            self.assertGreater(sum(submitted[0][2]), 0)
            for address, size in zip(submitted[0][0], submitted[0][2]):
                data, offset = resolve(address, size)
                self.assertEqual(data[offset : offset + size], bytes([65]) * size)
            ring = decoder.kv_caches["m.swa_cache"]
            ring_data = memory[ring.base]
            self.assertEqual(ring_data[:1024], bytes([37]) * 1024)
            stream.synchronize.assert_called_once()
            broken = copy.copy(meta)
            broken.state_snapshot = {
                **state.to_wire(),
                "transfer_id": "partial-transfer",
                "source_request_id": "partial-source",
            }
            broken.remote_request_id = "partial-source"
            broken_snapshot = protocol.StateSnapshot.from_wire(broken.state_snapshot)
            sender.add_state_lease(broken_snapshot, 0, p_ids)
            receiver.engine.batch_transfer_sync_read = lambda *a: -1
            self.assertEqual(
                receiver._handle_state_requests(remote, "test-host", 9000, {"partial": broken}), {"partial"}
            )
            self.assertIn(broken_snapshot.transfer_id, sender.state_leases)
            # A failed data operation may not publish success or release a
            # source while another worker may still be reading it.
            self.assertNotIn(broken_snapshot.source_request_id, sender.finished_request_ids)
            sender._state_control((protocol.STATE_DONE_MSG, broken_snapshot.transfer_id, "D/r"))
            # Aggregate completion retires the source. The same descriptor
            # cannot issue another READ after a destination slot is reused.
            sender._state_control((protocol.STATE_DONE_MSG, state.transfer_id, "D/r"))
            self.assertEqual(receiver._handle_state_requests(remote, "test-host", 9000, {"r": meta}), {"r"})
            self.assertEqual(len(submitted), 1)

    def test_wire_roundtrip_and_inconsistent_boundaries(self):
        for boundary in [
            1,
            2,
            31,
            32,
            33,
            63,
            64,
            65,
            127,
            128,
            129,
            255,
            256,
            257,
            8191,
            8192,
            8193,
            99999,
            100000,
            131071,
            131072,
        ]:
            with self.subTest(boundary=boundary):
                value = snapshot(boundary)
                self.assertEqual(value, protocol.StateSnapshot.from_wire(json.loads(json.dumps(value.to_wire()))))
                wire = value.to_wire()
                wire["window_next"] = (wire["window_next"] + 1) % 128
                with self.assertRaises(ValueError):
                    protocol.StateSnapshot.from_wire(wire)
        with self.assertRaises(ValueError):
            snapshot(0)

    def test_capacities_mapping_valid_ranges_and_sparse_scales(self):
        for boundary in [1, 127, 128, 129, 255, 256, 257, 8193, 100000, 131072]:
            pages = (boundary + 127) // 128
            local_ids = (list(range(100, 100 + pages)), [7], [5])
            remote_ids = (list(range(1, 1 + pages)), [2], [3])
            plan = protocol.build_copy_plan(snapshot(boundary), layout(4096, 17), layout(), local_ids, remote_ids)
            self.assertEqual(
                {(p.name, p.cache_index) for p in plan if "indexer" in p.name},
                set() if boundary == 1 else {("m.indexer.k_cache", 0), ("m.indexer.k_cache", 1)},
            )
            window = next(p for p in plan if p.name.endswith(".swa_cache"))
            self.assertEqual(
                (window.source_offset, window.destination_offset, window.nbytes),
                (2 * 1024, 7 * 1024, min(boundary, 128) * 8),
            )
            self.assertEqual(sum(p.nbytes for p in plan if p.name.endswith(".long_kv_cache")), boundary // 2 * 8)
            for item in plan:
                self.assertGreater(item.nbytes, 0)

    def test_missing_null_duplicate_and_incompatible_resources_fail_before_dma(self):
        for local_ids in [([], [7], [5]), ([0], [7], [5]), ([1], [], [5]), ([1], [7, 8], [5]), ([1, 1], [7], [5])]:
            with self.subTest(ids=local_ids), self.assertRaises(ValueError):
                protocol.build_copy_plan(snapshot(129), layout(), layout(), local_ids, ([3, 4], [2], [3]))
        for field, value in [("model_id", "other"), ("tp_size", 8), ("fingerprint", "different")]:
            remote = layout()
            remote[field] = value
            with self.assertRaises(ValueError):
                protocol.build_copy_plan(snapshot(128), layout(), remote, ([1], [2], [3]), ([4], [5], [6]))
        remote = layout()
        remote["planes"].pop()
        with self.assertRaises(ValueError):
            protocol.build_copy_plan(snapshot(128), layout(), remote, ([1], [2], [3]), ([4], [5], [6]))

    def test_ring_restore_and_first_continuous_decode_against_cpu_oracle(self):
        # Toy projections use the source kernel's absolute-position ring and
        # per-channel softmax pooling of each completed pair. This validates
        # handoff/order, not the full model, BF16 rounding, or NPU operators.
        for boundary in [1, 2, 31, 32, 33, 127, 128, 129, 255, 256, 257, 8193]:
            with self.subTest(boundary=boundary):
                history_ids = list(range(1, (boundary + 127) // 128 + 1))
                source_ids, destination_ids = (history_ids, [2], [3]), (history_ids[::-1], [7], [5])
                source, destination = {}, {}
                for plane in protocol.decode_planes(layout(256, 17)):
                    source[plane.key] = bytearray([19]) * (plane.capacity * plane.block_stride)
                    destination[plane.key] = bytearray([37]) * (plane.capacity * plane.block_stride)

                def projections(position):
                    return float(position % 31 - 15), float(position % 7) / 10

                for position in range(boundary):
                    raw = struct.pack("ff", *projections(position))
                    source[("m.swa_cache", 0)][
                        2 * 1024 + (position % 128) * 8 : 2 * 1024 + (position % 128) * 8 + 8
                    ] = raw
                    source[("m.compressor.state_cache", 0)][
                        3 * 256 + (position % 32) * 8 : 3 * 256 + (position % 32) * 8 + 8
                    ] = raw
                plan = protocol.build_copy_plan(
                    snapshot(boundary), layout(256, 17), layout(256, 17), destination_ids, source_ids
                )
                for key, offset, size in [
                    (("m.swa_cache", 0), 7 * 1024, 1024),
                    (("m.compressor.state_cache", 0), 5 * 256, 256),
                ]:
                    destination[key][offset : offset + size] = bytes(size)
                for item in plan:
                    key = item.name, item.cache_index
                    destination[key][item.destination_offset : item.destination_offset + item.nbytes] = source[key][
                        item.source_offset : item.source_offset + item.nbytes
                    ]
                for position in range(boundary, boundary + 65):
                    value, score = projections(position)
                    if position % 2:
                        prev = struct.unpack_from(
                            "ff", destination[("m.compressor.state_cache", 0)], 5 * 256 + ((position - 1) % 32) * 8
                        )
                        expected_prev = projections(position - 1)
                        actual = (prev[0] * math.exp(prev[1]) + value * math.exp(score)) / (
                            math.exp(prev[1]) + math.exp(score)
                        )
                        expected = (expected_prev[0] * math.exp(expected_prev[1]) + value * math.exp(score)) / (
                            math.exp(expected_prev[1]) + math.exp(score)
                        )
                        self.assertAlmostEqual(actual, expected, places=5)
                    for key, slot, rows in [(("m.swa_cache", 0), 7, 128), (("m.compressor.state_cache", 0), 5, 32)]:
                        struct.pack_into("ff", destination[key], (slot * rows + position % rows) * 8, value, score)
                    low = max(0, position - 127)
                    for absolute in range(low, position + 1):
                        actual = struct.unpack_from(
                            "ff", destination[("m.swa_cache", 0)], (7 * 128 + absolute % 128) * 8
                        )
                        expected = projections(absolute)
                        self.assertAlmostEqual(actual[0], expected[0], places=5)
                        self.assertAlmostEqual(actual[1], expected[1], places=5)
                self.assertEqual(destination[("m.swa_cache", 0)][:1024], bytes([37]) * 1024)

    def test_actual_scheduler_generation_lease_cancel_timeout_and_reuse(self):
        sender, env = runtime_sender()
        ids = ([1, 2], [3], [4])
        state = snapshot(129)
        sender.add_state_lease(state, 0, ids)
        acquire = (protocol.STATE_ACQUIRE_MSG, state.transfer_id, "D/request-1", 0, state.to_wire(), ids)
        self.assertEqual(sender._state_control(acquire), b"ACK")
        self.assertEqual(sender._state_control(acquire), b"ACK")
        self.assertEqual(sender._state_control((*acquire[:2], "other", *acquire[3:])), b"REJECT")
        self.assertEqual(sender._state_control((*acquire[:-1], ([2, 1], [3], [4]))), b"REJECT")
        sender._handle_finished_request(state.source_request_id)
        self.assertFalse(state.source_request_id in sender.finished_request_ids)
        env["time"].time = lambda: 100
        sender._retrieve_expired_requests_locked()
        self.assertFalse(state.source_request_id in sender.finished_request_ids)
        self.assertIn(state.transfer_id, sender.quarantined_state_leases)
        done = (protocol.STATE_DONE_MSG, state.transfer_id, "D/request-1")
        self.assertEqual(sender._state_control(done), b"ACK")
        self.assertEqual(sender._state_control(done), b"ACK")
        self.assertEqual(sender.finished_requests.get_nowait(), state.source_request_id)
        self.assertTrue(sender.finished_requests.empty())
        self.assertEqual(sender._state_control(acquire), b"REJECT")
        fresh = protocol.StateSnapshot.create("generation-2", state.source_request_id, "weights-v1", 129, 130, 4)
        sender.add_state_lease(fresh, 100, ids)
        self.assertEqual(sender._state_control(done), b"ACK")
        self.assertNotIn(fresh.source_request_id, sender.finished_request_ids)
        env["time"].time = lambda: 111
        sender._retrieve_expired_requests_locked()
        self.assertIn(fresh.source_request_id, sender.finished_request_ids)

    def test_joint_allocation_failure_rolls_back_new_appends_only(self):
        class Pool:
            def __init__(self):
                self.returned = []

            def free_blocks(self, blocks):
                self.returned.extend(blocks)

        managers = [
            SimpleNamespace(req_to_blocks=defaultdict(list), block_pool=Pool(), num_cached_block={}, new_block_ids=[])
            for _ in range(3)
        ]
        managers[0].req_to_blocks["r"] = [11]
        managers[0].num_cached_block["r"] = 7

        def allocate():
            for index, manager in enumerate(managers):
                manager.req_to_blocks["r"].append(20 + index)
                manager.num_cached_block["r"] = 100
                manager.new_block_ids.append(20 + index)
                if index == 1:
                    raise RuntimeError("Injected second-state-pool allocation failure")

        with self.assertRaises(RuntimeError):
            protocol.reserve_append_only(managers, "r", allocate)
        self.assertEqual(managers[0].req_to_blocks["r"], [11])
        self.assertEqual(managers[0].num_cached_block["r"], 7)
        self.assertNotIn("r", managers[1].req_to_blocks)
        self.assertEqual([m.block_pool.returned for m in managers], [[20], [21], []])
        self.assertTrue(all(not m.new_block_ids for m in managers))


if __name__ == "__main__":
    unittest.main(verbosity=2)
