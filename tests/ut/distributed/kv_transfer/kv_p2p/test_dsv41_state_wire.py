# SPDX-License-Identifier: Apache-2.0
"""Actual Mooncake MessagePack codec checks; requires msgspec, no vLLM/NPU.

Marker bases are stubbed; BlockIds mirrors the fixed vLLM alias. Dataclass
definitions and validation come from metadata.py. Handshakes use the real typed
msgspec decoder; the request check does not stand in for full vLLM IPC.
"""

import ast
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

import msgspec

ROOT = Path(__file__).resolve().parents[5]
DIRECTORY = ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p"
SPEC = importlib.util.spec_from_file_location("dsv41_wire_protocol", DIRECTORY / "dsv41_state.py")
assert SPEC is not None and SPEC.loader is not None
protocol = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = protocol
SPEC.loader.exec_module(protocol)


def metadata_types():
    source = DIRECTORY / "mooncake/metadata.py"
    tree = ast.parse(source.read_text())
    # vLLM imports supply marker bases and a type alias, not wire behavior.
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith("vllm.")
    ]
    module = types.ModuleType("dsv41_wire_metadata")
    module.KVConnectorHandshakeMetadata = object
    module.KVConnectorMetadata = object
    module.BlockIds = tuple[list[int], ...] | list[list[int]]
    sys.modules[module.__name__] = module
    exec(compile(tree, str(source), "exec"), module.__dict__)
    return module


class WireTests(unittest.TestCase):
    def setUp(self):
        self.types = metadata_types()
        self.snapshot = protocol.StateSnapshot.create("generation-1", "source", "weights", 129, 130, 4)
        self.layout = {
            "version": 1,
            "model_id": "weights",
            "tp_size": 4,
            "fingerprint": "format",
            "planes": [
                {
                    "name": "m.swa_cache",
                    "cache_index": 0,
                    "kind": "window",
                    "group": 1,
                    "capacity": 49,
                    "rows": 128,
                    "row_bytes": 1024,
                    "block_stride": 131072,
                    "tokens_per_row": 1,
                    "dtype": "torch.bfloat16",
                    "shape": (128, 1, 512),
                }
            ],
        }
        window = self.layout["planes"][0]
        self.layout["planes"].extend(
            [
                dict(
                    window,
                    name="m.long_kv_cache",
                    kind="history",
                    group=0,
                    capacity=8877,
                    rows=64,
                    block_stride=65536,
                    tokens_per_row=2,
                    shape=(64, 1, 512),
                ),
                dict(
                    window,
                    name="m.compressor.state_cache",
                    kind="compressor",
                    group=2,
                    rows=32,
                    row_bytes=4096,
                    dtype="torch.float32",
                    shape=(32, 1, 1024),
                ),
            ]
        )

    def test_typed_worker_and_merged_handshake(self):
        t = self.types
        planes = self.layout["planes"]
        worker = t.MooncakeTransferMetadata(
            engine_id="P",
            te_rpc_port=1234,
            block_size=128,
            num_blocks=8877,
            layer_names=[p["name"] for p in planes],
            layer_block_sizes=[128] * 3,
            group_indices=[p["group"] for p in planes],
            kv_caches_base_addr=[[4096 + i * 1000000000] for i in range(3)],
            block_strides=[[p["block_stride"]] for p in planes],
            block_lens=[[p["rows"] * p["row_bytes"]] for p in planes],
            block_shapes=[[p["shape"]] for p in planes],
            block_size_scales=[[1]] * 3,
            local_ip="127.0.0.1",
            handshake_port=14500,
            state_layout=self.layout,
        )
        decoded = msgspec.msgpack.Decoder(t.MooncakeTransferMetadata).decode(msgspec.msgpack.encode(worker))
        self.assertEqual(decoded.block_shapes, worker.block_shapes)
        self.assertEqual(
            next(p for p in protocol.decode_planes(decoded.state_layout) if p.kind == "window").shape, (128, 1, 512)
        )
        tp = t.MooncakeTPTransferMetadata(1234, list(range(3)), worker.kv_caches_base_addr, "127.0.0.1", 14500)
        pp = t.MooncakePPTransferMetadata(
            worker.block_size,
            worker.num_blocks,
            worker.layer_names,
            worker.layer_block_sizes,
            worker.group_indices,
            worker.block_strides,
            worker.block_lens,
            worker.block_shapes,
            worker.block_size_scales,
            {0: t.MooncakePCPTransferMetadata({rank: tp for rank in range(4)})},
            self.layout,
        )
        groups = t.MooncakeTransferMetadataGroups("P", "127.0.0.1", 14512, 1, 1, 1, 4, False, {0: pp})
        restored = msgspec.msgpack.Decoder(t.MooncakeTransferMetadataGroups).decode(msgspec.msgpack.encode(groups))
        self.assertEqual(
            set(restored.metadata_by_pp_rank[0].metadata_by_pcp_rank[0].metadata_by_tp_rank), set(range(4))
        )
        planes = protocol.decode_planes(restored.metadata_by_pp_rank[0].state_layout)
        self.assertEqual({p.kind: p.capacity for p in planes}, {"history": 8877, "window": 49, "compressor": 49})

    def test_http_snapshot_and_control_message_roundtrip(self):
        # HTTP JSON converts tuples to lists; MessagePack controls must still
        # compare with the original lease's exact generation and resource map.
        wire = json.loads(json.dumps(self.snapshot.to_wire()))
        self.assertEqual(protocol.StateSnapshot.from_wire(wire), self.snapshot)
        ids = ((81, 82), (7,), (3,))
        control = (protocol.STATE_ACQUIRE_MSG, "generation-1", "D/request", 2, wire, ids)
        decoded = msgspec.msgpack.Decoder().decode(msgspec.msgpack.encode(control))
        self.assertEqual(decoded[0], protocol.STATE_ACQUIRE_MSG)
        self.assertEqual(protocol.StateSnapshot.from_wire(decoded[4]), self.snapshot)
        self.assertEqual(tuple(tuple(group) for group in decoded[5]), ids)

    def test_scheduler_worker_request_metadata(self):
        t = self.types
        request = t.MooncakeConnectorMetadata()
        params = {
            "remote_block_ids": ((81, 82), (7,), (3,)),
            "remote_host": "127.0.0.1",
            "remote_port": 14512,
            "remote_engine_id": "P",
            "remote_request_id": "source",
            "remote_num_prompt_tokens": 129,
            "dsv41_state_snapshot": self.snapshot.to_wire(),
            "dsv41_state_consumer": "D/request",
        }
        request.add_new_req("request", ((14, 25), (5,), (9,)), 130, 129, params, ((14, 25), (5,), (9,)))
        value = request.requests["request"]
        # vLLM IPC is outside this isolated test. In particular its BlockIds
        # alias is an array-like union, not a supported typed msgspec union.
        restored = msgspec.msgpack.decode(msgspec.msgpack.encode(value))
        self.assertEqual(restored["local_full_block_ids"], [[14, 25], [5], [9]])
        self.assertEqual(restored["remote_block_ids"], [[81, 82], [7], [3]])
        self.assertEqual(protocol.StateSnapshot.from_wire(restored["state_snapshot"]).boundary, 129)


if __name__ == "__main__":
    unittest.main(verbosity=2)
