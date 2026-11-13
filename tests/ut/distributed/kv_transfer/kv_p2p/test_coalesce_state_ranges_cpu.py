# SPDX-License-Identifier: Apache-2.0
"""Byte-for-byte CPU oracles; run directly without torch/NPU imports."""

import ast
import importlib.util
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
PATH = ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p/dsv41_state.py"
SPEC = importlib.util.spec_from_file_location("coalesce_state_cpu", PATH)
protocol = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = protocol
SPEC.loader.exec_module(protocol)
R = protocol.CopyRange
merge = protocol.coalesce_copy_ranges


class CoalesceTests(unittest.TestCase):
    def test_forward_and_reverse(self):
        for rows in [(R("h", 0, 10, 40, 4), R("h", 0, 14, 44, 4)), (R("h", 0, 14, 44, 4), R("h", 0, 10, 40, 4))]:
            self.assertEqual(merge(rows), (R("h", 0, 10, 40, 8),))

    def test_gaps_directions_regions_and_overlap(self):
        a = R("h", 0, 10, 40, 4)
        for b in [
            R("h", 0, 15, 44, 4),
            R("h", 0, 14, 45, 4),
            R("h", 0, 14, 36, 4),
            R("other", 0, 14, 44, 4),
            R("h", 1, 14, 44, 4),
            R("h", 0, 12, 42, 4),
        ]:
            self.assertEqual(merge((a, b)), (a, b))

    def test_partial_tail_does_not_cover_padding(self):
        rows = (R("h", 0, 0, 0, 8), R("h", 0, 8, 8, 3), R("h", 0, 16, 16, 8))
        self.assertEqual(merge(rows), (R("h", 0, 0, 0, 11), rows[2]))

    def test_limit_and_invalid(self):
        rows = tuple(R("h", 0, i * 4, i * 4 + 16, 4) for i in range(5))
        self.assertEqual([x.nbytes for x in merge(rows, 8)], [8, 8, 4])
        self.assertEqual(merge((R("h", 0, 0, 0, 20),), 8), (R("h", 0, 0, 0, 20),))
        self.assertEqual(merge(()), ())
        for row in [R("h", 0, 0, 0, 0), R("h", 0, -1, 0, 4)]:
            with self.assertRaises(ValueError):
                merge((row,))
        with self.assertRaises(ValueError):
            merge((), 0)

    def test_random_byte_mapping_and_untouched_sentinels(self):
        rng = random.Random(1024)
        for _ in range(600):
            src = bytes(rng.randrange(256) for _ in range(512))
            source_ids, dest_ids = list(range(1, 50)), list(range(1, 50))
            if rng.choice([True, False]):
                source_ids.reverse()
                dest_ids.reverse()
            for ids in (source_ids, dest_ids):
                for _ in range(5):
                    i, j = rng.sample(range(49), 2)
                    ids[i], ids[j] = ids[j], ids[i]
            rows = tuple(R("h", 0, s * 8, d * 8, rng.choice([3, 8, 8, 8])) for s, d in zip(source_ids, dest_ids))
            combined = merge(rows, rng.choice([16, 32, 128]))

            def copy(plan, source=src):
                out = bytearray([173] * 512)
                for r in plan:
                    out[r.destination_offset : r.destination_offset + r.nbytes] = source[
                        r.source_offset : r.source_offset + r.nbytes
                    ]
                return out

            self.assertEqual(copy(rows), copy(combined))
            self.assertEqual(sum(r.nbytes for r in rows), sum(r.nbytes for r in combined))
            self.assertLessEqual(len(combined), len(rows))

    def test_runtime_address_adapter_uses_coalesced_plan(self):
        path = ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/dsv41.py"
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "transfer_addresses")
        rows = (R("h", 0, 16, 32, 8), R("h", 0, 8, 24, 8))
        env = {"build_copy_plan": lambda *a: rows, "coalesce_copy_ranges": merge}
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
        local = SimpleNamespace(state_layout={}, layer_names=["h"], kv_caches_base_addr=[[1000]])
        tp = SimpleNamespace(kv_caches_base_addr=[[2000]])
        remote = SimpleNamespace(
            state_layout={}, layer_names=["h"], metadata_by_pcp_rank=[SimpleNamespace(metadata_by_tp_rank=[tp])]
        )
        self.assertEqual(env["transfer_addresses"](None, local, remote, 0, [], []), ([2008], [1024], [16]))


if __name__ == "__main__":
    unittest.main()
