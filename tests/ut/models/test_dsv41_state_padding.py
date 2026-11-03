# SPDX-License-Identifier: Apache-2.0
"""Execute the actual kernel AST; invalid padded rows may not access storage."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/state_kernels.py"


class Pointer:
    __array_priority__ = 1000

    def __init__(self, backing, offsets=0):
        self.backing = backing.reshape(-1)
        self.offsets = np.asarray(offsets)

    def __add__(self, offsets):
        return Pointer(self.backing, self.offsets + offsets)


class Language:
    constexpr = object
    maximum = staticmethod(np.maximum)
    minimum = staticmethod(np.minimum)
    where = staticmethod(np.where)

    def __init__(self):
        self.program = (0, 0)

    def program_id(self, axis):
        return self.program[axis]

    @staticmethod
    def arange(start, end):
        return np.arange(start, end)

    @staticmethod
    def load(pointer, mask=True, other=0):
        offsets, mask = np.broadcast_arrays(pointer.offsets, mask)
        selected = offsets[mask]
        assert np.all((selected >= 0) & (selected < pointer.backing.size)), "Read outside allocation"
        output = np.full(offsets.shape, other, dtype=pointer.backing.dtype)
        output[mask] = pointer.backing[selected]
        return output

    @staticmethod
    def store(pointer, values, mask=True):
        offsets, values, mask = np.broadcast_arrays(pointer.offsets, values, mask)
        selected = offsets[mask]
        assert np.all((selected >= 0) & (selected < pointer.backing.size)), "Write outside allocation"
        pointer.backing[selected] = values[mask]


def kernel(name, language):
    node = next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    namespace = {"tl": language}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


class TestStatePadding(unittest.TestCase):
    def test_runner_graph_padding_can_have_positive_dummy_query(self):
        source = SOURCE.parents[2] / "worker/model_runner_v1.py"
        runner = next(
            n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner"
        )
        method = next(
            n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "_pad_query_start_loc_for_fia"
        )
        namespace = {"CUDAGraphMode": SimpleNamespace(FULL="full")}
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        for actual, padded in [(6, 8), (11, 12)]:
            with self.subTest(actual=actual, padded=padded):
                starts = np.full(65, -1, dtype=np.int32)
                starts[: actual + 1] = np.arange(actual + 1)
                copied = []
                buffer = SimpleNamespace(
                    np=starts, copy_to_gpu=lambda copied=copied, starts=starts: copied.append(starts.copy())
                )
                owner = SimpleNamespace(
                    uniform_decode_query_len=1,
                    arange_np=np.arange(65),
                    compilation_config=SimpleNamespace(cudagraph_mode="full_decode_only"),
                )
                count = namespace["_pad_query_start_loc_for_fia"](owner, buffer, padded, padded, actual, "full", padded)
                self.assertEqual(count, padded)
                self.assertEqual(len(copied), 1)
                np.testing.assert_array_equal(np.diff(copied[0][actual : padded + 1]), np.ones(padded - actual))

    def test_padded_and_stale_rows_never_touch_ring_or_current(self):
        language = Language()
        commit = kernel("_commit", language)
        assemble = kernel("_assemble", language)
        assemble_cp = kernel("_assemble_cp", language)
        for state, query, seq in [(0, 1, 0), (2, 1, 0), (0, 0, 0), (-1, 1, 0)]:
            with self.subTest(state=state, query=query, seq=seq):
                ring = np.full((3, 128, 512), 17, dtype=np.uint16)
                # An invalid positive query must not read its current row.
                current = np.empty((0, 512), dtype=np.uint16)
                ids = np.array([state], dtype=np.int32)
                starts = np.array([0, query], dtype=np.int32)
                lengths = np.array([seq], dtype=np.int32)
                base = np.array([0], dtype=np.int32)
                out = np.full((256, 512), 19, dtype=np.uint16)
                for block in range(8):
                    language.program = (0, block)
                    commit(*map(Pointer, [ring, current, ids, starts, lengths]), WIDTH=512, WINDOW=128, ROWS=16)
                    assemble(
                        *map(Pointer, [ring, current, ids, starts, lengths, base, out]), WIDTH=512, WINDOW=128, ROWS=16
                    )
                    assemble_cp(
                        *map(Pointer, [ring, current, ids, starts, lengths, starts, lengths, base, out]),
                        WIDTH=512,
                        WINDOW=128,
                        ROWS=16,
                    )
                self.assertTrue(np.all(ring == 17))
                self.assertTrue(np.all(out == 19))

    def test_valid_window_order_and_commit_remain_exact(self):
        language = Language()
        commit = kernel("_commit", language)
        assemble = kernel("_assemble", language)
        assemble_cp = kernel("_assemble_cp", language)
        rng = np.random.default_rng(20261103)
        for total in [1, 2, 127, 128, 129, 255, 256, 100001]:
            for query in [1, min(total, 2), min(total, 17)]:
                for cp in [False, True]:
                    with self.subTest(total=total, query=query, cp=cp):
                        ring = rng.integers(0, 256, (3, 128, 512), dtype=np.uint16)
                        initial = ring.copy()
                        current = rng.integers(256, 512, (query, 512), dtype=np.uint16)
                        state = 0 if total == 1 else 2
                        ids = np.array([state], dtype=np.int32)
                        starts = np.array([0, query], dtype=np.int32)
                        lengths = np.array([total], dtype=np.int32)
                        base = np.array([0], dtype=np.int32)
                        out = np.full((256, 512), 19, dtype=np.uint16)
                        computed = total - query
                        old = min(computed, 127)
                        expected = np.concatenate([initial[state, np.arange(computed - old, computed) % 128], current])
                        for block in range((query + 127 + 15) // 16):
                            language.program = (0, block)
                            arrays = [ring, current, ids, starts, lengths]
                            if cp:
                                arrays += [starts, lengths]
                            (assemble_cp if cp else assemble)(
                                *map(Pointer, arrays + [base, out]), WIDTH=512, WINDOW=128, ROWS=16
                            )
                        np.testing.assert_array_equal(out[: old + query], expected)
                        np.testing.assert_array_equal(ring, initial)
                        expected_ring = initial.copy()
                        for row in range(max(0, query - 128), query):
                            expected_ring[state, (computed + row) % 128] = current[row]
                        for block in range(8):
                            language.program = (0, block)
                            commit(*map(Pointer, [ring, current, ids, starts, lengths]), WIDTH=512, WINDOW=128, ROWS=16)
                        np.testing.assert_array_equal(ring, expected_ring)


if __name__ == "__main__":
    unittest.main()
