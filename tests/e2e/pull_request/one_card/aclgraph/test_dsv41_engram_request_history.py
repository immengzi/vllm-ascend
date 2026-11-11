# SPDX-License-Identifier: Apache-2.0
"""NPU Engram request-history hashing, long chunks, padding and slot reuse."""

import unittest
from types import SimpleNamespace as NS

import torch


class TestEngramRequestHistory(unittest.TestCase):
    @unittest.skipUnless(hasattr(torch, "npu") and torch.npu.is_available(), "NPU required")
    def test_hashes_and_reuse(self):
        import vllm_ascend.ops  # noqa: F401 - register operators before model package imports
        from vllm_ascend.models.deepseek_v41.engram.hash_state import AscendNgramHashState

        torch.npu.set_device(0)
        mult = torch.tensor([[1, 3, 5, 7], [11, 13, 17, 19]], dtype=torch.int64)
        primes = torch.tensor([[[17, 19], [23, 29], [31, 37]], [[41, 43], [47, 53], [59, 61]]], dtype=torch.int64)
        state = NS(
            use_slot_cache=True,
            _cache=torch.full((4 * 128,), -99, dtype=torch.int32, device="npu"),
            multipliers=mult.npu(),
            primes=primes.npu(),
            offsets=torch.zeros_like(primes).npu(),
            token_map=torch.arange(512, dtype=torch.int32, device="npu"),
            block_size=128,
            pad_id=0,
            request_state_virtual_blocks=64,
        )
        history = {1: [], 2: []}
        checks = []

        def run(chunks, pad=0):
            inputs, positions, starts, rows = [], [], [0], []
            for slot, values in chunks:
                start = len(history[slot])
                inputs.extend(values)
                positions.extend(range(start, start + len(values)))
                rows.append([slot])
                starts.append(len(inputs))
            real = len(inputs)
            inputs += [0] * pad
            positions += [0] * pad
            ids = torch.tensor(inputs, dtype=torch.int64, device="npu")
            pos = torch.tensor(positions, dtype=torch.int64, device="npu")
            dead = torch.arange(len(inputs), device="npu") >= real
            qsl = torch.tensor(starts, dtype=torch.int32, device="npu")
            lookback = torch.full((len(chunks), 3), -1, dtype=torch.int64, device="npu")
            output = AscendNgramHashState.forward(
                state,
                ids,
                pos,
                qsl,
                dead,
                lookback,
                torch.zeros_like(lookback, dtype=torch.bool),
                torch.full((len(inputs),), -1, dtype=torch.int64, device="npu"),
                torch.tensor(rows, dtype=torch.int32, device="npu"),
            ).cpu()[:real]
            expected = []
            for slot, values in chunks:
                previous = history[slot]
                full = previous + values
                for position in range(len(previous), len(full)):
                    layers = []
                    for layer in range(2):
                        rolling = 0
                        hashes = []
                        for shift in range(4):
                            value = full[position - shift] if position >= shift else 0
                            rolling ^= value * int(mult[layer, shift])
                            if shift:
                                hashes.extend(rolling % int(p) for p in primes[layer, shift - 1])
                        layers.append(hashes)
                    expected.append(layers)
                history[slot] = full
            torch.testing.assert_close(output, torch.tensor(expected, dtype=torch.int32), rtol=0, atol=0)
            cache = state._cache.cpu()
            for slot, _ in chunks:
                full = history[slot]
                for position in range(max(0, len(full) - 128), len(full)):
                    assert int(cache[slot * 128 + position % 128]) == full[position]
            checks.append({"lengths": [len(history[s]) for s, _ in chunks], "tokens": real, "padding": pad})

        run([(1, [(i * 7) % 511 + 1 for i in range(2048)])])
        for i in range(131):
            run([(1, [(i * 11) % 511 + 1])], pad=3)
        run([(1, [(i * 13) % 511 + 1 for i in range(257)])])
        history[1] = []  # Same physical slot, entirely new request, without clearing.
        run([(1, [8, 9, 10]), (2, [(i * 17) % 511 + 1 for i in range(129)])])
        run([(1, [11]), (2, [12])], pad=2)
        self.assertEqual(len(checks), 135)
