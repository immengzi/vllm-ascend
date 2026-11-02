# SPDX-License-Identifier: Apache-2.0
"""NPU component proof of T=N-1 Engram history reconstruction, without PD DMA."""

import json
import unittest
from types import SimpleNamespace as NS

import numpy as np
import torch


class TestPromptHandoffHistory(unittest.TestCase):
    @unittest.skipUnless(hasattr(torch, "npu") and torch.npu.is_available(), "NPU required")
    def test_full_prompt_lookback_and_local_generated_ring(self):
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        import vllm_ascend.ops  # noqa: F401 - register operators before model imports
        from vllm_ascend.models.deepseek_v41.engram.hash_state import DEAD_ID, AscendNgramHashState

        torch.npu.set_device(0)
        prepare = GPUModelRunner._prepare_lookback_token_ids
        mult = torch.tensor([[1, 3, 5, 7], [11, 13, 17, 19]], dtype=torch.int64)
        primes = torch.tensor([[[17, 19], [23, 29], [31, 37]], [[41, 43], [47, 53], [59, 61]]], dtype=torch.int64)
        mapping = (torch.arange(512, dtype=torch.int32) * 17 + 5).remainder(1024)
        shared = dict(
            use_slot_cache=True,
            multipliers=mult.npu(),
            primes=primes.npu(),
            offsets=torch.zeros_like(primes).npu(),
            token_map=mapping.npu(),
            block_size=128,
            pad_id=0,
            request_state_virtual_blocks=1172,
        )
        cases = []
        for length in [2, 31, 32, 127, 128, 129, 2047, 2048, 2049, 100001]:
            prompt = [(i * 13 + 7) % 498 + 1 for i in range(length)]
            if length > 4:
                prompt[-3] = 500  # image/dead sentinel survives prompt reconstruction
            generated = [501 if i % 37 == 0 else (i * 11 + 3) % 498 + 1 for i in range(150)]
            full = prompt + generated
            reference = NS(**shared, _cache=torch.full((4 * 128,), -99, dtype=torch.int32, device="npu"))
            receiver = NS(**shared, _cache=torch.full((4 * 128,), -77, dtype=torch.int32, device="npu"))
            boundary = length - 1
            seed = reference._cache.cpu()
            for pos in range(max(0, boundary - 128), boundary):
                token = full[pos]
                seed[3 * 128 + pos % 128] = DEAD_ID if token in [500, 501] else mapping[token]
            reference._cache.copy_(seed)
            cpu_tokens = np.full((1, length + 152), 42, dtype=np.int64)  # generated CPU placeholders
            cpu_tokens[0, :length] = prompt
            buffer = NS(np=np.empty((1, 3), dtype=np.int64))
            buffer.copy_to_gpu = lambda buf=buffer: torch.tensor(buf.np.copy(), device="npu")
            runner = NS(
                lookback_token_ids=buffer,
                input_batch=NS(
                    num_computed_tokens_cpu=np.array([boundary]),
                    num_prompt_tokens=np.array([length]),
                    token_ids_cpu=cpu_tokens,
                ),
            )
            for pos in range(boundary, length + 150):
                runner.input_batch.num_computed_tokens_cpu[0] = pos
                lookback = prepare(runner, 1)
                expected_lookback = [full[pos - 1 - j] if 0 <= pos - 1 - j < length else -1 for j in range(3)]
                self.assertEqual(buffer.np[0].tolist(), expected_lookback)
                token = full[pos]
                args = [
                    torch.tensor([token], device="npu"),
                    torch.tensor([pos], device="npu"),
                    torch.tensor([0, 1], dtype=torch.int32, device="npu"),
                    torch.tensor([token in [500, 501]], device="npu"),
                    lookback,
                    (lookback == 500) | (lookback == 501),
                    None,
                ]
                a = AscendNgramHashState.forward(
                    reference, *args, torch.tensor([[3]], dtype=torch.int32, device="npu")
                ).cpu()
                b = AscendNgramHashState.forward(
                    receiver, *args, torch.tensor([[1]], dtype=torch.int32, device="npu")
                ).cpu()
                expected = []
                for layer in range(2):
                    rolling = 0
                    blocked = False
                    hashes = []
                    for shift in range(4):
                        index = pos - shift
                        raw = full[index] if index >= 0 else 0
                        blocked = blocked or index < 0 or raw in [500, 501]
                        value = 0 if blocked else int(mapping[raw])
                        rolling ^= value * int(mult[layer, shift])
                        if shift:
                            hashes.extend(rolling % int(p) for p in primes[layer, shift - 1])
                    expected.append(hashes)
                oracle = torch.tensor([expected], dtype=torch.int32)
                torch.testing.assert_close(a, oracle, rtol=0, atol=0)
                torch.testing.assert_close(b, oracle, rtol=0, atol=0)
            torch.testing.assert_close(
                reference._cache.cpu()[3 * 128 : 4 * 128], receiver._cache.cpu()[128:256], rtol=0, atol=0
            )
            cases.append(
                {
                    "prompt_tokens": length,
                    "boundary": boundary,
                    "continued_steps": 151,
                    "local_state_ids": [3, 1],
                    "receiver_prefix_cache_restored": False,
                }
            )
        print(
            "PD_ENGRAM_COMPONENT "
            + json.dumps(
                {
                    "passed": True,
                    "cases": cases,
                    "source": GPUModelRunner.__module__ + "." + prepare.__name__,
                    "scope": (
                        "Real hash kernels plus pinned runner prompt-lookback function; "
                        "no connector, DMA or full-model PD validation."
                    ),
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
