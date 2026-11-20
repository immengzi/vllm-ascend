# SPDX-License-Identifier: Apache-2.0
"""Execute the window kernels' load/store expressions against CPU memory."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


class CPUAddresses:
    """One integer address space; masked accesses never touch invalid offsets."""

    constexpr = int
    int64 = torch.int64
    minimum = staticmethod(torch.clamp_max)
    where = staticmethod(torch.where)
    arange = staticmethod(torch.arange)

    def __init__(self, size):
        self.memory = torch.full((size,), -1, dtype=torch.int64)
        self.tile = 0

    def program_id(self, axis):
        return 0 if axis == 0 else self.tile

    def load(self, addresses, mask=None, other=0):
        if mask is None:
            return self.memory[addresses]
        mask = mask.expand_as(addresses)
        result = torch.full_like(addresses, other)
        result[mask] = self.memory[addresses[mask]]
        return result

    def store(self, addresses, value, mask):
        mask = mask.expand_as(addresses)
        self.memory[addresses[mask]] = value[mask]


@pytest.mark.parametrize("page_stride", [65536, 73856])
def test_chunked_window_addresses(page_stride):
    # Element strides of the 128 KiB and 144.25 KiB A3 physical pages in BF16.
    window, width, slot = 128, 512, 2
    chunks = [128, 1, 128, 257, 4096, 1]
    ring = 0
    current = 4 * page_stride
    output = current + max(chunks) * width
    ids = output + (window + max(chunks)) * width
    starts, seq, base = ids + 1, ids + 3, ids + 4
    cpu = CPUAddresses(base + 1)
    module_path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/state_kernels.py"
    spec = importlib.util.spec_from_file_location("single_pool_address_test_kernels", module_path)
    kernels = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules, {"vllm.triton_utils": SimpleNamespace(tl=cpu, triton=SimpleNamespace(jit=lambda f: f))}
    ):
        spec.loader.exec_module(kernels)

    # Reuse the same physical block for a new request with different values.
    for generation, lengths in enumerate((chunks, [1, 129])):
        processed = 0
        tag = generation * 10000
        for length in lengths:
            cpu.memory[ids : base + 1] = torch.tensor([slot, 0, length, processed + length, 0])
            cpu.memory[current : current + length * width].view(length, width).copy_(
                torch.arange(processed, processed + length)[:, None] + tag
            )
            old = min(processed, window - 1)
            for cpu.tile in range((old + length + 15) // 16):
                kernels._assemble(ring, current, ids, starts, seq, base, output, width, window, 16, page_stride)
            expected = torch.arange(processed - old, processed + length)[:, None] + tag
            assembled = cpu.memory[output : output + (old + length) * width].view(-1, width)
            assert torch.equal(assembled, expected.expand_as(assembled))
            for cpu.tile in range(8):
                kernels._commit(ring, current, ids, starts, seq, width, window, 16, page_stride)
            processed += length
            positions = torch.arange(max(0, processed - window), processed)
            persistent = cpu.memory[slot * page_stride : slot * page_stride + window * width].view(window, width)
            assert torch.equal(persistent[positions % window], (positions[:, None] + tag).expand(-1, width))
            assert (cpu.memory[: slot * page_stride] == -1).all()
            assert (cpu.memory[slot * page_stride + window * width : current] == -1).all()
