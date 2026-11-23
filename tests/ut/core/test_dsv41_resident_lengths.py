# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D cold-handoff residency sweep on the PR17 CPU cache manager. No model execution."""

from unittest.mock import patch

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm.v1.request import Request

from tests.ut.core import test_dsv41_compact_pages as pages
from tests.ut.core.test_dsv41_logical_pools import plan
from vllm_ascend.models.deepseek_v41.compact_cache import LOGICAL_CONFIG_KEY, PACKED_HISTORY_KEY
from vllm_ascend.patch.platform import patch_kv_cache_coordinator

flash = pages.flash


@pytest.mark.parametrize("layout", ["native", "uniform", "packed"])
def test_sweep(flash, layout):
    config, specs = flash
    config.additional_config = {LOGICAL_CONFIG_KEY: layout != "native", PACKED_HISTORY_KEY: layout == "packed"}
    config.max_in_flight_tokens = config.scheduler_config.max_num_batched_tokens = 512
    config.scheduler_config.max_num_seqs = 72
    cfg = plan(config, specs, int(13.23 * 2**30))[0]
    with patch(
        "vllm.v1.core.kv_cache_manager.get_kv_cache_coordinator", patch_kv_cache_coordinator.get_kv_cache_coordinator
    ):
        mgr = KVCacheManager(
            generate_scheduler_kv_cache_config([cfg]), 1048576, 128, 128, max_in_flight_tokens=512, enable_caching=False
        )
    pool = mgr.coordinator.block_pool
    stride = {"native": 540928, "uniform": 131072, "packed": 369280}[layout]
    cases = {
        str(n): [n]
        for n in (32, 128, 129, 257, 1024, 4096, 8192, 13184, 13185, 16384, 32768, 65536, 131072, 262144, 700001)
    }
    cases["7x4K+1x256K"] = [4096] * 7 + [262144]
    cases["1x4K+1x64K+1x256K"] = [4096, 65536, 262144]
    expected_capacity = {
        "65536": (50, 68, 72),
        "131072": (25, 34, 36),
        "262144": (12, 17, 18),
        "700001": (4, 6, 6),
        "1x4K+1x64K+1x256K": (30, 41, 44),
    }
    expected_bytes = {
        "128": (12 * 540928, 46 * 131072, 23 * 369280),
        "129": (23 * 540928, 49 * 131072, 24 * 369280),
        "4096": (43 * 540928, 139 * 131072, 54 * 369280),
    }
    column = ("native", "uniform", "packed").index(layout)
    for case, pattern in cases.items():
        initial = pool.get_num_free_blocks()
        live = []
        charged = []
        for i in range(72):
            n = pattern[i % len(pattern)]
            req = Request(str(i), [1] * n, SamplingParams(max_tokens=8), pooling_params=None)
            start = pool.get_num_free_blocks()
            allocated = mgr.allocate_slots(req, 0, num_external_computed_tokens=n - 1, delay_cache_blocks=True)
            if allocated is not None:
                req.num_computed_tokens = n - 1
                allocated = mgr.allocate_slots(req, 1)
            if allocated is None:
                mgr.free(req)
                break
            charged.append((start - pool.get_num_free_blocks()) * stride)
            live.append(req)
        assert len(live) == expected_capacity.get(case, (72, 72, 72))[column]
        if case in expected_bytes:
            assert charged[0] == expected_bytes[case][column]
        print("SWEEP", layout, case, len(live), charged[0], sum(charged))
        for req in live:
            mgr.free(req)
        assert pool.get_num_free_blocks() == initial
