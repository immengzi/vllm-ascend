# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
from unittest.mock import patch

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request

from tests.ut.core import test_dsv41_compact_pages as pages
from tests.ut.core import test_dsv41_compact_ring_handoff as handoff

flash = pages.flash


def test_native_read_bytes_at_swa_boundaries(flash):
    cfg, specs = flash
    cfg.additional_config = {}
    cfg.model_config.max_model_len = 4096
    pc, pv = pages.worker_views(cfg, specs, 96)
    dc, dv = pages.worker_views(cfg, specs, 104)
    pro = pages.register(cfg, pc, pv, 4, 0, multiple_regions=True)
    con = pages.register(cfg, dc, dv, 2, 1, multiple_regions=True)
    for n in (128, 129, 130, 256, 257, 258, 385, 386):
        pm, dm = handoff.manager_for(pc), handoff.manager_for(dc)
        pre = Request("p", [3000] * (n - 1), SamplingParams(max_tokens=1), pooling_params=None)
        dec = Request("d", [3000] * n, SamplingParams(max_tokens=1), pooling_params=None)
        hold = Request("held", [3050], SamplingParams(max_tokens=1), pooling_params=None)
        assert dm.allocate_slots(hold, 1) is not None
        assert pm.allocate_slots(pre, n - 1) is not None
        assert dm.allocate_slots(dec, 0, num_external_computed_tokens=n - 1, delay_cache_blocks=True) is not None
        # Stock Scheduler._connector_finished trims the obsolete SWA prefix
        # using processed tokens before it publishes producer block IDs.
        pre.num_computed_tokens = n - 1
        pm.remove_skipped_blocks(request_id="p", processed_computed_tokens=n - 1, num_prompt_tokens=n - 1)
        pi = pm.get_block_ids_for_computed_tokens(request_id="p", num_computed_tokens=n - 1)
        destination = dm.get_blocks("d")
        di = destination.get_block_ids()
        unhashed = destination.get_unhashed_block_ids_all_groups()
        assert all(0 not in group for group in unhashed)
        # stock_read_batch verifies the address bounds for every SDK segment.
        with patch.object(handoff.ctypes, "memmove"):
            read = handoff.stock_read_batch([pro] * 4, con, {"r": (pi, di, n)}, local_unhashed_ids={"r": unhashed})
        observed = {
            128: 6005376,
            129: 6005376,
            130: 11617536,
            256: 6374656,
            257: 6374656,
            258: 11986816,
            385: 6743936,
            386: 12356096,
        }
        assert read == observed[n]
