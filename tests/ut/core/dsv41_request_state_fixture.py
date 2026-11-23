# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Bind the scheduler and worker parts in a single process for component tests."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.v1.core.block_pool import BlockPool

from vllm_ascend.models.deepseek_v41.request_state_storage import RequestStateLeases, RequestStateStorage
from vllm_ascend.worker.v2 import attn_utils


def make_leases(history, windows, compressors):
    pool = BlockPool(num_gpu_blocks=history + 1, enable_caching=False, hash_block_size=128)
    return RequestStateLeases(pool, windows, compressors)


class Cache(RequestStateStorage):
    def __init__(self, config, specs, history, windows, compressors, *, device="cpu"):
        config.additional_config = {}
        config.kv_transfer_config = SimpleNamespace(kv_connector="MooncakePullConnector")
        with patch.object(attn_utils, "get_current_vllm_config", return_value=config):
            super().__init__(
                specs,
                history,
                windows,
                compressors,
                lambda size: attn_utils._allocate_int8_cache_tensor(size, 2**21, torch.device(device)),
            )
        self.requests = make_leases(history, windows, compressors)
