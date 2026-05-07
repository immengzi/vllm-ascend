import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerKVCache(unittest.TestCase):

    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.speculative_config = None
        runner.model_config = MagicMock()
        runner.model_config.is_encoder_decoder = False
        runner.model_config.use_mla = True
        runner.compilation_config = MagicMock()
        runner.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
        runner.pcp_size = 1
        runner.dcp_size = 1
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))

    @patch("vllm_ascend.worker.model_runner_v1.envs.VLLM_ASCEND_LAPS_SCHEDULING", True)
    def test_use_laps_prefill_graph_enabled_for_supported_full_graph_path(self):
        runner = self._build_runner()

        self.assertTrue(runner._use_laps_prefill_graph())

    @patch("vllm_ascend.worker.model_runner_v1.envs.VLLM_ASCEND_LAPS_SCHEDULING", True)
    def test_use_laps_prefill_graph_disabled_for_spec_decode(self):
        runner = self._build_runner()
        runner.speculative_config = MagicMock()

        self.assertFalse(runner._use_laps_prefill_graph())

    @patch("vllm_ascend.worker.model_runner_v1.envs.VLLM_ASCEND_LAPS_SCHEDULING", True)
    def test_use_laps_prefill_graph_disabled_for_non_full_graph(self):
        runner = self._build_runner()
        runner.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

        self.assertFalse(runner._use_laps_prefill_graph())


if __name__ == "__main__":
    unittest.main()
