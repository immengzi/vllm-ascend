import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
import torch
import reference_runtime as runtime


class RuntimeTests(unittest.TestCase):
    def tearDown(self):runtime.PENDING.clear();runtime.CONTEXT=None

    def test_mrv2_deferred_sampling_keeps_only_current_step_context(self):
        observed=[]
        class Runner:
            def execute_model(self,output):return None
            def update_requests(self,output):pass
            def sample_tokens(self):observed.append(runtime.CONTEXT)
        runtime.install_runner(Runner);runner=Runner()
        spec=dict(action='export',key='abc123',folder='stage',original_prompt=[1,2])
        req=NS(req_id='abc123-request',block_ids=((1,),),prompt_token_ids=[1])
        output=NS(finished_req_ids=set(),scheduled_new_reqs=[req],num_scheduled_tokens={req.req_id:1},
                  scheduled_cached_reqs=NS(req_ids=[],new_block_ids=[]))
        with patch.object(runtime,'control',return_value=spec):runner.execute_model(output)
        self.assertIsNone(runtime.CONTEXT)
        runner.sample_tokens();self.assertEqual(observed[-1][3],req.req_id)
        runner.sample_tokens();self.assertIsNone(observed[-1])

    def test_receiver_only_enqueues_cpu_metadata(self):
        class Receiver:
            tp_rank=1
            def _handle_requests(self,engine,host,port,requests):return {'key-failed'}
        runtime.install_receiver(Receiver)
        spec=dict(action='observe',key='key')
        reqs={rid:NS(state_snapshot={'boundary':129}) for rid in ('key-ok','key-failed','other')}
        with patch.object(runtime,'control',return_value=spec):
            Receiver()._handle_requests('engine','host',123,reqs)
        self.assertEqual(set(runtime.PENDING),{'key-ok'})
        self.assertIsNone(runtime.PENDING['key-ok']['actual_source_rank'])

    def test_hash_buffers_update_without_host_read(self):
        class Hash:
            multipliers=torch.zeros(2,4,dtype=torch.int32)
            primes=torch.zeros(2,2,dtype=torch.int32)
            def forward(self,ids,positions,starts,*a,**kw):
                return torch.ones(len(ids),2,6,dtype=torch.int32)
        class Model:
            def __init__(self,*,vllm_config):self.model=NS(engram_hash=Hash())
            def compute_logits(self,x):return x
        runtime.install_model(Model)
        cfg=NS(scheduler_config=NS(max_num_batched_tokens=256,max_num_seqs=4))
        model=Model(vllm_config=cfg);h=model.model.engram_hash
        pointers=(h._reference_hash.data_ptr(),h._reference_positions.data_ptr())
        for _ in range(2):h.forward(torch.arange(3),torch.arange(3),torch.tensor([0,3]))
        self.assertEqual(h._reference_epoch.item(),2)
        self.assertTrue(torch.equal(h._reference_hash[:3],torch.ones(3,2,6,dtype=torch.int32)))
        self.assertEqual(pointers,(h._reference_hash.data_ptr(),h._reference_positions.data_ptr()))

    def test_descriptor_enumerates_both_indexer_components(self):
        name='model.layers.0.self_attn.indexer.k_cache'
        layer=NS(spec=NS(tokens_per_state=2),kv_cache=[(torch.zeros(4,64,1,128,dtype=torch.int8),torch.zeros(4,64,1,1,dtype=torch.float16))])
        runner=NS(kv_cache_config=NS(kv_cache_groups=[NS(layer_names=[name])]),cache_config=NS(block_size=128),
                  compilation_config=NS(static_forward_context={name:layer}))
        values=runtime.descriptors(runner)
        self.assertEqual(len(values),2)
        self.assertEqual([p.dtype for p,_,_ in values],['int8','float16'])
        self.assertEqual([p.compression for p,_,_ in values],[2,2])


if __name__=='__main__':unittest.main()
