import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
from audit_transport import audit
from reference_contract import boundary_counters,progress_counters,prompt_digest
from tensor_reference import digest


class TransportAuditTests(unittest.TestCase):
    def fixture(self,root):
        for stage in ('r','p','d'):(root/stage).mkdir()
        prompt=[10,11];sha=prompt_digest(prompt);artifacts=[]
        def state(phase,rank):
            data=torch.arange(2,dtype=torch.float32)
            return dict(planes={'x':data},geometry={'x':{}},hashes={'x':digest(data)},metadata=dict(
                phase=phase,layout='native',runtime_identity='ctx',prompt_sha256=sha,producer_tp=4,dp=0,
                request_id='source-request' if phase=='source' else 'cmpl-abcd-d-0',
                tp=4 if phase=='source' else 2,source_rank=rank if phase=='source' else None,tp_rank=rank,
                counters=boundary_counters(1,2)))
        def save(path,value,phase=None,rank=None,step=None):
            torch.save(value,root/path)
            item=dict(path=path,sha256=hashlib.sha256((root/path).read_bytes()).hexdigest())
            if phase is not None:item.update(phase=phase,rank=rank)
            if step is not None:item['step']=step
            return item
        sources={}
        for rank in range(4):
            source=state('source',rank)
            save(f'p/abcd-source-t1-tp{rank}.pt',source)
            if rank in (0,2):
                path=f'r/source{rank}.pt';sources[str(rank)]=path
                artifacts.append(save(path,source,'source',rank))
        output = dict(choices=[dict(token_ids=[7], logprobs=dict(
            tokens=['x'], token_logprobs=[-0.1], top_logprobs=[{'x': -0.1}]))])
        rows=[];repeat_artifacts=[]
        for rank in (0,1):
            loaded=state('loaded',rank)
            save(f'd/abcd-loaded-t1-tp{rank}.pt',loaded)
            artifacts.append(save(f'r/loaded{rank}.pt',loaded,'loaded',rank))
            sample=state('sample',rank)
            sample['metadata'].update(original_boundary=1,counters=dict(original_prompt_tokens=2,**progress_counters(2)))
            sample.update(logits=torch.nn.functional.one_hot(torch.tensor([7]),8).float(),hidden=torch.ones(1,2),
                engram_hash=torch.tensor([1]),engram_lookback=torch.tensor([10]),engram_lookup={'0':torch.ones(2)})
            save(f'd/abcd-sample-t2-tp{rank}.pt',sample)
            for repeat in (0,1):
                path=f'r/key{repeat}-sample-t2-tp{rank}.pt'
                item=save(path,sample,'continuation',rank,1);repeat_artifacts.append(item)
                if repeat==0:artifacts.append(dict(item,phase='sample'))
        for repeat in (0,1):
            rows.append(dict(dp=0,prompt=prompt,output_count=1,repeat=repeat,key=f'key{repeat}',token_ids=[7],output=output,source_files=sources,
                reference_audit=dict(passed=True,transport='local_snapshot_replay',source_sha256='a'*64,loaded_sha256='b'*64,
                    first_logits_sha256='c'*64,artifacts=artifacts,repeat_audit=dict(passed=True,artifacts=repeat_artifacts,
                        samples=[dict(rank=rank,step=1,passed=True) for rank in (0,1)]))))
        reference=dict(passed=True,mode='reference',model='dsv41',reference_kind='hybrid_snapshot_replay',producer_tp=4,
            runtime_context=dict(producer=dict(TP=4,DP=3),consumer=dict(TP=2,DP=6)),
            consumer_tp=2,runtime_identity='ctx',rows=rows)
        execution=dict(mode='pd',execution_passed=True,variant='control',producer_tp=4,consumer_tp=2,runtime_identity='ctx',
            live_topology=dict(producer=dict(tp=4,dp=3),consumer=dict(tp=2,dp=6)),
            rows=[dict(dp=0,prompt=prompt,output_count=1,observer_key='abcd',request_id='abcd-d',token_ids=[7],output=output,
                producer=dict(kv_transfer_params=dict(remote_request_id='source-request')),
                decode_prompt_sources=dict(local_compute=1,local_cache_hit=0,external_kv_transfer=1))])
        (root/'reference.json').write_text(json.dumps(reference));(root/'execution.json').write_text(json.dumps(execution))
        (root/'read.log').write_text(''.join(f"KV cache transfer for requests ['cmpl-abcd-d-0'] took 1 ms (1 ranges, 10 bytes) local_device_id {rank} remote_session_id 127.0.0.1:123 source_rank={rank*2}\n" for rank in (0,1)))
        return (root,'p','d',root/'execution.json',root/'reference.json',root/'read.log')

    def test_complete_toy_transport_checks_still_require_lifecycle(self):
        with tempfile.TemporaryDirectory() as folder:
            result=audit(*self.fixture(Path(folder)))
            self.assertTrue(result['correctness_passed']);self.assertFalse(result['passed'])
            self.assertIn('lifecycle',result['pending_gates'])

    def test_loaded_state_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=self.fixture(root);path=root/'d/abcd-loaded-t1-tp1.pt'
            state=torch.load(path,weights_only=True);state['planes']['x'][0]+=1;torch.save(state,path)
            self.assertFalse(audit(*args)['correctness_passed'])

    def test_actual_native_mapping_is_used(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=self.fixture(root)
            path=root/'read.log'
            path.write_text(path.read_text().replace('source_rank=2','source_rank=3'))
            result=audit(*args)
            self.assertEqual(result['results'][0]['ranks'][1]['loaded']['reads'][0]['source_rank'], 3)

    def test_probability_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=self.fixture(root);path=root/'execution.json'
            execution=json.loads(path.read_text())
            execution['rows'][0]['output']['choices'][0]['logprobs']['token_logprobs']=[-2.0]
            path.write_text(json.dumps(execution))
            with self.assertRaisesRegex(ValueError,'probabilities differ'):audit(*args)

    def test_native_mapping_must_be_observed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=self.fixture(root);path=root/'read.log'
            path.write_text(path.read_text().replace(' source_rank=0',''))
            with self.assertRaisesRegex(ValueError,'lacks observed'):audit(*args)

    def test_http_execution_without_actual_read_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=self.fixture(root);(root/'read.log').write_text('HTTP 200 OK\n')
            with self.assertRaises(ValueError):audit(*args)


if __name__=='__main__':unittest.main()
