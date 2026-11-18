import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
import smoke


class SmokeGateTests(unittest.TestCase):
    def test_single_host_ep12_reference_cannot_qualify_dual_host_ep16(self):
        reference=dict(runtime_context=dict(producer=dict(TP=4,DP=3),consumer=dict(TP=2,DP=6)))
        live=dict(producer=dict(tp=4,dp=4),consumer=dict(tp=2,dp=8))
        with self.assertRaises(ValueError):smoke.check_reference_deployment(reference,live)

    def test_live_topology_rejects_mislabeled_tp(self):
        with patch.object(smoke,'fetch',return_value=json.dumps({'vllm_config':{'parallel_config':{'tensor_parallel_size':4}}})):
            with self.assertRaises(ValueError):smoke.live_topology('http://d',2)

    def test_heterogeneous_http_success_is_not_full_correctness(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ref=root/'ref.json';out=root/'out.json'
            ref.write_text(json.dumps(dict(passed=True,mode='reference',model='dsv41',reference_kind='hybrid_snapshot_replay',
                runtime_context=dict(producer=dict(TP=4,DP=3),consumer=dict(TP=2,DP=6)),
                rows=[dict(dp=0,prompt=[10,11],output_count=1,token_ids=[7])])))
            args=NS(mode='pd',endpoint='http://d',producer='http://p',reference=ref,out=out,model='dsv41',
                producer_tp=4,consumer_tp=2,runtime_identity='ctx',producer_dp_size=4,variant='state')
            params=dict(remote_num_prompt_tokens=1,dsv41_state_snapshot=dict(boundary=1,tp_size=4,transfer_id='generation'))
            with patch.object(smoke,'timeout_for_request',return_value=1),patch.object(smoke,'check_hybrid_reference'), \
                 patch.object(smoke,'live_topology',side_effect=[dict(tp=4,dp=3),dict(tp=2,dp=6)]), \
                 patch.object(smoke,'completion',return_value=(dict(kv_transfer_params=params),[7])), \
                 patch.object(smoke,'checked_result',return_value=({},[7])), \
                 patch.object(smoke,'fetch',return_value=''),patch.object(smoke,'sources_from_metrics',return_value={}), \
                 patch.object(smoke,'wait_for_source_delta',return_value=dict(local_compute=1,external_kv_transfer=1)):
                smoke.run(args)
            result=json.loads(out.read_text())
            self.assertTrue(result['execution_passed']);self.assertFalse(result['passed'])
            self.assertIn('source_loaded_state',result['pending_gates'])

    def test_consumer_tp_is_explicit(self):
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            smoke.parser().parse_args(['pd','--endpoint','http://d','--producer','http://p','--reference','ref.json','--out','out.json'])

    def test_full_d_reference_rejected_before_http(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ref=root/'reference.json'
            ref.write_text(json.dumps(dict(passed=True,mode='reference',model='dsv41',rows=[])))
            args=NS(mode='pd',endpoint='http://d',producer='http://p',reference=ref,out=root/'out.json',
                    model='dsv41',producer_tp=4,consumer_tp=2,runtime_identity='ctx',producer_dp_size=4)
            with patch.object(smoke,'timeout_for_request',return_value=1),patch.object(smoke,'fetch') as fetch:
                with self.assertRaisesRegex(ValueError,'hybrid reference'):smoke.run(args)
                fetch.assert_not_called()

    def test_confirmed_window_cannot_target_unlisted_endpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'window.json'
            path.write_text(json.dumps(dict(confirmed=True,start='2026-11-18T12:00:00+08:00',
                client_cutoff='2026-11-18T14:45:00+08:00',end='2026-11-18T15:00:00+08:00',endpoints=['http://c01'])))
            with self.assertRaises(ValueError):smoke.configure_window(path,['http://other'])

    def test_raw_artifacts_and_topology_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);artifacts=[]
            import hashlib
            for phase,ranks in (('source',(0,2)),('loaded',(0,1)),('sample',(0,1))):
                for rank in ranks:
                    path=root/f'{phase}-{rank}.pt';path.write_bytes(f'{phase}:{rank}'.encode())
                    artifacts.append(dict(phase=phase,rank=rank,path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            document=dict(passed=True,mode='reference',reference_kind='hybrid_snapshot_replay',producer_tp=4,consumer_tp=2,
                runtime_identity='ctx',rows=[dict(output_count=1,reference_audit=dict(passed=True,transport='local_snapshot_replay',
                    repeat_audit=dict(passed=True,samples=[dict(rank=rank,step=1,passed=True) for rank in (0,1)],
                        artifacts=[dict(phase='continuation',step=1,**{k:v for k,v in a.items() if k!='phase'}) for a in artifacts if a['phase']=='sample']*2),
                    source_sha256='a'*64,loaded_sha256='b'*64,first_logits_sha256='c'*64,artifacts=artifacts))])
            self.assertIs(smoke.check_hybrid_reference(document,4,2,'ctx',root),document)
            bad=copy.deepcopy(document);bad['rows'][0]['reference_audit']['artifacts'].pop()
            with self.assertRaises(ValueError):smoke.check_hybrid_reference(bad,4,2,'ctx',root)
            (root/'source-0.pt').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):smoke.check_hybrid_reference(document,4,2,'ctx',root)


if __name__=='__main__':unittest.main()
