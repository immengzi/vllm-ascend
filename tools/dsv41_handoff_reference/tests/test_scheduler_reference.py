import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
from scheduler_reference import apply_bootstrap


class SchedulerReferenceTests(unittest.TestCase):
    def fixture(self):
        tokens=list(range(130));rid='refR-test'
        request=NS(req_id=rid,prompt_token_ids=tokens,prefill_token_ids=tokens,num_computed_tokens=0)
        owner=NS(num_computed_tokens=130,num_in_flight_tokens=130)
        scheduler=NS(requests={rid:owner})
        output=NS(scheduled_new_reqs=[request],num_scheduled_tokens={rid:130},total_num_scheduled_tokens=130)
        return scheduler,output,dict(action='replay',key='refR-test',original_prompt=tokens)

    def test_schedule_and_inflight_agree(self):
        s,o,c=self.fixture();self.assertTrue(apply_bootstrap(s,o,c))
        self.assertEqual(o.scheduled_new_reqs[0].num_computed_tokens,129)
        self.assertEqual(s.requests['refR-test'].num_computed_tokens,130)
        self.assertEqual(s.requests['refR-test'].num_in_flight_tokens,1)
        self.assertEqual(o.total_num_scheduled_tokens,1)

    def test_normal_request_unchanged(self):
        s,o,c=self.fixture();c['key']='other'
        self.assertFalse(apply_bootstrap(s,o,c));self.assertEqual(o.total_num_scheduled_tokens,130)

    def test_export_unchanged(self):
        s,o,c=self.fixture();c['action']='export'
        self.assertFalse(apply_bootstrap(s,o,c));self.assertEqual(o.total_num_scheduled_tokens,130)

    def test_mismatch_fails_before_mutation(self):
        for mode in ('prompt','chunk','prior','batch'):
            s,o,c=self.fixture()
            if mode=='prompt':c['original_prompt']=[0]*130
            if mode=='chunk':o.num_scheduled_tokens['refR-test']=64
            if mode=='prior':o.scheduled_new_reqs[0].num_computed_tokens=1
            if mode=='batch':o.num_scheduled_tokens['another']=1
            with self.subTest(mode=mode),self.assertRaises(ValueError):apply_bootstrap(s,o,c)
            self.assertEqual(o.scheduled_new_reqs[0].num_computed_tokens,1 if mode=='prior' else 0)


if __name__=='__main__':unittest.main()
