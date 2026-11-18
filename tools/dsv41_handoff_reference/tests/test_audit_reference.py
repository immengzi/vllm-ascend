import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
import audit_reference as audit


def snapshot():
    return dict(metadata=dict(runtime_identity='ctx',prompt_sha256='prompt',counters={'processed_tokens':130},
                              tp_rank=0,tp=2,original_boundary=129),
                planes={'window':torch.arange(4)},logits=torch.ones(1,4),hidden=torch.ones(1,2),
                engram_hash=torch.arange(4),engram_lookback=torch.arange(4),engram_lookup={'0':torch.arange(4)})


class AuditTests(unittest.TestCase):
    def test_continuation_detects_state_logits_and_actual_lookup_corruption(self):
        original=snapshot()
        self.assertTrue(audit.audit_samples(original,copy.deepcopy(original))['passed'])
        for key in ('logits','hidden','engram_hash','engram_lookback','engram_lookup','planes'):
            other=copy.deepcopy(original)
            value=other[key]
            if isinstance(value,dict):value=next(iter(value.values()))
            value.reshape(-1)[0]+=1
            self.assertFalse(audit.audit_samples(original,other)['passed'],key)

    def test_missing_or_duplicate_repeat_is_rejected(self):
        row=dict(dp=0,prompt=[1,2],output_count=8,repeat=0,reference_audit={})
        for rows in ([row],[row,copy.deepcopy(row)]):
            with self.assertRaises(ValueError):audit.audit_repeats(Path('/tmp'),'stage',rows)

    def test_all_selected_steps_and_ranks_required(self):
        self.assertEqual(audit.sample_steps(8),[1,2])
        self.assertEqual(audit.sample_steps(130),[1,2,32,128,130])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'stage').mkdir()
            rows=[dict(prompt=[1,2],output_count=8,key=key,token_ids=list(range(8)),
                       output={'choices':[{'logprobs':{'token_logprobs':[-1.]*8}}]}) for key in ('a','b')]
            for row in rows:
                for step in (1,2):
                    for rank in (0,1):
                        state=snapshot();state['metadata']['tp_rank']=rank
                        torch.save(state,root/'stage'/f'{row["key"]}-sample-t{1+step}-tp{rank}.pt')
            result=audit.compare_case_samples(root,'stage',rows[0],'stage',rows[1])
            self.assertTrue(result['passed']);self.assertEqual(len(result['samples']),4)
            rows[1]['token_ids'][-1]=999
            self.assertFalse(audit.compare_case_samples(root,'stage',rows[0],'stage',rows[1])['passed'])
            (root/'stage'/'b-sample-t3-tp1.pt').unlink()
            with self.assertRaises(FileNotFoundError):audit.compare_case_samples(root,'stage',rows[0],'stage',rows[1])


if __name__=='__main__':unittest.main()
