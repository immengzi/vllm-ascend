import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
from reference_contract import Plane, audit_wire_boundary, boundary_counters, check_hybrid_reference, row_locations


def plane(kind,layout='pools',compression=1):
    rows=32 if kind=='compressor' else 128//compression if kind=='history' else 128
    return Plane('layer.'+kind,0,kind,layout,20,rows,128,compression,'float32',(2,))


class ContractTests(unittest.TestCase):
    def test_window_before_at_and_after_wrap(self):
        for total in (1,31,32,33,127,128,129,255,256,257):
            with self.subTest(total=total):
                p=plane('window');got=row_locations(p,[7],total)
                self.assertEqual(got,[(7,i%128) for i in range(max(0,total-128),total)])

    def test_native_window_uses_two_logical_pages_across_wrap(self):
        got=row_locations(plane('window','native'),[3,11,9],129)
        self.assertEqual(got[0],(3,1));self.assertEqual(got[-1],(11,0));self.assertEqual(len(got),128)
        self.assertEqual(row_locations(plane('window','native'),[0,11,9],256)[0],(11,0))

    def test_missing_active_native_page_is_rejected(self):
        with self.assertRaises(ValueError):row_locations(plane('window','native'),[0,11],129)

    def test_history_compression_and_partial_page(self):
        for ratio in (1,2):
            for total in (127,128,129,256):
                with self.subTest(ratio=ratio,total=total):
                    p=plane('history',compression=ratio);got=row_locations(p,[8,2,5],total)
                    self.assertEqual(len(got),total//ratio)
                    self.assertEqual(got,[( [8,2,5][i//p.rows], i%p.rows) for i in range(total//ratio)])

    def test_compressor_chronology(self):
        self.assertEqual(row_locations(plane('compressor'),[6],33),[(6,i) for i in range(1,32)]+[(6,0)])

    def test_invalid_or_alias_ids(self):
        for ids in ([0],[20],[-1],[True],[2,2]):
            with self.subTest(ids=ids),self.assertRaises(ValueError):row_locations(plane('history'),ids,129)

    def test_one_ring_id_only(self):
        with self.assertRaises(ValueError):row_locations(plane('compressor'),[3,4],33)

    def test_counter_even_odd_and_window(self):
        for total in (127,128,129,256):
            d=boundary_counters(total,total+1)
            self.assertEqual(d['window_next'],total%128)
            self.assertEqual(d['compressed_residual'],total%2)
            wire=dict(boundary=total,original_prompt_tokens=total+1,
                      **{k:v for k,v in d.items() if k.startswith(('window_','compressor_','compressed_'))})
            self.assertEqual(audit_wire_boundary(wire,total,total+1),d)
            wire['window_next']=(wire['window_next']+1)%128
            with self.assertRaises(ValueError):audit_wire_boundary(wire,total,total+1)

    def test_d_full_prefill_not_heterogeneous_oracle(self):
        d=dict(passed=True,mode='reference',rows=[dict(token_ids=[1])])
        with self.assertRaisesRegex(ValueError,'hybrid reference'):check_hybrid_reference(d,4,2,'runtime')
        self.assertIs(check_hybrid_reference(d,4,4,None),d)

    def test_hybrid_requires_identity_and_evidence(self):
        d=dict(passed=True,mode='reference',reference_kind='hybrid_snapshot_replay',producer_tp=4,consumer_tp=2,
               runtime_identity='runtime',rows=[dict(output_count=1,reference_audit=dict(passed=True,transport='local_snapshot_replay',
                   repeat_audit=dict(passed=True,samples=[dict(rank=rank,step=1,passed=True) for rank in (0,1)]),
                   source_sha256='a'*64,loaded_sha256='b'*64,first_logits_sha256='c'*64))])
        self.assertIs(check_hybrid_reference(d,4,2,'runtime'),d)
        for key in ('source_sha256','loaded_sha256','first_logits_sha256'):
            bad=copy.deepcopy(d);del bad['rows'][0]['reference_audit'][key]
            with self.assertRaises(ValueError):check_hybrid_reference(bad,4,2,'runtime')
        with self.assertRaises(ValueError):check_hybrid_reference(d,4,2,'other')


if __name__=='__main__':unittest.main()
