import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
import torch
from reference_contract import Plane
from tensor_reference import capture,compare,restore


class TensorReferenceTests(unittest.TestCase):
    def test_native_to_pools_and_back_different_ids_and_chronology(self):
        for total in (31,32,33,127,128,129,256):
            with self.subTest(total=total):
                native=Plane('swa',0,'window','native',12,128,128,1,'float32',(2,))
                pools=Plane('swa',0,'window','pools',5,128,128,1,'float32',(2,))
                src=torch.arange(12*128*2,dtype=torch.float32).reshape(12,128,2)
                dst=torch.full((5,128,2),-991.,dtype=torch.float32)
                snap=capture([(native,src,0)],[[3,8,9]],total)
                restore(snap,[(pools,dst,0)],[[2]],total)
                loaded=capture([(pools,dst,0)],[[2]],total)
                self.assertTrue(compare(snap,loaded)['passed'])
                self.assertTrue(bool((dst[1]==-991).all()));self.assertTrue(bool((dst[3]==-991).all()))
                if total<128:self.assertTrue(bool((dst[2,total:]==-991).all()))
                back=torch.full_like(src,-991.)
                restore(loaded,[(native,back,0)],[[6,4,10]],total)
                self.assertTrue(compare(snap,capture([(native,back,0)],[[6,4,10]],total))['passed'])

    def test_missing_history_does_not_silently_truncate(self):
        p=Plane('h',0,'history','pools',4,128,128,1,'float32',(1,))
        q=Plane('c',0,'compressor','pools',4,32,128,1,'float32',(1,))
        x=torch.ones(4,128,1);y=torch.ones(4,32,1)
        with self.assertRaises(ValueError):capture([(p,x,0),(q,y,1)],[[1],[1]],129)

    def test_corrupt_later_plane_cannot_partially_restore(self):
        p=Plane('h',0,'history','pools',4,128,128,1,'float32',(1,))
        q=Plane('c',0,'compressor','pools',4,32,128,1,'float32',(1,))
        x=torch.ones(4,128,1);y=torch.ones(4,32,1)
        snap=capture([(p,x,0),(q,y,1)],[[1,2],[1]],129)
        snap['planes']['c:0'][0]=999
        a=torch.full_like(x,-7);b=torch.full_like(y,-7)
        with self.assertRaises(ValueError):restore(snap,[(p,a,0),(q,b,1)],[[2,3],[2]],129)
        self.assertTrue(bool((a==-7).all()));self.assertTrue(bool((b==-7).all()))

    def test_valid_data_corruption_detected(self):
        p=Plane('h',0,'history','native',4,64,128,2,'float32',(1,))
        x=torch.arange(256,dtype=torch.float32).reshape(4,64,1)
        a=capture([(p,x,0)],[[1,2]],129);b=copy.deepcopy(a);b['planes']['h:0'][0]+=1
        self.assertFalse(compare(a,b)['passed'])


if __name__=='__main__':unittest.main()
