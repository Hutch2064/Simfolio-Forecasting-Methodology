from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys,unittest
from unittest.mock import patch
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
from candidates.asset_fading_vol import fade
from candidates.asset_fading_vol.candidate import FadingVolPool
from candidates.asset_fading_vol.kernels import advance_raw
from candidates.asset_circular_blocks.candidate import CircularPool

class FadingVolTests(unittest.TestCase):
    def test_positive_variance_and_historical_long_run_target(self):
        initial=np.array([.1,4.,1e-20]);decay=np.array([.94,.98,.999])
        scales=fade.scale_table(initial,decay,0,100)
        np.testing.assert_array_equal(scales[:,0],np.sqrt(initial))
        self.assertTrue(np.all(scales>0))
        np.testing.assert_array_equal(fade.scale_table(initial,decay,1000000,3),np.ones((3,3)))

    def test_raw_mean_is_preserved_under_complete_historical_row_enumeration(self):
        x=np.array([[-.02],[.01],[.03],[-.01]]);mu=x.mean(axis=0)
        indices=np.arange(4).reshape(4,1);delta=np.full((4,1),.003)
        output=advance_raw(x,indices,delta,mu,np.array([[2.]]),np.array([0.]),np.zeros((4,1)),np.zeros((4,1)),np.zeros(4))
        self.assertAlmostEqual(output.mean(),mu[0]+.003,places=16)

    def test_exact_incumbent_recovery_without_signal_and_unchanged_filtered_half(self):
        x=np.random.default_rng(417).normal(.0002,.012,(260,2))
        expected=np.concatenate([p for _,p in CircularPool(True).terminal_chunks(x,[.7,.3],91,100,677)],axis=1)
        for ewma in (False,True):
            with patch.object(fade,'scale_table',side_effect=lambda initial,decay,start,count:np.ones((len(initial),count))):
                got=np.concatenate([p for _,p in FadingVolPool(ewma).terminal_chunks(x,[.7,.3],91,100,677)],axis=1)
            np.testing.assert_array_equal(got,expected)
            actual=np.concatenate([p for _,p in FadingVolPool(ewma).terminal_chunks(x,[.7,.3],91,100,677)],axis=1)
            np.testing.assert_array_equal(actual[50:],expected[50:])

    def test_ewma_filter_and_chunk_invariance(self):
        y=np.array([.2,-1.,.7]);d=.8;v=1.;loss=0.
        for value in y:
            loss+=np.log(v)+value*value/v;v=d*v+(1-d)*value*value
        actual,next_v=fade.ewma_loss(y,d)
        self.assertAlmostEqual(actual,.5*loss/len(y),places=15);self.assertAlmostEqual(next_v,v,places=15)
        x=np.random.default_rng(618).normal(.0003,.01,(280,2))
        for ewma in (False,True):
            a,b=FadingVolPool(ewma),FadingVolPool(ewma)
            p=np.concatenate([p for _,p in a.terminal_chunks(x,[.6,.4],83,100,998,width=1)],axis=1)
            q=np.concatenate([p for _,p in b.terminal_chunks(x,[.6,.4],83,100,998,width=64)],axis=1)
            np.testing.assert_array_equal(p,q)

if __name__=='__main__':unittest.main()
