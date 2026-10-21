from pathlib import Path
import os,sys,unittest
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
from unittest.mock import patch
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
from candidates.asset_AR1_mean import fit_ar
from candidates.asset_AR1_mean.candidate import ARPool
from candidates.asset_AR1_mean.kernels import advance
from candidates.asset_fading_vol.candidate import FadingVolPool

class AssetAR1Tests(unittest.TestCase):
    def test_training_coefficient_recovery_invariance_and_stationarity(self):
        rng=np.random.default_rng(159);x=np.zeros(10000);noise=rng.normal(0,1,len(x))
        for i in range(1,len(x)):x[i]=.6*x[i-1]+noise[i]
        value=fit_ar.coefficient(x);self.assertLess(abs(value-.6),.03)
        self.assertAlmostEqual(value,fit_ar.coefficient(3+100*x),places=14)
        self.assertLess(abs(fit_ar.coefficient(np.arange(1000,dtype=float))),1.)

    def test_zero_coefficient_recovers_incumbent_bitwise(self):
        x=np.random.default_rng(817).normal(.0002,.01,(260,2));w=[.6,.4]
        expected=np.concatenate([p for _,p in FadingVolPool(False).terminal_chunks(x,w,93,100,413)],axis=1)
        with patch.object(fit_ar,'coefficient',return_value=0.):
            same=np.concatenate([p for _,p in ARPool().terminal_chunks(x,w,93,100,413)],axis=1)
        np.testing.assert_array_equal(same,expected)

    def test_raw_component_unchanged_and_chunk_invariance(self):
        x=np.random.default_rng(818).normal(.0002,.01,(280,2));w=[.7,.3]
        parent=np.concatenate([p for _,p in FadingVolPool(False).terminal_chunks(x,w,93,100,414)],axis=1)
        first=np.concatenate([p for _,p in ARPool().terminal_chunks(x,w,93,100,414,width=1)],axis=1)
        second=np.concatenate([p for _,p in ARPool().terminal_chunks(x,w,93,100,414,width=64)],axis=1)
        np.testing.assert_array_equal(first[:50],parent[:50]);np.testing.assert_array_equal(first,second)
        self.assertTrue(np.isfinite(first).all())

    def test_conditional_mean_fades_to_unchanged_scenario_mean(self):
        pool=np.zeros((5,1));indices=np.zeros((2,5),dtype=int);delta=np.array([[.002],[-.002]]);mu=np.array([.001]);target=np.array([.01]);alpha=beta=np.array([0.]);variance=np.ones((2,1));phi=np.array([.5]);previous=np.full((2,1),.02);weights=np.array([0.]);current=np.zeros((2,1));assets=np.zeros((2,1));portfolio=np.zeros(2)
        actual=advance(pool,indices,delta,mu,target,alpha,beta,variance,phi,previous,weights,current,assets,portfolio)
        means=mu+delta
        daily=means+(.5**np.arange(1,6))[None,:]*(.02-means)
        np.testing.assert_allclose(actual,np.cumsum(daily,axis=1),rtol=0,atol=3e-17)
if __name__=='__main__':unittest.main()
