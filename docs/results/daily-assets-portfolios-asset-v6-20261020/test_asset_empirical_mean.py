from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys,unittest
from unittest.mock import patch
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
from candidates.asset_empirical_mean import mean
from candidates.asset_empirical_mean.candidate import EmpiricalMeanPool
from candidates.asset_prior_best import mean as gaussian
from candidates.asset_circular_blocks.candidate import CircularPool

class EmpiricalMeanTests(unittest.TestCase):
    def test_exact_weighted_population_mean_and_covariance_with_unequal_batches(self):
        for n in (501,817):
            x=np.random.default_rng(n).normal(.0002,.013,(n,4))
            f=mean.estimate(x);p=f['probabilities'];d=f['support']
            np.testing.assert_allclose(p@d,0,atol=1e-18)
            np.testing.assert_allclose((d*p[:,None]).T@d,f['covariance'],rtol=2e-14,atol=1e-20)

    def test_finite_ensemble_has_exact_target_mean(self):
        x=np.random.default_rng(189).normal(.0002,.012,(513,3));f=mean.estimate(x)
        for symmetric in (False,True):
            d=mean.draws(f,1000,291,symmetric=symmetric)
            self.assertTrue(np.isfinite(d).all())
            np.testing.assert_allclose(d.mean(axis=0),f['center']-f['sample_mean'],atol=1e-18)

    def test_current_model_is_recovered_when_original_gaussian_draws_are_used(self):
        x=np.random.default_rng(173).normal(.0002,.011,(240,2))
        expected=np.concatenate([p for _,p in CircularPool(True).terminal_chunks(x,[.7,.3],87,100,611)],axis=1)
        for symmetric in (False,True):
            with patch.object(mean,'draws',side_effect=lambda f,n,s,**kw:gaussian.draws(f,n,s)):
                actual=np.concatenate([p for _,p in EmpiricalMeanPool(symmetric).terminal_chunks(x,[.7,.3],87,100,611)],axis=1)
            np.testing.assert_array_equal(actual,expected)

    def test_complete_model_is_chunk_invariant(self):
        x=np.random.default_rng(438).normal(.0002,.012,(260,2))
        for symmetric in (False,True):
            a,b=EmpiricalMeanPool(symmetric),EmpiricalMeanPool(symmetric)
            p=np.concatenate([p for _,p in a.terminal_chunks(x,[.8,.2],81,100,971,width=1)],axis=1)
            q=np.concatenate([p for _,p in b.terminal_chunks(x,[.8,.2],81,100,971,width=64)],axis=1)
            np.testing.assert_array_equal(p,q)

if __name__=='__main__':unittest.main()
