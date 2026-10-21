from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys,unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize._numdiff import approx_derivative
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
from candidates.asset_gjr import fit as g
from candidates.asset_gjr.candidate import GJRPool
from candidates.asset_prior_best import fit as symmetric
from candidates.asset_prior_best.candidate import AssetPool
from candidates.asset_circular_blocks.candidate import CircularPool

class GJRAssetTests(unittest.TestCase):
    def test_analytic_gradient(self):
        y=np.random.default_rng(991).standard_t(7,400);y=(y-y.mean())/y.std()
        kn=float(np.mean(np.where(y<0,y*y,0.)))
        for theta in np.random.default_rng(993).normal(0,2,(8,3)):
            gradient=g.objective(theta,y,1-kn,kn)[1]
            numeric=approx_derivative(lambda z:g.objective(z,y,1-kn,kn)[0],theta,method='3-point').reshape(-1)
            np.testing.assert_allclose(gradient,numeric,rtol=2e-5,atol=1e-8)

    def test_stationarity_and_empirical_variance_target(self):
        y=np.random.default_rng(118).standard_t(4,500);y=(y-y.mean())/y.std()
        for theta in np.random.default_rng(119).normal(0,5,(20,3)):
            ap,am,b,_=g.parameters(theta);self.assertLess(b+max(ap,am),1.)
            _,_,variance,_=g.filter_loss(y,ap,am,b,.4,.6)
            pool,omega,persistence,kn=g.target_pool(y,variance,ap,am,b)
            self.assertGreater(omega,0);self.assertLess(persistence,1)
            self.assertAlmostEqual(omega+b+ap*(1-kn)+am*kn,1.,places=14)
            np.testing.assert_allclose(pool.mean(),0,atol=1e-15)
            np.testing.assert_allclose(np.mean(pool*pool),1,atol=1e-15)

    def test_nested_model_exact_paths_and_raw_half_unchanged(self):
        x=np.random.default_rng(312).normal(.0002,.012,(220,2))
        def nested(history):
            p=symmetric.fit_variance_target(history)
            return {**p,'alpha_positive':p['alpha'],'alpha_negative':p['alpha'],'omega_forecast':p['omega']}
        for kind,parent in [('partition',AssetPool(False)),('circular',CircularPool(True))]:
            expected=np.concatenate([p for _,p in parent.terminal_chunks(x,[.6,.4],91,100,881)],axis=1)
            with patch.object(g,'fit',side_effect=nested):
                actual=np.concatenate([p for _,p in GJRPool(kind).terminal_chunks(x,[.6,.4],91,100,881)],axis=1)
            np.testing.assert_array_equal(actual,expected)
            actual=np.concatenate([p for _,p in GJRPool(kind).terminal_chunks(x,[.6,.4],91,100,881)],axis=1)
            np.testing.assert_array_equal(actual[:50],expected[:50])

    def test_recovers_sign_asymmetry_and_scale_equivariance(self):
        rng=np.random.default_rng(15);x=np.empty(4000);variance=1.
        for i in range(len(x)):
            x[i]=np.sqrt(variance)*rng.normal();variance=.11+(.16 if x[i]<0 else .02)*x[i]**2+.8*variance
        a,b=g.fit(x*.01),g.fit(x*.02)
        self.assertGreater(a['alpha_negative'],a['alpha_positive'])
        self.assertLessEqual(a['objective'],a['nested_parent_objective'])
        for key in ['alpha_positive','alpha_negative','beta','next_variance_normalized']:
            self.assertAlmostEqual(a[key],b[key],places=8)
        self.assertAlmostEqual(b['target_variance']/a['target_variance'],4.,places=12)

if __name__=='__main__':unittest.main()
