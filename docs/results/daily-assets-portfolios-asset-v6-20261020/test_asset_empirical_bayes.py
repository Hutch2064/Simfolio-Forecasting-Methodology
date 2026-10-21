from pathlib import Path
import sys,unittest
import numpy as np
from scipy.optimize import minimize_scalar
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_empirical_bayes import mean
from candidates.asset_empirical_bayes.candidate import prepare,EmpiricalBayesPool,BASE_CONFIG,CONFIGS
from candidates.asset_trained_pool.candidate import TrainedPool


def packet(mu,covariance):
    e,v=np.linalg.eigh(covariance)
    return {'sample_mean':np.asarray(mu,dtype=float),'center':np.asarray(mu,dtype=float),'covariance':covariance,'root':v*np.sqrt(np.maximum(e,0.))}


class MeanTests(unittest.TestCase):
    def test_correlated_fit_matches_direct_marginal_likelihood(self):
        C=np.array([[.7,.2],[.2,.5]]);mu=np.array([2.,1.]);fit=mean.estimate(packet(mu,C),True)
        energy=mu@np.linalg.solve(C,mu)
        objective=lambda t: .5*(2*np.log1p(t)+energy/(1+t))
        answer=minimize_scalar(objective,bounds=(0,energy),method='bounded',options={'xatol':1e-10})
        expected=max(0.,energy/2-1)
        self.assertAlmostEqual(fit['empirical_bayes']['prior_variance_ratio'],answer.x,places=6)
        np.testing.assert_allclose(fit['center'],mu*expected/(1+expected),rtol=1e-14)
        np.testing.assert_allclose(fit['root']@fit['root'].T,fit['covariance'],atol=2e-16)

    def test_zero_prior_and_zero_noise_directions(self):
        C=np.diag([1.,0.]);fit=mean.estimate(packet([.2,3.],C),True)
        np.testing.assert_allclose(fit['center'],[0.,3.],atol=1e-15)
        np.testing.assert_array_equal(fit['covariance'],np.zeros((2,2)))
        fit=mean.estimate(packet([2.,3.],np.zeros((2,2))),True)
        np.testing.assert_array_equal(fit['center'],[2.,3.])

    def test_duplicate_asset_invariance_and_scalar_formula(self):
        scalar=mean.estimate(packet([2.],np.ones((1,1))),True)
        duplicate=mean.estimate(packet([2.,2.],np.ones((2,2))),True)
        np.testing.assert_allclose(duplicate['center'],np.tile(scalar['center'],2),atol=1e-14)
        np.testing.assert_allclose(duplicate['covariance'],np.full((2,2),.75),atol=1e-14)
        self.assertEqual(duplicate['empirical_bayes']['rank'],1)

    def test_center_only_retains_original_covariance_and_input(self):
        p=packet([2.,1.],np.array([[.7,.2],[.2,.5]]));before=p['center'].copy();f=mean.estimate(p,False)
        np.testing.assert_array_equal(f['covariance'],p['covariance']);np.testing.assert_array_equal(f['root'],p['root']);np.testing.assert_array_equal(p['center'],before)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(7501);cls.x=rng.normal(size=(231,2))@np.array([[.01,0.],[.003,.009]]).T+.0002;cls.bundle=prepare(cls.x)

    def test_exact_parent_control_at100and1000(self):
        for n in(100,1000):
            a=np.concatenate([v for _,v in EmpiricalBayesPool({'mean':'parent'},self.bundle).terminal_chunks(self.x,[.6,.4],23,n,7502)],axis=1)
            b=np.concatenate([v for _,v in TrainedPool(BASE_CONFIG,self.bundle['parent']).terminal_chunks(self.x,[.6,.4],23,n,7502)],axis=1)
            np.testing.assert_array_equal(a,b)

    def test_joint_path_counts_accounting_chunk_and_prefix(self):
        for config in CONFIGS:
            m=EmpiricalBayesPool(config,self.bundle);a=np.concatenate([v for _,v in m.terminal_chunks(self.x,[.6,.4],71,100,7503)],axis=1)
            b=np.concatenate([v for _,v in EmpiricalBayesPool(config,self.bundle).terminal_chunks(self.x,[.6,.4],71,100,7503,width=1)],axis=1)
            self.assertEqual(a.shape,(100,71));self.assertTrue(np.isfinite(a).all());np.testing.assert_array_equal(a,b);np.testing.assert_allclose(m.last_asset_weights.sum(1),1.,atol=8e-15)
        changed=self.x.copy();changed[-1,0]+=.001
        with self.assertRaises(AssertionError):list(m.terminal_chunks(changed,[.6,.4],2,100,7503))

    def test_other_fits_unchanged_weight_refit_without_holdout(self):
        parent=self.bundle['parent']
        from candidates.asset_feature_batch import fits
        from candidates.asset_trained_pool import weight
        base=fits.base_fit(self.x)
        for config in CONFIGS:
            f=self.bundle[config['mean']];np.testing.assert_equal(f['raw'],parent['raw']);np.testing.assert_equal(f['filter'],parent['filter'])
            np.testing.assert_equal(f['pool_fit'],weight.estimate(self.x,f,base,BASE_CONFIG['raw_fade']))
            np.testing.assert_array_equal(f['mean']['sample_mean'],parent['mean']['sample_mean'])

if __name__=='__main__':unittest.main()
