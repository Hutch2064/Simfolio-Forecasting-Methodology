from pathlib import Path
import os,sys,unittest
for name in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_likelihood_sv import fit


class LikelihoodTests(unittest.TestCase):
    def test_training_likelihood_gradient_matches_independent_finite_difference(self):
        z=np.random.default_rng(931).normal(size=170)*2;phi,V=.83,.7;step=1e-6
        loss,gradient,_,_,_=fit.filter_loss(z,phi,V)
        numerical=np.array([(fit.filter_loss(z,phi+step,V)[0]-fit.filter_loss(z,phi-step,V)[0])/(2*step),(fit.filter_loss(z,phi,V+step)[0]-fit.filter_loss(z,phi,V-step)[0])/(2*step)])
        np.testing.assert_allclose(gradient,numerical,rtol=2e-6,atol=2e-8)

    def test_prediction_variance_precedes_current_observation_and_stationary_null(self):
        z=np.random.default_rng(932).normal(size=90);changed=z.copy();changed[40]+=20
        a=fit.filter_loss(z,.9,.7)[4];b=fit.filter_loss(changed,.9,.7)[4]
        np.testing.assert_array_equal(a[:41],b[:41]);self.assertNotEqual(a[41],b[41]);np.testing.assert_array_equal(fit.filter_loss(z,0.,0.)[4],np.zeros(len(z)))

    def test_likelihood_fit_resolves_persistent_latent_state_and_constant_boundary(self):
        rng=np.random.default_rng(933);h=np.empty(2200);h[0]=0
        for t in range(1,len(h)):h[t]=.97*h[t-1]+.23*rng.normal()
        y=np.exp(h/2)*rng.normal(size=len(h));y=(y-y.mean())/y.std();before=y.copy();f=fit.estimate(y)
        self.assertGreater(f['sv_phi'],.8);self.assertGreater(f['V'],.05);self.assertLess(f['objective'],f['null_objective']);np.testing.assert_array_equal(y,before)
        self.assertTrue(np.isfinite(f['pool']).all());self.assertGreaterEqual(f['P'],0);self.assertLess(f['sv_phi'],1)

    def test_all_twelve_single_laws_have_exact_path_count_joint_accounting_and_chunk_identity(self):
        from candidates.asset_feature_batch import fits
        from candidates.asset_likelihood_sv.candidate import LikelihoodSV,CONFIGS,identity,prepare
        rng=np.random.default_rng(934);x=rng.normal(size=(250,2))*.014;x[:,1]+=.6*x[:,0];bundle=prepare(x,fits.base_fit(x))
        self.assertEqual(len(CONFIGS),12);self.assertEqual(len({identity(c)for c in CONFIGS}),12)
        for config in CONFIGS:
            model=LikelihoodSV(config,bundle);a=np.concatenate([p for _,p in model.terminal_chunks(x,[.7,.3],71,100,935)],1)
            b=np.concatenate([p for _,p in LikelihoodSV(config,bundle).terminal_chunks(x,[.7,.3],71,100,935,width=1)],1)
            self.assertFalse(model.dependency_identity['fixed_component_mixture']);self.assertEqual(a.shape,(100,71));np.testing.assert_array_equal(a,b)
            np.testing.assert_allclose(model.last_asset_weights.sum(1),np.ones(100),rtol=0,atol=5e-15)

if __name__=='__main__':unittest.main()
