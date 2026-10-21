from pathlib import Path
import os,sys,unittest
for name in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_latent_mean import fit


class LatentMeanTests(unittest.TestCase):
    def test_heteroskedastic_likelihood_gradient_matches_finite_differences(self):
        rng=np.random.default_rng(971);y=rng.normal(size=150);variance=np.exp(.2*rng.normal(size=len(y)));step=1e-6;phi,share=-.65,.13
        gradient=fit.kalman(y,variance,phi,share)[1];numerical=np.array([(fit.kalman(y,variance,phi+step,share)[0]-fit.kalman(y,variance,phi-step,share)[0])/(2*step),(fit.kalman(y,variance,phi,share+step)[0]-fit.kalman(y,variance,phi,share-step)[0])/(2*step)])
        np.testing.assert_allclose(gradient,numerical,atol=3e-9,rtol=2e-6)

    def test_null_is_exact_and_state_prediction_precedes_observation(self):
        rng=np.random.default_rng(972);y=rng.normal(size=100);variance=np.exp(.2*rng.normal(size=len(y)));value,_,m,P,predicted=fit.kalman(y,variance,0.,0.)
        self.assertAlmostEqual(value,.5*np.mean(np.log(variance)+y*y/variance),places=14);self.assertEqual(m,0);self.assertEqual(P,0);np.testing.assert_array_equal(predicted,np.zeros(len(y)))
        changed=y.copy();changed[30]+=10;a=fit.kalman(y,variance,.9,.1)[4];b=fit.kalman(changed,variance,.9,.1)[4];np.testing.assert_array_equal(a[:31],b[:31]);self.assertNotEqual(a[31],b[31])

    def test_persistent_synthetic_mean_is_identified_without_horizon_cutoff(self):
        rng=np.random.default_rng(973);n=2400;state=0.;y=np.empty(n)
        for t in range(n):state=.96*state+np.sqrt(.2*(1-.96**2))*rng.normal();y[t]=state+np.sqrt(.8)*rng.normal()
        y=(y-y.mean())/y.std();f=fit.estimate(y,np.ones(n));self.assertGreater(f['phi'],.8);self.assertGreater(f['share'],.04);self.assertTrue(f['BIC_selects_latent']);self.assertLess(f['objective'],f['null_objective'])
        a=fit.expected_curve(f,0,99);b=np.concatenate([fit.expected_curve(f,t,1)for t in range(99)]);np.testing.assert_array_equal(a,b);self.assertLess(abs(a[-1]),abs(a[0]));self.assertLess(abs(f['phi']),1)

if __name__=='__main__':unittest.main()
