from pathlib import Path
import os,sys,unittest,itertools
for name in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
import numpy as np
from scipy.stats import norm
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_regime_rows import fit


class RegimeFitTests(unittest.TestCase):
    def test_forward_backward_matches_all_state_sequences(self):
        y=np.array([-.3,.7,-1.,.4,1.2,-.5]);means=np.array([-.4,.7]);variance=.8;transition=np.array([[.91,.09],[.17,.83]]);initial=np.array([.6,.4])
        ll,gamma,counts,filtered=fit.smooth(y,means,variance,transition,initial);mass=0.;g=np.zeros_like(gamma);c=np.zeros_like(counts)
        for states in itertools.product((0,1),repeat=len(y)):
            probability=initial[states[0]]
            for t,state in enumerate(states):
                probability*=norm.pdf(y[t],means[state],np.sqrt(variance))
                if t:probability*=transition[states[t-1],state]
            mass+=probability
            for t,state in enumerate(states):
                g[t,state]+=probability
                if t:c[states[t-1],state]+=probability
        self.assertAlmostEqual(ll,np.log(mass),places=13);np.testing.assert_allclose(gamma,g/mass,atol=3e-15,rtol=0);np.testing.assert_allclose(counts,c/mass,atol=3e-15,rtol=0);np.testing.assert_allclose(filtered,g[-1]/mass,atol=3e-15,rtol=0)

    def test_observed_likelihood_gradient_matches_finite_differences(self):
        y=np.random.default_rng(951).normal(size=130);theta=np.array([-.4,.5,np.log(.7),-1.6,1.9,.2]);step=1e-6
        value,gradient=fit.objective(theta,y);numerical=[]
        for j in range(len(theta)):
            delta=np.zeros(len(theta));delta[j]=step;numerical.append((fit.objective(theta+delta,y)[0]-fit.objective(theta-delta,y)[0])/(2*step))
        np.testing.assert_allclose(gradient,numerical,atol=3e-9,rtol=2e-6)

    def test_transition_conditioning_fades_at_the_fitted_chain_rate(self):
        transition=np.array([[.93,.07],[.13,.87]]);stationary=np.array([.13,.07])/.2;probability=np.array([.1,.9]);initial=probability-stationary
        for t in range(1,100):
            probability=probability@transition;np.testing.assert_allclose(probability-stationary,initial*.8**t,atol=100*np.finfo(float).eps,rtol=0)


class RegimeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from candidates.asset_regime_rows.candidate import prepare
        rng=np.random.default_rng(952);n=700;state=np.zeros(n,dtype=int)
        for t in range(1,n):state[t]=state[t-1]if rng.random()<.97 else 1-state[t-1]
        cls.x=rng.normal(size=(n,2))*.008*np.where(state[:,None],2.4,.7);cls.x[:,1]+=.4*cls.x[:,0];cls.bundle=prepare(cls.x)

    def test_fitted_common_regimes_use_joint_finite_paths_and_chunk_identity(self):
        from candidates.asset_regime_rows.candidate import RegimeRows,CONFIGS,identity
        self.assertEqual(len(CONFIGS),4);self.assertEqual(len({identity(c)for c in CONFIGS}),4);self.assertTrue(all(r['eligible']for r in self.bundle['reports']))
        self.assertLess(self.bundle['two_state_BIC'],self.bundle['one_state_BIC'])
        np.testing.assert_allclose(self.bundle['gamma'].sum(1),np.ones(len(self.x)),atol=5e-16,rtol=0)
        for config in CONFIGS:
            model=RegimeRows(config,self.bundle);a=np.concatenate([p for _,p in model.terminal_chunks(self.x,[.6,.4],79,100,953)],1)
            b=np.concatenate([p for _,p in RegimeRows(config,self.bundle).terminal_chunks(self.x,[.6,.4],79,100,953,width=1)],1)
            self.assertEqual(a.shape,(100,79));np.testing.assert_array_equal(a,b);self.assertTrue(np.isfinite(a).all());np.testing.assert_allclose(model.last_asset_weights.sum(1),np.ones(100),atol=5e-15,rtol=0)

    def test_genuine_constant_signal_has_disclosed_single_state_boundary(self):
        from candidates.asset_regime_rows.candidate import RegimeRows,CONFIGS,prepare
        x=np.tile(np.array([[-.01],[.01]]),(50,1));bundle=prepare(x);self.assertTrue(bundle['one_state'])
        for config in CONFIGS:
            model=RegimeRows(config,bundle);out=np.concatenate([p for _,p in model.terminal_chunks(x,[1.],31,1000,954)],1);self.assertEqual(out.shape,(1000,31));self.assertEqual(model.last_state_count,1)

if __name__=='__main__':unittest.main()
