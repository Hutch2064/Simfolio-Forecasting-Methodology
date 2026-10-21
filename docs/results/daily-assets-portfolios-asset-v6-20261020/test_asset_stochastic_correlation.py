from pathlib import Path
import os,sys,unittest
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6' if (O/'asset_level_v6').exists() else O;sys.path.insert(0,str(V))
from candidates.asset_stochastic_correlation import states
from candidates.asset_stochastic_correlation.candidate import StochasticCorrelationPool,BASE_CONFIG,CONFIGS,prepare
from candidates.asset_trained_pool.candidate import TrainedPool,prepare as parent_prepare

class RecursionTests(unittest.TestCase):
    def test_matches_independent_numpy_corrected_recursion(self):
        rng=np.random.default_rng(7101);s=np.array([[1.,.35,-.1],[.35,1.,.2],[-.1,.2,1.]])
        q0=np.array([[1.3,.4,.1],[.4,.8,.15],[.1,.15,1.1]]);q=np.tile(q0,(5,1,1));normal=rng.normal(size=(9,5,3));expected=np.empty((5,9,3));target=q.copy()
        for t in range(9):
            for p in range(5):
                v=np.linalg.cholesky(target[p])@normal[t,p];expected[p,t]=v/np.sqrt(np.diag(target[p]));target[p]=.08*s+.85*target[p]+.07*np.outer(v,v)
        actual=states.advance(normal,q,s,.07,.85)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=3e-14);np.testing.assert_allclose(q,target,rtol=0,atol=3e-14)
        self.assertTrue(np.all(np.linalg.eigvalsh(q)>0))

    def test_chunk_state_is_exact_and_not_reset(self):
        rng=np.random.default_rng(7102);s=np.array([[1.,.4],[.4,1.]]);normal=rng.normal(size=(19,11,2));whole=np.tile(s,(11,1,1));split=whole.copy()
        a=states.advance(normal,whole,s,.1,.82);b=np.concatenate([states.advance(normal[:7],split,s,.1,.82),states.advance(normal[7:],split,s,.1,.82)],axis=1)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(whole,split)

    def test_conditional_gaussian_covariance_and_expected_Q(self):
        rng=np.random.default_rng(7103);s=np.array([[1.,.25],[.25,1.]]);q0=np.array([[1.4,.7],[.7,.9]]);n=40000;normal=rng.normal(size=(5,n,2));q=np.tile(q0,(n,1,1));a=.12;b=.72
        z=states.advance(normal,q,s,a,b);r=q0/np.sqrt(np.diag(q0))[:,None]/np.sqrt(np.diag(q0))[None,:]
        products=z[:,0,:,None]*z[:,0,None,:];se=products.std(axis=0,ddof=1)/np.sqrt(n)
        self.assertTrue(np.all(np.abs(products.mean(axis=0)-r)<6*se))
        expected=s+(a+b)**5*(q0-s);se=q.std(axis=0,ddof=1)/np.sqrt(n)
        self.assertTrue(np.all(np.abs(q.mean(axis=0)-expected)<6*se));self.assertGreater(q[:,0,1].std(),.01)

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(7104);cls.x=rng.normal(size=(260,2))*.01;cls.x[:,1]+=.4*cls.x[:,0];cls.bundle=prepare(cls.x)

    def test_sample_control_is_byte_identical_parent_at_both_precisions(self):
        for n in (100,1000):
            p=np.concatenate([x for _,x in TrainedPool(BASE_CONFIG,self.bundle).terminal_chunks(self.x,[.7,.3],81,n,7105)],axis=1)
            q=np.concatenate([x for _,x in StochasticCorrelationPool({'correlation':'sample'},self.bundle).terminal_chunks(self.x,[.7,.3],81,n,7105)],axis=1)
            np.testing.assert_array_equal(p,q)

    def test_all_modes_joint_accounting_counts_and_chunk_identity(self):
        for cfg in CONFIGS:
            m=StochasticCorrelationPool(cfg,self.bundle);a=np.concatenate([x for _,x in m.terminal_chunks(self.x,[.7,.3],83,100,7106)],axis=1)
            b=np.concatenate([x for _,x in StochasticCorrelationPool(cfg,self.bundle).terminal_chunks(self.x,[.7,.3],83,100,7106,width=1)],axis=1)
            self.assertEqual(a.shape,(100,83));self.assertTrue(np.isfinite(a).all());np.testing.assert_array_equal(a,b)
            np.testing.assert_allclose(m.last_asset_weights.sum(axis=1),1.,rtol=0,atol=6e-15)
            changed=self.x.copy();changed[-1,0]+=.001
            with self.assertRaises(AssertionError):list(StochasticCorrelationPool(cfg,self.bundle).terminal_chunks(changed,[.7,.3],2,100,7106))

    def test_every_single_asset_mode_is_exact_parent(self):
        x=self.x[:,:1];bundle=prepare(x)
        for n in (100,1000):
            expected=np.concatenate([x for _,x in TrainedPool(BASE_CONFIG,bundle).terminal_chunks(x,[1.],73,n,7107)],axis=1)
            for cfg in CONFIGS:
                actual=np.concatenate([x for _,x in StochasticCorrelationPool(cfg,bundle).terminal_chunks(x,[1.],73,n,7107)],axis=1)
                np.testing.assert_array_equal(actual,expected)

    def test_fits_preserve_parent_weight_and_training_only_identity(self):
        parent=parent_prepare(self.x,BASE_CONFIG['family'],BASE_CONFIG['raw_fade']);np.testing.assert_equal(parent['pool_fit'],self.bundle['pool_fit']);dep=self.bundle['dependence']
        self.assertGreaterEqual(dep['a'],0);self.assertGreaterEqual(dep['b'],0);self.assertLess(dep['a']+dep['b'],1)
        self.assertEqual(len(CONFIGS),3);self.assertEqual(set(CONFIGS[0]),{'correlation'})

if __name__=='__main__':unittest.main()
