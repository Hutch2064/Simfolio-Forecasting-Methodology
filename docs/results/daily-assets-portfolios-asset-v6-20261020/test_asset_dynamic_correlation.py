from pathlib import Path
import os,sys,unittest
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_dynamic_correlation import correlation,state,shrinkage

class CorrelationTests(unittest.TestCase):
    def test_shrinkage_matches_independent_library_with_collinear_training_columns(self):
        from sklearn.covariance import ledoit_wolf
        rng=np.random.default_rng(83);x=rng.normal(size=(190,4));x[:,3]=x[:,0]+1e-8*x[:,1];z=(x-x.mean(0))/x.std(0)
        actual,_,weight=shrinkage.shrunk_correlation(z);expected,expected_weight=ledoit_wolf(z,assume_centered=True);d=np.sqrt(np.diag(expected));expected=expected/d[:,None]/d[None,:]
        np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=0);self.assertAlmostEqual(weight,expected_weight,places=14);self.assertGreater(np.linalg.eigvalsh(actual).min(),0)

    def test_corrected_recursion_gradient_matches_finite_differences(self):
        rng=np.random.default_rng(84);z=rng.normal(size=(95,3));s=np.array([[1.,.2,.1],[.2,1.,-.1],[.1,-.1,1.]])
        a,b=.08,.81;loss,gradient,q=state.likelihood(z,s,a,b,s);step=1e-6
        numerical=np.array([(state.likelihood(z,s,a+step,b,s)[0]-state.likelihood(z,s,a-step,b,s)[0])/(2*step),(state.likelihood(z,s,a,b+step,s)[0]-state.likelihood(z,s,a,b-step,s)[0])/(2*step)])
        np.testing.assert_allclose(gradient,numerical,rtol=2e-6,atol=2e-6);self.assertGreater(np.linalg.eigvalsh(q).min(),0)

    def test_mean_Q_is_corrected_law_conditional_expectation(self):
        rng=np.random.default_rng(85);s=np.array([[1.,.2],[.2,1.]]);q=np.array([[1.4,.45],[.45,.8]]);a,b=.08,.86;f={'target':s,'next_Q':q,'a':a,'b':b}
        r=correlation.normalized_correlation(q);z=rng.normal(size=(200000,2))@np.linalg.cholesky(r).T;w=z*np.sqrt(np.diag(q));expected=(1-a-b)*s+a*(w.T@w/len(w))+b*q
        np.testing.assert_allclose(expected,correlation.mean_Q(f,1,1)[0],rtol=0,atol=.0015)
        states=correlation.mean_Q(f,0,30)
        np.testing.assert_array_equal(states[0],q)
        for t in range(29):np.testing.assert_allclose(states[t+1],(1-a-b)*s+(a+b)*states[t],atol=5e-16,rtol=0)

    def test_roots_have_unit_marginal_variance_and_chunk_invariance(self):
        f={'target':np.array([[1.,.2],[.2,1.]]),'next_Q':np.array([[1.4,.45],[.45,.8]]),'a':.08,'b':.86}
        all_roots=correlation.roots(f,0,79,'conditional_mean',np.eye(2));pieces=np.concatenate([correlation.roots(f,i,1,'conditional_mean',np.eye(2))for i in range(79)])
        np.testing.assert_array_equal(all_roots,pieces)
        for root in all_roots:np.testing.assert_allclose(np.diag(root@root.T),np.ones(2),atol=10*np.finfo(float).eps,rtol=0)
        g={**f,'next_Q':f['target']};np.testing.assert_array_equal(correlation.roots(g,0,79,'conditional_mean',np.eye(2)),correlation.roots(g,0,79,'static_shrinkage',np.eye(2)))

    def test_scalar_identity_and_training_fit_no_mutation(self):
        rng=np.random.default_rng(86);z=rng.normal(size=(130,1));before=z.copy();f=correlation.prepare(z);np.testing.assert_array_equal(z,before);self.assertTrue(f['scalar_identity'])
        root=np.array([[np.nextafter(1.,0.)]]);np.testing.assert_array_equal(correlation.roots(f,0,25,'conditional_mean',root),np.repeat(root[None,:,:],25,0))

class CorrelationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from candidates.asset_feature_batch import fits
        rng=np.random.default_rng(871);z=rng.normal(size=(230,2));x=z*.012;x[115:,1]=.85*x[115:,0]+.005*z[115:,1]
        cls.x=x;cls.bundles={f:fits.prepare(x,f)for f in ('GARCH','SV_path','SV_RMS')}
        for bundle in cls.bundles.values():bundle['correlation']=correlation.prepare(bundle['filter']['pool'])

    def test_all_static_sample_controls_are_exact_parent_paths(self):
        from candidates.asset_dynamic_correlation.candidate import CONFIGS,CorrelationPool
        from candidates.asset_feature_batch.candidate import FeaturePool
        for config in CONFIGS:
            if config['correlation']!='sample':continue
            base={k:v for k,v in config.items()if k!='correlation'};bundle=self.bundles[config['family']]
            actual=np.concatenate([p for _,p in CorrelationPool(config,bundle).terminal_chunks(self.x,[.6,.4],73,100,872)],1)
            expected=np.concatenate([p for _,p in FeaturePool(base,bundle).terminal_chunks(self.x,[.6,.4],73,100,872)],1)
            np.testing.assert_array_equal(actual,expected,err_msg=str(config))

    def test_all_models_preserve_joint_finite_paths_and_chunk_identity(self):
        from candidates.asset_dynamic_correlation.candidate import CONFIGS,CorrelationPool,identity
        self.assertEqual(len(CONFIGS),19);self.assertEqual(len({identity(c)for c in CONFIGS}),19)
        for config in CONFIGS:
            bundle=self.bundles[config['family']];model=CorrelationPool(config,bundle)
            actual=np.concatenate([p for _,p in model.terminal_chunks(self.x,[.6,.4],73,100,873,width=64)],1)
            small=np.concatenate([p for _,p in CorrelationPool(config,bundle).terminal_chunks(self.x,[.6,.4],73,100,873,width=1)],1)
            self.assertEqual(actual.shape,(100,73));self.assertTrue(np.isfinite(actual).all());np.testing.assert_array_equal(actual,small,err_msg=str(config))
            np.testing.assert_allclose(model.last_asset_weights.sum(1),np.ones(100),atol=5e-15,rtol=0)

    def test_scalar_dynamic_configurations_retain_exact_parent_marginals(self):
        from candidates.asset_dynamic_correlation.candidate import CONFIGS,CorrelationPool
        from candidates.asset_feature_batch.candidate import FeaturePool
        from candidates.asset_feature_batch import fits
        history=self.x[:,:1];bundles={f:fits.prepare(history,f)for f in ('GARCH','SV_path','SV_RMS')}
        for config in CONFIGS:
            if config['correlation']=='sample':continue
            bundle=bundles[config['family']];base={k:v for k,v in config.items()if k!='correlation'}
            actual=np.concatenate([p for _,p in CorrelationPool(config,bundle).terminal_chunks(history,[1.],37,100,874)],1)
            expected=np.concatenate([p for _,p in FeaturePool(base,bundle).terminal_chunks(history,[1.],37,100,874)],1)
            np.testing.assert_array_equal(actual,expected,err_msg=str(config))

if __name__=='__main__':unittest.main()
