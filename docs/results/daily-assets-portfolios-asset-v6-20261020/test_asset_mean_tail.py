from pathlib import Path
import os,sys,unittest
for name in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
import numpy as np
from scipy.stats import t
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_mean_tail import mean

class MeanTests(unittest.TestCase):
    def test_Student_marginals_covariance_and_exact_center(self):
        root=np.array([[.2,0.],[.1,.3]]);fit={'sample_mean':np.array([.04,.03]),'center':np.array([.02,.01]),'root':root,'df':9}
        x=mean.draws(fit,60000,7301,'Student_joint');bias=fit['center']-fit['sample_mean'];z=x-bias
        np.testing.assert_allclose(x.mean(0),bias,rtol=0,atol=3e-15);np.testing.assert_allclose(z.T@z/len(z),root@root.T,rtol=.035,atol=.0003)
        for point in(-1.,0.,1.,2.):
            observed=np.mean(z[:,0]/.2<=point);expected=t.cdf(point/np.sqrt(7/9),9);self.assertLess(abs(observed-expected),.008)

    def test_infinite_df_is_exact_joint_Gaussian_limit(self):
        fit={'sample_mean':np.zeros(3),'center':np.zeros(3),'root':np.eye(3),'df':np.inf}
        np.testing.assert_array_equal(mean.draws(fit,100,7302,'Student_joint'),mean.draws(fit,100,7302,'Gaussian_joint'))

    def test_parent_draw_mode_is_exact(self):
        from candidates.asset_prior_best import mean as original
        fit=original.estimate(np.random.default_rng(7303).normal(size=(270,2))*.01)
        for n in(100,1000):np.testing.assert_array_equal(mean.draws(fit,n,7304,'parent'),original.draws(fit,n,7304))

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from candidates.asset_mean_tail.candidate import prepare
        rng=np.random.default_rng(7305);cls.x=rng.normal(size=(270,2))*.012;cls.x[:,1]+=.4*cls.x[:,0];cls.bundle=prepare(cls.x)

    def test_parent_model_is_exact_at100_and1000_paths(self):
        from candidates.asset_mean_tail.candidate import MeanTailPool,BASE_CONFIG
        from candidates.asset_trained_pool.candidate import TrainedPool
        for n in(100,1000):
            a=np.concatenate([p for _,p in MeanTailPool({'mean_law':'parent'},self.bundle).terminal_chunks(self.x,[.6,.4],73,n,7306)],axis=1)
            b=np.concatenate([p for _,p in TrainedPool(BASE_CONFIG,self.bundle).terminal_chunks(self.x,[.6,.4],73,n,7306)],axis=1);np.testing.assert_array_equal(a,b)

    def test_new_laws_counts_joint_accounting_chunks_and_prefix_identity(self):
        from candidates.asset_mean_tail.candidate import MeanTailPool,CONFIGS
        for cfg in CONFIGS:
            m=MeanTailPool(cfg,self.bundle);a=np.concatenate([p for _,p in m.terminal_chunks(self.x,[.6,.4],75,100,7307)],axis=1)
            b=np.concatenate([p for _,p in MeanTailPool(cfg,self.bundle).terminal_chunks(self.x,[.6,.4],75,100,7307,width=1)],axis=1)
            self.assertEqual(a.shape,(100,75));np.testing.assert_array_equal(a,b);self.assertTrue(np.isfinite(a).all());np.testing.assert_allclose(m.last_asset_weights.sum(1),1.,rtol=0,atol=6e-15)
            changed=self.x.copy();changed[-1,0]+=.001
            with self.assertRaises(AssertionError):list(MeanTailPool(cfg,self.bundle).terminal_chunks(changed,[.6,.4],2,100,7307))

    def test_parent_fit_weight_covariance_and_training_df_unchanged(self):
        from candidates.asset_mean_tail.candidate import BASE_CONFIG
        from candidates.asset_trained_pool.candidate import prepare
        parent=prepare(self.x,BASE_CONFIG['family'],BASE_CONFIG['raw_fade']);np.testing.assert_equal(parent['pool_fit'],self.bundle['pool_fit']);np.testing.assert_equal(parent['mean'],self.bundle['mean'])
        self.assertEqual(self.bundle['mean']['df'],int(np.sqrt(len(self.x)))-1)

if __name__=='__main__':unittest.main()
