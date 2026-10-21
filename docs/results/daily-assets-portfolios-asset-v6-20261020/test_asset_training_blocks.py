from pathlib import Path
import os,sys,unittest
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_training_blocks import dependence


class DependenceTests(unittest.TestCase):
    def test_fft_mean_covariance_matches_explicit_stationary_index_markov_chain(self):
        rng=np.random.default_rng(921);x=rng.normal(size=(17,3));z=x-x.mean(0);n=len(x)
        shift=np.roll(np.eye(n),1,axis=1);uniform=np.ones((n,n))/n
        for length in (1.,1.7,8.,100.):
            r=1-1/length;transition=r*shift+(1-r)*uniform;matrix=np.eye(n);total=n*(z.T@z/n)
            for k in range(1,n):
                matrix=matrix@transition;gamma=z.T@matrix@z/n;total+=(n-k)*(gamma+gamma.T)
            np.testing.assert_allclose(dependence.mean_covariance(x,length),total/n**2,rtol=2e-13,atol=3e-16)
        np.testing.assert_allclose(dependence.mean_covariance(x,1.),z.T@z/n**2,atol=2e-17,rtol=3e-16)

    def test_simulated_stationary_mean_covariance_and_uniform_row_exposure(self):
        x=np.column_stack((np.sin(np.arange(23)),np.cos(np.arange(23)/3)));stream=dependence.StationaryRows(len(x),180000,922,4.3)
        rows=stream.take(len(x));means=x[rows].mean(1)
        np.testing.assert_allclose(np.cov(means.T,bias=True),dependence.mean_covariance(x,4.3),rtol=.025,atol=.0002)
        counts=np.bincount(rows.ravel(),minlength=len(x))/rows.size
        np.testing.assert_allclose(counts,np.full(len(x),1/len(x)),rtol=.015,atol=0)

    def test_training_length_is_scale_invariant_and_follows_dependence(self):
        rng=np.random.default_rng(923);iid=rng.normal(size=(1500,2));serial=iid.copy()
        for i in range(1,len(serial)):serial[i]+=.9*serial[i-1]
        before=serial.copy();selected=dependence.select_length(serial)
        self.assertGreater(selected['length'],dependence.select_length(iid)['length'])
        np.testing.assert_allclose(selected['marginal_lengths'],dependence.select_length(serial*np.array([.01,30.]))['marginal_lengths'],rtol=3e-14,atol=2e-13)
        np.testing.assert_array_equal(before,serial)
        two_point=np.tile(np.array([[-1.],[1.]]),(100,1));self.assertEqual(dependence.select_length(two_point)['marginal_lengths'][1],1.)

    def test_joint_stream_chunk_identity_and_cyclic_continuation(self):
        a=dependence.StationaryRows(13,80,924,5.7).take(79);stream=dependence.StationaryRows(13,80,924,5.7)
        b=np.concatenate([stream.take(1)for _ in range(79)],1);np.testing.assert_array_equal(a,b)
        self.assertGreater(np.mean(a[:,1:]==(a[:,:-1]+1)%13),.8)


class GenerativeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from candidates.asset_feature_batch import fits
        from candidates.asset_training_blocks.candidate import prepare,LAWS,family
        rng=np.random.default_rng(925);cls.x=rng.normal(size=(190,2))*.012;cls.x[:,1]+=.4*cls.x[:,0]
        base=fits.base_fit(cls.x);bundles={f:fits.prepare(cls.x,f,base)for f in {family(law)for law in LAWS}}
        cls.prepared={law:prepare(cls.x,law,bundles[family(law)])for law in LAWS}

    def test_no_mixtures_joint_accounting_and_chunk_invariance_for_all_laws(self):
        from candidates.asset_training_blocks.candidate import TrainingBlocks,CONFIGS,identity
        self.assertEqual(len(CONFIGS),14);self.assertEqual(len({identity(c)for c in CONFIGS}),14)
        for config in CONFIGS:
            model=TrainingBlocks(config,self.prepared[config['law']]);self.assertFalse(model.dependency_identity['fixed_component_mixture'])
            actual=np.concatenate([p for _,p in model.terminal_chunks(self.x,[.6,.4],75,100,926)],1)
            pieces=np.concatenate([p for _,p in TrainingBlocks(config,self.prepared[config['law']]).terminal_chunks(self.x,[.6,.4],75,100,926,width=1)],1)
            self.assertEqual(actual.shape,(100,75));np.testing.assert_array_equal(actual,pieces);self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(model.last_asset_weights.sum(1),np.ones(100),rtol=0,atol=5e-15)

    def test_single_asset_raw_law_matches_manual_joint_row_and_mean_draws(self):
        from candidates.asset_feature_batch import fits
        from candidates.asset_prior_best import mean
        from candidates.asset_training_blocks.candidate import TrainingBlocks,prepare,seed
        x=self.x[:,:1];prepared=prepare(x,'raw',fits.prepare(x,'constant'));config={'law':'raw','mean_covariance':'stationary'}
        expected_rows=dependence.StationaryRows(len(x),1000,seed('training_blocks_rows_v1',927),prepared['dependence']['length']).take(33)
        delta=mean.draws(prepared['stationary_mean'],1000,seed('training_blocks_mean_v1',927))
        expected=np.cumsum(x[expected_rows,0]+delta,axis=1)
        actual=np.concatenate([p for _,p in TrainingBlocks(config,prepared).terminal_chunks(x,[1.],33,1000,927)],1)
        np.testing.assert_array_equal(actual,expected)

    def test_changed_training_matrix_cannot_reuse_another_prefix_fit(self):
        from candidates.asset_training_blocks.candidate import TrainingBlocks
        changed=self.x.copy();changed[-1,0]+=.001
        with self.assertRaises(AssertionError):list(TrainingBlocks({'law':'raw','mean_covariance':'batch'},self.prepared['raw']).terminal_chunks(changed,[.6,.4],3,100,928))

if __name__=='__main__':unittest.main()
