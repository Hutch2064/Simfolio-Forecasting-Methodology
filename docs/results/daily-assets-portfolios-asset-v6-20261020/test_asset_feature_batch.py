from pathlib import Path
import os,sys,unittest
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
from candidates.asset_feature_batch import fits
from candidates.asset_feature_batch.candidate import FeaturePool,CONFIGS,identity
from candidates.asset_fading_vol.candidate import FadingVolPool

class FeatureBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(701);scale=np.r_[np.full(130,.004),np.full(130,.025)]
        cls.x=rng.normal(size=(260,2))*scale[:,None]+.0003
        cls.bundles={family:fits.prepare(cls.x,family) for family in fits.FAMILIES}

    def test_complete_grid_has_distinct_declarations(self):
        self.assertEqual(len(CONFIGS),216);self.assertEqual(len({identity(c) for c in CONFIGS}),216)
        self.assertEqual(set(c['family'] for c in CONFIGS),set(fits.FAMILIES))

    def test_incumbent_anchor_is_bitwise_exact(self):
        config={'family':'GARCH','innovations':'empirical','raw_fade':True,'mean_shrinkage':True,'mean_uncertainty':True}
        actual=np.concatenate([p for _,p in FeaturePool(config,self.bundles['GARCH']).terminal_chunks(self.x,[.7,.3],93,100,674)],axis=1)
        expected=np.concatenate([p for _,p in FadingVolPool(False).terminal_chunks(self.x,[.7,.3],93,100,674)],axis=1)
        np.testing.assert_array_equal(actual,expected)

    def test_all_configurations_score_all_paths_and_horizons(self):
        for config in CONFIGS:
            model=FeaturePool(config,self.bundles[config['family']])
            actual=np.concatenate([p for _,p in model.terminal_chunks(self.x,[.7,.3],37,100,675)],axis=1)
            self.assertEqual(actual.shape,(100,37));self.assertTrue(np.isfinite(actual).all(),str(config))
            np.testing.assert_allclose(model.last_asset_weights.sum(axis=1),1.,rtol=0,atol=5e-15)

    def test_family_innovation_covariance_and_stationarity_targets(self):
        for family,bundle in self.bundles.items():
            f=bundle['filter'];cov=f['root']@f['root'].T
            np.testing.assert_allclose(np.diag(cov),np.ones(2),rtol=0,atol=3e-15)
            self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(),-1e-14)
            self.assertTrue(np.all(f['initial']>0));self.assertTrue(np.all(f['target']>0))
            if family=='HARCH':self.assertTrue(np.all(f['weights'].sum(axis=1)<1))
            if family.startswith('SV_'):
                self.assertTrue(np.all(f['sv_phi']<1));np.testing.assert_allclose(f['level']+.5*f['V'],0.,rtol=0,atol=1e-15)

    def test_chunk_invariance_for_all_filters_and_innovation_laws(self):
        for family in fits.FAMILIES:
            for innovations in ('empirical','Gaussian','Student'):
                config={'family':family,'innovations':innovations,'raw_fade':True,'mean_shrinkage':True,'mean_uncertainty':True}
                a=FeaturePool(config,self.bundles[family]);b=FeaturePool(config,self.bundles[family])
                first=np.concatenate([p for _,p in a.terminal_chunks(self.x,[.7,.3],75,100,676,width=1)],axis=1)
                second=np.concatenate([p for _,p in b.terminal_chunks(self.x,[.7,.3],75,100,676,width=64)],axis=1)
                np.testing.assert_array_equal(first,second,err_msg=str(config))

    def test_rms_volatility_matches_gaussian_state_second_moment(self):
        rng=np.random.default_rng(199);m,P=-.4,.3
        actual=np.mean(np.exp(rng.normal(m,np.sqrt(P),400000)))
        expected=np.exp(m+.5*P)
        self.assertLess(abs(actual/expected-1),.005)
if __name__=='__main__':unittest.main()
