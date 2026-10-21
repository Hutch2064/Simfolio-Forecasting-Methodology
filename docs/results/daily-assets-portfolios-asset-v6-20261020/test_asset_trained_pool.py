from pathlib import Path
import os,sys,unittest
for name in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
import numpy as np
from scipy.optimize import minimize_scalar
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_trained_pool import weight


class WeightTests(unittest.TestCase):
    def test_quadratic_matches_exact_leave_one_out_distribution_distances(self):
        rng=np.random.default_rng(941);pool=rng.normal(size=29);pool-=pool.mean();observed=pool+.12
        raw=np.exp(rng.normal(size=len(pool))*.2);filtered=np.exp(rng.normal(size=len(pool))*.2)
        c,l,q,bound,_=weight.coefficients(pool,observed,raw,filtered,relative_tolerance=1e-8)
        expected=np.zeros(3)
        for i in range(len(pool)):
            sample=raw[i]*np.delete(pool,i);a=np.abs(sample[:,None]-sample[None,:]).mean()/2;b=filtered[i]/np.sqrt(np.pi)
            d=np.mean([weight.normal_absolute(x,filtered[i])for x in sample]);A=np.abs(sample-observed[i]).mean();B=weight.normal_absolute(observed[i],filtered[i])
            expected+=np.array([B-b,A-B+2*b-d,d-a-b])
        np.testing.assert_allclose([c,l,q],expected,rtol=0,atol=bound*len(pool)+2e-14)
        objective=lambda w:expected[0]+expected[1]*w+expected[2]*w*w
        minimum=minimize_scalar(objective,bounds=(0,1),method='bounded',options={'xatol':1e-14})
        fitted=np.clip(-l/(2*q),0,1);self.assertLessEqual(objective(fitted),minimum.fun+bound*len(pool)/2+1e-12)

    def test_cross_interpolation_obeys_derived_bound_and_refinement(self):
        rng=np.random.default_rng(942);pool=np.r_[rng.standard_t(4,size=181),20.,-31.];scales=np.geomspace(.04,13.,217)
        actual,bound,_=weight.cross_interpolated(pool,scales);exact=weight.cross_values(pool,scales)
        self.assertGreaterEqual(float(np.min(actual-exact)),-2e-14);self.assertLessEqual(float(np.max(actual-exact)),bound+2e-14)
        finer,_,_=weight.cross_interpolated(pool,scales,relative_tolerance=1e-8);self.assertLess(np.max(np.abs(finer-exact)),np.max(np.abs(actual-exact)))

    def test_leave_one_out_terms_with_duplicate_and_zero_observations(self):
        pool=np.array([0.,0.,-1.,1.,1.,-1.,0.]);scale=np.ones(len(pool));c,l,q,bound,_=weight.coefficients(pool,pool,scale,scale)
        self.assertTrue(np.isfinite([c,l,q]).all());self.assertGreaterEqual(q,0)
        for w in(0.,.1,.8,1.):self.assertGreater(c+l*w+q*w*w,0)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from candidates.asset_feature_batch import fits
        from candidates.asset_trained_pool.candidate import CONFIGS,prepare
        rng=np.random.default_rng(943);cls.x=rng.normal(size=(270,2))*.012;cls.x[:,1]+=.5*cls.x[:,0];base=fits.base_fit(cls.x)
        cls.bundles={(c['family'],c['raw_fade']):prepare(cls.x,c['family'],c['raw_fade'],base)for c in CONFIGS}

    def test_forced_half_weight_control_is_exact_current_model(self):
        from candidates.asset_feature_batch.candidate import FeaturePool
        from candidates.asset_trained_pool.candidate import TrainedPool,CONFIGS
        config=next(c for c in CONFIGS if c['family']=='SV_RMS'and c['raw_fade']=='GARCH');bundle=self.bundles['SV_RMS','GARCH'];forced={**bundle,'pool_fit':{**bundle['pool_fit'],'raw_fraction':.5}}
        for paths in(100,1000):
            a=np.concatenate([p for _,p in TrainedPool(config,forced).terminal_chunks(self.x,[.6,.4],71,paths,944)],1)
            b=np.concatenate([p for _,p in FeaturePool({**config,'raw_fade':True},bundle).terminal_chunks(self.x,[.6,.4],71,paths,944)],1)
            np.testing.assert_array_equal(a,b)

    def test_all_fitted_laws_use_only_training_and_exact_declared_paths(self):
        from candidates.asset_trained_pool.candidate import TrainedPool,CONFIGS,identity
        self.assertEqual(len(CONFIGS),6);self.assertEqual(len({identity(c)for c in CONFIGS}),6)
        for config in CONFIGS:
            bundle=self.bundles[config['family'],config['raw_fade']];fraction=bundle['pool_fit']['raw_fraction'];self.assertGreaterEqual(fraction,0);self.assertLessEqual(fraction,1)
            model=TrainedPool(config,bundle);a=np.concatenate([p for _,p in model.terminal_chunks(self.x,[.6,.4],73,100,945)],1)
            b=np.concatenate([p for _,p in TrainedPool(config,bundle).terminal_chunks(self.x,[.6,.4],73,100,945,width=1)],1)
            self.assertEqual(a.shape,(100,73));self.assertEqual(sum(model.last_component_counts),100);self.assertLessEqual(abs(model.last_component_counts[0]/100-fraction),.01+1e-15);np.testing.assert_array_equal(a,b)
            np.testing.assert_allclose(model.last_asset_weights.sum(1),np.ones(100),atol=5e-15,rtol=0)
            changed=self.x.copy();changed[-1,0]+=.001
            with self.assertRaises(AssertionError):list(TrainedPool(config,bundle).terminal_chunks(changed,[.6,.4],2,100,945))

if __name__=='__main__':unittest.main()
