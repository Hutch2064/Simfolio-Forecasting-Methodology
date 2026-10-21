from pathlib import Path
import os,sys,unittest
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_component_weights.candidate import ComponentPool,CONFIGS,identity
from candidates.asset_feature_batch.candidate import FeaturePool
from candidates.asset_feature_batch import fits

class ComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(982);x=rng.normal(size=(230,2))*.015;x[150:,1]+=.5*x[150:,0];cls.x=x;cls.bundles={f:fits.prepare(x,f)for f in ('SV_RMS','SV_path','constant')}

    def test_unique_laws_and_single_pure_raw_definition(self):
        self.assertEqual(len(CONFIGS),13);self.assertEqual(len({identity(c)for c in CONFIGS}),13);self.assertEqual(sum(c['raw_fraction']==1 for c in CONFIGS),1)

    def test_equal_mix_control_exact_at_both_screen_and_official_path_counts(self):
        for family in self.bundles:
            config=next(c for c in CONFIGS if c['family']==family and c['raw_fraction']==.5);base={k:v for k,v in config.items()if k!='raw_fraction'}
            for count in (120,1000):
                a=np.concatenate([p for _,p in ComponentPool(config,self.bundles[family]).terminal_chunks(self.x,[.6,.4],35,count,983)],1);b=np.concatenate([p for _,p in FeaturePool(base,self.bundles[family]).terminal_chunks(self.x,[.6,.4],35,count,983)],1);np.testing.assert_array_equal(a,b)

    def test_all_ensembles_have_exact_declared_component_counts_and_no_path_drop(self):
        for config in CONFIGS:
            for count in (120,1000):
                model=ComponentPool(config,self.bundles[config['family']]);result=np.concatenate([p for _,p in model.terminal_chunks(self.x,[.6,.4],31,count,984)],1)
                self.assertEqual(result.shape,(count,31));self.assertTrue(np.isfinite(result).all());self.assertEqual(model.last_component_counts,(int(count*config['raw_fraction']),count-int(count*config['raw_fraction'])));np.testing.assert_allclose(model.last_asset_weights.sum(1),np.ones(count),atol=5e-15,rtol=0)

    def test_endpoints_do_not_depend_on_the_absent_component(self):
        raw=next(c for c in CONFIGS if c['raw_fraction']==1.);bundle=self.bundles['SV_RMS'];changed={**bundle,'filter':{k:(v.copy()if isinstance(v,np.ndarray)else v)for k,v in bundle['filter'].items()}};changed['filter']['target']*=4
        a=np.concatenate([p for _,p in ComponentPool(raw,bundle).terminal_chunks(self.x,[.6,.4],41,120,985)],1);b=np.concatenate([p for _,p in ComponentPool(raw,changed).terminal_chunks(self.x,[.6,.4],41,120,985)],1);np.testing.assert_array_equal(a,b)
        pure=next(c for c in CONFIGS if c['family']=='SV_RMS'and c['raw_fraction']==0.);changed={**bundle,'raw':{k:v.copy()for k,v in bundle['raw'].items()}};changed['raw']['alpha']*=.5;changed['raw']['next_variance_normalized']*=4
        a=np.concatenate([p for _,p in ComponentPool(pure,bundle).terminal_chunks(self.x,[.6,.4],41,120,985)],1);b=np.concatenate([p for _,p in ComponentPool(pure,changed).terminal_chunks(self.x,[.6,.4],41,120,985)],1);np.testing.assert_array_equal(a,b)

    def test_chunk_width_does_not_change_any_mixture(self):
        for config in CONFIGS:
            a=np.concatenate([p for _,p in ComponentPool(config,self.bundles[config['family']]).terminal_chunks(self.x,[.6,.4],71,120,986,width=1)],1);b=np.concatenate([p for _,p in ComponentPool(config,self.bundles[config['family']]).terminal_chunks(self.x,[.6,.4],71,120,986,width=64)],1);np.testing.assert_array_equal(a,b)
if __name__=='__main__':unittest.main()
