from pathlib import Path
import os,sys,unittest
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_selector_universe import bank as b
from candidates.asset_selector_universe_precision import candidate as c
from candidates.asset_fast_selector import bank as old

class BroaderOptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x=np.random.default_rng(212).normal(.0002,.01,size=(96,2));cls.w=np.array([.6,.4])
        cls.p,cls.invalid=b.prefix_fits(cls.x,list(range(len(b.BANK))))
    def test_declared_full_factorial_unique_definitions(self):
        self.assertEqual(len(b.FILTERED),162);self.assertEqual(len(b.BANK),166)
        self.assertEqual(len({b.identity(s)for s in b.BANK}),166)
        self.assertEqual({s['family']for s in b.FILTERED},set(b.fits.FAMILIES))
        self.assertEqual({s['mean_mode']for s in b.FILTERED},{'sample','shrink','zero'})
    def test_every_option_produces_finite_joint_paths(self):
        self.assertFalse(self.invalid)
        for index in range(len(b.BANK)):
            model=b.model(index,self.p);paths=np.concatenate([v for _,v in model.terminal_chunks(self.x,self.w,21,100,77)],axis=1)
            self.assertEqual(paths.shape,(100,21));self.assertTrue(np.isfinite(paths).all(),b.BANK[index])
            self.assertEqual(model.last_asset_log_paths.shape,(100,2))
            np.testing.assert_allclose(model.last_asset_weights.sum(axis=1),1.,atol=3e-15)
    def test_inherited_parent_shrunk_option_exact_parity(self):
        for config in old.CONFIGS:
            if not config['mean_shrinkage']:continue
            spec={'kind':'filtered','family':config['family'],'innovations':config['innovations'],'mean_mode':'shrink','mean_uncertainty':True}
            index=b.BANK.index(spec);prepared=self.p['families'][config['family']]
            a=np.concatenate([v for _,v in b.model(index,self.p).terminal_chunks(self.x,self.w,21,100,88)],axis=1)
            z=np.concatenate([v for _,v in old.PurePool(config,prepared).terminal_chunks(self.x,self.w,21,100,88)],axis=1)
            np.testing.assert_array_equal(a,z)
    def test_mean_overrides_are_request_local(self):
        source=self.p['families']['GARCH'];before=source['mean']['center'].copy()
        for mode in('zero','sample','shrink'):
            spec={'kind':'filtered','family':'GARCH','innovations':'Gaussian','mean_mode':mode,'mean_uncertainty':False};m=b.FilteredOption(spec,source)
            expected=np.zeros(2)if mode=='zero'else source['mean']['sample_mean']if mode=='sample'else before
            np.testing.assert_array_equal(m.bundle['mean']['center'],expected)
        np.testing.assert_array_equal(source['mean']['center'],before)
    def test_chunk_invariance_in_all_volatility_families(self):
        for family in b.fits.FAMILIES:
            spec={'kind':'filtered','family':family,'innovations':'Student','mean_mode':'zero','mean_uncertainty':True};index=b.BANK.index(spec)
            a=np.concatenate([v for _,v in b.model(index,self.p).terminal_chunks(self.x,self.w,21,100,89)],axis=1)
            z=np.concatenate([v for _,v in b.model(index,self.p).terminal_chunks(self.x,self.w,21,100,89,width=1)],axis=1)
            np.testing.assert_array_equal(a,z)
    def test_request_level_learning_and_short_history_no_parent_fallback(self):
        learned=c.prepare(self.x[:32],self.w,20)
        self.assertEqual(len(learned['valid_bank_indices']),166);self.assertEqual(learned['diagnostics']['inner_paths'],1000)
        for config in c.CONFIGS:
            m=c.HorizonPool(config,learned);a=np.concatenate([v for _,v in m.terminal_chunks(self.x[:32],self.w,20,100,89)],axis=1)
            self.assertEqual(a.shape,(100,20));self.assertTrue(np.isfinite(a).all());self.assertEqual(sum(m.last_selection['path_counts']),100)
        short=c.prepare(self.x[:8],self.w,20);self.assertEqual(short['valid_bank_indices'],[1,2,3]);self.assertFalse(short['diagnostics']['automatic_model_fallback'])

if __name__=='__main__':unittest.main()
