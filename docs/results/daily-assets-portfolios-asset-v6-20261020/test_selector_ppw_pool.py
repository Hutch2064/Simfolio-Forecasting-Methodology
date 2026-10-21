from pathlib import Path
import os,sys,unittest
from unittest.mock import patch
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_selector_universe import bank as base
from candidates.asset_selector_ppw import bank,candidate as c

class PPWMenuIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x=np.random.default_rng(2097).normal(.0003,.01,(28,2));cls.w=np.array([.6,.4]);cls.dates=np.datetime64('2000-01-01')+np.arange(28).astype('timedelta64[D]')
    def test_original166component_forecasts_unchanged(self):
        self.assertEqual(len(base.BANK),166);self.assertEqual(len(bank.BANK),168);self.assertEqual(bank.BANK[:166],base.BANK)
        original,errors=base.prefix_fits(self.x,list(range(166)));expanded,other=bank.prefix_fits(self.x,list(range(168)),self.w);self.assertEqual(errors,{});self.assertEqual(other,{})
        for i in range(166):
            a=np.concatenate([v for _,v in base.model(i,original).terminal_chunks(self.x,self.w,4,4,2098,width=2)],axis=1)
            b=np.concatenate([v for _,v in bank.model(i,expanded).terminal_chunks(self.x,self.w,4,4,2098,width=2)],axis=1)
            self.assertEqual(a.tobytes(),b.tobytes(),i)
    def test_dated_prefix_isolation_and_coherent_pool(self):
        with patch.object(c,'prefix_fits',wraps=c.prefix_fits)as fit:b=c.prepare(self.x,self.w,8,self.dates,inner_paths=4,width=2)
        self.assertEqual([len(call.args[0])for call in fit.call_args_list[:2]],[24,26])
        for trace in b['trace']:
            self.assertEqual(trace['PPW_dependence_fit']['training_rows'],trace['training_rows']);self.assertLessEqual(trace['validation_stop'],len(self.x));self.assertLess(trace['training_last_date'],trace['validation_last_known_date'])
        for mode in ('terminal','path'):
            learned=b['weights_by_mode'][mode];self.assertEqual(len(learned),168);self.assertAlmostEqual(learned.sum(),1.);self.assertTrue(np.all(learned>=0))
            model=c.PathCriterionPool({'criterion':mode,'inner_paths':4},self.dates,b)
            paths=np.concatenate([v for _,v in model.terminal_chunks(self.x,self.w,8,1000,2099,width=2)],axis=1)
            self.assertEqual(paths.shape,(1000,8));self.assertTrue(np.isfinite(paths).all());self.assertEqual(sum(model.last_selection['path_counts']),1000)
    def test_unsupported_option_has_explicit_reason_without_model_fallback(self):
        b=c.prepare(self.x[:8],self.w,20,self.dates[:8],inner_paths=4,width=2)
        self.assertFalse(b['diagnostics']['automatic_model_fallback'])
        for i in (166,167):
            self.assertIn(i,b['invalid_internal_options']);self.assertIn('lag window',b['invalid_internal_options'][i]['reason'])
            for w in b['weights_by_mode'].values():self.assertEqual(w[i],0.)

if __name__=='__main__':unittest.main()
