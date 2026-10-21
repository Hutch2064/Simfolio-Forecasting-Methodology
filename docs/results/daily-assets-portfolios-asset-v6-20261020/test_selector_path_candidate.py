from pathlib import Path
import os,sys,unittest
from unittest.mock import patch
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6' if(O/'asset_level_v6').exists() else O;sys.path.insert(0,str(V))
from candidates.asset_selector_path_objective import candidate as c
from candidates.asset_selector_universe import candidate as control

class DatedPathCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x=np.random.default_rng(871).normal(.0002,.01,size=(28,2));cls.w=np.array([.6,.4]);cls.dates=np.datetime64('2000-01-01')+np.arange(28).astype('timedelta64[D]');cls.b=c.prepare(cls.x,cls.w,8,cls.dates,inner_paths=4,width=2)
    def test_fits_only_receive_inner_prefixes_and_past_dates(self):
        with patch.object(c,'prefix_fits',wraps=c.prefix_fits)as fit:c.prepare(self.x,self.w,8,self.dates,inner_paths=4,width=2)
        self.assertEqual([len(call.args[0])for call in fit.call_args_list[:2]],[24,26])
        for call in fit.call_args_list:np.testing.assert_array_equal(call.args[0],self.x[:len(call.args[0])])
        for row in self.b['trace']:self.assertLess(row['training_last_date'],row['validation_last_known_date']);self.assertLessEqual(row['validation_stop'],len(self.x))
    def test_two_criteria_and_joint_output_request_binding(self):
        self.assertEqual(set(self.b['weights_by_mode']),{'terminal','path'})
        for mode in('terminal','path'):
            values=self.b['weights_by_mode'][mode];self.assertAlmostEqual(values.sum(),1.);self.assertTrue(np.all(values>=0))
            model=c.PathCriterionPool({'criterion':mode,'inner_paths':4},self.dates,self.b);paths=np.concatenate([v for _,v in model.terminal_chunks(self.x,self.w,8,1000,872)],axis=1)
            self.assertEqual(paths.shape,(1000,8));self.assertTrue(np.isfinite(paths).all());self.assertEqual(sum(model.last_selection['path_counts']),1000)
        with self.assertRaises(ValueError):list(c.PathCriterionPool({'criterion':'path','inner_paths':4},self.dates+np.timedelta64(1,'D'),self.b).terminal_chunks(self.x,self.w,8,1000,872))
    def test_chunk_invariance_and_short_history_eligibility(self):
        other=c.prepare(self.x,self.w,8,self.dates,inner_paths=4,width=1)
        for mode in('terminal','path'):
            self.assertEqual(self.b['coefficients'][mode]['A'].tobytes(),other['coefficients'][mode]['A'].tobytes());self.assertEqual(self.b['coefficients'][mode]['B'].tobytes(),other['coefficients'][mode]['B'].tobytes())
        short=c.prepare(self.x[:8],self.w,20,self.dates[:8],inner_paths=4)
        self.assertFalse(short['diagnostics']['automatic_model_fallback'])
        for w in short['weights_by_mode'].values():self.assertTrue(all(i in(1,2,3)for i in np.flatnonzero(w)))
    def test_endpoint_control_reproduces_existing100path_algorithm(self):
        old=control.prepare(self.x,self.w,8);new=c.prepare(self.x,self.w,8,self.dates,inner_paths=100)
        self.assertEqual(old['weights_by_mode']['stack'].tobytes(),new['weights_by_mode']['terminal'].tobytes())
        a=np.concatenate([v for _,v in control.HorizonPool({'pool':'stack'},old).terminal_chunks(self.x,self.w,8,1000,873)],axis=1)
        b=np.concatenate([v for _,v in c.PathCriterionPool({'criterion':'terminal','inner_paths':100},self.dates,new).terminal_chunks(self.x,self.w,8,1000,873)],axis=1)
        self.assertEqual(a.tobytes(),b.tobytes())

if __name__=='__main__':unittest.main()
