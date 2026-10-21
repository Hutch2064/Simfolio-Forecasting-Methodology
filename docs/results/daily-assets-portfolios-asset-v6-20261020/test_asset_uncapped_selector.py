from pathlib import Path
import os,sys,unittest
from unittest.mock import patch
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_uncapped_selector import candidate as c

class CompleteSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x=np.random.default_rng(88201).normal(.0002,.01,size=(32,2));cls.w=np.array([.6,.4]);cls.b=c.prepare(cls.x,cls.w,20)
    def test_all_eligible_origins_without_caps(self):
        eligible,plan=c.validation_plan(1000,21);self.assertEqual(len(eligible),19);self.assertEqual(len(plan),976);self.assertEqual(plan[0],(24,21));self.assertEqual(plan[-1],(999,1))
        self.assertEqual([r['training_rows']for r in self.b['trace']],list(range(24,32)))
    def test_long_requested_horizon_is_scored_not_fallback(self):
        d=self.b['diagnostics'];self.assertEqual(d['longest_directly_validated_lead'],8);self.assertFalse(d['entire_requested_horizon_directly_validated']);self.assertFalse(d['automatic_parent_fallback']);self.assertIsNone(d['origin_count_cap']);self.assertEqual(d['inner_paths'],1000)
    def test_direct_internal_score_reduction(self):
        scores={i:np.mean([r['scores'][i]for r in self.b['trace']])for i in self.b['eligible_bank_indices']}
        for i,q in scores.items():self.assertAlmostEqual(q,self.b['inner_scores'][i],places=14)
        self.assertEqual(self.b['choice'],min(scores,key=lambda i:(scores[i],i)))
    def test_short_history_eligible_options_not_prior_winner(self):
        x=self.x[:8];bundle=c.prepare(x,self.w,20);self.assertEqual(bundle['eligible_bank_indices'],[1,2]);self.assertIn(bundle['choice'],[1,2]);self.assertEqual(bundle['diagnostics']['inner_origins'],6)
        with self.assertRaises(ValueError):c.validation_plan(2,20)
    def test_inner_fits_only_receive_their_prefix(self):
        with patch.object(c.fits,'base_fit',wraps=c.fits.base_fit)as fit:c.prepare(self.x,self.w,3)
        self.assertEqual([len(call.args[0])for call in fit.call_args_list[:8]],list(range(24,32)))
        self.assertTrue(all(len(call.args[0])==32 for call in fit.call_args_list[8:]))
        for call in fit.call_args_list:np.testing.assert_array_equal(call.args[0],self.x[:len(call.args[0])])
    def test_selected_joint_paths_and_request_identity(self):
        m=c.CompleteHistorySelector(self.b);a=np.concatenate([z for _,z in m.terminal_chunks(self.x,self.w,20,1000,88202)],axis=1)
        b=np.concatenate([z for _,z in c.model(self.b['choice'],self.b).terminal_chunks(self.x,self.w,20,1000,88202,width=1)],axis=1);np.testing.assert_array_equal(a,b);self.assertEqual(a.shape,(1000,20));self.assertTrue(np.isfinite(a).all())
        for x,w,H in[(self.x[:-1],self.w,20),(self.x,self.w[::-1],20),(self.x,self.w,21)]:
            with self.assertRaises(ValueError):list(m.terminal_chunks(x,w,H,1000,88202))
    def test_flat_prefix_disqualifies_invalid_fits_not_a_model_fallback(self):
        x=self.x.copy();x[:24]=0.;b=c.prepare(x,self.w,20)
        self.assertTrue(b['invalid_internal_options']);self.assertIn(b['choice'],[1,2]);self.assertFalse(b['diagnostics']['automatic_parent_fallback']);self.assertEqual(len(b['trace']),8)
if __name__=='__main__':unittest.main()
