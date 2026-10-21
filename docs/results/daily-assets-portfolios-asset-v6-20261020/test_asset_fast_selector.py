from pathlib import Path
import os,sys,unittest,hashlib
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_fast_selector import candidate as c
from candidates.asset_trained_pool.candidate import TrainedPool
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon

class SelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(88101);z=rng.normal(size=(600,2))@np.array([[1.,0.],[.6,.8]]).T
        cls.x=z*np.linspace(.005,.025,600)[:,None]+.0003;cls.w=np.array([.7,.3]);cls.b=c.prepare(cls.x,cls.w,21)
    def test_nonoverlapping_target_complete_training_origins(self):
        self.assertEqual(c.rolling_origins(600,30,3),[570,540,510]);self.assertEqual(c.rolling_origins(600,100,3),[])
        for fold in self.b['folds']:
            stop=fold['training_stop_exclusive'];self.assertGreaterEqual(stop,504);self.assertLessEqual(fold['validation_stop_exclusive'],len(self.x));self.assertEqual(fold['training_hash'],c.signature(self.x[:stop]))
    def test_inner_parent_score_independent_reconstruction(self):
        fold=self.b['folds'][0];stop=fold['training_stop_exclusive'];x=self.x[:stop];prepared=c.shared_fits(x)
        actual=np.cumsum(c.portfolio_returns(self.x[stop:stop+21],self.w));loss=[]
        for start,paths in c.model(0,prepared).terminal_chunks(x,self.w,21,32,fold['seed']):loss.extend(empirical_crps_by_horizon(paths,actual[start:start+paths.shape[1]]))
        self.assertEqual(c.native_objective(loss,c.portfolio_returns(x,self.w)),self.b['inner_scores'][0][0])
    def test_exact_minimum_full_budget_and_deterministic_ties(self):
        for cfg in c.CONFIGS:
            n=cfg['maximum_inner_origins'];expected=int(np.argmin(np.mean(self.b['inner_scores'][:n],axis=0)));self.assertEqual(self.b['choices'][str(n)],expected)
        self.assertEqual(c.choose([[1.,1.,2.]],3),0);self.assertEqual(c.choose([],3),0)
    def test_joint_portfolio_accounting_and_all_paths(self):
        for cfg in c.CONFIGS:
            m=c.FastSelector(cfg,self.b);out=np.concatenate([z for _,z in m.terminal_chunks(self.x,self.w,21,100,88102)],axis=1)
            self.assertEqual(out.shape,(100,21));self.assertTrue(np.isfinite(out).all());self.assertEqual(m.last_selection['bank_size'],19)
        np.testing.assert_allclose(c.portfolio_returns(self.x,self.w),np.log(np.exp(self.x)@self.w),atol=3e-16)
    def test_selected_model_exact_equivalence_chunk_and_prefix_guard(self):
        cfg=c.CONFIGS[-1];model=c.FastSelector(cfg,self.b);a=np.concatenate([z for _,z in model.terminal_chunks(self.x,self.w,21,100,88103)],axis=1)
        index=self.b['choices']['3'];b=np.concatenate([z for _,z in c.model(index,self.b).terminal_chunks(self.x,self.w,21,100,88103,width=1)],axis=1);np.testing.assert_array_equal(a,b)
        for x,w,h in[(self.x[:-1],self.w,21),(self.x,self.w[::-1],21),(self.x,self.w,22)]:
            with self.assertRaises(ValueError):list(model.terminal_chunks(x,w,h,100,88103))
    def test_no_feasible_horizon_exact_parent_fallback(self):
        bundle=c.prepare(self.x,self.w,100);self.assertFalse(bundle['folds']);self.assertTrue(bundle['diagnostics']['fallback_no_full_target_validation'])
        for cfg in c.CONFIGS:
            a=np.concatenate([z for _,z in c.FastSelector(cfg,bundle).terminal_chunks(self.x,self.w,100,100,88104)],axis=1)
            b=np.concatenate([z for _,z in TrainedPool(c.BASE_CONFIG,bundle['parent']).terminal_chunks(self.x,self.w,100,100,88104)],axis=1);np.testing.assert_array_equal(a,b)
    def test_compute_budget_does_not_fit_extra_origins(self):
        b=c.prepare(self.x,self.w,21,maximum_origins=1);self.assertEqual(len(b['folds']),1);self.assertEqual(b['inner_scores'][0],self.b['inner_scores'][0]);self.assertEqual(b['choices']['1'],self.b['choices']['1'])
    def test_no_training_or_reference_mutation(self):
        x=self.x.copy();before=c.signature(x);c.prepare(x,self.w,21,maximum_origins=1);self.assertEqual(before,c.signature(x));self.assertEqual(len(c.BANK),19)
        self.assertEqual({r['family']for r in c.BANK if r['kind']=='filtered'},{'constant','EWMA_reverting','GARCH','SV_RMS'})
if __name__=='__main__':unittest.main()
