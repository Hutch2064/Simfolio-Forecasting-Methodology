from pathlib import Path
import os,sys,unittest
from unittest.mock import patch
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_selector_stacking import energy as e,candidate as c

class EmpiricalMixtureTests(unittest.TestCase):
    def test_cross_distances_and_weighted_empirical_CRPS(self):
        x=np.random.default_rng(712).normal(size=(4,37));y=.3
        A,B=e.coefficients(x,y)
        direct=np.mean(np.abs(x[:,None,:,None]-x[None,:,None,:]),axis=(2,3))
        np.testing.assert_allclose(B,direct,rtol=2e-15,atol=2e-15)
        w=np.array([.1,.2,.3,.4]);prob=np.repeat(w/37,37);flat=x.ravel()
        q=np.dot(prob,np.abs(flat-y))-.5*np.sum(prob[:,None]*prob[None,:]*np.abs(flat[:,None]-flat[None,:]))
        self.assertAlmostEqual(e.objective(w,A,B),q,places=14)
    def test_two_model_interior_optimum_and_grid(self):
        A,B=e.coefficients(np.array([[-1.,-1.],[1.,1.]]),0.)
        result=e.fit(A,B);np.testing.assert_allclose(result['weights'],[.5,.5],atol=1e-12)
        grid=[e.objective(np.array([t,1-t]),A,B)for t in np.linspace(0,1,1001)]
        self.assertLessEqual(result['objective'],min(grid)+1e-13);self.assertLess(result['KKT_gap'],1e-12)
    def test_vertex_and_duplicate_components(self):
        A,B=e.coefficients(np.array([[0.,0.],[1.,1.],[1.,1.]]),0.)
        result=e.fit(A,B);np.testing.assert_allclose(result['weights'],[1,0,0],atol=1e-12)
        x=np.tile(np.arange(10.),(3,1));A,B=e.coefficients(x,.4);result=e.fit(A,B)
        self.assertAlmostEqual(result['weights'].sum(),1.);self.assertAlmostEqual(result['objective'],A[0]-.5*B[0,0])
    def test_finite_ensemble_allocation(self):
        w=np.array([.0001,.2333,.7666]);counts=e.even_allocation(w,1000)
        self.assertEqual(counts.sum(),1000);self.assertTrue(np.all(counts%2==0));self.assertTrue(np.max(np.abs(counts/1000-w))<=.002)
        for paths in(1,999):
            with self.assertRaises(ValueError):e.even_allocation(w,paths)

class CausalPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x=np.random.default_rng(812).normal(.0002,.01,size=(32,2));cls.w=np.array([.6,.4]);cls.b=c.prepare(cls.x,cls.w,20)
    def test_target_horizon_and_all_nonoverlapping_blocks(self):
        eligible,folds=c.schedule(1000,21);self.assertEqual(len(eligible),19);self.assertEqual(len(folds),976//21)
        self.assertEqual(folds[-1],(979,21));self.assertTrue(all(folds[i][0]+21==folds[i+1][0]for i in range(len(folds)-1)))
        _,folds=c.schedule(1000,10000);self.assertEqual(folds,[(24,488),(512,488)])
        self.assertEqual(c.schedule(3,20)[1],[(2,1)])
        with self.assertRaises(ValueError):c.schedule(2,20)
    def test_causal_prefixes_and_observations(self):
        with patch.object(c.fits,'base_fit',wraps=c.fits.base_fit)as fit:c.prepare(self.x,self.w,20)
        self.assertEqual([len(call.args[0])for call in fit.call_args_list[:2]],[24,28])
        for call in fit.call_args_list:np.testing.assert_array_equal(call.args[0],self.x[:len(call.args[0])])
        for r in self.b['trace']:
            stop=r['training_rows'];H=r['validation_horizon'];self.assertLessEqual(stop+H,len(self.x))
            self.assertEqual(r['training_sha256'],c.signature(self.x[:stop]))
            self.assertEqual(r['terminal_observation'],float(c.portfolio_returns(self.x[stop:stop+H],self.w).sum()))
    def test_learned_simplex_and_selection_from_same_coefficients(self):
        b=self.b;A=b['coefficients']['A'];B=b['coefficients']['B'];ids=b['valid_bank_indices']
        hard=b['weights_by_mode']['select'];stack=b['weights_by_mode']['stack']
        self.assertEqual(np.argmax(hard),ids[int(np.argmin(A-.5*np.diag(B)))])
        self.assertAlmostEqual(stack.sum(),1.);self.assertTrue(np.all(stack>=0))
        self.assertLessEqual(e.objective(stack[ids],A,B),e.objective(hard[ids],A,B)+1e-12)
        self.assertEqual(b['diagnostics']['inner_origins'],2);self.assertEqual(b['diagnostics']['inner_horizon'],4)
        self.assertFalse(b['diagnostics']['automatic_model_fallback'])
    def test_joint_paths_exact_member_count_and_chunk_invariance(self):
        for config in c.CONFIGS:
            m=c.HorizonPool(config,self.b);a=np.concatenate([z for _,z in m.terminal_chunks(self.x,self.w,20,1000,8123)],axis=1)
            b=np.concatenate([z for _,z in c.HorizonPool(config,self.b).terminal_chunks(self.x,self.w,20,1000,8123,width=1)],axis=1)
            np.testing.assert_array_equal(a,b);self.assertEqual(a.shape,(1000,20));self.assertTrue(np.isfinite(a).all())
            self.assertEqual(sum(m.last_selection['path_counts']),1000)
        for x,w,H in[(self.x[:-1],self.w,20),(self.x,self.w[::-1],20),(self.x,self.w,21)]:
            with self.assertRaises(ValueError):list(c.HorizonPool(c.CONFIGS[1],self.b).terminal_chunks(x,w,H,1000,3))
    def test_short_or_flat_history_scores_raw_options(self):
        short=c.prepare(self.x[:8],self.w,20);self.assertEqual(short['valid_bank_indices'],[1,2])
        self.assertTrue(all(np.flatnonzero(w).tolist()==[i for i in np.flatnonzero(w)if i in(1,2)]for w in short['weights_by_mode'].values()))
        x=self.x.copy();x[:24]=0.;flat=c.prepare(x,self.w,20)
        self.assertTrue(flat['invalid_internal_options']);self.assertEqual(flat['valid_bank_indices'],[1,2])
        self.assertFalse(flat['diagnostics']['automatic_model_fallback'])

if __name__=='__main__':unittest.main()
