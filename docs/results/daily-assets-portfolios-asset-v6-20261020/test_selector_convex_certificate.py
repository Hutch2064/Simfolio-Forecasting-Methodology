from pathlib import Path
import os,sys,unittest
from unittest.mock import patch
from types import SimpleNamespace
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_selector_stacking import energy as prior
from candidates.asset_selector_universe_certified import energy as revised

class ConvexCertificateTests(unittest.TestCase):
    def setUp(self):self.A,self.B=prior.coefficients(np.array([[-1.,-1.],[1.,1.]]),0.)
    def test_certified_optimum_accepted_even_if_solver_iteration_status(self):
        answer=SimpleNamespace(x=np.array([.5,.5]),success=False,status=9,nit=1000,message='Iteration limit reached')
        with patch.object(revised,'minimize',return_value=answer):fit=revised.fit(self.A,self.B)
        np.testing.assert_array_equal(fit['weights'],[.5,.5]);self.assertLess(fit['KKT_gap'],1e-14)
        self.assertFalse(fit['solver_attempts'][0]['solver_success'])
    def test_uncertified_iteration_state_continues_with_larger_budget(self):
        bad=SimpleNamespace(x=np.array([1.,0.]),success=False,status=9,nit=1000,message='Iteration limit reached')
        good=SimpleNamespace(x=np.array([.5,.5]),success=True,status=0,nit=1,message='Converged')
        with patch.object(revised,'minimize',side_effect=[bad,good])as solver:fit=revised.fit(self.A,self.B)
        self.assertEqual(solver.call_count,2);self.assertEqual(solver.call_args_list[1].kwargs['options']['maxiter'],2000)
        self.assertEqual(fit['iterations'],1001);self.assertEqual(len(fit['solver_attempts']),2)
    def test_uncertified_noniteration_failure_is_visible(self):
        bad=SimpleNamespace(x=np.array([1.,0.]),success=False,status=4,nit=1,message='Incompatible constraints')
        with patch.object(revised,'minimize',return_value=bad):
            with self.assertRaises(ArithmeticError):revised.fit(self.A,self.B)
    def test_existing_successful_solutions_remain_bitwise_identical(self):
        samples=np.random.default_rng(231).normal(size=(19,1000));A,B=prior.coefficients(samples,.4)
        a=prior.fit(A,B);b=revised.fit(A,B);np.testing.assert_array_equal(a['weights'],b['weights'])
        self.assertEqual(a['objective'],b['objective']);self.assertEqual(a['KKT_gap'],b['KKT_gap'])

if __name__=='__main__':unittest.main()
