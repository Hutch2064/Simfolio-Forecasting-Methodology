from pathlib import Path
import sys,unittest
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
from candidates.asset_selector_lazy_coefficients.columns import accumulate
from candidates.asset_selector_lazy_coefficients.solver import fit
from candidates.asset_selector_path_objective.coefficients import accumulate as dense_accumulate
from candidates.asset_selector_universe_certified.energy import fit as dense_fit

class LazyTests(unittest.TestCase):
 def fixture(self,n=12,m=30,h=9):
  rng=np.random.default_rng(9142);x=rng.normal(size=(n,m,h));y=rng.normal(size=h);mass=np.arange(1,h+1,dtype=float);mass/=mass.sum();a=np.zeros(n);b=np.zeros((n,n));dense_accumulate(x,y,mass,a,b);return x,y,mass,a,b
 def test_requested_columns_and_diagonal_are_byte_identical(self):
  x,y,m,a,b=self.fixture();selected=np.array([8,0,11]);A=np.zeros(len(a));diag=np.zeros(len(a));C=np.zeros((len(a),len(selected)));accumulate(x,y,m,selected,A,diag,C);np.testing.assert_array_equal(A,a);np.testing.assert_array_equal(diag,np.diag(b));np.testing.assert_array_equal(C,b[:,selected])
 def test_chunk_continuation_and_column_only_pass(self):
  x,y,m,a,b=self.fixture();s=np.array([3,7]);A=np.zeros(len(a));d=np.zeros(len(a));C=np.zeros((len(a),2))
  for start,stop in [(0,2),(2,5),(5,9)]:accumulate(x[:,:,start:stop],y[start:stop],m[start:stop],s,A,d,C)
  np.testing.assert_array_equal(A,a);np.testing.assert_array_equal(d,np.diag(b));np.testing.assert_array_equal(C,b[:,s]);before=A.copy();D=np.zeros_like(C);accumulate(x,y,m,s,A,d,D,include_terms=False);np.testing.assert_array_equal(A,before);np.testing.assert_array_equal(D,C)
 def test_full_menu_certificate_agrees_with_dense_optimum(self):
  _,_,_,a,b=self.fixture(n=25);seen=[]
  def oracle(j):seen.append(j);return b[:,j]
  q=fit(a,np.diag(b),oracle);reference=dense_fit(a,b);self.assertEqual(len(seen),len(set(seen)));self.assertTrue(q['all_options_checked_in_global_certificate']);self.assertIsNone(q['total_iteration_or_component_cap']);self.assertLessEqual(abs(q['objective']-reference['objective']),q['KKT_gap']+reference['KKT_gap']+1e-12);self.assertLessEqual(q['scaled_KKT_gap'],q['tolerance']);np.testing.assert_allclose(q['weights'].sum(),1,rtol=0,atol=1e-14)
 def test_dominated_point_masses_do_not_require_dense_pairs(self):
  points=np.arange(100,dtype=float);a=abs(points-17);b=abs(points[:,None]-points);q=fit(a,np.zeros(100),lambda j:b[:,j]);self.assertEqual(q['requested_columns'],[17]);self.assertEqual(q['objective'],0.);self.assertEqual(q['weights'][17],1.)
 def test_scale_equivariance(self):
  _,_,_,a,b=self.fixture();q=fit(a,np.diag(b),lambda j:b[:,j]);large=fit(a*1e6,np.diag(b)*1e6,lambda j:b[:,j]*1e6);np.testing.assert_allclose(q['weights'],large['weights'],atol=1e-11,rtol=0)
 def test_invalid_oracle_is_rejected_without_a_fallback(self):
  _,_,_,a,b=self.fixture()
  with self.assertRaises(ArithmeticError):fit(a,np.diag(b),lambda j:np.zeros(len(a)))
  with self.assertRaises(ValueError):fit(np.array([-1.]),np.zeros(1),lambda j:np.zeros(1))

if __name__=='__main__':unittest.main()
