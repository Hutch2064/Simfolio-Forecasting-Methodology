from pathlib import Path
import sys,unittest
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
from candidates.asset_selector_matrix_free_online.gradient import accumulate
from candidates.asset_selector_matrix_free_online.gradient_numpy import accumulate as vectorized
from candidates.asset_selector_matrix_free_online.state import State
from candidates.asset_selector_coefficient_sweep.coefficients import accumulate as dense
from candidates.asset_selector_online_pool.online import fit_sequence

class MatrixFreeTests(unittest.TestCase):
 def arrays(self,n=12,m=40,h=11):
  rng=np.random.default_rng(51482);return rng,rng.normal(size=(n,m,h)),rng.normal(size=h),np.arange(1,h+1,dtype=float)/(h*(h+1)/2)
 def check(self,x,y,m,w):
  a=np.zeros(len(w));b=np.zeros((len(w),len(w)));dense(x,y,m,a,b);A=np.zeros(len(w));C=np.zeros(len(w));accumulate(x,y,m,w,A,C);np.testing.assert_array_equal(A,a);np.testing.assert_allclose(C,b@w,rtol=5e-12,atol=5e-13*max(1,np.max(abs(b))));return A,C
 def test_uniform_dense_and_sparse_mixtures(self):
  rng,x,y,m=self.arrays();w=rng.dirichlet(np.ones(len(x)));self.check(x,y,m,w);self.check(x,y,m,np.ones(len(x))/len(x));w=np.zeros(len(x));w[[1,8]]=[.2,.8];self.check(x,y,m,w)
 def test_ties_and_constant_distributions(self):
  rng,x,y,m=self.arrays();x=np.round(x);x[0]=0;x[1]=2;self.check(x,y,m,np.ones(len(x))/len(x))
 def test_extreme_rescaling(self):
  rng,x,y,m=self.arrays();w=rng.dirichlet(np.ones(len(x)))
  for scale in [1e-50,1e50]:self.check(x*scale,y*scale,m,w)
 def test_chunk_continuation_is_exact(self):
  rng,x,y,m=self.arrays();w=np.ones(len(x))/len(x);a=np.zeros(len(x));c=np.zeros(len(x));accumulate(x,y,m,w,a,c);A=np.zeros(len(x));C=np.zeros(len(x))
  for start,stop in [(0,1),(1,4),(4,11)]:accumulate(x[:,:,start:stop],y[start:stop],m[start:stop],w,A,C)
  np.testing.assert_array_equal(a,A);np.testing.assert_array_equal(c,C)
 def test_full_sequential_weights_agree_with_dense_control(self):
  rng,x,y,m=self.arrays();state=State(len(x));terms=[]
  for fold in range(5):
   cube=x+rng.normal(size=x.shape)*.2;a=np.zeros(len(x));b=np.zeros((len(x),len(x)));dense(cube,y,m,a,b);terms.append((a,b));A=np.zeros(len(x));C=np.zeros(len(x));accumulate(cube,y,m,state.weights,A,C);state.observe(A,C);control=fit_sequence(terms)
   np.testing.assert_allclose(state.weights,control['weights'],rtol=0,atol=2e-11)
   np.testing.assert_allclose(state.trace[-1]['weights_before_outcome'],control['trace'][-1]['weights_before_outcome'],rtol=0,atol=2e-11)
 def test_invalid_inputs_fail_without_a_fallback(self):
  _,x,y,m=self.arrays();a=np.zeros(len(x));c=np.zeros(len(x))
  with self.assertRaises(ValueError):accumulate(x,y,m,np.zeros(len(x)),a,c)
  with self.assertRaises(ValueError):State(0)
 def test_vectorized_kernel_matches_dense_across_ties_and_scales(self):
  rng,x,y,m=self.arrays();w=rng.dirichlet(np.ones(len(x)))
  for scale in [1e-50,1.,1e50]:
   cube=np.round(x,1)*scale;A=np.zeros(len(x));C=np.zeros(len(x));vectorized(cube,y*scale,m,w,A,C);a,c=self.check(cube,y*scale,m,w);np.testing.assert_array_equal(A,a);np.testing.assert_allclose(C,c,rtol=5e-12,atol=5e-13*scale)
 def test_vectorized_chronological_state_matches_dense(self):
  rng,x,y,m=self.arrays();state=State(len(x));terms=[]
  for fold in range(5):
   cube=x+rng.normal(size=x.shape)*.2;a=np.zeros(len(x));b=np.zeros((len(x),len(x)));dense(cube,y,m,a,b);terms.append((a,b));A=np.zeros(len(x));C=np.zeros(len(x));vectorized(cube,y,m,state.weights,A,C);state.observe(A,C);np.testing.assert_allclose(state.weights,fit_sequence(terms)['weights'],rtol=0,atol=2e-11)
 def test_sparse_cdf_queries_every_zero_weight_option(self):
  rng,x,y,m=self.arrays(n=166,m=80,h=3);w=np.zeros(len(x));w[[1,64,121]]=[.2,.3,.5];a=np.zeros(len(x));b=np.zeros((len(x),len(x)));dense(x,y,m,a,b);A=np.zeros(len(x));C=np.zeros(len(x));vectorized(x,y,m,w,A,C);np.testing.assert_array_equal(A,a);np.testing.assert_allclose(C,b@w,rtol=5e-12,atol=5e-12);self.assertTrue(np.isfinite(C[w==0]).all());self.assertTrue(np.all(C[w==0]>0))
 def test_joint_candidate_matches_established_online_weights_and_paths(self):
  from candidates.asset_selector_online_pool.candidate import prepare as control,OnlinePool
  from candidates.asset_selector_matrix_free_online.candidate import prepare,MatrixFreeOnline
  rng=np.random.default_rng(921);x=rng.normal(0,.01,size=(40,3));w=np.array([.2,.3,.5]);dates=np.arange(np.datetime64('2000-01-01'),np.datetime64('2000-01-01')+np.timedelta64(40,'D'));old=control(x,w,4,dates,inner_paths=100);new=prepare(x,w,4,dates,inner_paths=100)
  np.testing.assert_allclose(old['weights_by_mode']['online'],new['learned_weights'],rtol=0,atol=2e-11)
  left=OnlinePool({'weight_rule':'SOLO_FTRL','inner_paths':100},dates,old);right=MatrixFreeOnline({'weight_rule':'SOLO_FTRL','inner_paths':100,'gradient_backend':'matrix_free'},dates,new)
  for(a,p),(b,q)in zip(left.terminal_chunks(x,w,4,1000,884),right.terminal_chunks(x,w,4,1000,884)):self.assertEqual(a,b);np.testing.assert_array_equal(p,q)
  self.assertFalse(new['diagnostics']['full_pairwise_matrix_constructed']);self.assertTrue(new['diagnostics']['every_valid_option_in_every_gradient']);self.assertEqual(old['invalid_internal_options'],new['invalid_internal_options'])

if __name__=='__main__':unittest.main()
