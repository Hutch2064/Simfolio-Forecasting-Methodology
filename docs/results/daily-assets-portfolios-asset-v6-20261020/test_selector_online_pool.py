"""Sequential chronology, score gradient, scale invariance and joint-path control."""
from pathlib import Path
import sys,unittest
import numpy as np
from scipy.optimize import minimize
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
import common
from candidates.asset_selector_online_pool.online import project_simplex,fit_sequence
from candidates.asset_selector_coefficient_sweep.candidate import prepare as control,PathCriterionPool as Control
from candidates.asset_selector_online_pool.candidate import prepare,PathCriterionPool,OnlinePool
class OnlineTests(unittest.TestCase):
 def terms(self):
  rng=np.random.default_rng(9228);terms=[]
  from candidates.asset_selector_stacking.energy import coefficients
  for i in range(5):terms.append(coefficients(rng.normal(i*.1,.5,size=(8,100)),float(rng.normal())))
  return terms
 def test_simplex_projection_solves_regularized_problem(self):
  rng=np.random.default_rng(832)
  for k in(1,2,11):
   x=rng.normal(size=k);w=project_simplex(x);self.assertTrue(np.all(w>=0));self.assertAlmostEqual(w.sum(),1.)
   result=minimize(lambda v:.5*np.sum((v-x)**2),np.ones(k)/k,jac=lambda v:v-x,bounds=[(0,1)]*k,constraints=[{'type':'eq','fun':lambda v:v.sum()-1,'jac':lambda v:np.ones(k)}],method='SLSQP',options={'ftol':1e-13})
   np.testing.assert_allclose(w,result.x,rtol=0,atol=1e-9)
 def test_gradient_matches_feasible_score_derivative(self):
  A,B=self.terms()[0];w=np.ones(8)/8;direction=np.arange(8,dtype=float);direction-=direction.mean();epsilon=1e-7;objective=lambda v:A@v-.5*v@B@v
  observed=(objective(w+epsilon*direction)-objective(w-epsilon*direction))/(2*epsilon);self.assertAlmostEqual(observed,float((A-B@w)@direction),places=7)
 def test_history_prefix_weights_do_not_use_later_outcomes(self):
  terms=self.terms();prefix=fit_sequence(terms[:3]);full=fit_sequence(terms);self.assertEqual(prefix['trace'],full['trace'][:3]);changed=terms[:3]+[(terms[3][0]+10,terms[3][1]),terms[4]];self.assertEqual(prefix['trace'],fit_sequence(changed)['trace'][:3])
 def test_scale_and_common_loss_shift_invariance(self):
  terms=self.terms();base=fit_sequence(terms)
  for scale in(1e-8,1e8):np.testing.assert_allclose(base['weights'],fit_sequence([(a*scale,b*scale)for a,b in terms])['weights'],rtol=1e-10,atol=1e-12)
  np.testing.assert_allclose(base['weights'],fit_sequence([(a+3,b)for a,b in terms])['weights'],rtol=1e-10,atol=1e-12)
 def test_uninformative_and_single_component_have_no_previous_winner_fallback(self):
  r=fit_sequence([(np.ones(5),np.zeros((5,5)))for _ in range(3)]);np.testing.assert_array_equal(r['weights'],np.ones(5)/5);self.assertFalse(r['automatic_previous_winner_fallback']);np.testing.assert_array_equal(fit_sequence([(np.array([1.]),np.zeros((1,1)))])['weights'],[1.])
 def test_joint_forecast_preserves_both_batch_controls(self):
  rng=np.random.default_rng(921);x=rng.normal(0,.01,size=(40,3));w=np.array([.2,.3,.5]);dates=np.arange(np.datetime64('2000-01-01'),np.datetime64('2000-01-01')+np.timedelta64(40,'D'));old=control(x,w,4,dates,inner_paths=100);new=prepare(x,w,4,dates,inner_paths=100)
  self.assertEqual(old['trace'],new['trace'])
  for mode in('terminal','path'):
   np.testing.assert_array_equal(old['weights_by_mode'][mode],new['weights_by_mode'][mode]);config={'criterion':mode,'inner_paths':100};left=Control(config,dates,old);right=PathCriterionPool(config,dates,new)
   for(a,p),(b,q)in zip(left.terminal_chunks(x,w,4,1000,884),right.terminal_chunks(x,w,4,1000,884)):self.assertEqual(a,b);np.testing.assert_array_equal(p,q)
  online=OnlinePool({'weight_rule':'SOLO_FTRL','inner_paths':100},dates,new);parts=list(online.terminal_chunks(x,w,4,1000,884));self.assertEqual(np.concatenate([p for _,p in parts],axis=1).shape,(1000,4));self.assertFalse(online.last_selection['automatic_model_fallback']);np.testing.assert_array_equal(new['weights_by_mode']['path'],old['weights_by_mode']['path'])
if __name__=='__main__':unittest.main()
