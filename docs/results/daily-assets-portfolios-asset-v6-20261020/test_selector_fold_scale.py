from pathlib import Path
import sys,unittest
import importlib.util
from unittest.mock import patch
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';sys.path.insert(0,str(V))
import score_main
from candidates.asset_selector_fold_scale import candidate as c
from candidates.asset_selector_coefficient_sweep.candidate import identity as prior_identity
spec=importlib.util.spec_from_file_location('preserved_fold_scale_scoring',V/'candidates/asset_selector_fold_scale/canonical_scoring.py')
canonical=importlib.util.module_from_spec(spec);spec.loader.exec_module(canonical)
# Recovery tests use the preserved canonical formula, not an external local copy.
score_main.legacy=canonical

class ToyOption:
 def __init__(self,i):self.i=i
 def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64):
  scale=float(np.std(c.portfolio_returns(history,weights),ddof=1));z=np.linspace(-1,1,paths);lead=np.arange(1,horizon+1)
  values=scale*((self.i-1)*lead[None,:]+z[:,None]*np.sqrt(lead)[None,:]*(self.i+1))
  for start in range(0,horizon,width):yield start,values[:,start:start+width]

class FoldScaleTests(unittest.TestCase):
 def fixture(self,n=16):
  x=np.random.default_rng(638).normal(0,.01,(n,3));x[n//2:]*=3;w=np.array([.2,.3,.5]);dates=np.datetime64('2000-01-01')+np.arange(n);return x,w,dates
 def toy(self):
  return patch.multiple(c,BANK=[{'kind':'blocks'}]*3,schedule=lambda rows,H:([0,1,2],[(8,4),(12,4)]),prefix_fits=lambda history,eligible:({},{}),model=lambda i,b:ToyOption(i))
 def test_coefficients_match_independent_canonical_fold_scores(self):
  x,w,dates=self.fixture();mix=np.ones(3)/3
  with self.toy():b=c.prepare(x,w,4,dates,inner_paths=10)
  fold_scores=[]
  for stop,H in [(8,4),(12,4)]:
   past=x[:stop];observed=np.cumsum(c.portfolio_returns(x[stop:stop+H],w));pred=np.stack([next(ToyOption(i).terminal_chunks(past,w,H,10,0))[1]for i in range(3)])
   losses=[]
   for t in range(H):
    values=pred[:,:,t].reshape(-1);losses.append(np.mean(abs(values-observed[t]))-.5*np.mean(abs(values[:,None]-values[None,:])))
   fold_scores.append(score_main.origin_score(np.array(losses),c.portfolio_returns(past,w),dates[stop-1],dates[stop:stop+H]))
  a=b['coefficients']['path']['A'];B=b['coefficients']['path']['B'];self.assertAlmostEqual(float(mix@a-.5*mix@B@mix),float(np.mean(fold_scores)),places=13)
 def test_scale_uses_each_prefix_not_its_validation_or_full_request(self):
  x,w,d=self.fixture()
  with self.toy():a=c.prepare(x,w,4,d,inner_paths=10);changed=x.copy();changed[10]+=.2;b=c.prepare(changed,w,4,d,inner_paths=10)
  self.assertEqual(a['trace'][0]['training_portfolio_sd'],b['trace'][0]['training_portfolio_sd']);self.assertNotEqual(a['trace'][1]['training_portfolio_sd'],b['trace'][1]['training_portfolio_sd'])
  for row in a['trace']:self.assertEqual(row['training_portfolio_sd'],float(np.std(c.portfolio_returns(x[:row['training_rows']],w),ddof=1)))
 def test_normalized_objective_is_invariant_to_uniform_log_return_units(self):
  x,w,d=self.fixture();w=np.array([1.,0.,0.])
  with self.toy():a=c.prepare(x,w,4,d,inner_paths=10);b=c.prepare(x*7,w,4,d,inner_paths=10)
  for mode in ('terminal','path'):
   for term in ('A','B'):np.testing.assert_allclose(a['coefficients'][mode][term],b['coefficients'][mode][term],rtol=1e-13,atol=1e-13)
 def test_zero_prefix_scale_has_no_epsilon_or_fallback(self):
  x,w,d=self.fixture();x[:8]=0
  with self.toy(),self.assertRaisesRegex(ValueError,'no epsilon or fallback'):c.prepare(x,w,4,d,inner_paths=10)
 def test_incompatible_unscaled_bundle_is_rejected(self):
  x,w,d=self.fixture()
  with self.toy():b=c.prepare(x,w,4,d,inner_paths=10)
  del b['fold_score_scale'];m=c.PathCriterionPool({'criterion':'path','inner_paths':10},d,b)
  with self.assertRaises(ValueError):next(m.terminal_chunks(x,w,4,1000,15))
 def test_financial_fixture_chunk_invariance_and_final1000_paths(self):
  x,w,d=self.fixture(40);a=c.prepare(x,w,4,d,inner_paths=10,width=1);b=c.prepare(x,w,4,d,inner_paths=10,width=4)
  for mode in ('terminal','path'):
   for term in ('A','B'):np.testing.assert_array_equal(a['coefficients'][mode][term],b['coefficients'][mode][term])
   np.testing.assert_array_equal(a['weights_by_mode'][mode],b['weights_by_mode'][mode])
  config={'criterion':'path','inner_paths':10};m=c.PathCriterionPool(config,d,a);values=np.concatenate([v for _,v in m.terminal_chunks(x,w,4,1000,15,width=3)],axis=1);self.assertEqual(values.shape,(1000,4));self.assertTrue(np.isfinite(values).all());self.assertNotEqual(m.model_id,prior_identity(config));self.assertEqual(m.last_selection['historical_fold_scale'],'training_portfolio_sd_ddof1')

if __name__=='__main__':unittest.main()
