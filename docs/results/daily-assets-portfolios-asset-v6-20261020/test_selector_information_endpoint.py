from pathlib import Path
from unittest.mock import patch
import hashlib,sys,unittest
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
import common
from candidates.asset_selector_information_endpoint import candidate as c
from candidates.asset_selector_information_origins import candidate as path
from candidates.asset_selector_inner_precision_batched import candidate as old

class EndpointTests(unittest.TestCase):
 def fixture(self,rows=40):
  x=np.random.default_rng(812).normal(.0002,.01,(rows,3));return x,np.array([.2,.3,.5]),np.datetime64('2000-01-01')+np.arange(rows)
 def test_same_components_dates_and_random_paths_as_path_control(self):
  x,w,dates=self.fixture();native=c.all_chunks;captured=[[],[]]
  def recording(which):
   def wrap(*args,**kwargs):
    for start,cube in native(*args,**kwargs):captured[which].append((start,hashlib.sha256(cube.tobytes()).hexdigest()));yield start,cube
   return wrap
  with patch.object(c,'all_chunks',recording(0)):a=c.prepare(x,w,4,dates,inner_paths=4,schedule_rule='half_doubling')
  with patch.object(path,'all_chunks',recording(1)):b=path.prepare(x,w,4,dates,inner_paths=4,schedule_rule='half_doubling')
  self.assertEqual(captured[0],captured[1]);self.assertEqual(a['trace'],b['trace'])
 def test_all_origin_endpoint_recovers_previous_terminal_control(self):
  x,w,dates=self.fixture();a=c.prepare(x,w,4,dates,inner_paths=4,schedule_rule='all');b=old.prepare(x,w,4,dates,inner_paths=4)
  for key in('A','B'):np.testing.assert_array_equal(a['coefficients'][key],b['coefficients']['terminal'][key])
  np.testing.assert_array_equal(a['weights_by_mode']['stack'],b['weights_by_mode']['terminal'])
  ma=c.EndpointPool({'schedule':'all','inner_paths':4,'pool':'stack'},dates,a);mb=old.PathCriterionPool({'criterion':'terminal','inner_paths':4},dates,b)
  np.testing.assert_array_equal(np.concatenate([v for _,v in ma.terminal_chunks(x,w,4,1000,51)],axis=1),np.concatenate([v for _,v in mb.terminal_chunks(x,w,4,1000,51)],axis=1));self.assertEqual(ma.model_id,mb.model_id)
 def test_endpoint_coefficients_independently_reduced_with_coverage_counts(self):
  x,w,dates=self.fixture();native=c.all_chunks;saved=[]
  def capture(*args,**kwargs):
   last=None
   for start,cube in native(*args,**kwargs):last=cube[:,:,-1].copy();yield start,cube
   saved.append(last)
  with patch.object(c,'all_chunks',capture):b=c.prepare(x,w,4,dates,inner_paths=4,schedule_rule='doubling')
  ids=b['trace'][-1]['eligible_this_fold'];A=np.zeros(len(ids));B=np.zeros((len(ids),len(ids)));total=0
  for trace,terminal in zip(b['trace'],saved):
   stop,h,m=trace['training_rows'],trace['validation_horizon'],trace['represented_origins'];actual=float(old.portfolio_returns(x[stop:stop+h],w).sum())
   A+=m*np.mean(np.abs(terminal-actual),axis=1)/np.sqrt(h)
   for i in range(len(ids)):
    for j in range(len(ids)):B[i,j]+=m*np.mean(np.abs(terminal[i,:,None]-terminal[j,None,:]))/np.sqrt(h)
   total+=m
  np.testing.assert_allclose(A/total,b['coefficients']['A'],rtol=2e-14,atol=2e-17);np.testing.assert_allclose(B/total,b['coefficients']['B'],rtol=2e-14,atol=2e-17)
 def test_both_spacings_have_exact1000paths_and_causal_bindings(self):
  x,w,dates=self.fixture(80)
  for rule in('doubling','half_doubling'):
   b=c.prepare(x,w,4,dates,inner_paths=4,schedule_rule=rule);model=c.EndpointPool({'schedule':rule,'inner_paths':4,'pool':'stack'},dates,b)
   values=np.concatenate([v for _,v in model.terminal_chunks(x,w,4,1000,63)],axis=1)
   self.assertEqual(values.shape,(1000,4));self.assertTrue(np.isfinite(values).all());self.assertEqual(sum(model.last_selection['path_counts']),1000)
   self.assertFalse(b['diagnostics']['intermediate_lead_pair_matrices_computed'])
   self.assertTrue(all(f['validation_stop']<=len(x)for f in b['trace']))
   with self.assertRaises(ValueError):next(model.terminal_chunks(x,w,5,1000,63))
 def test_seed_repetitions_change_internal_draws_not_available_history(self):
  x,w,dates=self.fixture();a=c.prepare(x,w,4,dates,inner_paths=4,schedule_rule='half_doubling');b=c.prepare(x,w,4,dates,inner_paths=4,schedule_rule='half_doubling',seed_replication=1)
  self.assertNotEqual([t['seed']for t in a['trace']],[t['seed']for t in b['trace']]);self.assertEqual([t['training_sha256']for t in a['trace']],[t['training_sha256']for t in b['trace']])

if __name__=='__main__':unittest.main()
