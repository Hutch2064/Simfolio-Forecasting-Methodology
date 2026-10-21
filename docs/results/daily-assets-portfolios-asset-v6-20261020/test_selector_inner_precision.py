from pathlib import Path
import sys,unittest
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';sys.path.insert(0,str(V))
from candidates.asset_selector_inner_precision.candidate import prepare,PathCriterionPool,identity
from candidates.asset_selector_coefficient_sweep.candidate import prepare as control,PathCriterionPool as Control

class PrecisionTests(unittest.TestCase):
 def fixture(self):
  rng=np.random.default_rng(921);x=rng.normal(0,.01,size=(40,3));w=np.array([.2,.3,.5]);dates=np.arange(np.datetime64('2000-01-01'),np.datetime64('2000-01-01')+np.timedelta64(40,'D'));return x,w,dates
 def test_zero_replication_reproduces_all_coefficients_weights_and_final_paths(self):
  x,w,d=self.fixture();old=control(x,w,4,d,inner_paths=10);new=prepare(x,w,4,d,inner_paths=10,seed_replication=0);self.assertEqual(old['trace'],new['trace'])
  for mode in ('terminal','path'):
   for key in ('A','B'):np.testing.assert_array_equal(old['coefficients'][mode][key],new['coefficients'][mode][key])
   np.testing.assert_array_equal(old['weights_by_mode'][mode],new['weights_by_mode'][mode]);config={'criterion':mode,'inner_paths':10};a=Control(config,d,old);b=PathCriterionPool(config,d,new)
   for(i,p),(j,q)in zip(a.terminal_chunks(x,w,4,1000,885),b.terminal_chunks(x,w,4,1000,885)):self.assertEqual(i,j);np.testing.assert_array_equal(p,q)
 def test_replication_changes_only_internal_seed_context(self):
  x,w,d=self.fixture();a=prepare(x,w,4,d,inner_paths=10,seed_replication=1);b=prepare(x,w,4,d,inner_paths=10,seed_replication=1);zero=prepare(x,w,4,d,inner_paths=10,seed_replication=0);self.assertEqual(a['trace'],b['trace']);self.assertEqual(a['history_sha256'],zero['history_sha256']);self.assertNotEqual([r['seed']for r in a['trace']],[r['seed']for r in zero['trace']]);self.assertEqual(a['seed_replication'],1)
 def test_outer_path_count_is1000_at_each_internal_precision(self):
  x,w,d=self.fixture()
  for count in (10,30,100):
   b=prepare(x,w,4,d,inner_paths=count);m=PathCriterionPool({'criterion':'path','inner_paths':count},d,b);values=np.concatenate([p for _,p in m.terminal_chunks(x,w,4,1000,885)],axis=1);self.assertEqual(values.shape,(1000,4));self.assertTrue(np.isfinite(values).all());self.assertEqual(m.last_selection['config']['inner_paths'],count)
 def test_invalid_precision_and_seed_fail_closed(self):
  x,w,d=self.fixture()
  for count in (1,9):
   with self.assertRaises(ValueError):prepare(x,w,4,d,inner_paths=count)
  with self.assertRaises(ValueError):prepare(x,w,4,d,inner_paths=10,seed_replication=-1)
 def test_source_clone_changes_only_explicit_precision_context(self):
  source=(V/'candidates/asset_selector_coefficient_sweep/candidate.py').read_text();changed=(V/'candidates/asset_selector_inner_precision/candidate.py').read_text();changed=changed.replace('from candidates.asset_selector_coefficient_sweep.coefficients import accumulate','from .coefficients import accumulate',1).replace('width=64,seed_replication=0):','width=64):',1).replace("    if not isinstance(seed_replication,int) or seed_replication<0:raise ValueError('Nonnegative internal seed replication required')\n    seed_suffix='' if seed_replication==0 else f':replication={seed_replication}'\n",'',1).replace('{inner_paths}{seed_suffix}', '{inner_paths}',1).replace("'inner_paths':inner_paths,'seed_replication':seed_replication,'weights_by_mode'","'inner_paths':inner_paths,'weights_by_mode'",1);self.assertEqual(changed,source)
 def test_model_identity_ignores_execution_replication(self):
  x,w,d=self.fixture();config={'criterion':'path','inner_paths':10};a=prepare(x,w,4,d,inner_paths=10,seed_replication=0);b=prepare(x,w,4,d,inner_paths=10,seed_replication=1);self.assertEqual(PathCriterionPool(config,d,a).model_id,PathCriterionPool(config,d,b).model_id);self.assertNotEqual(identity(config),identity({**config,'inner_paths':30}))

if __name__=='__main__':unittest.main()
