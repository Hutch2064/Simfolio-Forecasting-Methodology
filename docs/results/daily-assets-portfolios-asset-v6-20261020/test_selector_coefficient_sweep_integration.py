"""Synthetic joint histories: all folds, coefficients, weights and output bytes."""
from pathlib import Path
import sys,unittest
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
import common
from candidates.asset_selector_path_objective.candidate import prepare as reference,PathCriterionPool as Reference
from candidates.asset_selector_coefficient_sweep.candidate import prepare as optimized,PathCriterionPool as Optimized
class IntegrationTests(unittest.TestCase):
 def test_all_work_on_joint_dated_histories(self):
  rng=np.random.default_rng(99381)
  for rows,assets,horizon,paths in((40,1,4,100),(48,3,7,1000)):
   x=rng.normal(0,.008,size=(rows,assets));w=np.arange(1,assets+1,dtype=float);w/=w.sum();dates=np.arange(np.datetime64('2001-01-01'),np.datetime64('2001-01-01')+np.timedelta64(rows,'D'))
   first=reference(x,w,horizon,dates,inner_paths=paths,width=3);second=optimized(x,w,horizon,dates,inner_paths=paths,width=3)
   self.assertEqual(first['trace'],second['trace']);self.assertEqual(first['invalid_internal_options'],second['invalid_internal_options'])
   for mode in('terminal','path'):
    for key in('A','B'):np.testing.assert_array_equal(first['coefficients'][mode][key],second['coefficients'][mode][key])
    np.testing.assert_array_equal(first['weights_by_mode'][mode],second['weights_by_mode'][mode]);self.assertEqual(first['fit'][mode]['objective'],second['fit'][mode]['objective'])
    config={'criterion':mode,'inner_paths':paths};left=Reference(config,dates,first);right=Optimized(config,dates,second);a=list(left.terminal_chunks(x,w,horizon,1000,997,width=3));b=list(right.terminal_chunks(x,w,horizon,1000,997,width=3));self.assertEqual(len(a),len(b))
    for (i,values),(j,other)in zip(a,b):self.assertEqual(i,j);np.testing.assert_array_equal(values,other)
 def test_request_binding_survives_acceleration(self):
  rng=np.random.default_rng(553);x=rng.normal(0,.01,size=(32,2));w=np.array([.4,.6]);dates=np.arange(np.datetime64('2001-01-01'),np.datetime64('2001-01-01')+np.timedelta64(len(x),'D'));bundle=optimized(x,w,4,dates,inner_paths=100);model=Optimized({'criterion':'path','inner_paths':100},dates,bundle)
  with self.assertRaises(ValueError):list(model.terminal_chunks(x+.0001,w,4,100,5))
  with self.assertRaises(ValueError):list(model.terminal_chunks(x,w[::-1],4,100,5))
if __name__=='__main__':unittest.main()
