from pathlib import Path
import sys,unittest
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
import common
from candidates.asset_selector_information_origins import candidate as c
from candidates.asset_selector_information_origins.validation import schedule,RULES
from candidates.asset_selector_design import candidate as reference

class InformationOriginsTests(unittest.TestCase):
 def fixture(self,rows=40):
  x=np.random.default_rng(729).normal(.0001,.01,(rows,3));return x,np.array([.2,.3,.5]),np.datetime64('2001-01-01')+np.arange(rows)
 def test_all_represented_origins_are_counted_and_fits_are_causal(self):
  for rows in(3,25,40,1000,30000):
   for horizon in(1,43,100000):
    _,full,_=schedule(rows,horizon,'all')
    for rule in RULES:
     _,folds,total=schedule(rows,horizon,rule)
     self.assertEqual(sum(count for _,_,count in folds),total);self.assertEqual(total,len(full));self.assertEqual(folds[0][:2],full[0][:2]);self.assertEqual(folds[-1][:2],full[-1][:2])
     self.assertEqual(folds,sorted(set(folds)))
     for stop,h,count in folds:self.assertGreaterEqual(count,1);self.assertLessEqual(stop+h,rows);self.assertIn((stop,h,1),full)
 def test_origin_count_has_no_fixed_maximum_and_long_history_is_reduced(self):
  for rule in('doubling','half_doubling'):
   small=schedule(1000,1,rule)[1];large=schedule(1000000,1,rule)[1]
   self.assertGreater(len(large),len(small))
   self.assertLess(len(schedule(26403,43,rule)[1]),30)
  self.assertGreater(len(schedule(26403,43,'all')[1]),600)
 def test_counts_are_nearest_origin_voronoi_counts(self):
  for rule in('doubling','half_doubling'):
   _,full,_=schedule(500,7,'all');_,folds,total=schedule(500,7,rule)
   selected=[stop for stop,_,_ in folds];counts=np.bincount([int(np.argmin([abs(stop-s)for s in selected]))for stop,_,_ in full],minlength=len(selected))
   np.testing.assert_array_equal(counts,[count for _,_,count in folds])
 def test_all_origin_control_is_exact_to_established_candidate(self):
  x,w,dates=self.fixture();a=reference.prepare(x,w,4,dates,inner_paths=4,schedule_rule='established');b=c.prepare(x,w,4,dates,inner_paths=4,schedule_rule='all')
  for key in('A','B'):np.testing.assert_array_equal(a['coefficients'][key],b['coefficients'][key])
  np.testing.assert_array_equal(a['weights_by_mode']['stack'],b['weights_by_mode']['stack'])
  ma=reference.DesignPool({'schedule':'established','inner_paths':4,'pool':'stack'},dates,a);mb=c.DesignPool({'schedule':'all','inner_paths':4,'pool':'stack'},dates,b)
  np.testing.assert_array_equal(np.concatenate([v for _,v in ma.terminal_chunks(x,w,4,1000,71)],axis=1),np.concatenate([v for _,v in mb.terminal_chunks(x,w,4,1000,71)],axis=1));self.assertEqual(ma.model_id,mb.model_id)
 def test_no_coarsening_means_exact_existing_paths(self):
  x,w,dates=self.fixture();a=c.prepare(x,w,100,dates,inner_paths=4,schedule_rule='all')
  for rule in('doubling','half_doubling'):
   b=c.prepare(x,w,100,dates,inner_paths=4,schedule_rule=rule)
   self.assertEqual([(f['training_rows'],f['validation_horizon'])for f in a['trace']],[(f['training_rows'],f['validation_horizon'])for f in b['trace']])
   self.assertTrue(all(f['represented_origins']==1 for f in b['trace']))
   for key in('A','B'):np.testing.assert_array_equal(a['coefficients'][key],b['coefficients'][key])
 def test_final1000_paths_and_bindings_for_each_spaced_learner(self):
  x,w,dates=self.fixture(80)
  for rule in('doubling','half_doubling'):
   b=c.prepare(x,w,4,dates,inner_paths=4,schedule_rule=rule)
   model=c.DesignPool({'schedule':rule,'inner_paths':4,'pool':'stack'},dates,b);values=np.concatenate([v for _,v in model.terminal_chunks(x,w,4,1000,91)],axis=1)
   self.assertEqual(values.shape,(1000,4));self.assertTrue(np.isfinite(values).all());self.assertEqual(sum(model.last_selection['path_counts']),1000)
   self.assertEqual(sum(b['diagnostics']['origin_representation_counts']),b['diagnostics']['available_internal_origins'])
   with self.assertRaises(ValueError):next(model.terminal_chunks(x,w,5,1000,91))

if __name__=='__main__':unittest.main()
