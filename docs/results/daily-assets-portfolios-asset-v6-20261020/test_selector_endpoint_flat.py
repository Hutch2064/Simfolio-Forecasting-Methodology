from pathlib import Path
from unittest.mock import patch
import sys,unittest
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
import common
from candidates.asset_selector_endpoint_flat import candidate as c
from candidates.asset_selector_endpoint_flat.cache import PreparedCache
from candidates.asset_selector_endpoint_components import candidate as control

class FlatTests(unittest.TestCase):
 def fixture(self):
  x=np.random.default_rng(836).normal(.0002,.01,(40,3));return x,np.array([.2,.3,.5]),np.datetime64('2003-01-01')+np.arange(len(x))
 def test_exact_coefficients_weights_identity_and1000paths_across_precisions_and_seeds(self):
  x,w,dates=self.fixture()
  for n in(2,8):
   for rep in(0,1,2):
    a=control.prepare(x,w,4,dates,inner_paths=n,schedule_rule='half_doubling',seed_replication=rep);b=c.prepare(x,w,4,dates,inner_paths=n,schedule_rule='half_doubling',seed_replication=rep)
    for key in('A','B'):np.testing.assert_array_equal(a['coefficients']['flat'][key],b['coefficients']['flat'][key])
    np.testing.assert_array_equal(a['weights_by_mode']['flat'][:170],b['weights_by_mode']['flat']);cfg={'menu':'flat','schedule':'half_doubling','inner_paths':n};ma,mb=control.MenuPool(cfg,dates,a),c.MenuPool(cfg,dates,b)
    np.testing.assert_array_equal(np.concatenate([v for _,v in ma.terminal_chunks(x,w,4,1000,51)],axis=1),np.concatenate([v for _,v in mb.terminal_chunks(x,w,4,1000,51)],axis=1));self.assertEqual(ma.model_id,mb.model_id)
 def test_request_cache_reproduces_each_precision_without_unused_fits(self):
  x,w,dates=self.fixture();ids,folds,_=c.schedule(len(x),4,'half_doubling');ids=[i for i in ids if i!=0]+([166,167]if 0 in ids else[]);cache=PreparedCache(ids,w)
  for n,rep in((2,0),(8,1)):
   direct=c.prepare(x,w,4,dates,inner_paths=n,schedule_rule='half_doubling',seed_replication=rep)
   with patch.object(c,'prefix_fits',cache.get):cached=c.prepare(x,w,4,dates,inner_paths=n,schedule_rule='half_doubling',seed_replication=rep)
   for key in('A','B'):np.testing.assert_array_equal(direct['coefficients']['flat'][key],cached['coefficients']['flat'][key])
   np.testing.assert_array_equal(direct['weights_by_mode']['flat'],cached['weights_by_mode']['flat'])
  self.assertEqual(cache.actual_fits,len(folds)+1)
  for fitted,errors in cache.entries.values():
   self.assertNotIn('PPW',fitted);self.assertTrue(all('SAS_fit'not in f for f in fitted['families'].values()));self.assertFalse(errors)
 def test_no_nested_or_PPWSAS_weights_and_request_binding(self):
  x,w,dates=self.fixture();b=c.prepare(x,w,4,dates,inner_paths=2,schedule_rule='half_doubling');weights=b['weights_by_mode']['flat'];self.assertEqual(weights[0],0);self.assertTrue(np.all(weights[168:]==0));self.assertEqual(set(b['weights_by_mode']),{'flat'})
  model=c.MenuPool({'menu':'flat','schedule':'half_doubling','inner_paths':2},dates,b);values=np.concatenate([v for _,v in model.terminal_chunks(x,w,4,1000,61)],axis=1);self.assertEqual(values.shape,(1000,4));self.assertEqual(sum(model.last_selection['path_counts']),1000)
  with self.assertRaises(ValueError):next(model.terminal_chunks(x,w,5,1000,61))

if __name__=='__main__':unittest.main()
