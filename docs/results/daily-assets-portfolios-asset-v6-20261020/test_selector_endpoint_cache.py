from pathlib import Path
import hashlib,sys,unittest
from unittest.mock import patch
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
import common
from candidates.asset_selector_endpoint_precision.cache import PreparedCache
from candidates.asset_selector_information_endpoint import candidate as c
from candidates.asset_selector_universe.bank import BANK,model,prefix_fits

def fingerprint(value):
    if isinstance(value,np.ndarray):return (value.shape,value.dtype.str,hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value,dict):return {k:fingerprint(v)for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [fingerprint(v)for v in value]
    return value

class CacheTests(unittest.TestCase):
 def fixture(self):
  x=np.random.default_rng(518).normal(.0002,.01,(40,3));return x,np.array([.2,.3,.5]),np.datetime64('2001-01-01')+np.arange(len(x))
 def test_every_recipe_is_exact_under_cached_superset_fits(self):
  x,w,_=self.fixture();cache=PreparedCache(range(len(BANK)));cached,invalid=cache.get(x,range(len(BANK)));self.assertFalse(invalid)
  before=fingerprint(cache.entries)
  for i in range(len(BANK)):
   direct,errors=prefix_fits(x,[i]);a,failures=cache.get(x,[i]);self.assertEqual(errors,failures)
   ma,mb=model(i,direct),model(i,a)
   np.testing.assert_array_equal(np.concatenate([v for _,v in ma.terminal_chunks(x,w,5,4,39)],axis=1),np.concatenate([v for _,v in mb.terminal_chunks(x,w,5,4,39)],axis=1))
  self.assertEqual(cache.actual_fits,1);self.assertEqual(before,fingerprint(cache.entries));self.assertGreater(cache.numpy_bytes(),0)
 def test_precision_and_seed_variants_reuse_fits_without_changing_forecasts(self):
  x,w,dates=self.fixture();eligible,folds,_=c.schedule(len(x),4,'half_doubling');cache=PreparedCache(eligible)
  for n,rep in((2,0),(4,0),(8,1),(30,2)):
   direct=c.prepare(x,w,4,dates,inner_paths=n,schedule_rule='half_doubling',seed_replication=rep)
   with patch.object(c,'prefix_fits',cache.get):cached=c.prepare(x,w,4,dates,inner_paths=n,schedule_rule='half_doubling',seed_replication=rep)
   for key in('A','B'):np.testing.assert_array_equal(direct['coefficients'][key],cached['coefficients'][key])
   np.testing.assert_array_equal(direct['weights_by_mode']['stack'],cached['weights_by_mode']['stack'])
   cfg={'schedule':'half_doubling','inner_paths':n,'pool':'stack'};a,b=c.EndpointPool(cfg,dates,direct),c.EndpointPool(cfg,dates,cached)
   np.testing.assert_array_equal(np.concatenate([v for _,v in a.terminal_chunks(x,w,4,1000,51)],axis=1),np.concatenate([v for _,v in b.terminal_chunks(x,w,4,1000,51)],axis=1))
  self.assertEqual(cache.actual_fits,len(folds)+1)
 def test_history_signatures_and_eligible_universe_are_bound(self):
  x,w,_=self.fixture();cache=PreparedCache([1,2,3]);cache.get(x,[1]);cache.get(x.copy(),[2]);self.assertEqual(cache.actual_fits,1)
  changed=x.copy();changed[0,0]+=.01;cache.get(changed,[1]);self.assertEqual(cache.actual_fits,2)
  with self.assertRaises(ValueError):cache.get(x,[0])

if __name__=='__main__':unittest.main()
