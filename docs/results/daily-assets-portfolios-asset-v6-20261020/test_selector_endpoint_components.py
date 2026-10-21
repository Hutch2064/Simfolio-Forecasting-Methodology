from pathlib import Path
from unittest.mock import patch
import hashlib,sys,unittest
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
import common
from candidates.asset_selector_endpoint_components import candidate as c
from candidates.asset_selector_endpoint_components.cache import PreparedCache
from candidates.asset_selector_information_endpoint import candidate as original

class ComponentTests(unittest.TestCase):
 def fixture(self):
  x=np.random.default_rng(716).normal(.0001,.01,(40,3));return x,np.array([.2,.3,.5]),np.datetime64('2002-01-01')+np.arange(len(x))
 def expanded_eligible(self,x,h):
  ids,_,_=c.schedule(len(x),h,'half_doubling');return ids+([166,167,*range(170,len(c.BANK))]if 0 in ids else[])+[168,169]
 def test_original_coefficients_weights_and1000paths_replay_exactly(self):
  x,w,dates=self.fixture()
  for n,rep in((4,0),(30,1)):
   old=original.prepare(x,w,4,dates,inner_paths=n,schedule_rule='half_doubling',seed_replication=rep);new=c.prepare(x,w,4,dates,inner_paths=n,schedule_rule='half_doubling',seed_replication=rep)
   for k in('A','B'):np.testing.assert_array_equal(new['coefficients']['original'][k],old['coefficients'][k])
   np.testing.assert_array_equal(new['weights_by_mode']['original'][:166],old['weights_by_mode']['stack'])
   a=original.EndpointPool({'schedule':'half_doubling','inner_paths':n,'pool':'stack'},dates,old);b=c.MenuPool({'schedule':'half_doubling','inner_paths':n,'menu':'original'},dates,new)
   np.testing.assert_array_equal(np.concatenate([v for _,v in a.terminal_chunks(x,w,4,1000,51)],axis=1),np.concatenate([v for _,v in b.terminal_chunks(x,w,4,1000,51)],axis=1));self.assertEqual(a.model_id,b.model_id)
 def test_menu_definitions_expose_exact_legs_and_keep1000paths(self):
  x,w,dates=self.fixture();b=c.prepare(x,w,4,dates,inner_paths=4,schedule_rule='half_doubling')
  self.assertEqual(len(c.BANK),224)
  for name,ids in c.MENUS.items():
   weights=b['weights_by_mode'][name];self.assertTrue(np.all(weights[[i for i in range(len(c.BANK))if i not in ids]]==0))
   if name!='original':self.assertEqual(weights[0],0.)
   model=c.MenuPool({'menu':name,'inner_paths':4,'schedule':'half_doubling'},dates,b);values=np.concatenate([v for _,v in model.terminal_chunks(x,w,4,1000,61)],axis=1)
   self.assertEqual(values.shape,(1000,4));self.assertTrue(np.isfinite(values).all());self.assertEqual(sum(model.last_selection['path_counts']),1000)
 def test_endpoint_submatrices_are_exact_separate_menu_reductions(self):
  x,w,dates=self.fixture();saved=[];native=c.all_chunks
  def capture(*args,**kwargs):
   last=None
   for start,cube in native(*args,**kwargs):last=cube[:,:,-1].copy();yield start,cube
   saved.append(last)
  with patch.object(c,'all_chunks',capture):b=c.prepare(x,w,4,dates,inner_paths=4,schedule_rule='half_doubling')
  count=b['diagnostics']['available_internal_origins']
  for menu in c.MENUS:
   ids=b['coefficients'][menu]['valid'];A=np.zeros(len(ids));B=np.zeros((len(ids),len(ids)))
   for trace,last in zip(b['trace'],saved):
    positions=[trace['eligible_this_fold'].index(i)for i in ids];stop,h,m=trace['training_rows'],trace['validation_horizon'],trace['represented_origins'];actual=float(c.portfolio_returns(x[stop:stop+h],w).sum());a,z=c.energy.coefficients(last[positions],actual);A+=m*a/np.sqrt(h);B+=m*z/np.sqrt(h)
   np.testing.assert_array_equal(A/count,b['coefficients'][menu]['A']);np.testing.assert_array_equal(B/count,b['coefficients'][menu]['B'])
 def test_shared_fits_leave_each_menu_forecast_exact(self):
  x,w,dates=self.fixture();cache=PreparedCache(self.expanded_eligible(x,4),w)
  for n,rep in((4,0),(30,1)):
   direct=c.prepare(x,w,4,dates,inner_paths=n,schedule_rule='half_doubling',seed_replication=rep)
   with patch.object(c,'prefix_fits',cache.get):cached=c.prepare(x,w,4,dates,inner_paths=n,schedule_rule='half_doubling',seed_replication=rep)
   for name in c.MENUS:
    for k in('A','B'):np.testing.assert_array_equal(direct['coefficients'][name][k],cached['coefficients'][name][k])
    cfg={'menu':name,'inner_paths':n,'schedule':'half_doubling'};a=c.MenuPool(cfg,dates,direct);b=c.MenuPool(cfg,dates,cached)
    np.testing.assert_array_equal(np.concatenate([v for _,v in a.terminal_chunks(x,w,4,1000,41)],axis=1),np.concatenate([v for _,v in b.terminal_chunks(x,w,4,1000,41)],axis=1))
  self.assertEqual(cache.actual_fits,len(cached['trace'])+1)
 def test_history_dates_weights_and_training_diagnostics_are_bound(self):
  x,w,dates=self.fixture();b=c.prepare(x,w,4,dates,inner_paths=4,schedule_rule='half_doubling');model=c.MenuPool({'menu':'flat_PPW_SAS','inner_paths':4,'schedule':'half_doubling'},dates,b)
  with self.assertRaises(ValueError):next(model.terminal_chunks(x,w,5,1000,51))
  with self.assertRaises(ValueError):next(model.terminal_chunks(x,w[::-1],4,1000,51))
  for trace in b['trace']:
   self.assertLessEqual(trace['validation_stop'],len(x));self.assertEqual(trace['PPW_dependence_fit']['training_rows'],trace['training_rows'])
   self.assertTrue(all(f['training_rows']==trace['training_rows']for entries in trace['SAS_fits'].values()for f in entries))
  cache=PreparedCache(self.expanded_eligible(x,4),w)
  with self.assertRaises(ValueError):cache.get(x,[1],w[::-1])

if __name__=='__main__':unittest.main()
