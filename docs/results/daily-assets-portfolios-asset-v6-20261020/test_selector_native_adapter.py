"""Synthetic native inputs, checkpoint resume, explicit past dates and parity."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib,json,sys,tempfile,unittest
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';sys.path.insert(0,str(V))
import common
from asset_models import build_replacement,build_without_replacement
from asset_forecast import forecast as original_forecast
from candidates.asset_selector_native_adapter.dated_forecast import forecast
from candidates.asset_selector_native_adapter.model import build_path,build_endpoint,build_online,NativeSelector
from candidates.asset_selector_native_adapter import sweep as api

class AdapterTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);rng=np.random.default_rng(55129);self.dates=np.arange(np.datetime64('2000-01-01'),np.datetime64('2000-01-01')+np.timedelta64(44,'D'));self.values=rng.normal(0,.01,44);np.save(self.root/'dates.npy',self.dates);np.save(self.root/'values.npy',self.values);self.row={'series_id':'SYNTHETIC','panel_branch':'assets','dates_file':'dates.npy','returns_file':'values.npy'};(self.root/'panel.json').write_text(json.dumps({'series':[self.row]}));self.origin=SimpleNamespace(position=39,max_horizon=4,origin_date=str(self.dates[39]),origin_label='synthetic',evaluation_split='synthetic',train_fraction=None)
 def tearDown(self):self.temp.cleanup()
 def test_undated_baseline_delegation_is_byte_identical(self):
  for factory in(build_replacement,build_without_replacement):
   a,d=original_forecast(factory(),self.root,self.row,self.values,self.dates,self.origin,122);b,e=forecast(factory(),self.root,self.row,self.values,self.dates,self.origin,122);np.testing.assert_array_equal(a,b);self.assertEqual(set(d),set(e))
   for k in d:
    if k!='forecast_and_score_seconds':self.assertEqual(d[k],e[k])
 def test_bindings_are_independent_and_do_not_share_mutable_dates(self):
  model=build_path();dates=self.dates[:40].copy();a=model.with_training_dates(dates);b=model.with_training_dates(self.dates[:32]);dates[0]-=np.timedelta64(10,'D');self.assertNotEqual(a.dates[0],dates[0]);self.assertEqual(len(a.dates),40);self.assertEqual(len(b.dates),32);self.assertIsNone(a.bundle);self.assertIsNone(b.bundle);self.assertFalse(a.dates.flags.writeable)
 def test_unbound_calls_and_invalid_dates_fail_closed(self):
  m=build_path()
  with self.assertRaises(ValueError):m.terminal_chunks(None,None,4,1000,2)
  with self.assertRaises(ValueError):m.with_training_dates(self.dates[:1])
  with self.assertRaises(ValueError):m.with_training_dates(self.dates[:3][::-1])
 def test_future_calendar_and_values_are_not_given_to_request_binding(self):
  captured=[]
  class Bound:
   model_id='observed'
   def terminal_chunks(self,history,weights,horizon,paths,seed,**kwargs):captured.append(history.copy());yield 0,np.zeros((paths,horizon))
  class Observer:
   model_id='observed'
   def with_training_dates(self,dates):captured.append(dates.copy());return Bound()
  loss,d=forecast(Observer(),self.root,self.row,self.values,self.dates,self.origin,4);np.testing.assert_array_equal(captured[0],self.dates[:40]);np.testing.assert_array_equal(captured[1][:,0],self.values[:40]);self.assertEqual(d['training_last_date'],str(self.dates[39]));self.assertEqual(loss.shape,(4,))
 def test_native_execute_checkpoint_and_resume_for_all_rules(self):
  panel={'series':[self.row]};origin=vars(self.origin);series=(self.dates,self.values,[origin])
  for factory in(build_endpoint,build_path,build_online):
   model=factory();folder=self.root/model.model_id;seed=api.task_identity('fixture',model.model_id,self.row['series_id'],origin)['seed'];direct,_=forecast(model,self.root,self.row,self.values,self.dates,self.origin,seed)
   self.assertEqual(api.execute_task(model,panel,series,folder,'fixture',(0,0),panel_root=self.root),'done');cp,_=api.task_paths(folder,(0,0));identity=api.task_identity('fixture',model.model_id,self.row['series_id'],origin);saved,diag=api.original._read_checkpoint(cp,identity);np.testing.assert_array_equal(saved,direct);self.assertEqual(diag['simulations'],1000);self.assertTrue(diag['historical_calendar_prefix_bound']);digest=hashlib.sha256(cp.read_bytes()).hexdigest()
   with patch.object(NativeSelector,'with_training_dates',side_effect=AssertionError('Resume must not forecast')):self.assertEqual(api.execute_task(model,panel,series,folder,'fixture',(0,0),panel_root=self.root),'done')
   self.assertEqual(hashlib.sha256(cp.read_bytes()).hexdigest(),digest)
 def test_dependency_identity_binds_parent_numerical_sources(self):
  model=build_path();identity=model.dependency_identity;self.assertIn('candidates/asset_selector_universe/bank.py',identity['source_hashes']);self.assertIn('candidates/asset_selector_native_adapter/numerical_sources.json',identity['source_hashes']);original=Path.read_bytes
  with patch.object(Path,'read_bytes',lambda p:b'changed-source'if p.name=='bank.py'else original(p)):
   with self.assertRaises(ValueError):model.dependency_identity
 def test_sweep_clone_changes_only_context_hook_and_protocol_identity(self):
  original=(V/'sweep_v6.py').read_text();changed=(V/'candidates/asset_selector_native_adapter/sweep.py').read_text();restored=changed.replace('from candidates.asset_selector_native_adapter.dated_forecast import forecast as forecast_asset','from asset_forecast import forecast as forecast_asset',1).replace("PROTOCOL = 'daily-cross-asset-joint-dated-selectors-crps-1000-v1'","PROTOCOL = 'daily-cross-asset-joint-paths-crps-1000-v6'",1);self.assertEqual(original,restored)
if __name__=='__main__':unittest.main()
