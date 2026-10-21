"""Actual worker I/O on a synthetic joint portfolio, including resume guards."""
from pathlib import Path
from unittest.mock import patch
import hashlib,json,sys,tempfile,unittest
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O));sys.path.insert(0,str(O/'asset_level_v6'))
import run_selector_skew_components72 as worker

class WorkerTests(unittest.TestCase):
 def test_joint_portfolio_all_precisions_and_seed_repeats_resume_without_refitting(self):
  with tempfile.TemporaryDirectory()as temporary:
   root=Path(temporary);inputs=root/'inputs';inputs.mkdir();out=root/'study'
   for name in ('workers','losses','records','case_records'):(out/name).mkdir(parents=True,exist_ok=True)
   rng=np.random.default_rng(6229);dates=np.arange(np.datetime64('2000-01-01'),np.datetime64('2000-01-01')+np.timedelta64(45,'D'));matrix=rng.normal(0,.01,size=(45,3));weights=np.array([.2,.3,.5]);assets=[]
   for i in range(3):
    np.save(inputs/f'A{i}.dates.npy',dates);np.save(inputs/f'A{i}.returns.npy',matrix[:,i]);assets.append({'series_id':f'A{i}','panel_branch':'assets','dates_file':f'A{i}.dates.npy','returns_file':f'A{i}.returns.npy'})
   target=np.log1p(np.sum(np.expm1(matrix[1:])*weights,axis=1));np.save(inputs/'P.dates.npy',dates[1:]);np.save(inputs/'P.returns.npy',target)
   origin={'position':39,'max_horizon':4,'origin_date':str(dates[40]),'origin_label':'synthetic40','evaluation_split':'synthetic','train_fraction':None};(inputs/'P.origins.json').write_text(json.dumps([origin]))
   row={'series_id':'P','panel_branch':'portfolios','construction':'constant_weights_common_observed_endpoint','rational_weights':[['A0',1,5],['A1',3,10],['A2',1,2]],'provenance':{'valuation_anchor':str(dates[0])},'dates_file':'P.dates.npy','returns_file':'P.returns.npy','origins_file':'P.origins.json','observations':44,'first_date':str(dates[1]),'last_date':str(dates[-1]),'origins_count':1,'max_horizon':4}
   (inputs/'panel.json').write_text(json.dumps({'series':assets+[row]}));configs=[{'menu':menu,'inner_paths':n,'seed_replication':0}for n in (4,30)for menu in worker.MENUS]
   plan={'cases':[{'series_id':'P','origin':origin}],'configs':configs,'source_hashes':{'worker_source':hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest()}}
   with patch.multiple(worker,V=root,D=out,PLAN=plan,ROWS={'P':row},COUNT=0,create=True):
    from candidates.asset_selector_inner_precision_batched import candidate as old
    x,w,_=worker.AssetInputs(inputs).training(row,40); past_dates=dates[1:41]
    references={}
    for n in (4,30):
     b=old.prepare(x,w,4,past_dates,inner_paths=n);model=old.PathCriterionPool({'criterion':'path','inner_paths':n},past_dates,b)
     seed=worker.api.original.coherent_daily_seed(origin['origin_date'],4,1000)
     predictive=np.concatenate([v for _,v in model.terminal_chunks(x,w,4,1000,seed)],axis=1)
     loss=worker.empirical_crps_by_horizon(predictive,np.cumsum(target[40:44]));cp=root/f'original{n}.npz';np.savez_compressed(cp,losses=loss)
     references['original_'+str(n)]={'checkpoint':str(cp)}
    (out/'references.json').write_text(json.dumps({'cases':{'0':references}}))
    first=worker.evaluate(0);self.assertFalse(first['errors']);self.assertEqual(len(first['records']),10);self.assertEqual(len({r['model_id']for r in first['records']}),10);self.assertEqual(len({r['final_seed_shared_across_inner_counts']for r in first['records']}),1)
    for r in first['records']:
     self.assertEqual(r['outer_paths'],1000);self.assertEqual(len(r['selection']['path_counts']),224);self.assertEqual(r['constituent_ids'],('A0','A1','A2'));self.assertEqual(r['selection']['config']['inner_paths'],r['inner_paths']);self.assertTrue(np.isfinite(r['score']))
     with np.load(out/'losses'/f"{r['config_index']:03d}-000.npz")as data:self.assertEqual(data['losses'].shape,(4,));self.assertEqual(hashlib.sha256(data['losses'].tobytes()).hexdigest(),r['loss_sha256'])
    fingerprints={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(out/'losses').glob('*.npz')}
    with patch.object(worker,'prepare',side_effect=AssertionError('Resume must not refit or resimulate')):
     resumed=worker.evaluate(0);self.assertFalse(resumed['errors']);self.assertEqual(len(resumed['records']),10)
    self.assertEqual(fingerprints,{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(out/'losses').glob('*.npz')})
    # A present file is not sufficient: changed precision metadata must fail.
    record=out/'records/000-000.json';corrupt=json.loads(record.read_text());corrupt['outer_paths']=10;record.write_text(json.dumps(corrupt))
    with self.assertRaises(AssertionError):worker.evaluate(0)

if __name__=='__main__':unittest.main()
