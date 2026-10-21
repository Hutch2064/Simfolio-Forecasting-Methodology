"""Independent checkpoint/score reduction and fresh request-level selector replay."""
import os
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import collections,datetime,hashlib,json,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_fast_selector';D=C/'development1000';sys.path.insert(0,str(V))
import common,score_main,sweep_v6 as api
from inputs import AssetInputs
from candidates.asset_fast_selector.candidate import FastSelector,CONFIGS,prepare,BANK,rolling_origins
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=read(D/'summary.json');plan=read(D/'plan.json');refs=read(D/'references.json')['cases'];panel=api.load_panel(V/'inputs',verify_files=False);metadata={r['series_id']:r for r in panel['series']}
assert s['complete_execution']and not s['errors']and s['paths']==1000 and s['successful_selected_forecasts']==710
assert read(D/'active.json')['status']=='terminal'
assert all(sha(O/p)==h for p,h in plan['source_hashes'].items())
records={};verified=0;reduced={};runtime=[];choices={}
for index,config in enumerate(CONFIGS):
    rows=[];selection=collections.Counter()
    for number,case in enumerate(plan['cases']):
        source=read(D/'case_progress'/f'{number:03d}.json');assert not source['errors'];row=next(r for r in source['records']if r['config_index']==index)
        path=D/'losses'/f'{index:03d}-{number:03d}.npz';assert sha(path)==row['checkpoint_sha256']
        with np.load(path,allow_pickle=False)as z:loss=z['losses'].copy()
        assert hashlib.sha256(loss.tobytes()).hexdigest()==row['loss_sha256'];origin=case['origin'];assert len(loss)==origin['max_horizon']and row['paths']==1000
        dates,values,_=api.load_series(V/'inputs',metadata[case['series_id']],verify_files=False);stop=origin['position']+1
        q=score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+len(loss)]);assert q==row['score']
        expected=rolling_origins(stop,len(loss),config['maximum_inner_origins']);assert row['inner_origins_used']==len(expected)
        if not expected:assert row['selector_choice']==0 and row['loss_sha256']==refs[str(number)]['current_best']['loss_sha256']
        selection[row['selector_choice']]+=1;rows.append(row);verified+=1
    order={r['case']:j for j,r in enumerate(s['records'])if r['config_index']==index};rows.sort(key=lambda r:order[r['case']])
    reduced[index]=score_main.aggregate_records(rows,metadata)['score'];assert reduced[index]==s['models'][rows[0]['model_id']]['main_Q'];choices[index]=dict(selection)
for label in('current_best','fixed_reference'):
    rows=[]
    for number,case in enumerate(plan['cases']):
        row=refs[str(number)][label];path=Path(row['checkpoint']);assert sha(path)==row['checkpoint_sha256']
        with np.load(path,allow_pickle=False)as z:assert hashlib.sha256(z['losses'].tobytes()).hexdigest()==row['loss_sha256']
        rows.append(row)
    assert score_main.aggregate_records(rows,metadata)['score']==s['reference_main_Q'][label]
# Fresh multivariate replay with actual selection (not an unsupported-horizon fallback).
number=next(i for i,c in enumerate(plan['cases'])if metadata[c['series_id']]['panel_branch']=='portfolios'and rolling_origins(c['origin']['position']+1,c['origin']['max_horizon'],3))
case=plan['cases'][number];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon'];x,w,ids=AssetInputs(V/'inputs').training(metadata[case['series_id']],stop)
bundle=prepare(x,w,H);dates,values,_=api.load_series(V/'inputs',metadata[case['series_id']],verify_files=False);actual=np.cumsum(values[stop:stop+H]);seed=api.original.coherent_daily_seed(origin['origin_date'],H,1000)
for index,config in enumerate(CONFIGS):
    loss=np.empty(H)
    for start,paths in FastSelector(config,bundle).terminal_chunks(x,w,H,1000,seed):loss[start:start+paths.shape[1]]=empirical_crps_by_horizon(paths,actual[start:start+paths.shape[1]])
    with np.load(D/'losses'/f'{index:03d}-{number:03d}.npz',allow_pickle=False)as z:np.testing.assert_array_equal(loss,z['losses'])
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified':True,'selected_checkpoints_verified':verified,'reference_checkpoints_verified':710,'cases_per_model':355,'outer_paths':1000,'main_Q_recomputed':reduced,'reference_main_Q':s['reference_main_Q'],'selection_counts_by_config':choices,'fresh_multivariate_replay':{'case':number,'series_id':case['series_id'],'constituents':ids,'both_configurations_bitwise_equal':True},'outer_future_accessible_to_selector':False,'source_closure_verified':True,'native_failures':0,'scoring_integrity_errors':0,'independent_validation_or_significance_claim':False,'scope':'Initial19option prototype; all355cases including210exactparentfallbacks. No claim of complete future option universe.'}
(C/'validation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
