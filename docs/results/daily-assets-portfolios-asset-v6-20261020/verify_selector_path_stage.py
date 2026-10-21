"""Independent checkpoint/causality/main-score audit of a closed criterion stage."""
from pathlib import Path
import datetime,hashlib,json,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_selector_path_objective';D=C/sys.argv[1];sys.path.insert(0,str(V))
import common,sweep_v6 as api,score_main
from inputs import AssetInputs
from candidates.asset_selector_universe_precision.candidate import signature
from candidates.asset_selector_stacking.energy import even_allocation
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=read(D/'summary.json');plan=read(D/'plan.json');assert s['qualification_valid']and not s['errors']and read(D/'active.json')['status']=='terminal'
assert s['paths']==s['inner_paths']==plan['paths']and all(sha(O/p)==h for p,h in plan['source_hashes'].items())
metadata={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};count=s['paths'];assert len(s['records'])==144
if count==100:
    pre=V/'candidates/asset_selector_universe/screen100';refs=read(pre/'references.json')['cases'];old=read(pre/'summary.json')
    for r in old['records']:
        if r['config_index']==1:refs[str(r['case'])]['terminal_pool']={**r,'checkpoint':str(pre/'losses'/f"001-{r['case']:03d}.npz")}
else:refs=read(D/'references.json')['cases']
for r in s['records']:
    i=r['case'];case=plan['cases'][i];row=metadata[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon'];p=D/'losses'/f"{r['config_index']:03d}-{i:03d}.npz"
    assert sha(p)==r['checkpoint_sha256']and r['paths']==r['inner_paths']==count and r['seed']==api.original.coherent_daily_seed(origin['origin_date'],H,count)
    with np.load(p,allow_pickle=False)as z:loss=z['losses'].copy()
    assert len(loss)==H and np.isfinite(loss).all()and hashlib.sha256(loss.tobytes()).hexdigest()==r['loss_sha256']
    dates,values,_=api.load_series(V/'inputs',row,verify_files=False);assert score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H])==r['score']
    x,w,ids=AssetInputs(V/'inputs').training(row,stop);assert list(ids)==r['constituent_ids']
    for trace in r['validation_trace']:
        prefix=trace['training_rows'];lead=trace['validation_horizon'];assert prefix+lead<=stop and trace['training_sha256']==signature(x[:prefix])
        assert trace['training_last_date']==str(dates[prefix-1].astype('datetime64[D]'))and trace['validation_last_known_date']==str(dates[prefix+lead-1].astype('datetime64[D]'))
    weights=np.asarray(r['selection']['ideal_weights']);assert np.all(weights>=0)and abs(weights.sum()-1)<1e-12
    np.testing.assert_array_equal(even_allocation(weights,count),r['selection']['path_counts'])
    if r['config_index']==0:assert r['loss_sha256']==refs[str(i)]['terminal_pool']['loss_sha256']
for mid,m in s['models'].items():
    rows=[r for r in s['records']if r['model_id']==mid];assert len(rows)==72 and len({r['case']for r in rows})==72
    assert score_main.aggregate_records(rows,metadata)['score']==m['main_Q']
for label in ('fixed_reference','current_best','terminal_pool'):
    records=[]
    for i in range(72):
        r=refs[str(i)][label];p=Path(r['checkpoint']);assert r['paths']==count and sha(p)==r['checkpoint_sha256']
        with np.load(p,allow_pickle=False)as z:assert hashlib.sha256(z['losses'].tobytes()).hexdigest()==r['loss_sha256']
        records.append(r)
    assert score_main.aggregate_records(records,metadata)['score']==s['reference_main_Q'][label]
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified':True,'stage':sys.argv[1],'paths':count,'inner_paths':count,'candidate_checkpoints_verified':144,'reference_checkpoints_verified':216,'endpoint_control_exact_loss_replays':72,'historical_prefix_signatures_and_past_dates_verified':144,'all_main_Q_reductions_recomputed':True,'models':s['models'],'reference_main_Q':s['reference_main_Q'],'official':False,'full_panel_catalogue_increment':0,'no_independence_or_significance_claim':True,'source_closure_verified':True,'summary_sha256':sha(D/'summary.json')}
api.original._atomic_json(D/'verification_receipt.json',out);print(json.dumps(out,indent=2),flush=True)
