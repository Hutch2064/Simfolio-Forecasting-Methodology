"""Independently verify all four shared-menu outputs, controls and display deltas."""
from pathlib import Path
import datetime,hashlib,json,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';D=V/'candidates/asset_selector_ppw/screen100';sys.path.insert(0,str(V))
import common,sweep_v6 as api,score_main
from inputs import AssetInputs
from candidates.asset_selector_universe_precision.candidate import signature
from candidates.asset_selector_stacking.energy import even_allocation
from score_percentage import normalize_model_percentages
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=read(D/'summary.json');plan=read(D/'plan.json');refs=read(D/'references.json')['cases'];assert read(D/'active.json')['status']=='terminal'and s['qualification_valid']and not s['errors']
assert s['paths']==s['inner_paths']==100 and s['successful_outer_forecasts']==288 and s['original_control_replays']==s['new_candidate_forecasts']==144
assert all(sha(O/p)==h for p,h in plan['source_hashes'].items())and not normalize_model_percentages(s)[1]
metadata={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']}
for r in s['records']:
    number=r['case'];case=plan['cases'][number];row=metadata[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon'];cp=D/'losses'/f"{r['config_index']:03d}-{number:03d}.npz"
    assert sha(cp)==r['checkpoint_sha256']and r['paths']==r['inner_paths']==100 and r['seed']==api.original.coherent_daily_seed(origin['origin_date'],H,100)
    with np.load(cp,allow_pickle=False)as z:loss=z['losses'].copy()
    assert len(loss)==H and np.isfinite(loss).all()and hashlib.sha256(loss.tobytes()).hexdigest()==r['loss_sha256']
    dates,values,_=api.load_series(V/'inputs',row,verify_files=False);assert r['score']==score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H])
    x,w,ids=AssetInputs(V/'inputs').training(row,stop);assert list(ids)==r['constituent_ids']
    for trace in r['validation_trace']:
        prefix=trace['training_rows'];lead=trace['validation_horizon'];assert prefix+lead<=stop and trace['training_sha256']==signature(x[:prefix])
        assert trace['training_last_date']==str(dates[prefix-1].astype('datetime64[D]'))and trace['validation_last_known_date']==str(dates[prefix+lead-1].astype('datetime64[D]'))
    weights=np.asarray(r['selection']['ideal_weights']);assert np.all(weights>=0)and abs(weights.sum()-1)<1e-12;np.testing.assert_array_equal(even_allocation(weights,100),r['selection']['path_counts'])
    if r['specification']['menu']=='original166':
        reference=refs[str(number)][r['specification']['criterion']];assert r['loss_sha256']==reference['loss_sha256']and r['validation_trace']==reference['validation_trace']
for mid,m in s['models'].items():
    rows=[r for r in s['records']if r['model_id']==mid];assert len(rows)==72 and len({r['case']for r in rows})==72 and score_main.aggregate_records(rows,metadata)['score']==m['main_Q']
for label in ('terminal','path','current_best','fixed_reference'):
    records=[]
    for number in range(72):
        r=refs[str(number)][label];cp=Path(r['checkpoint']);assert r['paths']==100 and sha(cp)==r['checkpoint_sha256']
        with np.load(cp,allow_pickle=False)as z:assert hashlib.sha256(z['losses'].tobytes()).hexdigest()==r['loss_sha256']
        records.append(r)
    assert score_main.aggregate_records(records,metadata)['score']==s['reference_main_Q'][label]
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified':True,'candidate_checkpoints_verified':288,'reference_checkpoints_verified':288,'original_control_loss_vectors_and_historical_traces_exactly_replayed':144,'all_origin_and_main_scores_recomputed':True,'all_reported_percentage_fields_recomputed':True,'historical_prefix_and_calendar_causality_verified':288,'source_closure_verified':True,'paths':100,'inner_paths':100,'summary_sha256':sha(D/'summary.json'),'verifier_source_sha256':sha(Path(__file__)),'official':False,'full_panel_catalogue_increment':0,'independence_or_significance_claim':False}
api.original._atomic_json(D/'verification_receipt.json',receipt);print(json.dumps(receipt,indent=2),flush=True)
