"""Independently reduce completed reserved losses and check all parity receipts."""
from pathlib import Path
import datetime,hashlib,json,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';D=V/'candidates/asset_selector_batched/validation_and_reserved';PRE=V/'candidates/asset_selector_universe_certified/development1000';sys.path.insert(0,str(V))
import common,sweep_v6 as api,score_main
from inputs import AssetInputs
from candidates.asset_selector_universe_precision.candidate import schedule,signature
from candidates.asset_selector_stacking.energy import even_allocation
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    summary=read(D/'summary.json');plan=read(D/'plan.json');prior=read(PRE/'summary.json');refs=read(D/'references.json')['cases']
    assert read(D/'active.json')['status']=='terminal'and summary['complete_execution']and not summary['errors']
    assert summary['all355_development_requests_exactly_replayed']and summary['qualification_valid']and summary['new_reserved_forecasts']==355
    assert all(sha(O/p)==h for p,h in plan['source_hashes'].items())
    metadata={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};previous={(r['case'],r['config_index']):r for r in prior['records']}
    assert len(summary['validation_receipts'])==355 and len({r['case']for r in summary['validation_receipts']})==355
    verified_parity=0
    for receipt in summary['validation_receipts']:
        case=receipt['case'];assert receipt['job_kind']=='parity'and not receipt['errors']and len(receipt['records'])==2
        disk=read(D/'parity_receipts'/f'{case:03d}.json');assert disk==receipt
        for r in receipt['records']:
            old=previous[case,r['config_index']];p=Path(r['source_checkpoint']);assert sha(p)==r['source_checkpoint_sha256']==old['checkpoint_sha256']
            with np.load(p,allow_pickle=False)as z:assert hashlib.sha256(z['losses'].tobytes()).hexdigest()==r['loss_sha256']==old['loss_sha256']
            assert r['exact_outer_loss_replay'];verified_parity+=1
    rows=summary['records'];assert len(rows)==355 and len({r['case']for r in rows})==355
    for r in rows:
        i=r['case'];case=plan['reserved_cases'][i];row=metadata[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon'];p=D/'reserved/losses'/f"{r['config_index']:03d}-{i:03d}.npz"
        assert r['paths']==1000 and sha(p)==r['checkpoint_sha256']
        with np.load(p,allow_pickle=False)as z:loss=z['losses'].copy()
        assert len(loss)==H and np.isfinite(loss).all()and hashlib.sha256(loss.tobytes()).hexdigest()==r['loss_sha256']
        dates,values,_=api.load_series(V/'inputs',row,verify_files=False);assert r['score']==score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H])
        assert r['series_id']==case['series_id']and r['origin_date']==origin['origin_date']and r['origin_label']==origin['origin_label']
        assert r['seed']==api.original.coherent_daily_seed(origin['origin_date'],H,1000)
        x,w,ids=AssetInputs(V/'inputs').training(row,stop);assert list(ids)==r['constituent_ids']
        _,folds=schedule(len(x),H);assert len(folds)==len(r['validation_trace'])
        for (prefix,lead),trace in zip(folds,r['validation_trace']):
            assert trace['training_rows']==prefix and trace['validation_horizon']==lead and prefix+lead<=stop
            assert trace['training_sha256']==signature(x[:prefix])
        weights=np.asarray(r['selection']['ideal_weights']);counts=np.asarray(r['selection']['path_counts']);assert np.isfinite(weights).all()and np.all(weights>=0)and abs(weights.sum()-1)<1e-12
        np.testing.assert_array_equal(even_allocation(weights,1000),counts);assert counts.sum()==1000
        certificate=r['fit_certificate'];assert certificate['objective']<=certificate['best_vertex_objective']+1e-12
    q=score_main.aggregate_records(rows,metadata)['score'];assert q==summary['reserved_main_Q']
    ref_Q={};verified_refs=0
    for label in ('current_best','fixed_reference'):
        records=[]
        for i in range(355):
            r=refs[str(i)][label];p=Path(r['checkpoint']);assert r['paths']==1000 and sha(p)==r['checkpoint_sha256']
            with np.load(p,allow_pickle=False)as z:loss=z['losses'].copy()
            assert hashlib.sha256(loss.tobytes()).hexdigest()==r['loss_sha256'];case=plan['reserved_cases'][i];origin=case['origin'];dates,values,_=api.load_series(V/'inputs',metadata[case['series_id']],verify_files=False);stop=origin['position']+1
            assert r['series_id']==case['series_id']and r['origin_date']==origin['origin_date']and len(loss)==origin['max_horizon']
            assert r['score']==score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+len(loss)])
            records.append(r);verified_refs+=1
        ref_Q[label]=score_main.aggregate_records(records,metadata)['score'];assert ref_Q[label]==summary['reference_main_Q'][label]
    output={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified':True,'source_closure_verified':True,'development_parity_loss_vectors_verified':verified_parity,'reserved_candidate_checkpoints_verified':355,'reserved_reference_checkpoints_verified':verified_refs,'inner_support_and_training_prefix_signatures_verified':355,'all_reserved_origin_and_main_scores_recomputed':True,'reserved_main_Q':q,'reference_main_Q':ref_Q,'percent_better_vs_fixed_reference':100*(1-q/ref_Q['fixed_reference']),'beats_current_working_on_exact_reserved_coverage':q<ref_Q['current_best'],'no_independence_or_significance_claim':True,'official':False,'full_panel_catalogue_increment':0,'native_failures':0,'scoring_integrity_errors':0,'summary_sha256':sha(D/'summary.json'),'plan_sha256':sha(D/'plan.json')}
    api.original._atomic_json(D/'verification_receipt.json',output);print(json.dumps(output,indent=2),flush=True)
if __name__=='__main__':main()
