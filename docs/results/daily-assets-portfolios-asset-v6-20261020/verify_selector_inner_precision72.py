"""Rebuild the controlled precision study's scores from every saved loss."""
from pathlib import Path
import datetime,hashlib,json,statistics,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';D=V/'candidates/asset_selector_inner_precision/study1000';sys.path.insert(0,str(V));sys.path.insert(0,str(O))
import common,sweep_v6 as api,score_main
from inputs import AssetInputs
from candidates.asset_selector_universe_precision.candidate import signature
from score_percentage import percent_better
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    plan=read(D/'plan.json');summary=read(D/'summary.json');assert summary['complete_execution']and summary['qualification_valid']and not summary['errors']and summary['outer_paths']==1000
    assert all(sha(O/p)==h for p,h in plan['source_hashes'].items());assert sha(D/'references.json')==plan['references_sha256'];panel=api.load_panel(V/'inputs',verify_files=False);rows={r['series_id']:r for r in panel['series']};refs=read(D/'references.json')['cases'];records=summary['records'];assert len(records)==864
    expected={(i,j)for i in range(72)for j in range(12)};assert {(r['case'],r['config_index'])for r in records}==expected
    reference_rows={k:[]for k in ('fixed_reference','precision1000_path')};verified=[];recalculated={};prefixes={}
    for i,case in enumerate(plan['cases']):
        row=rows[case['series_id']];origin=case['origin'];dates,values,origins=api.load_series(V/'inputs',row,verify_files=False);assert origin in origins;stop=origin['position']+1;H=origin['max_horizon'];x,w,ids=AssetInputs(V/'inputs').training(row,stop);prefixes[i]=(row,dates,values,stop,H,x,ids)
        for label in reference_rows:
            r=refs[str(i)][label];cp=Path(r['checkpoint']);assert r['series_id']==case['series_id']and r['origin_label']==origin['origin_label']and r['paths']==1000 and sha(cp)==r['checkpoint_sha256']
            with np.load(cp,allow_pickle=False)as data:loss=data['losses'];assert loss.shape==(H,)and np.isfinite(loss).all()and hashlib.sha256(loss.tobytes()).hexdigest()==r['loss_sha256']
            q=score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H]);assert q==r['score'];reference_rows[label].append({**r,'score':q})
    for r in records:
        i=r['case'];j=r['config_index'];case=plan['cases'][i];origin=case['origin'];row,dates,values,stop,H,x,ids=prefixes[i];cfg=plan['configs'][j];n=cfg['inner_paths'];rep=cfg['seed_replication']
        assert r['configuration']==cfg and r['outer_paths']==1000 and r['inner_paths']==n and r['inner_seed_replication']==rep and r['series_id']==case['series_id']and r['origin_label']==origin['origin_label']and tuple(r['constituent_ids'])==ids
        assert r['source_closure_digest']==api.original.stable_digest(plan['source_hashes']);assert r['final_seed_shared_across_inner_counts']==api.original.coherent_daily_seed(origin['origin_date'],H,1000)
        assert r['selection']['config']=={'criterion':'path','inner_paths':n}and sum(r['selection']['path_counts'])==1000
        for fold in r['validation_trace']:
            fstop=fold['training_rows'];fh=fold['validation_horizon'];assert fstop+fh==fold['validation_stop']and fstop+fh<=stop and fold['training_sha256']==signature(x[:fstop])
            assert fold['training_last_date']==str(dates[fstop-1])and fold['validation_last_known_date']==str(dates[fstop+fh-1]);suffix=''if rep==0 else f':replication={rep}';seed=int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{signature(x[:fstop])}:{fh}:{n}{suffix}'.encode()).digest()[:4],'little');assert seed==fold['seed']
        cp=D/'losses'/f'{j:03d}-{i:03d}.npz';assert sha(cp)==r['checkpoint_sha256']
        with np.load(cp,allow_pickle=False)as data:loss=data['losses'];assert loss.shape==(H,)and np.isfinite(loss).all()and hashlib.sha256(loss.tobytes()).hexdigest()==r['loss_sha256']
        q=score_main.origin_score(loss,values[:stop],origin['origin_date'],dates[stop:stop+H]);assert q==r['score'];recalculated[i,j]={**r,'score':q};verified.append({'case':i,'configuration':j,'checkpoint_sha256':r['checkpoint_sha256'],'loss_sha256':r['loss_sha256']})
    baseline=score_main.aggregate_records(reference_rows['fixed_reference'],rows)['score'];high=score_main.aggregate_records(reference_rows['precision1000_path'],rows)['score'];assert baseline==summary['main_reference_Q']and high==summary['preserved1000inner_reference_Q'];models=[]
    for j,cfg in enumerate(plan['configs']):
        r=[recalculated[i,j]for i in range(72)];q=score_main.aggregate_records(r,rows)['score'];old=next(m for m in summary['models']if m['config_index']==j);assert q==old['main_Q']and percent_better(q,baseline)==old['percent_better_vs_fixed_reference'];models.append({'configuration':cfg,'main_Q':q,'percent_better_vs_fixed_reference':percent_better(q,baseline),'worker_cpu_seconds':sum(x['cpu_seconds']for x in r),'prepare_cpu_seconds':sum(x['prepare_cpu_seconds']for x in r)})
    repetitions=[]
    for n in sorted({c['inner_paths']for c in plan['configs']}):
        qs=[m['main_Q']for m in models if m['configuration']['inner_paths']==n];repetitions.append({'inner_paths':n,'seed_repetitions':len(qs),'mean_main_Q':statistics.mean(qs),'median_main_Q':statistics.median(qs),'minimum_main_Q':min(qs),'maximum_main_Q':max(qs),'sample_standard_deviation_of_main_Q':statistics.stdev(qs),'interpretation':'Descriptive Monte Carlo variation on reused cases, not independent financial validation or a stability certificate'})
    out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified':True,'summary_sha256':sha(D/'summary.json'),'plan_sha256':sha(D/'plan.json'),'saved_candidate_forecasts_verified':len(verified),'reference_forecasts_verified':144,'all_candidate_and_reference_scores_recomputed':True,'all_internal_seeds_and_causal_prefixes_verified':True,'outer_paths':1000,'models':models,'seed_variation':repetitions,'main_reference_Q':baseline,'preserved1000inner_reference_Q':high,'high_precision_reference_seed_variation_not_established':True,'no_full_panel_promotion_or_production_precision_choice':True,'no_independence_or_significance_claim':True,'runtime_scope':'Within-stage per-configuration processCPU; high-precision reference used an older coefficient kernel and two final outputs, so it is not an isolated causal speed comparison','verified_checkpoint_identities':verified,'verifier_sha256':sha(Path(__file__))};api.original._atomic_json(D/'verification_receipt.json',out);print(json.dumps({k:v for k,v in out.items()if k!='verified_checkpoint_identities'},indent=2),flush=True)

if __name__=='__main__':main()
