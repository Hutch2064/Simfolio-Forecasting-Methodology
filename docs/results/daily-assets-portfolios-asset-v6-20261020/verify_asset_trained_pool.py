"""Independently reduce all validated selected-model loss vectors; no full admission."""
from pathlib import Path
import os
for name in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
import datetime,hashlib,json,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_trained_pool';sys.path.insert(0,str(V))
import common,sweep_v6 as api
from audit_asset_v6_full_pair import independent_origin,independent_reduce
from inputs import AssetInputs
from candidates.asset_trained_pool.candidate import TrainedPool,identity
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=api.original._atomic_json
CONFIG={'family':'SV_RMS','innovations':'Gaussian','raw_fade':'EWMA','mean_shrinkage':True,'mean_uncertainty':True}


def main():
    panel=api.load_panel(V/'inputs',verify_files=True);metadata={r['series_id']:r for r in panel['series']};mid=identity(CONFIG);commitment=hashlib.sha256();phases={};identities=set();verified=0
    for phase,count in(('development1000',355),('reserved1000',355),('broad1000',7418)):
        D=C/phase;summary=read(D/'summary.json');plan=read(D/'plan.json');assert read(D/'active.json')['status']=='terminal'
        assert summary['complete_execution']and summary['paths']==plan['paths']==1000 and not summary['errors']and len(plan['cases'])==count
        assert all(sha(O/path)==digest for path,digest in plan['source_hashes'].items())
        selected=[r for r in summary['records']if r['model_id']==mid];assert len(selected)==count and len({r['case']for r in selected})==count
        reduced=[];anchors=[];references=[];refs=read(D/'references.json')['cases']
        for record in selected:
            number=record['case'];case=plan['cases'][number];row=metadata[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon']
            assert record['series_id']==row['series_id']and record['origin_date']==origin['origin_date']and record['paths']==1000
            assert record['seed']==api.original.coherent_daily_seed(origin['origin_date'],H,1000)
            dates,values,origins=api.load_series(V/'inputs',row,verify_files=False);assert origin in origins
            cp=D/'losses'/f"{record['config_index']:03d}-{number:03d}.npz";assert sha(cp)==record['checkpoint_sha256']
            with np.load(cp,allow_pickle=False)as payload:loss=payload['losses'].copy()
            assert loss.shape==(H,)and hashlib.sha256(loss.tobytes()).hexdigest()==record['loss_sha256']
            score=independent_origin(loss,values[:stop],origin['origin_date'],dates[stop:stop+H]);assert abs(score-record['score'])<2e-12
            reduced.append((row['series_id'],origin['origin_date'],score));commitment.update(f'{phase}/{number}:{sha(cp)}\n'.encode());verified+=1
            key=(row['series_id'],origin['origin_date'],origin['origin_label'],H);assert key not in identities;identities.add(key)
            for label,out in(('current_best',anchors),('fixed_reference',references)):
                reference=refs[str(number)][label];path=Path(reference['checkpoint']);assert sha(path)==reference['checkpoint_sha256']
                with np.load(path,allow_pickle=False)as payload:
                    key='losses'if'losses'in payload.files else 'loss';vector=payload[key].copy()
                assert hashlib.sha256(vector.tobytes()).hexdigest()==reference['loss_sha256']
                value=independent_origin(vector,values[:stop],origin['origin_date'],dates[stop:stop+H]);assert abs(value-reference['score'])<2e-12;out.append((row['series_id'],origin['origin_date'],value))
        q=independent_reduce(reduced,metadata)['main_score'];anchor=independent_reduce(anchors,metadata)['main_score'];fixed=independent_reduce(references,metadata)['main_score']
        assert abs(q-summary['models'][mid]['main_Q'])<2e-12 and abs(anchor-summary['reference_main_Q']['current_best'])<2e-12 and abs(fixed-summary['reference_main_Q']['fixed_reference'])<2e-12 and q<anchor
        phases[phase]={'successful_cases':count,'main_Q':q,'prior_working_Q':anchor,'fixed_WR_Q':fixed,'percent_better_vs_fixed_WR':100*(1-q/fixed),'summary_sha256':sha(D/'summary.json')}
    # Replay one frozen broad case through a fresh fit and complete generator.
    plan=read(C/'broad1000/plan.json');summary=read(C/'broad1000/summary.json');record=min((r for r in summary['records']if r['model_id']==mid),key=lambda r:r['work_units']);case=plan['cases'][record['case']];row=metadata[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon']
    history,weights,_=AssetInputs(V/'inputs').training(row,stop);_,values,_=api.load_series(V/'inputs',row,verify_files=False);realized=np.cumsum(values[stop:stop+H]);loss=np.empty(H)
    for start,paths in TrainedPool(CONFIG).terminal_chunks(history,weights,H,1000,record['seed']):loss[start:start+paths.shape[1]]=empirical_crps_by_horizon(paths,realized[start:start+paths.shape[1]])
    assert hashlib.sha256(loss.tobytes()).hexdigest()==record['loss_sha256']
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete_development_verification':True,'model_id':mid,'config':CONFIG,'phases':phases,'unique_validated_cases':len(identities),'loss_checkpoints_verified':verified,'checkpoint_commitment_sha256':commitment.hexdigest(),'fresh_native_generator_replay_case':record['case'],'native_failures':0,'scoring_integrity_errors':0,'all_selected_model_loss_hashes_and_independent_origin_and_hierarchy_scores_verified':True,'full_panel_verified':False,'official_scoreboard_changed':False,'goal_stopping_condition_met':False}
    write(C/'validation_receipt.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
