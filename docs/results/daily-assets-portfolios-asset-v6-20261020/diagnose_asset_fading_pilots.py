"""Reconcile horizon/class diagnostic contributions to the unchanged overall Q."""
from pathlib import Path
from collections import defaultdict
import datetime,hashlib,json,os,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_fading_vol'
sys.path.insert(0,str(V))
import common,sweep_v6 as api
from diagnose_asset_hybrid_full import origin_weights
read=lambda p:json.loads(p.read_text())
def main():
    panel=api.load_panel(V/'inputs',verify_files=False)
    lookup={r['series_id']:(i,r) for i,r in enumerate(panel['series'])}
    plan=read(C/'plan.json');summaries={}
    labels=['at_most_31_calendar_days','32_to_366_calendar_days','367_to_1826_calendar_days','1827_to_7305_calendar_days','over_7305_calendar_days']
    for phase in ('development','reserved','broad'):
        if not (C/phase/'summary.json').exists():continue
        result=read(C/phase/'summary.json');assert result['complete'] and result['native_failures']==result['scoring_integrity_errors']==0
        cases=plan['phases'][phase];roster=[];contexts=[]
        for case in cases:
            sid,origin=case['series_id'],case['origin'];i,row=lookup[sid]
            dates,values,origins=api.load_series(V/'inputs',row,verify_files=False);j=origins.index(origin)
            key=(i,j);roster.append((key,row,origin['origin_date']));contexts.append((key,row,origin,dates,values))
        weights=origin_weights(roster);pieces={};commitments={}
        for mid in result['sole_main_scores']:
            root=api.model_directory(C/phase/'results' if mid in result['candidate_model_ids'] else V/'candidates/asset_circular_blocks/full/results' if mid==result['current_best_reference_model_id'] else V/'results',mid)
            fingerprint=read(root/'execution.json')['fingerprint'];parts={name:defaultdict(float) for name in ('horizon','class','branch')};commit=hashlib.sha256()
            for key,row,origin,dates,values in contexts:
                cp,_=api.task_paths(root,key);loss,d=api.original._read_checkpoint(cp,api.task_identity(fingerprint,mid,row['series_id'],origin));assert d['simulations']==1000
                n=origin['position']+1;h=origin['max_horizon'];days=(dates[n:n+h].astype('datetime64[D]')-np.datetime64(origin['origin_date'][:10],'D')).astype(int)
                edges=np.log1p(days.astype(float));masses=np.diff(np.r_[0.,edges])/edges[-1]
                cells=weights[key]*loss*masses/(np.std(values[:n],ddof=1)*np.sqrt(np.arange(1,h+1)))
                band=np.bincount(np.searchsorted([31,366,1826,7305],days,side='left'),weights=cells,minlength=5)
                for label,value in zip(labels,band):parts['horizon'][label]+=float(value)
                parts['class'][row['asset_class']]+=float(cells.sum());parts['branch'][row['panel_branch']]+=float(cells.sum())
                commit.update(f'{key}:{hashlib.sha256(cp.read_bytes()).hexdigest()}\n'.encode())
            for dimension,values in parts.items():assert abs(sum(values.values())-result['sole_main_scores'][mid])<2e-12,dimension
            pieces[mid]={dimension:dict(values) for dimension,values in parts.items()};commitments[mid]=commit.hexdigest()
        reference=result['worse_asset_baseline_reference_model_id'];qref=result['sole_main_scores'][reference]
        contributions={mid:{dimension:{label:100*(pieces[reference][dimension].get(label,0)-value)/qref for label,value in groups.items()} for dimension,groups in dimensions.items()} for mid,dimensions in pieces.items()}
        summaries[phase]={'cases':len(cases),'main_scores':result['sole_main_scores'],'additive_Q':pieces,'percent_better_contributions_vs_fixed_reference':contributions,'checkpoint_commitments':commitments}
    out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'paths':1000,'source_plan_sha256':hashlib.sha256((C/'plan.json').read_bytes()).hexdigest(),'phases':summaries,'each_dimension_sums_to_same_canonical_Q':True,'diagnostics_have_zero_additional_score_or_selection_weight':True,'not_independent_validation_or_significance':True}
    api.original._atomic_json(C/'horizon_diagnostics.json',out)
    for phase,s in summaries.items():
        print(json.dumps({'phase':phase,'diagnostic_horizon_percentage_contributions_vs_fixed_reference':{mid:v['horizon'] for mid,v in s['percent_better_contributions_vs_fixed_reference'].items()}}),flush=True)
if __name__=='__main__':main()
