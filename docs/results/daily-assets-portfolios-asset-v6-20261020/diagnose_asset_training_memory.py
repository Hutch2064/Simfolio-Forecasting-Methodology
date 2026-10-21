"""Training-prefix-only AR1 mechanism diagnostic; no extra selection score."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import datetime,hashlib,json,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';sys.path.insert(0,str(V))
import common,sweep_v6 as api
from inputs import AssetInputs

def stats(x):
    y=np.asarray(x,dtype=float);y=y-y.mean();lag=y[:-1];target=y[1:];den=lag@lag
    if den<=0:raise ValueError('Positive training variation required')
    phi=float(lag@target/den);residual=target-phi*lag;score=lag*residual
    score-=score.mean();bandwidth=int(len(score)**(1/3))
    value=float(score@score)
    for k in range(1,bandwidth+1):value+=2*(1-k/(bandwidth+1))*float(score[k:]@score[:-k])
    if value< -1e-12*float(score@score):raise ArithmeticError('Invalid Bartlett HAC variance')
    se=np.sqrt(max(0.,value))/den
    return {'raw_training_AR1':phi,'Bartlett_HAC_standard_error':float(se),'training_bandwidth':bandwidth,'training_rows':len(x),'absolute_phi_over_se':abs(phi)/se if se>0 else None}

def main():
    plan=json.loads((V/'candidates/asset_fading_vol/plan.json').read_text());panel=api.load_panel(V/'inputs',verify_files=False);rows={r['series_id']:r for r in panel['series']};inputs=AssetInputs(V/'inputs');seen=set();records=[]
    for case in plan['phases']['development']:
        row=rows[case['series_id']];n=case['origin']['position']+1;x,w,ids=inputs.training(row,n)
        for a,sid in enumerate(ids):
            digest=hashlib.sha256(np.ascontiguousarray(x[:,a]).tobytes()).hexdigest()
            key=sid,digest
            if key in seen:continue
            seen.add(key);records.append({'constituent_id':sid,'example_series_id':row['series_id'],'origin_date':case['origin']['origin_date'],'asset_class':row['asset_class'],'training_values_sha256':digest,**stats(x[:,a])})
    grouped={}
    for name in sorted({r['asset_class'] for r in records}):
        subset=[r for r in records if r['asset_class']==name];absolute=np.array([abs(r['raw_training_AR1']) for r in subset]);ratios=[r['absolute_phi_over_se'] for r in subset if r['absolute_phi_over_se'] is not None]
        grouped[name]={'training_contexts':len(subset),'absolute_AR1_median':float(np.median(absolute)),'absolute_AR1_p90':float(np.quantile(absolute,.9)),'absolute_AR1_over_HAC_se_median':float(np.median(ratios)),'absolute_AR1_over_HAC_se_p90':float(np.quantile(ratios,.9))}
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_only':True,'held_out_returns_or_losses_used':False,'source_plan_sha256':hashlib.sha256((V/'candidates/asset_fading_vol/plan.json').read_bytes()).hexdigest(),'contexts':len(records),'class_summaries':grouped,'records':records,'diagnostics_have_zero_score_or_model_promotion_weight':True,'no_independence_or_significance_claim':True,'note':'Overlapping prefixes and holdings are correlated. Lag1 association can reflect stale prices or sampling calendars; this does not show forecast improvement or justify a model route.'}
    api.original._atomic_json(V/'training_short_memory_diagnostic.json',result);print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
if __name__=='__main__':main()
