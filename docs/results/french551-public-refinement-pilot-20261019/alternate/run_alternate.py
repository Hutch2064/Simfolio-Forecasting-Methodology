import os
for key in ('OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','MKL_NUM_THREADS',
            'OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,hashlib
from pathlib import Path
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'implementation'))
from run_pilot import INPUT,PANEL,write,context,diagnose
from candidate import native_history,public_indices
from candidate_garch import simulate,variances
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon


def plan():
    return {'identity':'public_mean_reverting_residual_sbb_pilot_v1',
        'initial_pilot_plan_sha256':hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest(),
        'cases':'identical 11 series x four origins x240 paths; full horizons unchanged',
        'baseline':'reuse exact public arrays from first complete pilot; no refits',
        'reason':'Initial recursive EWMA lost 14.21 percent vs public; near-zero typical future variance collapses long-horizon ensemble spread. Add only a training-fitted restoring variance intercept.',
        'method':'v_next=(1-rho)*training_variance + rho*(lambda*v+(1-lambda)*residual^2). Fit lambda,rho in[0,1] training Gaussian predictive likelihood, variance targeted sample ddof1; nested EWMA rho1 and constant variance rho0 evaluated explicitly. Raw mean, empirical centered unit-RMS residual SBB, indices and native wealth accounting unchanged.',
        'optimization':'two deterministic L-BFGS-B starts: (EWMA lambda,.99),(.5,.5); endpoints EWMA and constant; no outcome fitting or parameter cutoffs',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(Path(__file__).parent.glob('*.py'))},
        'claim_limit':'one exploratory incremental alternative on same bounded panel; no untouched validation set or full-panel result'}


def summary(p):
    result={}
    for variant,key in [('public','public_loss'),('candidate','candidate_loss')]:
        rr=[]
        for row in p['series']:
            records=[]
            for o in p['origins']:
                loc=ROOT/'results' if variant=='public' else ROOT/'alternate/results'
                path=loc/row['series_id']/(o['origin_label']+'.npz')
                if not path.exists():break
                with np.load(path) as a:records.append((a[key].copy(),float(a['train_std']),int(a['train_n'])))
            if len(records)!=len(p['origins']):continue
            H=max(len(r[0]) for r in records);count=np.zeros(H);raw=np.zeros(H);q=np.zeros(H);sens=np.zeros(H)
            for loss,std,n in records:
                h=np.arange(1,len(loss)+1);count[:len(h)]+=1;raw[:len(h)]+=loss
                q[:len(h)]+=loss/(std*np.sqrt(h+h*h/n));sens[:len(h)]+=loss/(std*np.sqrt(h))
            w=np.log1p(1/np.arange(1,H+1))
            rr.append({'series_id':row['series_id'],'family':row['family'],'weighting':row['weighting'],
                'v2':float(np.dot(w,q/count)/np.log1p(H)),
                'sensitivity':float(np.dot(w,sens/count)/np.log1p(H)),
                'raw':float(np.mean(raw/count))})
        result[variant]={'complete_series':len(rr),'series':rr}
        if rr:result[variant].update({k:float(np.mean([r[k] for r in rr])) for k in ['v2','sensitivity','raw']})
    if result['candidate']['complete_series']==11:
        result['skill_percent']=100*(1-result['candidate']['v2']/result['public']['v2'])
    result['scope']='exploratory alternate on same bounded cases; no untouched validation; no full-panel ranking'
    write(ROOT/'alternate/summary.json',result)
    return result


def run():
    if json.loads((ROOT/'alternate/plan.json').read_text())!=plan():raise ValueError('plan changed')
    p=json.loads((ROOT/'plan.json').read_text());dates=np.load(INPUT/'dates.npy').astype('datetime64[ns]')
    values=np.load(INPUT/'daily_log_returns.npy',mmap_mode='r')
    positions={r['series_id']:i for i,r in enumerate(PANEL['series'])}
    # Verify the EWMA nested boundary formula without scoring held-out outcomes.
    from simfolio_forecasting_methodology.french_panel.control_adapters import _ewma_variances
    resid=np.array([.1,-.2,.15,-.03]);initial=.04
    a,b=variances(resid,initial,.9,1.);c,d=_ewma_variances(resid,initial,.9)
    np.testing.assert_allclose(a,c,rtol=1e-14);np.testing.assert_allclose(b,d,rtol=1e-14)
    for row in p['series']:
        for o in p['origins']:
            output=ROOT/'alternate/results'/row['series_id']/(o['origin_label']+'.npz')
            if output.exists():continue
            vals=values[positions[row['series_id']]];training,ctx=context(row,o,vals,dates)
            hist=native_history(training.portfolio_log_returns);start=perf_counter()
            indices,block=public_indices(hist,240,ctx.horizon_days,ctx.seed)
            index_s=perf_counter()-start;start=perf_counter()
            daily,fit=simulate(hist,indices);simulate_s=perf_counter()-start
            if daily.shape!=(240,ctx.horizon_days) or not np.all(np.isfinite(daily)):raise ValueError('bad paths')
            stop=o['position']+1;target=np.cumsum(vals[stop:stop+ctx.horizon_days]);terminal=np.cumsum(daily,axis=1)
            start=perf_counter();loss=empirical_crps_by_horizon(terminal,target);score_s=perf_counter()-start
            with np.load(ROOT/'results'/row['series_id']/(o['origin_label']+'.npz')) as ref:
                std=float(ref['train_std']);n=int(ref['train_n']);selected=ref['selected_horizons'].copy()
            output.parent.mkdir(parents=True,exist_ok=True)
            np.savez(output,candidate_loss=loss,train_std=std,train_n=n,
                     selected_horizons=selected,candidate_selected_terminal=terminal[:,selected-1])
            write(output.with_suffix('.json'),{'series':row,'origin':o,'seed':ctx.seed,'block_length':block,
                'fit':fit,'index_seconds':index_s,'simulate_seconds':simulate_s,'score_seconds':score_s,
                'candidate':diagnose(terminal,training.portfolio_log_returns,ctx.horizon_days)})
            print(json.dumps({'series':row['series_id'],'origin':o['origin_label'],'status':'saved',
                'persistence':fit['persistence'],'decay':fit['decay'],'simulate_s':simulate_s}),flush=True)
            summary(p)
    print(json.dumps(summary(p)),flush=True)


if __name__=='__main__':
    if '--declare' in sys.argv:
        if (ROOT/'alternate/plan.json').exists():raise ValueError('already declared')
        write(ROOT/'alternate/plan.json',plan());print('alternate declared before scores')
    else:run()
