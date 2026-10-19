"""Research-only conventional-SV conditional integration, exact canonical tasks.

The RMS integration is an existing ablation (neither multiplier), not a new
normalization fix. Conditional expectation exp(-variance/8) is a distinct
distribution change, never an output-parity claim.
"""
from dataclasses import asdict
import argparse
import cProfile
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pstats
import sys
import time
import numpy as np
from scipy.special import logsumexp

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m
FAST=load('variance_fast',ROOT/'no_rough_fast/models.py');REF=FAST.reference
FROZEN=load('variance_inputs',ROOT/'verified_frozen_tasks.py')
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon

LABELS=('production_saved_control','rough_off_saved_control','deterministic_RMS','conditional_mean_multiplier')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def forbid(*a,**kw):raise AssertionError('rough computation forbidden')
def initialize():
    os.nice(10);FAST.initialize()
    body=REF.MODELS.parent.body
    body.rough.rough_fit=forbid;body.pathwise.prepared=forbid

def context(t):
    return REF.ForecastContext(model_id=REF.MODEL_ID,portfolio_id=t.portfolio_id,
        origin_label=t.origin_label,horizon_days=t.horizon_days,simulations=240,
        seed=t.seed,future_dates=t.future_dates,origin_date=t.origin_date)

def forecast(t,kind):
    """Only future conventional multiplier integration changes; fitted law stays."""
    t.training.validate();c=context(t);body=REF.MODELS.parent.body;shell=REF.MODELS.shell
    past,future=shell._validate_calendar(t.training,c)
    assets=np.asarray(t.training.asset_log_returns,np.float64)
    data=[assets[:,a].tobytes() for a in range(assets.shape[1])]
    stage={};start=time.perf_counter()
    for d in data:body.refit(d)
    stage['conventional_fit_seconds']=time.perf_counter()-start;start=time.perf_counter()
    uniforms=REF.REFERENCE._return_uniforms(assets,c)
    stage['dependence_seconds']=time.perf_counter()-start;start=time.perf_counter()
    paths=np.empty_like(uniforms);fits=[]
    from streamed_paths import map_asset_inplace
    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import moment_return_curves
    for a,d in enumerate(data):
        mean,_=moment_return_curves(body.learned.student.original_fit(d),c.horizon_days)
        phis,coef,sd,initial,level,mm,mv,return_sd=body.configuration(d,c.horizon_days)
        base=np.ascontiguousarray(uniforms[:,:,a])
        map_asset_inplace(base,mean,return_sd,body.learned.nodes(d,c.simulations))
        if kind=='conditional_mean_multiplier':
            base[:]=mean+(base-mean)*np.exp(-mv/8.)
        elif kind=='deterministic_RMS':
            # Preserve the existing ablation's subtraction/addition rounding.
            base[:]=mean+(base-mean)*1.
        else:raise ValueError(kind)
        paths[:,:,a]=np.clip(base,-1.,1.)
        fits.append({'innovation_sd':float(sd),'state_dimension':len(initial),
            'conditional_variance_first_last':[float(mv[0]),float(mv[-1])],
            'conditional_expected_multiplier_first_last':np.exp(-mv[[0,-1]]/8.).tolist()})
    stage['asset_paths_seconds']=time.perf_counter()-start
    dates=shell._historical_rebalance_dates(past.append(future),t.training.policy.rebalance)
    return shell.rejoin(paths,t.training.policy.weights,np.asarray([d in dates for d in future])),stage,fits

def saved(i):
    folder=ROOT/'long-splits52' if i%51>=48 else (ROOT if (ROOT/'tasks'/f'task-{i:04d}.npz').exists() else ROOT/'remaining52')
    with np.load(folder/'tasks'/f'task-{i:04d}.npz') as z:return z['losses'].copy()

def growth(paths,training):
    terminal=paths.sum(axis=1);annualizer=252/paths.shape[1]
    return {'training_CAGR':float(np.expm1(np.mean(training)*252)),
        'mean_path_CAGR':float(np.mean(np.expm1(terminal*annualizer))),
        'median_path_CAGR':float(np.median(np.expm1(terminal*annualizer))),
        'CAGR_of_mean_terminal_wealth':float(np.expm1((logsumexp(terminal)-np.log(len(terminal)))*annualizer)),
        'terminal_log_return_quantiles':np.quantile(terminal,[.01,.05,.25,.5,.75,.95,.99]).tolist()}

def evaluate(i,t,verify=False):
    t.validate();start=time.perf_counter();REF.MODELS.clear_path_cache()
    truth=np.cumsum(t.realized_future_daily_log_returns);control=saved(i);losses=[*control[:2]]
    record={'index':i,'portfolio':t.portfolio_id,'origin':str(t.origin_date),'horizon':t.horizon_days,
        'variants':{},'rough_computation_forbidden':True,'saved_controls_loaded':True}
    if verify:
        tick=time.perf_counter();base=FAST.Candidate().simulate_daily_log_returns(t.training,context(t))
        assert empirical_crps_by_horizon(np.cumsum(base,axis=1),truth).tobytes()==control[1].tobytes()
        record['rough_off_replay']={'seconds':time.perf_counter()-tick,'loss_byte_equal':True,'stages':dict(FAST.TIMINGS)}
    summaries=[]
    for kind in LABELS[2:]:
        tick=time.perf_counter();paths,stage,fits=forecast(t,kind)
        assert paths.shape==(240,t.horizon_days) and np.isfinite(paths).all()
        loss=empirical_crps_by_horizon(np.cumsum(paths,axis=1),truth);losses.append(loss)
        if kind=='deterministic_RMS' and i%51<48:
            assert loss.tobytes()==control[3].tobytes(),'existing neither-multiplier replay changed'
        if verify and kind=='conditional_mean_multiplier':
            _,fit=REF.MODELS.parent.body.refit(np.asarray(t.training.asset_log_returns[:,0],np.float64).tobytes())
            assert fit['success']
        summaries.append(np.stack((paths[:,:126].mean(0),paths[:,:126].std(0),
            *np.quantile(paths[:,:126],[.01,.05,.5,.95,.99],axis=0))))
        record['variants'][kind]={'seconds_including_score':time.perf_counter()-tick,'stages':stage,
            'fit':fits,'growth':growth(paths,t.training.asset_log_returns)}
    record['seconds']=time.perf_counter()-start
    REF.MODELS.clear_path_cache()
    return np.stack(losses),np.stack(summaries),record

def analytical_tests():
    state=REF.MODELS.parent.body.state
    f=np.array([[.8]]);initial=np.array([.3]);sd=np.array([.2]);h=11
    m,v=state.moments(f,initial,sd,h);days=np.arange(1,h+1)
    np.testing.assert_allclose(m,.3*.8**days,rtol=1e-14)
    np.testing.assert_allclose(v,.04*(1-.8**(2*days))/(1-.64),rtol=1e-14)
    # Gaussian MGF, exactly E[M^2]=1 and E[M]=exp(-v/8).
    np.testing.assert_allclose(np.exp(-.5*v+.5*v),np.ones(h),rtol=0,atol=0)
    np.testing.assert_allclose(np.exp(-.25*v+.125*v),np.exp(-v/8),rtol=0,atol=0)
    return {'AR1_mean_variance_oracle':True,'conditional_second_moment_one':True,
        'conditional_first_moment_exp_minus_variance_over_8':True,
        'initial_state_deterministic':True,'no_lookahead':'forecast reads training and future calendar only; realized target read by scorer'}

def tasks_and_manifest():
    old=json.loads((ROOT/'state-anchor52/manifest.json').read_text())
    tasks=FROZEN.single52(ROOT.parent/'data',old['data_manifest_sha256'])
    identities=[asdict(REF.task_identity(t,REF.MODEL_ID,240)) for t in tasks]
    assert json.loads(json.dumps(identities))==old['task_identities'] and len(tasks)==2652
    files=[Path(__file__),ROOT/'verified_frozen_tasks.py',ROOT/'run.py',ROOT/'no_rough_fast/models.py',
        ROOT/'no_rough_fast/native.py',ROOT/'no_rough_fast/paths.cpp',ROOT/'manifest.json',
        ROOT/'remaining52/manifest.json',ROOT/'long-splits52/manifest.json',
        ROOT/'canonical52/production.npz',ROOT/'canonical52/no_rough_multiplier.npz',ROOT.parent/'data/canonical_data_manifest.json']
    manifest={'scope':'canonical52 exact CRPS same 2652 original tasks240 paths; research only',
        'labels':LABELS,'workers':1,'task_identities':identities,'source_hashes':{str(p):sha(p) for p in files},
        'snapshot_sources':{str(p):sha(p) for folder in ('src','tools') for p in sorted((REF.SNAPSHOT/folder).rglob('*')) if p.suffix in ('.py','.cpp')},
        'math':{'normalization_candidate':'identity; not run','deterministic_RMS':'sqrt(E[M²|training])=1; existing neither_multiplier ablation',
            'conditional_mean_multiplier':'E[M|training]=exp(-conditional_variance/8); changes distribution',
            'numerical_constants':'8 derives from Gaussian exponential moment; clip +/-1 is original bound'},
        'data_manifest_sha256':old['data_manifest_sha256']}
    manifest=json.loads(json.dumps(manifest,sort_keys=True));digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    manifest['experiment_id']='sv-integration-'+digest[:16]
    return tasks,manifest,digest

def aggregate(tasks,complete,digest,out,final=False):
    sums=np.zeros((4,52,8766));counts=np.zeros((52,8766),int);seen=np.zeros(52,int)
    names=list(dict.fromkeys(t.portfolio_id for t in tasks));rows={n:i for i,n in enumerate(names)}
    for i in sorted(complete):
        t=tasks[i];a=rows[t.portfolio_id]
        with np.load(out/'tasks'/f'task-{i:04d}.npz') as z:
            assert str(z['manifest'])==digest;sums[:,a,:t.horizon_days]+=z['losses']
        counts[a,:t.horizon_days]+=1;seen[a]+=1
    selected=np.flatnonzero(seen==51);result={'complete_origins':len(complete),'complete_assets':len(selected)}
    if len(selected):
        cells=sums[:,selected]/counts[selected];scores=cells.mean((1,2))
        for j,label in ((0,'production'),(1,'no_rough_multiplier')):
            with np.load(ROOT/'canonical52'/f'{label}.npz') as z:
                assert cells[j].tobytes()==z['losses'][selected].tobytes()
                assert np.array_equal(counts[selected],z['counts'][selected])
        result.update(scores=dict(zip(LABELS,scores.tolist())),saved_control_cells_byte_equal=True,
            percent_better_than_production={LABELS[j]:100*(1-scores[j]/scores[0]) for j in (2,3)},
            percent_better_than_rough_off={LABELS[j]:100*(1-scores[j]/scores[1]) for j in (2,3)})
        if final:
            assert len(complete)==2652 and len(selected)==52 and (counts>0).all()
            np.savez_compressed(out/'cells.npz',portfolio=np.array(names),losses=sums/counts,counts=counts)
    REF.atomic(out/('results.json' if final else 'interim.json'),result);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('profile','pilot','full'),required=True)
    args=parser.parse_args();tasks,manifest,digest=tasks_and_manifest()
    out=HERE/manifest['experiment_id'];out.mkdir(exist_ok=True);(out/'tasks').mkdir(exist_ok=True)
    if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())==manifest
    else:REF.atomic(out/'manifest.json',manifest)
    initialize();REF.atomic(out/'analytical-tests.json',analytical_tests())
    if args.mode=='profile':
        profiles=[]
        for name in ('single_SPYSIM','single_TLTSIM','single_SPXUSIM'):
            i=next(i for i,t in enumerate(tasks) if t.portfolio_id==name and i%51==48);t=tasks[i]
            for state in ('fresh_fit','cached_fit'):
                REF.MODELS.clear_path_cache();p=cProfile.Profile();p.enable();tick=time.perf_counter()
                paths=FAST.Candidate().simulate_daily_log_returns(t.training,context(t));elapsed=time.perf_counter()-tick;p.disable()
                assert empirical_crps_by_horizon(np.cumsum(paths,axis=1),np.cumsum(t.realized_future_daily_log_returns)).tobytes()==saved(i)[1].tobytes()
                target=out/f'profile-{name}-{state}.pstats';p.dump_stats(target)
                stats=pstats.Stats(p);rows=sorted([{'file':k[0],'line':k[1],'function':k[2],'calls':v[1],
                    'self_seconds':v[2],'cumulative_seconds':v[3]} for k,v in stats.stats.items()],key=lambda r:r['cumulative_seconds'],reverse=True)[:35]
                profiles.append({'index':i,'portfolio':name,'state':state,'horizon':t.horizon_days,'seconds':elapsed,
                    'stages':dict(FAST.TIMINGS),'top_cumulative':rows,'saved_rough_off_replay_byte_equal':True})
        REF.atomic(out/'profile.json',{'scope':'representative local compute diagnostics; not CTV','profiles':profiles})
        print(json.dumps({'output':str(out),'profile_samples':len(profiles)}),flush=True);return
    # Cross-asset pilot includes first/middle rolling and original long contexts.
    selected_names=('single_SPYSIM','single_TLTSIM','single_SPXUSIM','single_XLESIM','single_IWCSIM','single_REITSIM')
    pilot=[i for i,t in enumerate(tasks) if t.portfolio_id in selected_names and i%51 in (0,24,48)]
    assert len(pilot)==18
    indices=pilot if args.mode=='pilot' else list(range(2652));complete=set();failed={};started=time.time()
    for i in indices:
        target=out/'tasks'/f'task-{i:04d}.npz'
        try:
            if target.exists():
                with np.load(target) as z:assert str(z['manifest'])==digest and z['losses'].shape==(4,tasks[i].horizon_days) and np.isfinite(z['losses']).all()
                assert target.with_suffix('.json').exists();complete.add(i);continue
            loss,summary,record=evaluate(i,tasks[i],verify=(args.mode=='pilot' or not complete))
            temporary=target.with_suffix('.tmp.npz');np.savez_compressed(temporary,manifest=digest,losses=loss,daily_summary=summary)
            REF.atomic(target.with_suffix('.json'),record);temporary.replace(target);complete.add(i)
        except Exception as error:
            failed[i]={'type':type(error).__name__,'message':str(error)}
            REF.atomic(out/'tasks'/f'failure-{i:04d}.json',failed[i])
            if not complete:raise
        state={'pid':os.getpid(),'mode':args.mode,'complete_origins':len(complete),'required_origins':len(indices),
            'failures':failed,'elapsed_seconds':time.time()-started,'updated':time.time()}
        REF.atomic(out/(args.mode+'-progress.json'),state);print(json.dumps(state),flush=True)
        if args.mode=='full' and len(complete)%51==0:aggregate(tasks,complete,digest,out)
    assert all(sha(Path(p))==h for p,h in manifest['snapshot_sources'].items())
    if args.mode=='full' and len(complete)==2652:state['result']=aggregate(tasks,complete,digest,out,True)
    if args.mode=='pilot':
        losses=[]
        for i in sorted(complete):
            with np.load(out/'tasks'/f'task-{i:04d}.npz') as z:losses.append(z['losses'].mean(1))
        state['pilot_mean_origin_CRPS']=dict(zip(LABELS,np.mean(losses,axis=0).tolist()))
        state['scope']='18 matched cross-asset contexts; not canonical52 score'
    REF.atomic(out/(args.mode+'-completion.json'),state)

if __name__=='__main__':main()
