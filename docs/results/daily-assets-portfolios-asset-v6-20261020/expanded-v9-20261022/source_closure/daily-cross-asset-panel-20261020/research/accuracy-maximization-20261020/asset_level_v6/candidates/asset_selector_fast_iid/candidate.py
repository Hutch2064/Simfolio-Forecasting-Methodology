"""History-only endpoint/path objective contrast; calendar dates are past only."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import common
from candidates.asset_selector_universe.bank import BANK,prefix_fits
from candidates.asset_selector_universe_certified import energy
from candidates.asset_selector_universe_precision.candidate import schedule,portfolio_returns,signature
from candidates.asset_selector_path_objective.coefficients import lead_masses
from candidates.asset_selector_coefficient_sweep.coefficients import accumulate
from .paths import all_chunks,model

def identity(config):
    return 'model_testing_asset_calendar_path_CRPS_pool_v1_'+hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]

def prepare(history,weights,horizon,training_dates,*,inner_paths=1000,width=64,seed_replication=0):
    started=time.perf_counter();x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float);dates=np.asarray(training_dates,dtype='datetime64[D]')
    if(x.ndim!=2 or len(x)!=len(dates) or not np.isfinite(x).all() or not np.isfinite(w).all() or x.shape[1]!=len(w)
       or np.any(w<0) or not np.isclose(w.sum(),1,rtol=0,atol=1e-14) or np.isnat(dates).any() or np.any(np.diff(dates)<=np.timedelta64(0,'D'))
       or inner_paths<2 or inner_paths%2 or width<1):raise ValueError('Finite aligned dated joint history and even predictive ensembles required')
    if not isinstance(seed_replication,int) or seed_replication<0:raise ValueError('Nonnegative internal seed replication required')
    seed_suffix='' if seed_replication==0 else f':replication={seed_replication}'
    eligible,folds=schedule(len(x),horizon);invalid={};trace=[];totals={mode:(np.zeros(len(BANK)),np.zeros((len(BANK),len(BANK))))for mode in('terminal','path')}
    for stop,H in folds:
        past=x[:stop];prepared,failures=prefix_fits(past,[i for i in eligible if i not in invalid])
        for i,error in failures.items():invalid[i]={'training_rows':stop,**error}
        valid=[i for i in eligible if i not in invalid];returns=portfolio_returns(x[stop:stop+H],w);actual=np.cumsum(returns);terminal_y=float(returns.sum())
        days=(dates[stop:stop+H]-dates[stop-1]).astype('timedelta64[D]').astype(int);masses=lead_masses(days)
        seed=int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{signature(past)}:{H}:{inner_paths}{seed_suffix}'.encode()).digest()[:4],'little')
        A=np.zeros(len(valid));B=np.zeros((len(valid),len(valid)));last=None
        for start,cube in all_chunks(past,w,H,inner_paths,seed,valid,prepared,width=width):
            count=cube.shape[2];accumulate(cube,actual[start:start+count],masses[start:start+count],A,B);last=np.ascontiguousarray(cube[:,:,-1])
        endpoint_A,endpoint_B=energy.coefficients(last,terminal_y);endpoint_A/=np.sqrt(H);endpoint_B/=np.sqrt(H)
        for mode,a,b in(('terminal',endpoint_A,endpoint_B),('path',A,B)):
            totals[mode][0][valid]+=a;totals[mode][1][np.ix_(valid,valid)]+=b
        trace.append({'training_rows':stop,'validation_horizon':H,'validation_stop':stop+H,'training_sha256':signature(past),'training_last_date':str(dates[stop-1]),'validation_last_known_date':str(dates[stop+H-1]),'seed':seed,'eligible_this_fold':valid,'terminal_y_sum':terminal_y,'terminal_ensemble_sha256':hashlib.sha256(last.tobytes()).hexdigest()})
    valid=[i for i in eligible if i not in invalid];fitted={};weights_by_mode={};coefficients={}
    for mode,(a,b)in totals.items():
        A=a[valid]/len(folds);B=b[np.ix_(valid,valid)]/len(folds);fit=energy.fit(A,B);full=np.zeros(len(BANK));full[valid]=fit['weights'];weights_by_mode[mode]=full;fitted[mode]=fit;coefficients[mode]={'A':A,'B':B}
    needed=sorted({i for values in weights_by_mode.values()for i in np.flatnonzero(values>0)});full,failures=prefix_fits(x,needed)
    if failures:raise ArithmeticError(f'Selected full-history fit failed: {failures}')
    return {**full,'history_sha256':signature(x),'training_dates_sha256':hashlib.sha256(dates.tobytes()).hexdigest(),'dates':dates.copy(),'weights':w.copy(),'horizon':horizon,'inner_paths':inner_paths,'seed_replication':seed_replication,'weights_by_mode':weights_by_mode,'fit':fitted,'coefficients':coefficients,'trace':trace,'invalid_internal_options':invalid,'diagnostics':{'selection_and_refit_seconds':time.perf_counter()-started,'inner_origins':len(folds),'inner_horizon':folds[0][1],'requested_horizon':horizon,'all_known_validation_leads_in_path_objective':True,'automatic_model_fallback':False,'maximum_origin_count':None,'no_outer_future_dates_used':True}}

class PathCriterionPool:
    def __init__(self,config,training_dates,bundle=None):
        if config['criterion']not in('terminal','path')or config['inner_paths']<2 or config['inner_paths']%2:raise ValueError('Valid criterion and even internal ensemble required')
        self.config=dict(config);self.dates=np.asarray(training_dates,dtype='datetime64[D]');self.bundle=bundle;self.model_id=identity(config)
    @property
    def dependency_identity(self):return {'method':self.model_id,'forecast_level':'joint_assets','config':self.config,'past_calendar_required':True,'bank':BANK,'objective':'Same166options; endpoint versus requested-horizon weighted path CRPS, trained only on past dated history','source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(__file__).parent.glob('*.py')}}
    def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64):
        b=self.bundle if self.bundle is not None else prepare(history,weights,horizon,self.dates,inner_paths=self.config['inner_paths'])
        if(b['history_sha256']!=signature(history)or b['horizon']!=horizon or not np.array_equal(b['weights'],weights)or b['inner_paths']!=self.config['inner_paths']or b['training_dates_sha256']!=hashlib.sha256(self.dates.tobytes()).hexdigest()):raise ValueError('Forecast differs from dated fitted request')
        learned=b['weights_by_mode'][self.config['criterion']];counts=energy.even_allocation(learned,paths);generators=[(i,model(i,b).terminal_chunks(history,weights,horizon,int(n),context_seed,width=width))for i,n in enumerate(counts)if n]
        self.last_selection={'config':self.config,'ideal_weights':learned.tolist(),'path_counts':counts.tolist(),'invalid_internal_options':b['invalid_internal_options'],**b['diagnostics']}
        for start in range(0,horizon,width):
            pieces=[]
            for i,generator in generators:
                position,values=next(generator)
                if position!=start or values.shape[0]!=counts[i]:raise ArithmeticError('Incomplete selected path')
                pieces.append(values)
            result=np.concatenate(pieces)
            if not np.isfinite(result).all():raise ArithmeticError('Nonfinite forecast; no path exclusions')
            yield start,result
