"""History-only sequential selector without full pairwise coefficient matrices."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import common
from candidates.asset_selector_universe.bank import BANK,model,prefix_fits
from candidates.asset_selector_universe_precision.candidate import schedule,portfolio_returns,signature
from candidates.asset_selector_path_objective.coefficients import lead_masses
from candidates.asset_selector_stacking.energy import even_allocation
from .gradient_numpy import accumulate
from .state import State

def identity(config):
    return 'model_testing_asset_matrix_free_online_CRPS_pool_v1_'+hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]

def prepare(history,weights,horizon,training_dates,*,inner_paths=1000,width=64):
    started=time.perf_counter();x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float);dates=np.asarray(training_dates,dtype='datetime64[D]')
    if(x.ndim!=2 or len(x)!=len(dates) or x.shape[1]!=len(w) or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w<0)
       or not np.isclose(w.sum(),1,rtol=0,atol=1e-14) or np.isnat(dates).any() or np.any(np.diff(dates)<=np.timedelta64(0,'D')) or inner_paths<2 or inner_paths%2 or width<1):raise ValueError('Finite dated joint history, simplex weights and even predictive ensembles required')
    eligible,folds=schedule(len(x),horizon);invalid={};trace=[];timing={k:0. for k in ('eligibility_fit_cpu_seconds','validation_refit_cpu_seconds','component_simulation_cpu_seconds','gradient_cpu_seconds','full_refit_cpu_seconds')}
    # Match the established learner's final eligible-bank intersection. This
    # pass uses known training prefixes only, not any outer outcome. Retain no
    # unbounded collection of fitted history arrays across short-horizon folds.
    begin=time.process_time()
    for stop,H in folds:
        _,failures=prefix_fits(x[:stop],[i for i in eligible if i not in invalid])
        for i,error in failures.items():invalid[i]={'training_rows':stop,**error}
    timing['eligibility_fit_cpu_seconds']=time.process_time()-begin
    valid=[i for i in eligible if i not in invalid]
    if not valid:raise ArithmeticError('No valid causal option; no fixed-model fallback')
    state=State(len(valid))
    for stop,H in folds:
        past=x[:stop];begin=time.process_time();prepared,failures=prefix_fits(past,valid);timing['validation_refit_cpu_seconds']+=time.process_time()-begin
        if failures:raise ArithmeticError('Deterministic prefix eligibility changed on replay')
        actual=np.cumsum(portfolio_returns(x[stop:stop+H],w));days=(dates[stop:stop+H]-dates[stop-1]).astype('timedelta64[D]').astype(int);masses=lead_masses(days)
        seed=int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{signature(past)}:{H}:{inner_paths}'.encode()).digest()[:4],'little')
        generators=[model(i,prepared).terminal_chunks(past,w,H,inner_paths,seed,width=width)for i in valid]
        A=np.zeros(len(valid));cross=np.zeros(len(valid));last=None
        for start in range(0,H,width):
            begin=time.process_time();count=min(width,H-start);pieces=[]
            for generator in generators:
                position,paths=next(generator)
                if position!=start or paths.shape!=(inner_paths,count) or not np.isfinite(paths).all():raise ArithmeticError('Incomplete historical predictive path')
                pieces.append(paths)
            cube=np.stack(pieces);timing['component_simulation_cpu_seconds']+=time.process_time()-begin;begin=time.process_time()
            accumulate(cube,actual[start:start+count],masses[start:start+count],state.weights,A,cross);timing['gradient_cpu_seconds']+=time.process_time()-begin;last=np.ascontiguousarray(cube[:,:,-1])
        before=state.weights.copy();state.observe(A,cross)
        trace.append({'training_rows':stop,'validation_horizon':H,'validation_stop':stop+H,'training_sha256':signature(past),'training_last_date':str(dates[stop-1]),'validation_last_known_date':str(dates[stop+H-1]),'seed':seed,'eligible_this_fold':valid,'weights_before_outcome':before.tolist(),'terminal_ensemble_sha256':hashlib.sha256(last.tobytes()).hexdigest()})
    learned=np.zeros(len(BANK));learned[valid]=state.weights;needed=np.flatnonzero(learned>0);begin=time.process_time();full,failures=prefix_fits(x,needed);timing['full_refit_cpu_seconds']=time.process_time()-begin
    if failures:raise ArithmeticError('Selected full-history fit failed; no automatic fallback')
    return {**full,'history_sha256':signature(x),'training_dates_sha256':hashlib.sha256(dates.tobytes()).hexdigest(),'weights':w.copy(),'horizon':horizon,'inner_paths':inner_paths,'learned_weights':learned,'online_trace':state.trace,'trace':trace,'invalid_internal_options':invalid,'diagnostics':{'selection_and_refit_seconds':time.perf_counter()-started,'inner_origins':len(folds),'inner_horizon':folds[0][1],'requested_horizon':horizon,'all_known_validation_leads_in_path_objective':True,'automatic_model_fallback':False,'maximum_origin_count':None,'no_outer_future_dates_used':True,'full_pairwise_matrix_constructed':False,'every_valid_option_in_every_gradient':True,**timing}}

class MatrixFreeOnline:
    def __init__(self,config,training_dates,bundle=None):
        if config!={'weight_rule':'SOLO_FTRL','inner_paths':config['inner_paths'],'gradient_backend':'matrix_free'}:raise ValueError('Unknown frozen sequential selector')
        self.config=dict(config);self.dates=np.array(training_dates,dtype='datetime64[D]',copy=True);self.bundle=bundle;self.model_id=identity(config)
    @property
    def dependency_identity(self):return {'method':self.model_id,'forecast_level':'joint_assets','config':self.config,'bank':BANK,'selection':'Same scale-free sequential CRPS gradients, weighted-CDF evaluation without full pair matrix','source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(__file__).parent.glob('*.py')}}
    def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64):
        b=self.bundle if self.bundle is not None else prepare(history,weights,horizon,self.dates,inner_paths=self.config['inner_paths'])
        if(b['history_sha256']!=signature(history) or b['horizon']!=horizon or not np.array_equal(b['weights'],weights) or b['inner_paths']!=self.config['inner_paths'] or b['training_dates_sha256']!=hashlib.sha256(self.dates.tobytes()).hexdigest()):raise ValueError('Forecast differs from dated fitted request')
        allocation=even_allocation(b['learned_weights'],paths);generators=[(i,model(i,b).terminal_chunks(history,weights,horizon,int(n),context_seed,width=width))for i,n in enumerate(allocation)if n]
        self.last_selection={'config':self.config,'ideal_weights':b['learned_weights'].tolist(),'path_counts':allocation.tolist(),'invalid_internal_options':b['invalid_internal_options'],**b['diagnostics']}
        for start in range(0,horizon,width):
            pieces=[]
            for i,generator in generators:
                position,values=next(generator)
                if position!=start or values.shape[0]!=allocation[i]:raise ArithmeticError('Incomplete selected asset path')
                pieces.append(values)
            result=np.concatenate(pieces)
            if not np.isfinite(result).all():raise ArithmeticError('Nonfinite forecast; no exclusions')
            yield start,result
