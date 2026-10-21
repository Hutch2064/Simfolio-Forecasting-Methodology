"""Horizon-matched historical selection or learned predictive-distribution pooling."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import common
from candidates.asset_fast_selector.candidate import BANK,model,portfolio_returns,signature
from candidates.asset_uncapped_selector.candidate import validation_plan
from candidates.asset_feature_batch import fits
from candidates.asset_trained_pool.candidate import prepare as parent_prepare
from candidates.asset_fast_selector import bank
from . import energy

CONFIGS=[{'pool':'select'},{'pool':'stack'}]
INNER_PATHS=1000

def identity(config):return 'model_testing_asset_horizon_rolling_CRPS_pool_v1_'+hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]

def schedule(rows,requested_horizon):
    eligible,_=validation_plan(rows,requested_horizon)
    minimum=max(2 if BANK[i]['kind']=='blocks'else 24 for i in eligible)
    capacity=rows-minimum
    # Two completed blocks permit observing between-block score variation;
    # this is a minimum, never a maximum number of origins or a stability claim.
    horizon=min(requested_horizon,max(1,capacity//2));count=capacity//horizon
    return eligible,[(rows-j*horizon,horizon)for j in range(count,0,-1)]

def prefix_fits(history,eligible):
    prepared={'families':{}};invalid={}
    if all(BANK[i]['kind']=='blocks'for i in eligible):return prepared,invalid
    try:base=fits.base_fit(history)
    except Exception as e:return prepared,{i:{'type':type(e).__name__,'reason':str(e)}for i in eligible if BANK[i]['kind']!='blocks'}
    needed={BANK[i]['family']for i in eligible if BANK[i]['kind']=='filtered'}|({'SV_RMS'}if 0 in eligible else set())
    for family in sorted(needed):
        try:prepared['families'][family]=bank.prepare(history,family,base)
        except Exception as e:
            for i in eligible:
                if(BANK[i]['kind']=='filtered'and BANK[i]['family']==family)or(i==0 and family=='SV_RMS'):invalid[i]={'type':type(e).__name__,'reason':str(e)}
    if 0 in eligible and 0 not in invalid:
        try:prepared['parent']=parent_prepare(history,'SV_RMS','EWMA',base,prepared['families']['SV_RMS'])
        except Exception as e:invalid[0]={'type':type(e).__name__,'reason':str(e)}
    return prepared,invalid

def prepare(history,weights,horizon):
    started=time.perf_counter();x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float)
    if x.ndim!=2 or x.shape[1]!=len(w)or not np.isfinite(x).all()or np.any(w<0)or not np.isclose(w.sum(),1.,rtol=0,atol=1e-14):raise ValueError('Finite normalized joint history required')
    eligible,folds=schedule(len(x),horizon);invalid={};trace=[];terms=[];fit_seconds=0.;simulation_seconds=0.;coefficient_seconds=0.
    for stop,H in folds:
        past=x[:stop];begin=time.perf_counter();prepared,failures=prefix_fits(past,[i for i in eligible if i not in invalid]);fit_seconds+=time.perf_counter()-begin
        for i,error in failures.items():invalid[i]={'training_rows':stop,**error}
        seed=int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{signature(past)}:{H}:{INNER_PATHS}'.encode()).digest()[:4],'little')
        current=[i for i in eligible if i not in invalid];predictions=[];begin=time.perf_counter()
        for i in current:
            final=None;cursor=0
            for start,paths in model(i,prepared).terminal_chunks(past,w,H,INNER_PATHS,seed):
                assert start==cursor and paths.shape[0]==INNER_PATHS;cursor+=paths.shape[1];final=paths[:,-1].copy()
            if cursor!=H or not np.isfinite(final).all():raise ArithmeticError('Incomplete inner joint forecast')
            predictions.append(final)
        simulation_seconds+=time.perf_counter()-begin
        realized=float(portfolio_returns(x[stop:stop+H],w).sum());begin=time.perf_counter();A,B=energy.coefficients(np.stack(predictions),realized)
        # Full-request SD would cancel from all comparisons; use sqrt lead only.
        A/=np.sqrt(H);B/=np.sqrt(H);coefficient_seconds+=time.perf_counter()-begin
        terms.append((current,A,B));trace.append({'training_rows':stop,'validation_horizon':H,'validation_stop':stop+H,'training_sha256':signature(past),'seed':seed,'eligible_this_fold':current,'terminal_observation':realized,'ensemble_sha256':hashlib.sha256(np.stack(predictions).tobytes()).hexdigest()})
    valid=[i for i in eligible if i not in invalid];A=np.zeros(len(valid));B=np.zeros((len(valid),len(valid)))
    for ids,a,b in terms:
        positions=[ids.index(i)for i in valid];A+=a[positions];B+=b[np.ix_(positions,positions)]
    A/=len(terms);B/=len(terms);fitted=energy.fit(A,B);selected=int(np.argmin(A-.5*np.diag(B)));hard=np.zeros(len(valid));hard[selected]=1.
    weights_by_mode={}
    for kind,values in [('select',hard),('stack',fitted['weights'])]:
        full=np.zeros(len(BANK));full[valid]=values;weights_by_mode[kind]=full
    selection_seconds=time.perf_counter()-started;begin=time.perf_counter();needed=sorted({i for mode in weights_by_mode.values()for i in np.flatnonzero(energy.even_allocation(mode,1000))});full,failures=prefix_fits(x,needed)
    if failures:raise ArithmeticError(f'Selected full-history component fit failed: {failures}')
    return {**full,'history_sha256':signature(x),'weights':w.copy(),'horizon':horizon,'weights_by_mode':weights_by_mode,'valid_bank_indices':valid,'invalid_internal_options':invalid,'coefficients':{'A':A,'B':B},'fit':fitted,'trace':trace,'diagnostics':{'selection_seconds':selection_seconds,'inner_shared_fit_seconds':fit_seconds,'inner_simulation_seconds':simulation_seconds,'inner_CRPS_coefficient_seconds':coefficient_seconds,'shared_full_refit_seconds':time.perf_counter()-begin,'inner_paths':INNER_PATHS,'inner_origins':len(folds),'inner_horizon':folds[0][1],'requested_horizon':horizon,'full_requested_horizon_supported':folds[0][1]==horizon,'automatic_model_fallback':False,'maximum_origin_count':None,'valid_options':len(valid)}}

class HorizonPool:
    def __init__(self,config,bundle=None):
        if config not in CONFIGS:raise ValueError('Unknown frozen configuration')
        self.config=dict(config);self.bundle=bundle;self.model_id=identity(config)
    @property
    def dependency_identity(self):return {'method':self.model_id,'forecast_level':'joint_assets','config':self.config,'bank':BANK,'inner_paths':INNER_PATHS,'selection':'History-only terminal target-horizon CRPS, natural nonoverlapping blocks; longest support with two blocks when available; all available blocks included','source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(__file__).parent.glob('*.py')}}
    def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64):
        if paths!=1000:raise ValueError('This admitted outer evaluation requires1000paths')
        b=self.bundle if self.bundle is not None else prepare(history,weights,horizon)
        if b['history_sha256']!=signature(history)or b['horizon']!=horizon or not np.array_equal(b['weights'],weights):raise ValueError('Request differs from fitted selector')
        learned=b['weights_by_mode'][self.config['pool']];allocation=energy.even_allocation(learned,paths);generators=[(i,model(i,b).terminal_chunks(history,weights,horizon,int(number),context_seed,width=width))for i,number in enumerate(allocation)if number]
        self.last_selection={'mode':self.config['pool'],'ideal_weights':learned.tolist(),'path_counts':allocation.tolist(),'invalid_internal_options':b['invalid_internal_options'],**b['diagnostics']}
        for start in range(0,horizon,width):
            pieces=[]
            for i,generator in generators:
                position,values=next(generator);assert position==start and values.shape[0]==allocation[i];pieces.append(values)
            result=np.concatenate(pieces);assert result.shape==(paths,min(width,horizon-start))and np.isfinite(result).all();yield start,result

def build_000():return HorizonPool(CONFIGS[0])
def build_001():return HorizonPool(CONFIGS[1])
