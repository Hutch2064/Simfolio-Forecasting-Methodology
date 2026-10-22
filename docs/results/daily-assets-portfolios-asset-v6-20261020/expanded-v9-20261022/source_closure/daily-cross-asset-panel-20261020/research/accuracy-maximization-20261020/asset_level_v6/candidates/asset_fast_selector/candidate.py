"""Fast-method bank selected by causal, horizon-matched inner rolling forecasts."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import common
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
from candidates.asset_feature_batch import fits
from candidates.asset_trained_pool.candidate import TrainedPool,prepare as parent_prepare
from asset_paths import JointAssetBootstrap
from . import bank

BASE_CONFIG={'family':'SV_RMS','innovations':'Gaussian','raw_fade':'EWMA','mean_shrinkage':True,'mean_uncertainty':True}
# Computational budgets, not fitted financial parameters; frozen before results.
CONFIGS=[{'maximum_inner_origins':n,'inner_paths':32}for n in(1,3)]
MINIMUM_TRAINING_ROWS=504 # Existing panel's training floor.
BANK=[{'kind':'parent'}, {'kind':'blocks','replacement':True}, {'kind':'blocks','replacement':False},
      *[{'kind':'filtered',**c}for c in bank.CONFIGS]]

def identity(config):return 'model_testing_asset_fast_rolling_selector_v1_'+hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
def signature(x):return hashlib.sha256(np.ascontiguousarray(x,dtype=float).tobytes()).hexdigest()
def rolling_origins(rows,horizon,maximum):
    if horizon<1 or maximum<1:raise ValueError('Positive horizon and search budget required')
    return [stop for j in range(1,maximum+1)if(stop:=rows-j*horizon)>=MINIMUM_TRAINING_ROWS]

def native_objective(losses,history):
    h=np.arange(1,len(losses)+1,dtype=float);mass=np.log1p(h)-np.log(h);sd=float(np.std(history,ddof=1))
    if sd<=0 or not np.isfinite(sd):raise ArithmeticError('Positive historical portfolio scale required')
    return float(np.dot(mass,np.asarray(losses)/sd/np.sqrt(h))/mass.sum())

def portfolio_returns(history,weights):
    logw=np.full(len(weights),-np.inf);positive=np.asarray(weights)>0;logw[positive]=np.log(np.asarray(weights)[positive])
    return np.logaddexp.reduce(np.asarray(history)+logw,axis=1)

def shared_fits(x):
    base=fits.base_fit(x);bundles={}
    for family in('constant','EWMA_reverting','GARCH','SV_RMS'):bundles[family]=bank.prepare(x,family,base)
    parent=parent_prepare(x,'SV_RMS','EWMA',base,bundles['SV_RMS'])
    return {'families':bundles,'parent':parent}

def model(index,prepared):
    config=BANK[index]
    if config['kind']=='parent':return TrainedPool(BASE_CONFIG,prepared['parent'])
    if config['kind']=='blocks':return JointAssetBootstrap(replacement=config['replacement'])
    config={k:v for k,v in config.items()if k!='kind'}
    return bank.PurePool(config,prepared['families'][config['family']])

def choose(losses,maximum):
    if len(losses)==0:return 0
    scores=np.asarray(losses[:maximum],dtype=float).mean(axis=0)
    if not np.isfinite(scores).any():raise ArithmeticError('No fully scored inner candidate')
    return int(np.argmin(scores)) # Frozen bank order gives deterministic ties.

def prepare(history,weights,horizon,maximum_origins=3):
    started=time.perf_counter();x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float)
    if x.ndim!=2 or len(x)<MINIMUM_TRAINING_ROWS or x.shape[1]!=len(w)or not np.isfinite(x).all()or np.any(w<0)or not np.isclose(w.sum(),1.,rtol=0,atol=1e-14):raise ValueError('Valid dated-history-equivalent joint training prefix required')
    origins=rolling_origins(len(x),horizon,maximum_origins);scores=[];folds=[];fit_seconds=0.;forecast_seconds=0.
    for stop in origins:
        past=x[:stop];future=x[stop:stop+horizon];assert len(future)==horizon
        begin=time.perf_counter();prepared=shared_fits(past);fit_seconds+=time.perf_counter()-begin
        observations=portfolio_returns(future,w);actual=np.cumsum(observations);train=portfolio_returns(past,w)
        seed=int.from_bytes(hashlib.sha256(f'fast_inner_v1:{signature(past)}:{horizon}:32'.encode()).digest()[:4],'little')
        values=[];begin=time.perf_counter()
        for index in range(len(BANK)):
            loss=np.empty(horizon);cursor=0
            for start,paths in model(index,prepared).terminal_chunks(past,w,horizon,32,seed):
                assert start==cursor and paths.shape[0]==32;cursor+=paths.shape[1]
                loss[start:cursor]=empirical_crps_by_horizon(paths,actual[start:cursor])
            if cursor!=horizon or not np.isfinite(loss).all():raise ArithmeticError('Incomplete inner forecast')
            values.append(native_objective(loss,train))
        forecast_seconds+=time.perf_counter()-begin;scores.append(values)
        folds.append({'training_stop_exclusive':stop,'validation_stop_exclusive':stop+horizon,'training_hash':signature(past),'seed':seed})
    choices={str(c['maximum_inner_origins']):choose(scores,c['maximum_inner_origins'])for c in CONFIGS}
    selection_seconds=time.perf_counter()-started;begin=time.perf_counter()
    # Only refit the selected families, plus the exact parent for paired replay.
    base=fits.base_fit(x);parent=parent_prepare(x,'SV_RMS','EWMA',base)
    families={}
    for index in set(choices.values()):
        spec=BANK[index]
        if spec['kind']=='filtered'and spec['family']not in families:families[spec['family']]=bank.prepare(x,spec['family'],base)
    refit_seconds=time.perf_counter()-begin
    return {'history_sha256':signature(x),'weights':w.copy(),'horizon':horizon,'choices':choices,'inner_scores':scores,'folds':folds,'parent':parent,'families':families,
            'diagnostics':{'selection_seconds':selection_seconds,'inner_shared_fit_seconds':fit_seconds,'inner_forecast_CRPS_seconds':forecast_seconds,'full_refit_seconds':refit_seconds,'bank_size':len(BANK),'inner_paths':32,'available_inner_origins':len(origins),'fallback_no_full_target_validation':not origins,'no_outer_outcomes_used':True}}

class FastSelector:
    def __init__(self,config,bundle=None):
        if config not in CONFIGS:raise ValueError('Unknown frozen search budget')
        self.config=dict(config);self.bundle=bundle;self.model_id=identity(config)
    @property
    def dependency_identity(self):
        return {'method':self.model_id,'forecast_level':'joint_assets','selection':'Training-prefix-only complete target-horizon rolling validation; no outer scores','config':self.config,'bank':BANK,'fallback':'Current trained model if no complete inner target horizon','source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(__file__).parent.glob('*.py')}}
    def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64):
        bundle=self.bundle if self.bundle is not None else prepare(history,weights,horizon,self.config['maximum_inner_origins'])
        if bundle['history_sha256']!=signature(history)or bundle['horizon']!=horizon or not np.array_equal(bundle['weights'],weights):raise ValueError('Selector request differs from its fitted history, weights or horizon')
        index=bundle['choices'][str(self.config['maximum_inner_origins'])]
        self.last_selection={'bank_index':index,'specification':BANK[index],'config':self.config,'folds_used':min(len(bundle['folds']),self.config['maximum_inner_origins']),**bundle['diagnostics']}
        for start,result in model(index,bundle).terminal_chunks(history,weights,horizon,paths,context_seed,width=width):yield start,result

def build_000():return FastSelector(CONFIGS[0])
def build_001():return FastSelector(CONFIGS[1])
