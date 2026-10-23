"""Finite41/current control versus mapped ratio geometry; unchanged numerical laws."""
import hashlib,json,time
from collections import Counter
import numpy as np
from candidates.asset_selector_endpoint_fit_reuse_v3 import candidate as parent
from candidates.asset_selector_endpoint_active_qp_unique_v3 import energy
from .geometry import map_folds
from .provenance import source_manifest
CORE=tuple([1,2,3]+list(range(4,16))+list(range(22,34))+list(range(112,124))+[166,167])
BANK=tuple(parent.BANK);CONFIG={'menu':'flat','inner_paths':2,'schedule':'half_doubling'};MODES=('current','mapped_ratio')
schedule=parent.schedule;all_chunks=parent.all_chunks;coefficients=parent.compiled_coefficients;sources=source_manifest

def folds_for(rows,horizon,mode):
    if mode not in MODES:raise ValueError('Known fixed geometry required')
    eligible,folds,count=schedule(rows,horizon,'half_doubling');eligible=[i for i in eligible if i!=0]+([166,167]if 0 in eligible else[]);eligible=[i for i in eligible if i in CORE]
    if not eligible:raise ArithmeticError('No eligible finite41 recipe; no fallback')
    minimum=max(2 if BANK[i]['kind']=='blocks'else 24 for i in eligible)
    if mode=='mapped_ratio':folds,metadata=map_folds(rows,horizon,minimum,folds,count)
    else:metadata={'minimum_support':minimum,'old_selected_origins':len(folds),'mapped_selected_origins':len(folds),'represented_origins':int(count),'support_projection_applied':False,'origin_measure':'unchanged weighted representatives'}
    return eligible,folds,count,metadata

class GroupWork(parent.GroupWork):
    def __init__(self,*args,modes=MODES,**kwargs):
        if not modes or len(set(modes))!=len(modes)or any(mode not in MODES for mode in modes):raise ValueError('Unique known cache geometries required')
        super().__init__(*args,**kwargs);self.geometry_modes=tuple(modes);self.registered_folds={};self.expected=Counter()
        for horizon in self.horizons:
            for mode in self.geometry_modes:
                _,folds,_,_=folds_for(len(self.x),horizon,mode);self.registered_folds[(horizon,mode)]=tuple(folds);self.expected.update(stop for stop,h,mass in folds);self.expected[len(self.x)]+=1
    def validate_geometry(self,horizon,mode,folds):
        if self.registered_folds.get((horizon,mode))!=tuple(folds):raise ValueError('Cache must explicitly register the exact mapped prefixes/horizons')
    def prefix_fits(self,history,eligible,weights):
        if self.closed:raise ValueError('Closed request-local geometry cache')
        registered={len(self.x)}|{stop for folds in self.registered_folds.values()for stop,h,mass in folds}
        if len(history)not in registered or not set(eligible)<=set(CORE):raise ValueError('Only registered finite41 prefixes may reach financial fitting')
        return super().prefix_fits(history,eligible,weights)

def identity(mode):
    if mode not in MODES:raise ValueError('Known geometry required')
    spec={'parent':parent.identity(CONFIG),'mode':mode,'finite41_ids':CORE,'sources':sources()}
    return 'model_testing_mapped_ratio_geometry_v1_'+hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]

def prepare(history,weights,horizon,training_dates,*,mode='mapped_ratio',seed_replication=0,fit_cache=None,solver=None,width=64):
    if fit_cache is None:
        with GroupWork(history,weights,training_dates,horizons=(horizon,),modes=(mode,),skip_unused_parent_calibration=True)as work:return prepare(history,weights,horizon,training_dates,mode=mode,seed_replication=seed_replication,fit_cache=work,solver=solver,width=width)
    started=time.process_time();x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float);dates=np.asarray(training_dates,dtype='datetime64[D]')
    if x.ndim!=2 or len(x)!=len(dates)or x.shape[1]!=len(w)or not np.isfinite(x).all()or not np.isfinite(w).all()or np.any(w<0)or not np.isclose(w.sum(),1,rtol=0,atol=1e-14)or np.isnat(dates).any()or np.any(np.diff(dates)<=np.timedelta64(0,'D'))or not isinstance(seed_replication,int)or seed_replication<0 or width<1:raise ValueError('Finite exact dated prefix/unit weights and original seed policy required')
    fit_cache.validate_request(x,w,horizon,dates,'half_doubling')
    if fit_cache.skip_unused_parent_calibration is not True:raise ValueError('Qualified current parent calibration mode required')
    eligible,folds,count,geometry=folds_for(len(x),horizon,mode);fit_cache.validate_geometry(horizon,mode,folds);solver=energy.fit if solver is None else solver
    Aall=np.zeros(len(BANK));Ball=np.zeros((len(BANK),len(BANK)));invalid={};trace=[]
    for stop,h,represented in folds:
        past=x[:stop];prepared,failures=fit_cache.prefix_fits(past,[i for i in eligible if i not in invalid],w)
        for i,error in failures.items():invalid[i]={'training_rows':stop,**error}
        valid=[i for i in eligible if i not in invalid]
        if not valid:raise ArithmeticError('No valid finite41 option; no fallback')
        actual=float(parent.portfolio_returns(x[stop:stop+h],w).sum());suffix=''if seed_replication==0 else f':replication={seed_replication}';seed=int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{parent.signature(past)}:{h}:2{suffix}'.encode()).digest()[:4],'little');last=None;cursor=0
        for position,cube in all_chunks(past,w,h,2,seed,valid,prepared,width=width):
            if position!=cursor:raise ArithmeticError('Incomplete historical cursor')
            cursor+=cube.shape[2];last=np.ascontiguousarray(cube[:,:,-1])
        if cursor!=h or last is None:raise ArithmeticError('Missing historical endpoint')
        A,B=coefficients(last,actual);A/=np.sqrt(h);B/=np.sqrt(h);Aall[valid]+=represented*A;Ball[np.ix_(valid,valid)]+=represented*B
        trace.append({'training_rows':stop,'validation_horizon':h,'validation_stop':stop+h,'represented_origins':represented,'training_sha256':parent.signature(past),'training_last_date':str(dates[stop-1]),'validation_last_known_date':str(dates[stop+h-1]),'seed':seed,'eligible_this_fold':valid,'PPW_dependence_fit':prepared.get('PPW')})
    valid=[i for i in eligible if i not in invalid]
    if not valid:raise ArithmeticError('No valid finite41 option; no fallback')
    A=Aall[valid]/count;B=Ball[np.ix_(valid,valid)]/count;fitted=solver(A,B);v=np.asarray(fitted['weights'],dtype=float)
    if v.shape!=(len(valid),)or not np.isfinite(v).all()or np.any(v<0)or abs(float(v.sum())-1)>64*np.finfo(float).eps*len(valid):raise ArithmeticError('Invalid simplex; no fallback')
    vector=np.zeros(len(BANK));vector[valid]=v;needed=list(np.flatnonzero(vector>0));full,failures=fit_cache.prefix_fits(x,needed,w)
    if failures:raise ArithmeticError('Selected current fit failed; no substitution: '+str(failures))
    return {**full,'history_sha256':parent.signature(x),'training_dates_sha256':hashlib.sha256(dates.tobytes()).hexdigest(),'weights':w.copy(),'horizon':horizon,'inner_paths':2,'schedule_rule':'half_doubling','seed_replication':seed_replication,'weights_by_mode':{'flat':vector},'fit':{'flat':fitted},'coefficients':{'flat':{'A':A,'B':B,'valid':valid}},'trace':trace,'invalid_internal_options':invalid,'research_mode':mode,'geometry':geometry,'implementation_sources':sources(),'diagnostics':{'geometry':mode,'geometry_metadata':geometry,'selection_and_refit_cpu_seconds':time.process_time()-started,'internal_criterion':'terminal_CRPS/sqrt(native_validation_horizon)','eligible_menu_size':41,'automatic_model_fallback':False,'no_outer_future_dates_used':True,'no_path_clipping_or_exclusions':True,'skip_unused_parent_calibration':True}}

def prepare_all(history,weights,horizon,training_dates,*,modes=MODES,**kwargs):
    if not isinstance(modes,(tuple,list))or not modes or len(set(modes))!=len(modes)or any(mode not in MODES for mode in modes):raise ValueError('Unique known geometries required')
    if kwargs.get('fit_cache')is None:
        with GroupWork(history,weights,training_dates,horizons=(horizon,),modes=modes,skip_unused_parent_calibration=True)as work:return prepare_all(history,weights,horizon,training_dates,modes=modes,**{**kwargs,'fit_cache':work})
    result={}
    for mode in modes:
        try:result[mode]=prepare(history,weights,horizon,training_dates,mode=mode,**kwargs)
        except Exception as error:result[mode]={'research_mode':mode,'implementation_sources':sources(),'unsupported':{'type':type(error).__name__,'reason':str(error)}}
    return result
class Pool(parent.MenuPool):
    def __init__(self,mode,training_dates,bundle):
        if mode not in MODES or bundle.get('research_mode')!=mode or bundle.get('implementation_sources')!=sources():raise ValueError('Matching source-bound geometry bundle required')
        if bundle.get('unsupported'):raise ArithmeticError('Unsupported geometry fit; no fallback')
        if any(i not in CORE for t in bundle['trace']for i in t['eligible_this_fold'])or any(i not in CORE for i in bundle['coefficients']['flat']['valid'])or any(i not in CORE for i in np.flatnonzero(bundle['weights_by_mode']['flat']>0)):raise ValueError('Forbidden recipe reached finite41 geometry')
        super().__init__(CONFIG,training_dates,bundle);self.model_id=identity(mode)
def build_model(mode,training_dates,bundle):return Pool(mode,training_dates,bundle)
