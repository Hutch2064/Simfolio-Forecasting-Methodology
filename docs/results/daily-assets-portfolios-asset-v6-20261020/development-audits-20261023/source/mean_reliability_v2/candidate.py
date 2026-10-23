"""Prespecified zero-offset/reliability factorial; shared original N2 path bank."""
import hashlib,json,time
import numpy as np
from candidates.asset_selector_endpoint_fit_reuse_v3 import candidate as parent
from candidates.asset_selector_endpoint_active_qp_unique_v3 import energy
from .provenance import source_manifest
BANK=tuple(parent.BANK)
ZERO_IDS=tuple(i for i in range(1,168)if BANK[i].get('kind')=='filtered'and BANK[i].get('mean_mode')=='zero')
FINITE_IDS=tuple(i for i in range(1,168)if i in(1,2,3,166,167)or(BANK[i].get('family')in('constant','EWMA_reverting','SV_RMS')and BANK[i].get('innovations')in('Gaussian','empirical')))
MENUS={'full167':tuple(range(1,168)),'full113':tuple(i for i in range(1,168)if i not in ZERO_IDS),'finite41':FINITE_IDS,'finite29':tuple(i for i in FINITE_IDS if i not in ZERO_IDS)}
RULES=('current','mean_estimation_reference')
MODES=tuple(menu+'__'+rule for menu in MENUS for rule in RULES)
CONFIG={'menu':'flat','inner_paths':2,'schedule':'half_doubling'}
GroupWork=parent.GroupWork;schedule=parent.schedule;all_chunks=parent.all_chunks;coefficients=parent.compiled_coefficients;sources=source_manifest
assert len(ZERO_IDS)==54 and tuple(map(len,MENUS.values()))==(167,113,41,29)
def split_mode(mode):
    if mode not in MODES:raise ValueError('Unknown prespecified menu/reliability contrast')
    return mode.split('__')
def reliability(rule,stop,h):
    if rule not in RULES or stop<=0 or h<=0:raise ValueError('Positive training support/native lead and fixed weighting rule required')
    return 1. if rule=='current'else 1./np.sqrt(1.+h/stop)
def identity(mode):
    menu,rule=split_mode(mode)
    spec={'parent':parent.identity(CONFIG),'menu':menu,'retained_ids':MENUS[menu],'weighting':rule,'source_sha256':sources()}
    return 'model_testing_mean_reliability_v2_'+hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]
def prepare_all(history,weights,horizon,training_dates,*,seed_replication=0,fit_cache=None,solver=None,width=64):
    if fit_cache is None:
        with GroupWork(history,weights,training_dates,horizons=(horizon,),skip_unused_parent_calibration=True)as work:return prepare_all(history,weights,horizon,training_dates,seed_replication=seed_replication,fit_cache=work,solver=solver,width=width)
    started=time.process_time();x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float);dates=np.asarray(training_dates,dtype='datetime64[D]')
    if x.ndim!=2 or len(x)!=len(dates)or x.shape[1]!=len(w)or not np.isfinite(x).all()or not np.isfinite(w).all()or np.any(w<0)or not np.isclose(w.sum(),1,rtol=0,atol=1e-14)or np.isnat(dates).any()or np.any(np.diff(dates)<=np.timedelta64(0,'D'))or not isinstance(seed_replication,int)or seed_replication<0 or width<1:raise ValueError('Finite exact dated prefix/unit weights and original seed policy required')
    fit_cache.validate_request(x,w,horizon,dates,'half_doubling')
    if fit_cache.skip_unused_parent_calibration is not True:raise ValueError('Qualified current parent calibration mode required')
    solver=energy.fit if solver is None else solver
    eligible,folds,count=schedule(len(x),horizon,'half_doubling');eligible=[i for i in eligible if i!=0]+([166,167]if 0 in eligible else[])
    totalA={rule:np.zeros(len(BANK))for rule in RULES};totalB={rule:np.zeros((len(BANK),len(BANK)))for rule in RULES};mass={rule:0. for rule in RULES};invalid={};trace=[]
    for stop,h,represented in folds:
        past=x[:stop];prepared,failures=fit_cache.prefix_fits(past,[i for i in eligible if i not in invalid],w)
        for i,error in failures.items():invalid[i]={'training_rows':stop,**error}
        valid=[i for i in eligible if i not in invalid]
        if not valid:raise ArithmeticError('No valid historical option; no fallback')
        actual=float(parent.portfolio_returns(x[stop:stop+h],w).sum());suffix=''if seed_replication==0 else f':replication={seed_replication}';seed=int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{parent.signature(past)}:{h}:2{suffix}'.encode()).digest()[:4],'little');last=None;cursor=0
        for position,cube in all_chunks(past,w,h,2,seed,valid,prepared,width=width):
            if position!=cursor:raise ArithmeticError('Incomplete historical cursor')
            cursor+=cube.shape[2];last=np.ascontiguousarray(cube[:,:,-1])
        if cursor!=h or last is None:raise ArithmeticError('Missing historical endpoint')
        A,B=coefficients(last,actual);A/=np.sqrt(h);B/=np.sqrt(h)
        for rule in RULES:
            factor=reliability(rule,stop,h)
            # Preserve the original current arithmetic exactly, including multiplication order.
            weightedA=A if rule=='current'else factor*A;weightedB=B if rule=='current'else factor*B
            totalA[rule][valid]+=represented*weightedA;totalB[rule][np.ix_(valid,valid)]+=represented*weightedB;mass[rule]+=represented*factor
        trace.append({'training_rows':stop,'validation_horizon':h,'validation_stop':stop+h,'represented_origins':represented,'training_sha256':parent.signature(past),'training_last_date':str(dates[stop-1]),'validation_last_known_date':str(dates[stop+h-1]),'seed':seed,'eligible_this_fold':valid,'PPW_dependence_fit':prepared.get('PPW'),'reliability':{r:reliability(r,stop,h)for r in RULES}})
    learned={};fits={};coefs={};solver_failures={}
    for mode in MODES:
        menu,rule=split_mode(mode);valid=[i for i in eligible if i in MENUS[menu]and i not in invalid]
        denominator=count if rule=='current'else mass[rule]
        A=totalA[rule][valid]/denominator;B=totalB[rule][np.ix_(valid,valid)]/denominator
        coefs[mode]={'A':A,'B':B,'valid':valid,'represented_mass':float(denominator)}
        try:
            if not valid:raise ArithmeticError('No valid historical option; no fallback')
            fitted=solver(A,B);v=np.asarray(fitted['weights'],dtype=float)
            if v.shape!=(len(valid),)or not np.isfinite(v).all()or np.any(v<0)or abs(float(v.sum())-1)>64*np.finfo(float).eps*len(valid):raise ArithmeticError('Invalid simplex; no fallback')
            vector=np.zeros(len(BANK));vector[valid]=v;learned[mode]=vector;fits[mode]=fitted
        except Exception as error:
            learned[mode]=None;fits[mode]=None;solver_failures[mode]={'type':type(error).__name__,'reason':str(error)}
    needed=sorted({int(i)for v in learned.values()if v is not None for i in np.flatnonzero(v>0)})
    if needed:
        try:full,failures=fit_cache.prefix_fits(x,needed,w)
        except Exception as error:
            full={};failures={i:{'type':type(error).__name__,'reason':str(error)}for i in needed}
    else:full={};failures={}
    selected_failures={mode:{i:e for i,e in failures.items()if learned[mode]is not None and learned[mode][i]>0}for mode in MODES}
    manifest=sources();common={**full,'history_sha256':parent.signature(x),'training_dates_sha256':hashlib.sha256(dates.tobytes()).hexdigest(),'weights':w.copy(),'horizon':horizon,'inner_paths':2,'schedule_rule':'half_doubling','seed_replication':seed_replication,'implementation_sources':manifest}
    return {mode:{**common,'research_mode':mode,'unsupported_solver':solver_failures.get(mode,{}),'unsupported_selected_current':selected_failures[mode],'weights_by_mode':{'flat':learned[mode]},'fit':{'flat':fits[mode]},'coefficients':{'flat':coefs[mode]},'trace':[{**t,'eligible_this_fold':[i for i in t['eligible_this_fold']if i in MENUS[split_mode(mode)[0]]]}for t in trace],'invalid_internal_options':{i:e for i,e in invalid.items()if i in MENUS[split_mode(mode)[0]]},'diagnostics':{'factorial_menu':split_mode(mode)[0],'internal_weight_rule':split_mode(mode)[1],'eligible_menu_size':len(MENUS[split_mode(mode)[0]]),'selection_and_shared_refit_cpu_seconds':time.process_time()-started,'shared_historical_path_bank':True,'internal_criterion':'terminal_CRPS/sqrt(native_validation_horizon)','weighted_represented_mass':coefs[mode]['represented_mass'],'reference_weight_formula':'1/sqrt(1+h/stop)','reference_weight_is_benchmark_heuristic_not_universal_uncertainty':True,'no_outer_future_dates_used':True,'automatic_model_fallback':False,'no_path_clipping_or_exclusions':True,'skip_unused_parent_calibration':True}}for mode in MODES}
class MenuPool(parent.MenuPool):
    def __init__(self,mode,training_dates,bundle):
        menu,rule=split_mode(mode)
        if bundle.get('research_mode')!=mode or bundle.get('implementation_sources')!=sources():raise ValueError('Matching source-bound factorial bundle required')
        if bundle.get('unsupported_solver')or bundle.get('unsupported_selected_current'):raise ArithmeticError('Unsupported factorial mode; no substitution: '+str({'solver':bundle.get('unsupported_solver'),'selected_current':bundle.get('unsupported_selected_current')}))
        if any(i not in MENUS[menu]for i in bundle['coefficients']['flat']['valid'])or any(i not in MENUS[menu]for t in bundle['trace']for i in t['eligible_this_fold'])or any(i not in MENUS[menu]for i in np.flatnonzero(bundle['weights_by_mode']['flat']>0)):raise ValueError('Excluded option reached fitted menu; no substitution')
        super().__init__(CONFIG,training_dates,bundle);self.mode=mode;self.model_id=identity(mode)
    def terminal_chunks(self,*args,**kwargs):
        for start,values in super().terminal_chunks(*args,**kwargs):
            self.last_selection['factorial_mode']=self.mode;yield start,values
def build_model(mode,training_dates,bundle):return MenuPool(mode,training_dates,bundle)
