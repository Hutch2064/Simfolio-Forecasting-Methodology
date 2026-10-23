"""Finite41 plus36 existing deterministic-variance joint residual PPW derivatives."""
import hashlib,json,time
import numpy as np
from candidates.asset_filtered_residual_ppw_v1 import candidate as ppw
from candidates.asset_selector_endpoint_active_qp_unique_v3 import energy
from .provenance import source_manifest
BANK=tuple(ppw.BANK);CONFIG=dict(ppw.CONFIG)
CORE=tuple([1,2,3]+list(range(4,16))+list(range(22,34))+list(range(112,124))+[166,167])
DERIVATIVES=tuple(i for i in ppw.NEW_IDS if BANK[i]['family']in('constant','EWMA_reverting','SV_RMS'))
MENUS={'finite41':CORE,'finite77':CORE+DERIVATIVES};MODES=tuple(MENUS)
GroupWork=ppw.GroupWork;schedule=ppw.leader.schedule;all_chunks=ppw.all_chunks;coefficients=ppw.leader.compiled_coefficients;sources=source_manifest
assert len(CORE)==41 and DERIVATIVES==tuple(list(range(170,194))+list(range(242,254)))and len(MENUS['finite77'])==77
assert all(BANK[i]['innovations']=='empirical'and BANK[i]['base_iid_index']in CORE for i in DERIVATIVES)
def identity(mode,seed_replication=0):
    if mode not in MODES or not isinstance(seed_replication,int)or seed_replication<0:raise ValueError('Known menu and original nonnegative seed replication required')
    spec={'base':ppw.leader.identity(CONFIG),'mode':mode,'ids':MENUS[mode],'seed_replication':seed_replication,'source_sha256':sources()}
    return 'model_testing_finite77_v1_'+hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]
def prepare_all(history,weights,horizon,training_dates,*,seed_replication=0,fit_cache=None,solver=None,width=64):
    if fit_cache is None:
        with GroupWork(history,weights,training_dates,horizons=(horizon,),skip_unused_parent_calibration=True)as work:return prepare_all(history,weights,horizon,training_dates,seed_replication=seed_replication,fit_cache=work,solver=solver,width=width)
    started=time.process_time();x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float);dates=np.asarray(training_dates,dtype='datetime64[D]')
    if x.ndim!=2 or len(x)!=len(dates)or x.shape[1]!=len(w)or not np.isfinite(x).all()or not np.isfinite(w).all()or np.any(w<0)or not np.isclose(w.sum(),1,rtol=0,atol=1e-14)or np.isnat(dates).any()or np.any(np.diff(dates)<=np.timedelta64(0,'D'))or not isinstance(seed_replication,int)or seed_replication<0 or width<1:raise ValueError('Finite exact dated prefix/unit weights and original seed policy required')
    fit_cache.validate_request(x,w,horizon,dates,'half_doubling')
    if fit_cache.skip_unused_parent_calibration is not True:raise ValueError('Qualified current parent calibration mode required')
    solver=energy.fit if solver is None else solver
    eligible,folds,count=schedule(len(x),horizon,'half_doubling');old=[i for i in eligible if i!=0]+([166,167]if 0 in eligible else[]);old=[i for i in old if i in CORE]
    extra=[]if horizon==1 else[i for i in DERIVATIVES if BANK[i]['base_iid_index']in old];eligible=old+extra
    Aall=np.zeros(len(BANK));Ball=np.zeros((len(BANK),len(BANK)));invalid={};trace=[]
    for stop,h,represented in folds:
        past=x[:stop];prepared,failures=fit_cache.prefix_fits(past,[i for i in eligible if i not in invalid],w)
        for i,error in failures.items():invalid[i]={'training_rows':stop,**error}
        valid=[i for i in eligible if i not in invalid]
        if not valid:raise ArithmeticError('No valid historical option; no fallback')
        actual=float(ppw.leader.portfolio_returns(x[stop:stop+h],w).sum());suffix=''if seed_replication==0 else f':replication={seed_replication}';seed=int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{ppw.leader.signature(past)}:{h}:2{suffix}'.encode()).digest()[:4],'little');cursor=0;last=None
        for position,cube in all_chunks(past,w,h,2,seed,valid,prepared,width=width):
            if position!=cursor:raise ArithmeticError('Incomplete historical cursor')
            cursor+=cube.shape[2];last=np.ascontiguousarray(cube[:,:,-1])
        if cursor!=h or last is None:raise ArithmeticError('Missing historical endpoint')
        A,B=coefficients(last,actual);A/=np.sqrt(h);B/=np.sqrt(h);Aall[valid]+=represented*A;Ball[np.ix_(valid,valid)]+=represented*B
        trace.append({'training_rows':stop,'validation_horizon':h,'validation_stop':stop+h,'represented_origins':represented,'training_sha256':ppw.leader.signature(past),'training_last_date':str(dates[stop-1]),'validation_last_known_date':str(dates[stop+h-1]),'seed':seed,'eligible_this_fold':valid,'PPW_dependence_fit':None,'residual_PPW':prepared.get('residual_PPW',{})})
    learned={};fits={};coefs={};solver_failures={}
    for mode in MODES:
        valid=[i for i in eligible if i in MENUS[mode]and i not in invalid];A=Aall[valid]/count;B=Ball[np.ix_(valid,valid)]/count;coefs[mode]={'A':A,'B':B,'valid':valid}
        try:
            if not valid:raise ArithmeticError('No valid historical option; no fallback')
            if mode=='finite77'and horizon==1:
                if learned['finite41']is None:raise ArithmeticError('Identical H1 control solver unavailable')
                learned[mode]=learned['finite41'].copy();fits[mode]=fits['finite41'];continue
            fitted=solver(A,B);v=np.asarray(fitted['weights'],dtype=float)
            if v.shape!=(len(valid),)or not np.isfinite(v).all()or np.any(v<0)or abs(float(v.sum())-1)>64*np.finfo(float).eps*len(valid):raise ArithmeticError('Invalid simplex; no fallback')
            vector=np.zeros(170 if mode=='finite41'else len(BANK));vector[valid]=v;learned[mode]=vector;fits[mode]=fitted
        except Exception as error:learned[mode]=None;fits[mode]=None;solver_failures[mode]={'type':type(error).__name__,'reason':str(error)}
    needed=sorted({int(i)for v in learned.values()if v is not None for i in np.flatnonzero(v>0)})
    if needed:
        try:full,failures=fit_cache.prefix_fits(x,needed,w)
        except Exception as error:full={};failures={i:{'type':type(error).__name__,'reason':str(error)}for i in needed}
    else:full={};failures={}
    selected_failures={mode:{i:e for i,e in failures.items()if learned[mode]is not None and i<len(learned[mode])and learned[mode][i]>0}for mode in MODES};aliases={mode:{}for mode in MODES}
    for mode,v in learned.items():
        if v is None or selected_failures[mode]:continue
        for i in np.flatnonzero(v>0):
            if i in DERIVATIVES and ppw.effective_length(full,int(i))==1:
                base=BANK[i]['base_iid_index'];v[base]+=v[i];v[i]=0.;aliases[mode][int(i)]=base
    manifest=sources();common={**full,'history_sha256':ppw.leader.signature(x),'training_dates_sha256':hashlib.sha256(dates.tobytes()).hexdigest(),'weights':w.copy(),'horizon':horizon,'inner_paths':2,'schedule_rule':'half_doubling','seed_replication':seed_replication,'implementation_sources':manifest}
    return {mode:{**common,'research_mode':mode,'unsupported_solver':solver_failures.get(mode,{}),'unsupported_selected_current':selected_failures[mode],'weights_by_mode':{'flat':learned[mode]},'fit':{'flat':fits[mode]},'coefficients':{'flat':coefs[mode]},'trace':[{**t,'eligible_this_fold':[i for i in t['eligible_this_fold']if i in MENUS[mode]]}for t in trace],'invalid_internal_options':{i:e for i,e in invalid.items()if i in MENUS[mode]},'diagnostics':{'finite_catalogue':mode,'semantic_menu_size':len(MENUS[mode]),'effective_menu_size':len(old)if horizon==1 else len(old)+(len(extra)if mode=='finite77'else 0),'selection_and_shared_refit_cpu_seconds':time.process_time()-started,'H1_PPW_excluded_before_QP':horizon==1,'final_IID_aliases':aliases[mode],'training_QP_weights_are_before_final_law_aliasing':True,'no_outer_future_dates_used':True,'automatic_model_fallback':False,'no_path_clipping_or_exclusions':True,'skip_unused_parent_calibration':True,'moment_scope':'conditional finite fitted states, finite horizon, long-only; no uniform tail magnitude guarantee'}}for mode in MODES}
class Pool(ppw.Pool):
    def __init__(self,mode,training_dates,bundle):
        if mode not in MODES or bundle.get('research_mode')!=mode or bundle.get('implementation_sources')!=sources():raise ValueError('Matching source-bound finite menu required')
        if bundle.get('unsupported_solver')or bundle.get('unsupported_selected_current'):raise ArithmeticError('Unsupported finite menu; no substitution')
        if any(i not in MENUS[mode]for i in bundle['coefficients']['flat']['valid'])or any(i not in MENUS[mode]for t in bundle['trace']for i in t['eligible_this_fold'])or any(i not in MENUS[mode]for i in np.flatnonzero(bundle['weights_by_mode']['flat']>0)):raise ValueError('Forbidden recipe reached finite menu')
        if bundle['horizon']==1 and any(i in DERIVATIVES for i in bundle['coefficients']['flat']['valid']):raise ValueError('H1 derivatives must be excluded before QP')
        super().__init__(CONFIG,training_dates,bundle);self.mode=mode;self.model_id=identity(mode,bundle['seed_replication'])
def build_model(mode,training_dates,bundle):return Pool(mode,training_dates,bundle)
