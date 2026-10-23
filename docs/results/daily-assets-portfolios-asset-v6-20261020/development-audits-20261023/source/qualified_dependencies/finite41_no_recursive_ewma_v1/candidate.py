"""Prespecified flat149: exclude integrated EWMA before fitting, never clip paths."""
import hashlib,json,time
import numpy as np
from candidates.asset_selector_endpoint_fit_reuse_v3 import candidate as parent
from candidates.asset_selector_endpoint_active_qp_unique_v3 import energy as solver_module
from .provenance import source_manifest
CONFIG={'menu':'flat','inner_paths':2,'schedule':'half_doubling'}
FINITE_CONFIG={'menu':'finite_wealth41','inner_paths':2,'schedule':'half_doubling'}
BANK=tuple(parent.BANK);EXCLUDED=tuple(i for i,s in enumerate(BANK)if s.get('family')=='EWMA_recursive');ELIGIBLE=tuple(i for i in range(1,168)if i not in EXCLUDED)
FINITE_ELIGIBLE=tuple(i for i in range(1,168)if i in(1,2,3,166,167)or(BANK[i].get('family')in('constant','EWMA_reverting','SV_RMS')and BANK[i].get('innovations')in('Gaussian','empirical')))
GroupWork=parent.GroupWork;all_chunks=parent.all_chunks;coefficients=parent.compiled_coefficients;schedule=parent.schedule
assert EXCLUDED==tuple(range(148,166))and len(ELIGIBLE)==149 and len(FINITE_ELIGIBLE)==41
def catalogue(menu):
    if menu=='flat':return ELIGIBLE
    if menu=='finite_wealth41':return FINITE_ELIGIBLE
    raise ValueError('Only predeclared feedback149 or finite-wealth41 menu')
sources=source_manifest
def identity(config=CONFIG):
    if config not in(CONFIG,FINITE_CONFIG):raise ValueError('Only frozen N2/half_doubling menu configs')
    spec={'parent':parent.identity(CONFIG),'menu':config['menu'],'retained_ids':catalogue(config['menu']),'implementation_sources':sources()}
    return 'model_testing_prespecified_volatility_menu_v1_'+hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]
def prepare(history,weights,horizon,training_dates,*,inner_paths=2,schedule_rule='half_doubling',seed_replication=0,width=64,fit_cache=None,solver=None,skip_unused_parent_calibration=True,menu='flat'):
    if inner_paths!=2 or schedule_rule!='half_doubling'or skip_unused_parent_calibration is not True:raise ValueError('Frozen N2/half_doubling/qualified leg-only preparation required')
    if fit_cache is None:
        with GroupWork(history,weights,training_dates,horizons=(horizon,),skip_unused_parent_calibration=True)as work:return prepare(history,weights,horizon,training_dates,seed_replication=seed_replication,width=width,fit_cache=work,solver=solver,menu=menu)
    started=time.perf_counter();x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float);dates=np.asarray(training_dates,dtype='datetime64[D]')
    if x.ndim!=2 or len(x)!=len(dates)or x.shape[1]!=len(w)or not np.isfinite(x).all()or not np.isfinite(w).all()or np.any(w<0)or not np.isclose(w.sum(),1,rtol=0,atol=1e-14)or np.isnat(dates).any()or np.any(np.diff(dates)<=np.timedelta64(0,'D'))or not isinstance(seed_replication,int)or seed_replication<0 or width<1:raise ValueError('Finite exact dated prefix, weights and original seed policy required')
    fit_cache.validate_request(x,w,horizon,dates,schedule_rule)
    if fit_cache.skip_unused_parent_calibration is not True:raise ValueError('Cache must use qualified parent-leg preparation')
    fit=solver_module.fit if solver is None else solver
    retained=catalogue(menu);excluded=tuple(i for i in range(1,168)if i not in retained)
    eligible,folds,count=schedule(len(x),horizon,schedule_rule);eligible=[i for i in eligible if i!=0]+([166,167]if 0 in eligible else[]);eligible=[i for i in eligible if i in retained]
    totalA=np.zeros(len(BANK));totalB=np.zeros((len(BANK),len(BANK)));invalid={};trace=[]
    for stop,h,represented in folds:
        past=x[:stop];prepared,failures=fit_cache.prefix_fits(past,[i for i in eligible if i not in invalid],w)
        for i,e in failures.items():invalid[i]={'training_rows':stop,**e}
        valid=[i for i in eligible if i not in invalid]
        if not valid:raise ArithmeticError('No valid retained option; no fallback')
        actual=float(parent.portfolio_returns(x[stop:stop+h],w).sum());suffix=''if seed_replication==0 else f':replication={seed_replication}';seed=int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{parent.signature(past)}:{h}:2{suffix}'.encode()).digest()[:4],'little');last=None;cursor=0
        for position,cube in all_chunks(past,w,h,2,seed,valid,prepared,width=width):
            if position!=cursor:raise ArithmeticError('Historical cursor differs')
            cursor+=cube.shape[2];last=np.ascontiguousarray(cube[:,:,-1])
        if cursor!=h or last is None:raise ArithmeticError('Missing historical endpoint')
        A,B=coefficients(last,actual);A/=np.sqrt(h);B/=np.sqrt(h);totalA[valid]+=represented*A;totalB[np.ix_(valid,valid)]+=represented*B
        trace.append({'training_rows':stop,'validation_horizon':h,'validation_stop':stop+h,'represented_origins':represented,'training_sha256':parent.signature(past),'training_last_date':str(dates[stop-1]),'validation_last_known_date':str(dates[stop+h-1]),'seed':seed,'eligible_this_fold':valid,'PPW_dependence_fit':prepared.get('PPW')})
    valid=[i for i in eligible if i not in invalid]
    if not valid:raise ArithmeticError('No valid retained option; no fallback')
    A=totalA[valid]/count;B=totalB[np.ix_(valid,valid)]/count;fitted=fit(A,B);weights=np.asarray(fitted['weights'],dtype=float)
    if weights.shape!=(len(valid),)or not np.isfinite(weights).all()or np.any(weights<0)or abs(float(weights.sum())-1)>64*np.finfo(float).eps*len(valid):raise ArithmeticError('Solver left certified simplex; no fallback')
    vector=np.zeros(len(BANK));vector[valid]=weights;needed=list(np.flatnonzero(vector>0));full,failures=fit_cache.prefix_fits(x,needed,w)
    if failures:raise ArithmeticError('Selected current fit failed; no substitution: '+str(failures))
    return {**full,'history_sha256':parent.signature(x),'training_dates_sha256':hashlib.sha256(dates.tobytes()).hexdigest(),'weights':w.copy(),'horizon':horizon,'inner_paths':2,'schedule_rule':schedule_rule,'seed_replication':seed_replication,'weights_by_mode':{'flat':vector},'fit':{'flat':fitted},'coefficients':{'flat':{'A':A,'B':B,'valid':valid}},'trace':trace,'invalid_internal_options':invalid,'research_menu':menu,'excluded_menu_ids':excluded,'implementation_sources':sources(),'diagnostics':{'selection_and_refit_seconds':time.perf_counter()-started,'prespecified_menu_exclusions':list(excluded),'eligible_menu_size':len(retained),'internal_criterion':'terminal_CRPS/sqrt(native_validation_horizon)','automatic_model_fallback':False,'no_path_clipping_or_exclusions':True,'no_outer_future_dates_used':True,'partial_feedback_repair_not_finite_wealth_moment_guarantee':menu=='flat','conditional_finite_wealth_moment_catalogue':menu=='finite_wealth41','skip_unused_parent_calibration':True}}
class MenuPool(parent.MenuPool):
    def __init__(self,config,training_dates,bundle=None):
        if config not in(CONFIG,FINITE_CONFIG)or bundle is None:raise ValueError('Explicit source-bound menu bundle required; no inherited automatic fallback')
        retained=catalogue(config['menu']);excluded=tuple(i for i in range(1,168)if i not in retained)
        if bundle.get('research_menu')!=config['menu']or tuple(bundle.get('excluded_menu_ids',()))!=excluded or bundle.get('implementation_sources')!=sources():raise ValueError('Explicit source-bound menu bundle required; no inherited automatic fallback')
        super().__init__(CONFIG,training_dates,bundle);self.research_config=dict(config);self.excluded=excluded;self.model_id=identity(config)
    def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64):
        b=self.bundle
        if np.any(b['weights_by_mode']['flat'][list(self.excluded)]!=0)or any(i in self.excluded for i in b['coefficients']['flat']['valid'])or any(i in self.excluded for t in b['trace']for i in t['eligible_this_fold']):raise ValueError('Forbidden recipe reached selection; no final substitution')
        for start,cube in super().terminal_chunks(history,weights,horizon,paths,context_seed,width=width):
            self.last_selection['research_config']=self.research_config;yield start,cube
