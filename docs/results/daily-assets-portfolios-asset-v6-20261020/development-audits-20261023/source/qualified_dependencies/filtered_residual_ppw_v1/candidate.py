"""Paired flat167 versus filtered residual PPW; original paths stay verbatim."""
import hashlib,json,time
import numpy as np
from candidates.asset_selector_endpoint_fit_reuse_v3 import candidate as leader
from candidates.asset_selector_endpoint_active_qp_unique_v3.energy import fit as current_solver
from candidates.asset_selector_menu_components.bank import model as old_model
from candidates.asset_selector_ppw.blocks import fit_lengths,signature as ppw_signature
from .bank import BANK,NEW_IDS,EGARCH_IDS
from candidates.asset_gaussian_egarch_component_v1 import candidate as egarch
from .paths import group_chunks
MODES=('endpoint_n2','filtered_PPW','gaussian_EGARCH','filtered_PPW_EGARCH')
CONFIG={'menu':'flat','inner_paths':2,'schedule':'half_doubling'}

class GroupWork(leader.GroupWork):
    def prefix_fits(self,history,eligible,weights):
        original=[i for i in eligible if i<168]
        for i in eligible:
            if i in NEW_IDS and BANK[i]['base_iid_index'] not in original:original.append(BANK[i]['base_iid_index'])
        prepared,failures=super().prefix_fits(history,original,weights);prepared={**prepared,'residual_PPW':{}};invalid={i:e for i,e in failures.items() if i in eligible}
        for family in sorted({BANK[i]['family'] for i in eligible if i in NEW_IDS}):
            ids=[i for i in eligible if i in NEW_IDS and BANK[i]['family']==family]
            try:
                if family not in prepared['families']:raise ArithmeticError('Underlying filtered family unavailable')
                pool=prepared['families'][family]['filter']['pool'];key=(len(history),'residual_PPW',ppw_signature(pool)+ppw_signature(weights))
                result=self._get(key,lambda:fit_lengths(pool,weights))
                prepared['residual_PPW'][family]={**result,'family':family,'raw_history_sha256':leader.signature(history),'residual_pool_sha256':ppw_signature(pool),'innovation_transform':'Existing same-prefix standardized filter pool kept verbatim; centering only inside PPW estimator features'}
            except Exception as error:
                for i in ids:invalid[i]={'type':type(error).__name__,'reason':str(error)}
        if any(i in EGARCH_IDS for i in eligible):
            try:
                batch=next(iter(prepared['families'].values()))['mean'] if prepared['families'] else prepared.get('parent',{}).get('mean')
                prepared['EGARCH']=self._get((len(history),'Gaussian_EGARCH',''),lambda:egarch.prepare(history,base=None if batch is None else {'mean':batch}))
            except Exception as error:
                for i in EGARCH_IDS:
                    if i in eligible:invalid[i]={'type':type(error).__name__,'reason':str(error)}
        return prepared,invalid

def effective_length(prepared,index):
    spec=BANK[index];length=prepared['residual_PPW'][spec['family']]['lengths'][spec['sampling']]
    return length if spec['sampling']=='stationary' else int(np.floor(length+.5))

def new_model(index,prepared):
    if index<170:return old_model(index,prepared)
    if index in EGARCH_IDS:return egarch.GaussianEGARCH({k:BANK[index][k] for k in ('mean_mode','mean_uncertainty')},prepared['EGARCH'])
    return ResidualOption(index,prepared)

class ResidualOption:
    def __init__(self,index,prepared):self.index=index;self.prepared=prepared
    def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64):
        spec=BANK[self.index];bundle=self.prepared['families'][spec['family']];dependence=self.prepared['residual_PPW'][spec['family']]
        if dependence['raw_history_sha256']!=leader.signature(history) or dependence['residual_pool_sha256']!=ppw_signature(bundle['filter']['pool']) or dependence['weights_sha256']!=ppw_signature(weights):raise ValueError('Residual PPW fit differs from dated family prefix')
        for start,values in group_chunks(history,weights,horizon,paths,context_seed,[self.index],bundle,width=width,kind=spec['sampling'],length=dependence['lengths'][spec['sampling']]):yield start,values[0]

def all_chunks(history,weights,horizon,paths,seed,valid,prepared,*,width=64):
    positions={i:j for j,i in enumerate(valid)};original=[i for i in valid if i<170];groups={}
    generators=[]
    if original:generators.append((original,leader.all_chunks(history,weights,horizon,paths,seed,original,prepared,width=width)))
    draw_cache={'arrays':{},'bytes':0,'limit':8*1024**2}
    for i in valid:
        if i in NEW_IDS:groups.setdefault((BANK[i]['family'],BANK[i]['sampling']),[]).append(i)
    for i in valid:
        if i in EGARCH_IDS:
            def singleton(index=i):
                for position,values in new_model(index,prepared).terminal_chunks(history,weights,horizon,paths,seed,width=width):yield position,values[None,:,:]
            generators.append(([i],singleton()))
    for (family,kind),ids in groups.items():generators.append((ids,group_chunks(history,weights,horizon,paths,seed,ids,prepared['families'][family],width=width,kind=kind,length=prepared['residual_PPW'][family]['lengths'][kind],draw_cache=draw_cache)))
    for start in range(0,horizon,width):
        count=min(width,horizon-start);cube=np.empty((len(valid),paths,count))
        for ids,generator in generators:
            position,values=next(generator)
            if position!=start or values.shape!=(len(ids),paths,count) or not np.isfinite(values).all():raise ArithmeticError('Incomplete residual block bank')
            for j,i in enumerate(ids):cube[positions[i]]=values[j]
        yield start,cube

def prepare_all(history,weights,horizon,training_dates,*,seed_start=0,fit_cache=None,solver=None,include_egarch=True):
    if seed_start not in (0,3):raise ValueError('Predeclared production seed0 or diagnostic seed3 required')
    solver=current_solver if solver is None else solver
    if fit_cache is None:
        with GroupWork(history,weights,training_dates,horizons=(horizon,),skip_unused_parent_calibration=True) as work:return prepare_all(history,weights,horizon,training_dates,seed_start=seed_start,fit_cache=work,solver=solver,include_egarch=include_egarch)
    x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float);dates=np.asarray(training_dates,dtype='datetime64[D]');fit_cache.validate_request(x,w,horizon,dates,'half_doubling')
    if x.ndim!=2 or len(x)!=len(dates) or x.shape[1]!=len(w) or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w<0) or not np.isclose(w.sum(),1,rtol=0,atol=1e-14) or np.isnat(dates).any() or np.any(np.diff(dates)<=np.timedelta64(0,'D')):raise ValueError('Finite exact dated joint prefix required')
    eligible,folds,full_count=leader.schedule(len(x),horizon,'half_doubling');old=[i for i in eligible if i!=0]+([166,167] if 0 in eligible else[]);ppw=[] if horizon==1 else [i for i in NEW_IDS if BANK[i]['base_iid_index'] in old];extra=list(EGARCH_IDS) if include_egarch and any(BANK[i]['kind']=='filtered' for i in old) else [];eligible=old+ppw+extra
    invalid={};trace=[];total_A=np.zeros(len(BANK));total_B=np.zeros((len(BANK),len(BANK)));started=time.process_time()
    for stop,h,represented in folds:
        past=x[:stop];prepared,failures=fit_cache.prefix_fits(past,[i for i in eligible if i not in invalid],w)
        for i,error in failures.items():invalid[i]={'training_rows':stop,**error}
        valid=[i for i in eligible if i not in invalid];actual=float(leader.portfolio_returns(x[stop:stop+h],w).sum());suffix='' if seed_start==0 else f':replication={seed_start}';seed=int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{leader.signature(past)}:{h}:2{suffix}'.encode()).digest()[:4],'little')
        last=None;cursor=0
        for position,cube in all_chunks(past,w,h,2,seed,valid,prepared):
            if position!=cursor:raise ArithmeticError('Historical block path cursor differs')
            cursor+=cube.shape[2];last=np.ascontiguousarray(cube[:,:,-1])
        if cursor!=h or last is None:raise ArithmeticError('Missing historical endpoint')
        A,B=leader.compiled_coefficients(last,actual);A/=np.sqrt(h);B/=np.sqrt(h);total_A[valid]+=represented*A;total_B[np.ix_(valid,valid)]+=represented*B
        trace.append({'training_rows':stop,'validation_horizon':h,'validation_stop':stop+h,'represented_origins':represented,'training_sha256':leader.signature(past),'training_last_date':str(dates[stop-1]),'validation_last_known_date':str(dates[stop+h-1]),'seed':seed,'eligible_this_fold':valid,'residual_PPW':prepared['residual_PPW'],'Gaussian_EGARCH':None if 'EGARCH' not in prepared else {'parameters':prepared['EGARCH']['parameters'].tolist(),'reports':[{k:v for k,v in f.items() if k not in ('log_variance','standardized_residuals')} for f in prepared['EGARCH']['fits']]}})
    learned={};fits={};coefficients={}
    for mode,ids in (('endpoint_n2',old),('filtered_PPW',old+ppw),('gaussian_EGARCH',old+extra),('filtered_PPW_EGARCH',eligible)):
        duplicate='endpoint_n2' if (horizon==1 and mode=='filtered_PPW') or (not include_egarch and mode=='gaussian_EGARCH') else 'filtered_PPW' if not include_egarch and mode=='filtered_PPW_EGARCH' else 'gaussian_EGARCH' if horizon==1 and mode=='filtered_PPW_EGARCH' else None
        if duplicate is not None:
            learned[mode]=learned[duplicate];fits[mode]=fits[duplicate];coefficients[mode]=coefficients[duplicate];continue
        valid=[i for i in ids if i not in invalid];A=total_A[valid]/full_count;B=total_B[np.ix_(valid,valid)]/full_count;fitted=solver(A,B)
        vector=np.zeros(170 if mode=='endpoint_n2' or (mode=='filtered_PPW' and horizon==1) else len(BANK));vector[valid]=fitted['weights'];learned[mode]=vector;fits[mode]=fitted;coefficients[mode]={'A':A,'B':B,'valid':valid}
    needed=sorted({int(i) for vector in learned.values() for i in np.flatnonzero(vector>0)});full,failures=fit_cache.prefix_fits(x,needed,w)
    if failures:raise ArithmeticError(f'Selected current financial/dependence fit failed: {failures}')
    canonical={mode:vector.copy() for mode,vector in learned.items()};aliases={mode:{} for mode in MODES}
    for mode in ('filtered_PPW','filtered_PPW_EGARCH'):
        for i in np.flatnonzero(canonical[mode]>0):
            if i in NEW_IDS and effective_length(full,int(i))==1:
                base=BANK[i]['base_iid_index'];canonical[mode][base]+=canonical[mode][i];canonical[mode][i]=0.;aliases[mode][int(i)]=base
    common={**full,'history_sha256':leader.signature(x),'training_dates_sha256':hashlib.sha256(dates.tobytes()).hexdigest(),'weights':w.copy(),'horizon':horizon,'inner_paths':2,'schedule_rule':'half_doubling','seed_replication':seed_start,'trace':trace}
    return {mode:{**common,'trace':([{**{k:v for k,v in t.items() if k not in ('residual_PPW','Gaussian_EGARCH')},'eligible_this_fold':[i for i in t['eligible_this_fold'] if i<170 or (mode=='gaussian_EGARCH' and i in EGARCH_IDS)],'PPW_dependence_fit':None} for t in trace] if mode in ('endpoint_n2','gaussian_EGARCH') else trace),'weights_by_mode':{'flat':vector},'fit':{'flat':fits[mode]},'coefficients':{'flat':coefficients[mode]},'invalid_internal_options':{i:e for i,e in invalid.items() if i in (old+ppw if mode=='filtered_PPW' else old+extra if mode=='gaussian_EGARCH' else eligible if mode=='filtered_PPW_EGARCH' else old)},'diagnostics':{'residual_PPW':mode in ('filtered_PPW','filtered_PPW_EGARCH'),'Gaussian_EGARCH':mode in ('gaussian_EGARCH','filtered_PPW_EGARCH'),'H1_PPW_excluded_before_QP':horizon==1,'selection_cpu_seconds':time.process_time()-started,'final_IID_aliases':aliases[mode],'training_QP_weights_are_before_final_law_aliasing':True,'no_outer_future_dates_used':True,'automatic_model_fallback':False,'unsupported_new_components':(['Gaussian_EGARCH'] if mode in ('gaussian_EGARCH','filtered_PPW_EGARCH') and not any(i in EGARCH_IDS for i in coefficients[mode]['valid']) else [])+(['residual_PPW'] if mode in ('filtered_PPW','filtered_PPW_EGARCH') and horizon>1 and not any(i in NEW_IDS for i in coefficients[mode]['valid']) else [])}} for mode,vector in canonical.items()}

def identity(mode,seed_start=0):
    if mode not in MODES or seed_start not in (0,3):raise ValueError('Unknown predeclared mode/seed')
    if mode=='endpoint_n2' and seed_start==0:return leader.identity(CONFIG)
    key=json.dumps({'base':leader.identity(CONFIG),'mode':mode,'seed':seed_start,'residual_PPW_bank':list(NEW_IDS) if mode in ('filtered_PPW','filtered_PPW_EGARCH') else [],'Gaussian_EGARCH_bank':list(EGARCH_IDS) if mode in ('gaussian_EGARCH','filtered_PPW_EGARCH') else [],'law':'PPW on same-prefix standardized joint empirical residual rows; no conditional-martingale claim'},sort_keys=True)
    return 'model_testing_filtered_joint_residual_PPW_v1_'+hashlib.sha256(key.encode()).hexdigest()[:16]

class Pool(leader.MenuPool):
    def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64):
        b=self.bundle
        if b['history_sha256']!=leader.signature(history) or b['horizon']!=horizon or not np.array_equal(b['weights'],weights) or b['training_dates_sha256']!=hashlib.sha256(self.dates.tobytes()).hexdigest():raise ValueError('Forecast differs from fitted dated prefix')
        learned=b['weights_by_mode']['flat'];counts=leader.energy.even_allocation(learned,paths);generators=[(i,new_model(i,b).terminal_chunks(history,weights,horizon,int(n),context_seed,width=width)) for i,n in enumerate(counts) if n]
        self.last_selection={'config':self.config,'ideal_weights':learned.tolist(),'path_counts':counts.tolist(),'invalid_internal_options':b['invalid_internal_options'],**b['diagnostics']}
        for start in range(0,horizon,width):
            pieces=[]
            for i,generator in generators:
                position,values=next(generator)
                if position!=start or values.shape!=(counts[i],min(width,horizon-start)):raise ArithmeticError('Incomplete selected residual block path')
                pieces.append(values)
            values=np.concatenate(pieces)
            if not np.isfinite(values).all():raise ArithmeticError('Nonfinite residual forecast; no path exclusions')
            yield start,values

def build_model(mode,training_dates,bundle,seed_start=0):
    if bundle['seed_replication']!=seed_start:raise ValueError('Selected mode/seed differs from fitted request')
    cls=leader.MenuPool if mode=='endpoint_n2' or (mode=='filtered_PPW' and bundle['horizon']==1) else Pool
    model=cls(CONFIG,training_dates,bundle);model.model_id=identity(mode,seed_start);return model
