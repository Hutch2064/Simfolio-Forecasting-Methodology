"""Aligned batch V2: identical pair arithmetic, compiled outer pair dispatch."""
import hashlib
import json
import time
import numpy as np
from candidates.asset_selector_score_alignment import candidate as reference
from .coefficients import coefficients
control=reference.control
CONFIG={'criterion':'uniform_days','scale':'train_SD_sqrt_lead','learner':'batch',
        'inner_paths':2,'schedule':'half_doubling'}
MODE='uniform_days:train_SD_sqrt_lead:batch'
AXIS=('uniform_days','train_SD_sqrt_lead')


def identity(config):
    return 'research_aligned_compiled_pair_loop_v2_'+hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]


def prepare(history, weights, horizon, training_dates, *, inner_paths=2,
            schedule_rule='half_doubling', seed_replication=0, width=64):
    started=time.perf_counter();x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float)
    dates=np.asarray(training_dates,dtype='datetime64[D]')
    if (x.ndim!=2 or len(x)!=len(dates) or x.shape[1]!=len(w)
            or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w<0)
            or not np.isclose(w.sum(),1.,rtol=0,atol=1e-14) or np.isnat(dates).any()
            or np.any(np.diff(dates)<=np.timedelta64(0,'D')) or inner_paths<2
            or inner_paths%2 or width<1 or not isinstance(seed_replication,int) or seed_replication<0):
        raise ValueError('Finite dated joint history, normalized weights and even inner paths required')
    eligible,folds,full_count=control.schedule(len(x),horizon,schedule_rule)
    eligible=[i for i in eligible if i!=0]+([166,167]if 0 in eligible else[])
    invalid={};trace=[];valid_before=None;total_A=None;total_B=None
    for stop,h,represented in folds:
        past=x[:stop];prepared,failures=control.prefix_fits(past,[i for i in eligible if i not in invalid],w)
        for i,error in failures.items():invalid[i]={'training_rows':stop,**error}
        valid=[i for i in eligible if i not in invalid]
        if not valid or (valid_before is not None and valid!=valid_before):
            raise ArithmeticError('Stable nonempty recipe universe required across dated folds')
        if total_A is None:total_A=np.zeros(len(valid));total_B=np.zeros((len(valid),len(valid)))
        valid_before=valid
        returns=control.portfolio_returns(x[stop:stop+h],w);observed=np.cumsum(returns)
        observed[-1]=float(returns.sum())
        sd=float(np.std(control.portfolio_returns(past,w),ddof=1))
        if not np.isfinite(sd) or sd<=0:raise ArithmeticError('Positive finite fold-prefix SD required')
        suffix=''if seed_replication==0 else f':replication={seed_replication}'
        seed=int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{control.signature(past)}:{h}:{inner_paths}{suffix}'.encode()).digest()[:4],'little')
        A=np.zeros(len(valid));B=np.zeros((len(valid),len(valid)));cursor=0
        for start,cube in control.all_chunks(past,w,h,inner_paths,seed,valid,prepared,width=width):
            if start!=cursor or cube.shape[:2]!=(len(valid),inner_paths) or not np.isfinite(cube).all():
                raise ArithmeticError('Incomplete/nonfinite joint historical path cube')
            for offset in range(cube.shape[2]):
                lead=start+offset+1
                a,b=coefficients(np.ascontiguousarray(cube[:,:,offset]),observed[lead-1])
                divisor=reference.denominator('train_SD_sqrt_lead',sd,lead)
                with np.errstate(over='raise',under='raise',invalid='raise',divide='raise'):
                    try:
                        # Preserve the all-work reference's chronological arithmetic.
                        A+=1.*(a/divisor);B+=1.*(b/divisor)
                    except FloatingPointError as error:
                        raise ArithmeticError('Unrepresentable aligned coefficient; no clipping/fallback')from error
            cursor+=cube.shape[2]
        if cursor!=h:raise ArithmeticError('Missing native validation leads')
        total_A+=represented*A;total_B+=represented*B
        trace.append({'training_rows':stop,'validation_horizon':h,'validation_stop':stop+h,
            'represented_origins':represented,'training_sha256':control.signature(past),
            'training_last_date':str(dates[stop-1]),'validation_last_known_date':str(dates[stop+h-1]),
            'seed':seed,'eligible_this_fold':valid,'PPW_dependence_fit':prepared.get('PPW'),
            'training_portfolio_SD_ddof1':sd})
    if sum(t['represented_origins']for t in trace)!=full_count:raise ArithmeticError('Origin mass differs')
    total_A/=full_count;total_B/=full_count;fit=control.energy.fit(total_A,total_B)
    learned=np.zeros(len(control.BANK));learned[valid]=fit['weights']
    # Preserve the complete reference's full-prefix fit context, not a subset.
    # Its progressive prior guarantees full support in the all-mode union.
    full,failures=control.prefix_fits(x,sorted(valid),w)
    if failures:raise ArithmeticError(f'Full-prefix fit failed: {failures}')
    return {**full,'history_sha256':control.signature(x),
        'training_dates_sha256':hashlib.sha256(dates.tobytes()).hexdigest(),
        'weights':w.copy(),'horizon':horizon,'inner_paths':inner_paths,'schedule_rule':schedule_rule,
        'seed_replication':seed_replication,'weights_by_mode':{MODE:learned},
        'coefficients':{AXIS:{'A':total_A,'B':total_B,'valid':valid}},'fit':{AXIS:fit},
        'trace':trace,'invalid_internal_options':invalid,
        'diagnostics':{'selection_and_refit_seconds':time.perf_counter()-started,
            'inner_origins':len(folds),'available_internal_origins':full_count,
            'no_outer_future_dates_used':True,'automatic_model_fallback':False,
            'maximum_origin_count':None,'score_variants':1,'cache_bytes':0,
            'internal_criterion':'SUM_h CRPS_h/(fold_training_SD_ddof1*sqrt(native_h)); batch simplex',
            'cache_scope':'No persistent cache; one request-local full fit context'}}


class ScorePool:
    def __init__(self,config,training_dates,bundle=None):
        if config!=CONFIG:raise ValueError('Only the frozen aligned batch config18 is supported')
        self.config=dict(config);self.dates=np.asarray(training_dates,dtype='datetime64[D]')
        self.bundle=bundle;self.model_id=identity(config)
    def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64):
        cfg=self.config
        bundle=self.bundle if self.bundle is not None else prepare(history,weights,horizon,self.dates,
            inner_paths=cfg['inner_paths'],schedule_rule=cfg['schedule'])
        adapter=object.__new__(reference.sequential.MenuPool)
        adapter.config={'menu':MODE,'inner_paths':cfg['inner_paths'],'schedule':cfg['schedule']}
        adapter.dates=self.dates;adapter.bundle=bundle
        yield from adapter.terminal_chunks(history,weights,horizon,paths,context_seed,width=width)
        self.last_selection={**adapter.last_selection,'score_config':cfg}
