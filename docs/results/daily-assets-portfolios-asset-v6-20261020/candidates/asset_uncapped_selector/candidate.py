"""Complete history-supported inner validation; no fold cap or automatic fallback."""
from pathlib import Path
import hashlib,time
import numpy as np
import common
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
from candidates.asset_fast_selector.candidate import BANK,model,shared_fits,portfolio_returns,native_objective,signature,BASE_CONFIG
from candidates.asset_feature_batch import fits
from candidates.asset_trained_pool.candidate import prepare as parent_prepare
from candidates.asset_fast_selector import bank

MODEL_ID='model_testing_asset_complete_history_rolling_selector_v1'
INNER_PATHS=1000

def validation_plan(rows,horizon):
    if horizon<1:raise ValueError('Positive requested horizon required')
    eligible=[i for i,c in enumerate(BANK)if rows>(2 if c['kind']=='blocks'else 24)]
    if not eligible:raise ValueError('Insufficient history for any completed internal out-of-sample test')
    minimum=max(2 if BANK[i]['kind']=='blocks'else 24 for i in eligible)
    return eligible,[(stop,min(horizon,rows-stop))for stop in range(minimum,rows)]

def prepare(history,weights,horizon):
    x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float);started=time.perf_counter()
    if x.ndim!=2 or not np.isfinite(x).all()or x.shape[1]!=len(w)or np.any(w<0)or not np.isclose(w.sum(),1.,rtol=0,atol=1e-14):raise ValueError('Finite joint history and long-only normalized weights required')
    eligible,plan=validation_plan(len(x),horizon);scores=np.zeros(len(BANK));trace=[];fit_seconds=0.;forecast_seconds=0.;invalid={}
    for stop,H in plan:
        past=x[:stop];actual=np.cumsum(portfolio_returns(x[stop:stop+H],w));train=portfolio_returns(past,w)
        begin=time.perf_counter();prepared={'families':{}}
        if any(BANK[i]['kind']!='blocks'and i not in invalid for i in eligible):
            try:base=fits.base_fit(past)
            except Exception as exc:
                for i in eligible:
                    if BANK[i]['kind']!='blocks':invalid[i]={'training_rows':stop,'type':type(exc).__name__,'reason':str(exc)}
            else:
                for family in {BANK[i]['family']for i in eligible if BANK[i]['kind']=='filtered'and i not in invalid}|({'SV_RMS'}if 0 in eligible and 0 not in invalid else set()):
                    try:prepared['families'][family]=bank.prepare(past,family,base)
                    except Exception as exc:
                        for i in eligible:
                            if(BANK[i]['kind']=='filtered'and BANK[i]['family']==family)or(i==0 and family=='SV_RMS'):invalid[i]={'training_rows':stop,'type':type(exc).__name__,'reason':str(exc)}
                if 0 in eligible and 0 not in invalid:
                    try:prepared['parent']=parent_prepare(past,'SV_RMS','EWMA',base,prepared['families']['SV_RMS'])
                    except Exception as exc:invalid[0]={'training_rows':stop,'type':type(exc).__name__,'reason':str(exc)}
        fit_seconds+=time.perf_counter()-begin
        seed=int.from_bytes(hashlib.sha256(f'complete_inner_v1:{signature(past)}:{H}:{INNER_PATHS}'.encode()).digest()[:4],'little');values={};begin=time.perf_counter()
        for index in eligible:
            if index in invalid:continue
            loss=np.empty(H);cursor=0
            for start,paths in model(index,prepared).terminal_chunks(past,w,H,INNER_PATHS,seed):
                assert start==cursor and paths.shape[0]==INNER_PATHS;cursor+=paths.shape[1]
                loss[start:cursor]=empirical_crps_by_horizon(paths,actual[start:cursor])
            if cursor!=H or not np.isfinite(loss).all():raise ArithmeticError('Incomplete internal forecast')
            # A constant full-request normalization would cancel in every
            # candidate comparison; omit it instead of inventing a zero-SD rule.
            lead=np.arange(1,H+1,dtype=float);mass=np.log1p(lead)-np.log(lead)
            q=float(np.dot(mass,loss/np.sqrt(lead))/mass.sum())
            values[index]=q;scores[index]+=q
        forecast_seconds+=time.perf_counter()-begin
        trace.append({'training_rows':stop,'validated_leads':H,'training_sha256':signature(past),'seed':seed,'scores':values})
    scores/=len(plan);valid=[i for i in eligible if i not in invalid]
    if not valid:raise ArithmeticError('No option can produce all internal validation forecasts')
    choice=min(valid,key=lambda i:(scores[i],i));selection=time.perf_counter()-started;begin=time.perf_counter();spec=BANK[choice]
    prepared={'families':{}}
    if spec['kind']=='parent':prepared['parent']=parent_prepare(x,'SV_RMS','EWMA')
    elif spec['kind']=='filtered':prepared['families'][spec['family']]=bank.prepare(x,spec['family'])
    refit=time.perf_counter()-begin
    return {**prepared,'history_sha256':signature(x),'weights':w.copy(),'horizon':horizon,'choice':choice,'eligible_bank_indices':eligible,'inner_scores':{i:float(scores[i])for i in valid},'invalid_internal_options':invalid,'trace':trace,'diagnostics':{'selection_seconds':selection,'inner_shared_fit_seconds':fit_seconds,'inner_forecast_CRPS_seconds':forecast_seconds,'full_refit_seconds':refit,'inner_paths':INNER_PATHS,'inner_origins':len(plan),'longest_directly_validated_lead':max(H for _,H in plan),'requested_horizon':horizon,'entire_requested_horizon_directly_validated':max(H for _,H in plan)==horizon,'automatic_parent_fallback':False,'origin_count_cap':None,'eligible_options':len(eligible),'fully_valid_options':len(valid),'invalid_internal_options':invalid}}

class CompleteHistorySelector:
    model_id=MODEL_ID
    def __init__(self,bundle=None):self.bundle=bundle
    @property
    def dependency_identity(self):return {'method':MODEL_ID,'forecast_level':'joint_assets','bank':BANK,'selection':'All eligible training-only origins, all supported leads up to requestedH; no capped folds or automatic parent fallback','inner_paths':INNER_PATHS,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(__file__).parent.glob('*.py')}}
    def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64):
        b=self.bundle if self.bundle is not None else prepare(history,weights,horizon)
        if b['history_sha256']!=signature(history)or b['horizon']!=horizon or not np.array_equal(b['weights'],weights):raise ValueError('Request does not match selector fit')
        self.last_selection={'bank_index':b['choice'],'specification':BANK[b['choice']],**b['diagnostics']}
        yield from model(b['choice'],b).terminal_chunks(history,weights,horizon,paths,context_seed,width=width)

def build():return CompleteHistorySelector()
