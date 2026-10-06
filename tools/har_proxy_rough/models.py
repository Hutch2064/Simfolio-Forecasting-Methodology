"""M212 rough overlay with log-HAR proxy volatility moments, mean held exact."""
from functools import lru_cache
import hashlib,importlib.util
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import har
spec=importlib.util.spec_from_file_location('har_proxy_private_m212',ROOT/'tools/consistent_noise_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
streamed=parent.parent.streamed
SOURCE=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(HERE.glob('*.py')))).hexdigest()

@lru_cache(maxsize=64)
def har_fit(data):
    def compute():
        original=streamed.predecessor_fit(data)
        x=np.frombuffer(data,np.float64)
        y,_=shell.bd._sv_observed_log_variance(x,float(x.mean()),winsorize=True)
        level,phi,eta,_,_=original['posterior_center']
        _,h,_=shell.bd._sv_kalman_rts_smoother_mean(y,level,phi,eta)
        return har.fit(h)
    return shell.controls.cache('har_proxy_mle',hashlib.sha256(data+SOURCE.encode()).hexdigest(),compute)


def curves(data,horizon):
    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import moment_return_curves,predictive_state_moments
    original=streamed.predecessor_fit(data)
    mean,_=moment_return_curves(original,horizon)
    mh,vh=har.moments(har_fit(data),horizon)
    _,_,mu,vu=predictive_state_moments(original,horizon)
    base=original['base_fit'];sigma=base['sigma']
    expected_sd=np.exp(.5*mh+.125*vh)/100
    expected_var=np.exp(mh+.5*vh)/10000
    implied_mean=mu*expected_sd/sigma
    variance=expected_var*(1+(vu+mu*mu)/sigma**2)-implied_mean*implied_mean
    if not np.isfinite(variance).all():raise ArithmeticError('nonfinite log-HAR return variance')
    return mean,np.sqrt(np.maximum(variance,0.))


def asset_paths(data,uniforms):
    from streamed_paths import map_asset_inplace
    mean,sd=curves(data,uniforms.shape[1]);n=uniforms.shape[0]
    nodes=np.quantile(streamed.predecessor_fit(data)['innovation_pool'],np.linspace(.5/n,1-.5/n,n));nodes-=nodes.mean();nodes/=np.sqrt(np.mean(nodes*nodes))
    paths=uniforms.copy();map_asset_inplace(paths,mean,sd,nodes)
    return mean,paths

streamed.predecessor_asset_paths=asset_paths
class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        result=super().simulate_daily_log_returns(training,context)
        data=np.asarray(training.asset_log_returns,float)
        for a in range(data.shape[1]):
            d=data[:,a].tobytes();record=FIT_DIAGNOSTICS[self.model_id,hashlib.sha256(d).hexdigest()]
            record['baseline_volatility']='log_HAR_filtered_proxy';f=har_fit(d)
            record['har_fit']={k:np.asarray(v).tolist() if isinstance(v,np.ndarray) else v for k,v in f.items()}
            record['return_mean_curve']='unchanged_predecessor_curve'
        return result

initialize=parent.initialize
CANDIDATES=(Candidate(model_id='asset_map_log_har_proxy_consistent_noise_dynamic_rough_map'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=parent.clear_path_cache
TIMINGS=parent.TIMINGS
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell
overlay=parent.overlay
load_paths=parent.load_paths
