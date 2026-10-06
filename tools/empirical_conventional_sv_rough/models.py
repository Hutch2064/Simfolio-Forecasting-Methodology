"""M212 with empirical noise also used in conventional parameter fitting.

Observation noise is estimated from the original filtered innovation pool, then
held fixed in both volatility fits. This is a two-stage plug-in quasi-likelihood,
not a joint raw-return posterior. Return mean is preserved exactly.
"""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import gaussian_sv

spec=importlib.util.spec_from_file_location('empirical_conventional_private_m212',ROOT/'tools/consistent_noise_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
streamed=parent.parent.streamed
original_fit=streamed.predecessor_fit
original_noise=parent.parent.base.noise.measurement_scale
SOURCE=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(HERE.glob('*.py')))).hexdigest()

@lru_cache(maxsize=64)
def refit(data):
    def compute():
        original=original_fit(data);x=np.frombuffer(data,np.float64)
        y,eps=shell.bd._sv_observed_log_variance(x,float(x.mean()),winsorize=True)
        _,noise=original_noise(data)
        theta,diagnostics=gaussian_sv.fit(y,noise,original['posterior_center'][:3])
        ll,h,v=gaussian_sv.smooth_states(y,*theta,noise)
        pool=shell.bd._standardized_empirical_innovation_pool(eps*np.exp(-.5*h),clip=None,method='mean_std')
        if pool is None:raise ArithmeticError('invalid empirical-noise SV innovation pool')
        level,phi,eta=theta
        innov=(h[1:]-level-phi*(h[:-1]-level))/eta
        rho=shell.bd._finite_correlation(pool[:len(innov)],innov)
        result=original.copy();result.update(posterior_center=(level,phi,eta,float(h[-1]),float(rho)),innovation_pool=pool,bdes_multiscale_vol=shell.bd._bdes_multiscale_components(h,4,shell.bd.BDES_MULTISCALE_GRID_FIXED),state_loglikelihood=float(ll),state_path_variance_last=float(v[-1]),empirical_conventional_fit=diagnostics)
        return result
    key=hashlib.sha256(data+SOURCE.encode()).hexdigest()
    return shell.controls.cache('empirical_conventional_MAP',key,compute)

streamed.predecessor_fit=refit
# Noise calibration remains fixed at the original-pool estimate. Re-estimating
# it from the new pool would silently change the two-stage likelihood contract.
parent.parent.base.noise.predecessor_fit=original_fit

@lru_cache(maxsize=64)
def observed(data):
    x=np.frombuffer(data,np.float64);y,eps=shell.bd._sv_observed_log_variance(x,float(x.mean()))
    level,phi,eta,_,_=refit(data)['posterior_center'];_,noise=original_noise(data)
    return y-parent.causal_predictor(y,level,phi,eta,noise),eps
parent.parent.base.base.observed=observed
parent.parent.base.noise.observed=observed
# Bind a distinct source cache contract for the unchanged rough target.
parent.parent.SOURCE=hashlib.sha256((parent.parent.SOURCE+SOURCE).encode()).hexdigest()


def asset_paths(data,uniforms):
    from streamed_paths import map_asset_inplace

    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
        moment_return_curves,
    )
    mean,_=moment_return_curves(original_fit(data),uniforms.shape[1]);_,sd=moment_return_curves(refit(data),uniforms.shape[1])
    n=len(uniforms);nodes=np.quantile(refit(data)['innovation_pool'],np.linspace(.5/n,1-.5/n,n));nodes-=nodes.mean();nodes/=np.sqrt(np.mean(nodes*nodes))
    paths=uniforms.copy();map_asset_inplace(paths,mean,sd,nodes)
    return mean,paths
streamed.predecessor_asset_paths=asset_paths

class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        result=super().simulate_daily_log_returns(training,context)
        data=np.asarray(training.asset_log_returns,float)
        for a in range(data.shape[1]):
            d=data[:,a].tobytes();record=FIT_DIAGNOSTICS[self.model_id,hashlib.sha256(d).hexdigest()]
            record['conventional_parameter_fit']='same priors with initial empirical-pool observation noise'
            record['empirical_conventional_fit']=refit(d)['empirical_conventional_fit']
            record['return_mean_curve']='unchanged_original_predecessor'
        return result

initialize=parent.initialize
CANDIDATES=(Candidate(model_id='asset_map_empirical_conventional_noise_multiscale_dynamic_rough_map'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=parent.clear_path_cache
TIMINGS=parent.TIMINGS
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell
overlay=parent.overlay
load_paths=parent.load_paths
