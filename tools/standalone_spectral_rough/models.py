"""Variance-targeted standalone rough volatility with unchanged return mean.

The rough process supplies all stochastic volatility and is fitted to the
original log-square return proxy, without a conventional-volatility offset.
The conventional predecessor fit is retained only for the unchanged mean and
filtered innovation/noise inputs. No conventional multiscale variance curve is
used in forecasting. All estimation remains plug-in Gaussian quasi likelihood.
"""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('standalone_spectral_private_m212',ROOT/'tools/consistent_noise_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
streamed=parent.parent.streamed
SOURCE=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(HERE.glob('*.py')))).hexdigest()

@lru_cache(maxsize=64)
def observed(data):
    x=np.frombuffer(data,np.float64)
    return parent.shell.bd._sv_observed_log_variance(x,float(x.mean()))
parent.parent.base.base.observed=observed;parent.parent.base.noise.observed=observed
parent.parent.SOURCE=hashlib.sha256((parent.parent.SOURCE+SOURCE).encode()).hexdigest()

def asset_paths(data,uniforms):
    from streamed_paths import map_asset_inplace

    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
        moment_return_curves,
    )
    fit=streamed.predecessor_fit(data);mean,_=moment_return_curves(fit,uniforms.shape[1])
    # Stationary rough multiplier has E[M^2]=1, so this level matches the
    # observed centered-return variance, with no second volatility process.
    sd=np.full(uniforms.shape[1],fit['base_fit']['sigma']);n=len(uniforms)
    nodes=np.quantile(fit['innovation_pool'],np.linspace(.5/n,1-.5/n,n));nodes-=nodes.mean();nodes/=np.sqrt(np.mean(nodes*nodes))
    paths=uniforms.copy();map_asset_inplace(paths,mean,sd,nodes);return mean,paths
streamed.predecessor_asset_paths=asset_paths

class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        result=super().simulate_daily_log_returns(training,context)
        for key,r in FIT_DIAGNOSTICS.items():
            if key[0]==self.model_id:
                r['volatility_forecast']='stationary_variance_targeted_pure_rough_no_conventional_multiscale_variance_curve'
                r['conventional_offset_noise']='no conventional offset'
                r['rough_observations']='raw log-square return proxy'
                r['return_mean_curve']='unchanged predecessor; its conventional fit retained for mean/innovation/noise inputs only'
        return result
initialize=parent.initialize
CANDIDATES=(Candidate(model_id='asset_map_standalone_variance_targeted_spectral_dynamic_rough'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=parent.clear_path_cache;TIMINGS=parent.TIMINGS;FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths
