"""Common exact research kernels; compiler warm-up precedes benchmarks."""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from predictive_sv_candidates import fast_rebalanced, fast_uniforms
from research_jit_pilot import make_accelerators

from simfolio_forecasting_methodology.models.asset_level.filtered_innovation_moment_sv import (
    FilteredInnovationFixedMeanMomentSV,
)
from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd
from simfolio_forecasting_methodology.models.numerical import dynamic_gaussian as dg

DATA=Path(os.environ.get('SIMFOLIO_OOS_DATA',str(Path.cwd()/'.simfolio-oos-data')))
MODELS=None
def initialize(tuned):
    global MODELS
    if tuned:
        from parameter_mcmc_tuned import CANDIDATES, install
    else:
        from parameter_mcmc import CANDIDATES, install
    install();MODELS=(FilteredInnovationFixedMeanMomentSV(),CANDIDATES[0])
    _,_,dlm,sv=make_accelerators();bd._dlm_ar1_loglik=dlm;bd._sv_kalman_filter=sv
    dg.simulate_future_gaussian_uniforms=fast_uniforms;dg.rebalanced_portfolio_log_paths=fast_rebalanced
    from compiled_smoothers import make
    rts,ewma=make();rts(np.array([.1,.2]),.1,.9,.2);ewma(np.array([.1,.2]),.1)
    bd._sv_kalman_rts_smoother_mean=rts;bd._bdes_ewma=ewma
    original_fit=bd.fit_bdes_fastmap
    @lru_cache(maxsize=4096)
    def cached(data,filtered,fixed):return original_fit(np.frombuffer(data,np.float64),filtered_innovations=filtered,fixed_mean=fixed)
    def fit(x,*,filtered_innovations=False,fixed_mean=False):return cached(np.asarray(x,np.float64).tobytes(),filtered_innovations,fixed_mean)
    bd.fit_bdes_fastmap=fit
