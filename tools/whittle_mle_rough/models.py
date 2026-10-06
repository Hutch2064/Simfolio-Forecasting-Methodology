"""M200 debiased Whittle inference without covariance-parameter penalties.

Same empirical-noise residual rough process, numerical lift and return shell.
Hurst already has no interior penalty in M200. This candidate also removes the
Gaussian penalties on log timescale and log amplitude, keeping support bounds.
It is bounded quasi maximum likelihood, not posterior sampling.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parent = load('mle_private_whittle', ROOT/'tools/debiased_whittle_rough/models.py')
# An isolated overlay avoids changing the already-scored MAP adapters.
overlay = load('mle_private_overlay', ROOT/'tools/rough_bayesian/overlay.py')
original_prior = overlay.log_prior


def support_only(theta):
    return 0. if original_prior(theta) > float('-inf') else float('-inf')


overlay.log_prior = support_only
parent.overlay = parent.base.overlay = parent.base.base.overlay = overlay
parent.SOURCE = hashlib.sha256((parent.SOURCE+''.join(
    hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py')))).encode()).hexdigest()

original_fit = parent.rough_fit


def rough_fit(data, lag, seconds):
    result = dict(original_fit(data, lag, seconds))
    result.update(estimator='bounded_debiased_Whittle_quasi_MLE',
                  covariance_parameter_penalties='none; original support bounds retained')
    return result


parent.base.base.rough_fit = rough_fit

initialize = parent.initialize
clear_path_cache = parent.clear_path_cache
TIMINGS = parent.TIMINGS
FIT_DIAGNOSTICS = parent.FIT_DIAGNOSTICS
shell = parent.shell
Candidate = parent.Candidate
CANDIDATES = (Candidate(model_id='asset_map_multiscale_conditional_empirical_whittle_mle_rough'),)
MODEL_IDS = (parent.MODEL_IDS[0], CANDIDATES[0].model_id)
