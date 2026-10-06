"""M200 covariance inference with a conditional GLS rough-observation level.

Covariance parameters retain M200's debiased Whittle MAP target. The terminal
filter estimates the constant observation level by exact Gaussian GLS instead
of its arithmetic sample mean. This is a plug-in point estimate, not joint Bayes
or REML covariance estimation. The return-mean model remains unchanged.
"""
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'tools/reml_rough'))
from gaussian_level_filter import restricted_filter

spec = importlib.util.spec_from_file_location('gls_private_whittle', ROOT/'tools/debiased_whittle_rough/models.py')
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)


def conditional_terminal(y, phi, w, q, scale, measurement_variance):
    level = restricted_filter(y, phi, w, q, measurement_variance)[1]
    state, covariance = parent.overlay.terminal_filter(
        y*scale, phi, w, q*scale**2, float(level)*scale)
    return level, state/scale, covariance/scale**2


@lru_cache(maxsize=256)
def prepared(data, theta_bytes, lag, horizon):
    y, _ = parent.base.base.observed(data)
    scale, measurement_variance = parent.base.noise.measurement_scale(data)
    phi, w, q = parent.overlay.configuration(np.frombuffer(theta_bytes, np.float64), f'dynamic:{lag}')
    _, state, p = conditional_terminal(y, phi, w, q, scale, measurement_variance)
    eigen, vectors = np.linalg.eigh((p+p.T)/2)
    initial_root = vectors*np.sqrt(np.maximum(eigen, 0))
    root = np.diag(np.sqrt(np.diag(q)))
    stationary = q/(1-phi[:, None]*phi[None, :])
    return phi, w, root, state, initial_root, np.zeros(horizon), np.full(horizon, float(w@stationary@w))


parent.base.base.prepared = prepared


def initialize(cache_root=None, vine_threads=1, state_workers=1):
    parent.initialize(cache_root, vine_threads, state_workers)
    restricted_filter(np.zeros(4), np.array([.9]), np.ones(1), np.array([[.1]]), 4.)


clear_path_cache = parent.clear_path_cache
TIMINGS = parent.TIMINGS
FIT_DIAGNOSTICS = parent.FIT_DIAGNOSTICS
shell = parent.shell
overlay = parent.overlay
Candidate = parent.Candidate
CANDIDATES = (Candidate(model_id='asset_map_multiscale_conditional_empirical_whittle_gls_rough_map'),)
MODEL_IDS = (parent.MODEL_IDS[0], CANDIDATES[0].model_id)
