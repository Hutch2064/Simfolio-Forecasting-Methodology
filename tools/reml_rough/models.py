"""M197 rough covariance with analytic integration of its nuisance level.

Restricted Gaussian likelihood (Patterson & Thompson, 1971) plus unchanged
rough-parameter priors. This integrates the observation intercept with a flat
prior; covariance parameters remain MAP points. Return mean/dependence unchanged.
"""
import hashlib
import importlib.util
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(HERE))
from gaussian_level_filter import restricted_filter

spec = importlib.util.spec_from_file_location('reml_private_conditional_empirical', ROOT/'tools/conditional_empirical_rough/models.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
SOURCE = hashlib.sha256((base.SOURCE+''.join(hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(HERE.glob('*.py')))).encode()).hexdigest()


@lru_cache(maxsize=256)
def terminal(data,theta_bytes,lag):
    y,_ = base.base.observed(data)
    _,R = base.noise.measurement_scale(data)
    theta = np.frombuffer(theta_bytes,np.float64)
    phi,w,q = base.overlay.configuration(theta,f'dynamic:{lag}')
    return restricted_filter(y,phi,w,q,R)


@lru_cache(maxsize=64)
def rough_fit(data,lag,seconds):
    identity = hashlib.sha256(data).hexdigest()
    contract = f'rough_REML_empirical_noise_integrated_level_v1:{lag}'
    key = hashlib.sha256((SOURCE+identity+contract).encode()).hexdigest()
    def build():
        y,_ = base.base.observed(data)
        _,R = base.noise.measurement_scale(data)
        def target(theta):
            prior = base.overlay.log_prior(theta)
            if not np.isfinite(prior):
                return -np.inf
            phi,w,q = base.overlay.configuration(theta,f'dynamic:{lag}')
            return prior+restricted_filter(y,phi,w,q,R)[0]
        result = base.noise.fit_map(target,[.1,math.log(1/63),math.log(.7)],
            [(.03,.49),(math.log(1/2520),math.log(.5)),(math.log(.05),math.log(3))],seconds)
        summary = terminal(data,result['map'].tobytes(),lag)
        result.update(contract=contract,data_sha256=identity,measurement_noise_variance=R,
            estimator='rough_covariance_MAP_restricted_Gaussian_likelihood',
            integrated_level_mean=summary[1],integrated_level_variance=summary[2],
            observation_level_prior='flat',nuisance_level_uncertainty=True)
        return result
    started = time.perf_counter()
    result = base.shell.controls.cache('predecessor_MAP_rough_fit',key,build)
    base.shell.controls.timed('rough_fit_worker',started)
    return result


@lru_cache(maxsize=256)
def prepared(data,theta_bytes,lag,horizon):
    phi,w,q = base.overlay.configuration(np.frombuffer(theta_bytes,np.float64),f'dynamic:{lag}')
    _,_,_,state,p = terminal(data,theta_bytes,lag)
    eigen,vectors = np.linalg.eigh((p+p.T)/2)
    initial_root = vectors*np.sqrt(np.maximum(eigen,0))
    root = np.diag(np.sqrt(np.diag(q)))
    stationary = q/(1-phi[:,None]*phi[None,:])
    return phi,w,root,state,initial_root,np.zeros(horizon),np.full(horizon,float(w@stationary@w))


base.base.rough_fit = rough_fit
base.base.prepared = prepared


def initialize(cache_root=None,vine_threads=1,state_workers=1):
    base.initialize(cache_root,vine_threads,state_workers)
    restricted_filter(np.zeros(4),np.array([.9]),np.ones(1),np.array([[.1]]),4.)


clear_path_cache = base.clear_path_cache
TIMINGS = base.TIMINGS
FIT_DIAGNOSTICS = base.FIT_DIAGNOSTICS
shell = base.shell
overlay = base.overlay
Candidate = base.Candidate
CANDIDATES = (Candidate(model_id='asset_map_multiscale_conditional_empirical_reml_rough_map'),)
MODEL_IDS = (base.MODEL_IDS[0],CANDIDATES[0].model_id)
