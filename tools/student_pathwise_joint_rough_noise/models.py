"""M221 with an estimated effective rough-proxy observation variance.

Original rough parameter priors are retained. The extra variance uses flat
variance-scale support and maximum likelihood, not a new posterior sampler.
Student moments still supply the unchanged conventional causal predictor.
"""
import hashlib
import importlib.util
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('student_rough_noise_private_m221',ROOT/'tools/pathwise_multiscale_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
backend=parent.parent.parent.parent.parent
SOURCE=hashlib.sha256((backend.SOURCE+''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py')))).encode()).hexdigest()
FITTED_NOISE={}


def whittle_target(theta,periodogram,n,reference_variance):
    prior=backend.log_prior(theta[:3])
    if not np.isfinite(prior) or theta[3]<0:return -np.inf
    variance=reference_variance*math.expm1(theta[3])
    expected=math.exp(2*theta[2])*backend.unit_expected_periodogram(float(theta[0]),float(theta[1]),n)+variance*backend.noise_expected_periodogram(n)
    if not np.isfinite(expected).all() or (expected<=0).any():
        raise ArithmeticError('invalid joint-noise expected periodogram')
    terms=np.log(expected)+periodogram/expected
    terms[0]*=.5
    if n%2==0:terms[-1]*=.5
    return prior-float(terms.sum())


@lru_cache(maxsize=64)
def rough_fit(data,lag,seconds):
    identity=hashlib.sha256(data).hexdigest()
    contract=f'M221_effective_rough_noise_ML_analytic_differenced_Whittle_MAP_v1:{lag}'
    key=hashlib.sha256((SOURCE+identity+contract).encode()).hexdigest()
    def build():
        y,_=backend.base.base.observed(data)
        _,reference_variance=backend.base.noise.measurement_scale(data)
        differences=np.diff(y);n=len(differences)
        periodogram=np.abs(np.fft.rfft(differences))**2/n
        # Above max(I/N), every component of the noise derivative is positive:
        # the optimum cannot be outside this data-derived search interval.
        upper=float(np.max(periodogram/backend.noise_expected_periodogram(n)))
        if not np.isfinite(upper) or upper<=0:raise ArithmeticError('degenerate rough noise likelihood')
        bound=math.log1p(upper/reference_variance)
        start=min(math.log(2),bound*.5)
        result=backend.base.noise.fit_map(lambda theta:whittle_target(theta,periodogram,n,reference_variance),
            [.1,math.log(1/63),math.log(.7),start],
            [(.01,.49),(math.log(1/2520),math.log(.5)),(math.log(.05),math.log(3)),(0.,bound)],seconds)
        joint=result['map'].copy();variance=reference_variance*math.expm1(joint[3])
        result.update(map=joint[:3],joint_map=joint.tolist(),contract=contract,data_sha256=identity,
            measurement_noise_variance=variance,reference_student_noise_variance=reference_variance,
            estimator='rough_parameter_MAP_effective_observation_variance_ML_debiased_Whittle_quasi_likelihood',
            noise_support='R >= 0; data-derived derivative bound; same optimizer interior guards',
            noise_search_upper=upper,noise_coordinate='log1p(R / Student_variance); no Jacobian prior term',
            frequency_count=len(periodogram),zero_frequency='included_known_zero_difference_mean')
        return result
    started=time.perf_counter();result=parent.shell.controls.cache('predecessor_MAP_rough_fit',key,build)
    parent.shell.controls.timed('rough_fit_worker',started)
    FITTED_NOISE[identity,result['map'].tobytes(),lag]=result['measurement_noise_variance']
    return result


@lru_cache(maxsize=256)
def prepared(data,theta_bytes,lag,horizon):
    y,_=backend.base.base.observed(data);theta=np.frombuffer(theta_bytes,np.float64)
    phi,w,q=parent.overlay.configuration(theta,f'dynamic:{lag}')
    variance=FITTED_NOISE[hashlib.sha256(data).hexdigest(),theta_bytes,lag]
    scale=math.sqrt(4.934802200544679/variance)
    state,p=parent.overlay.terminal_filter(y*scale,phi,w,q*scale**2,float(y.mean())*scale)
    state=state/scale;p=p/scale**2
    eigen,vectors=np.linalg.eigh((p+p.T)/2)
    initial_root=vectors*np.sqrt(np.maximum(eigen,0));root=np.diag(np.sqrt(np.diag(q)))
    stationary=q/(1-phi[:,None]*phi[None,:])
    return phi,w,root,state,initial_root,np.zeros(horizon),np.full(horizon,float(w@stationary@w))


parent.rough_fit=rough_fit
backend.streamed.rough_fit=rough_fit
parent.prepared=prepared
initialize=parent.initialize
Candidate=parent.Candidate
CANDIDATES=(Candidate(model_id='asset_map_student_pathwise_joint_rough_noise_dynamic_rough'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=parent.clear_path_cache
TIMINGS=parent.TIMINGS
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell
overlay=parent.overlay
load_paths=parent.load_paths
