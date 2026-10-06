"""Exact discrete white-noise and ordinary-OU residual volatility models.

These are canonical covariance families, not approximate fractional lifts.
MAP uses the retained log-scale/log-decay priors and debiased Whittle target.
"""
import math
import time
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize


@lru_cache(maxsize=128)
def noise_spectrum(n):
    return 2*(1-(1-1/n)*np.cos(2*np.pi*np.arange(n//2+1)/n))


def spectrum(covariance,n):
    differences=np.empty(n);differences[0]=2*(covariance[0]-covariance[1])
    differences[1:]=2*covariance[1:n]-covariance[:n-1]-covariance[2:n+1]
    return 2*np.fft.rfft(differences*(1-np.arange(n)/n)).real-differences[0]


def expected(log_kappa,n):
    kappa=math.exp(log_kappa);lags=np.arange(n+1);covariance=np.exp(-kappa*lags)
    derivative=-kappa*lags*covariance
    return spectrum(covariance,n),spectrum(derivative,n)


def objective_n(u,periodogram,n,noise_variance,kind):
    if kind=='white':
        unit=noise_spectrum(n);du=None;log_amp=math.log(.7)+1.5*u[0]
    else:
        log_kappa=math.log(1/63)+2*u[0];log_amp=math.log(.7)+1.5*u[1]
        if log_kappa>math.log(np.finfo(float).max)-math.log(n):return 1e100,np.zeros_like(u)
        unit,du=expected(log_kappa,n)
    if 2*log_amp>math.log(np.finfo(float).max):return 1e100,np.zeros_like(u)
    amplitude_squared=math.exp(2*log_amp)
    prediction=amplitude_squared*unit+noise_variance*noise_spectrum(n)
    if np.any(prediction<=0) or not np.isfinite(prediction).all():return 1e100,np.zeros_like(u)
    weights=np.ones(len(prediction));weights[0]=.5
    if n%2==0:weights[-1]=.5
    common=weights*(1/prediction-periodogram/prediction**2)
    gradient=np.array([common@(3*amplitude_squared*unit)]) if kind=='white' else np.array([common@(2*amplitude_squared*du),common@(3*amplitude_squared*unit)])
    value=weights@(np.log(prediction)+periodogram/prediction)+.5*(u@u)
    return float(value),gradient+u


def fit(periodogram,n,noise_variance,kind,seconds=10.):
    started=time.perf_counter();evaluations=0
    def target(u):
        nonlocal evaluations
        evaluations+=1
        if time.perf_counter()-started>seconds:raise RuntimeError('endpoint MAP exceeded fitting resource ceiling')
        return objective_n(u,periodogram,n,noise_variance,kind)
    result=minimize(target,np.zeros(1 if kind=='white' else 2),jac=True,method='L-BFGS-B',options={'ftol':1e-10,'gtol':1e-5,'maxiter':200,'maxls':100})
    if not result.success or result.fun>=1e100:raise ArithmeticError(f'endpoint MAP unresolved: {result.message}')
    log_kappa=0. if kind=='white' else math.log(1/63)+2*result.x[0]
    log_amp=math.log(.7)+1.5*result.x[-1]
    return {'map':np.array([0. if kind=='white' else .5,log_kappa,log_amp]),'covariance_family':kind,
            'estimator':'MAP_exact_covariance_differenced_debiased_Whittle_quasi_likelihood',
            'posterior_uncertainty':False,'evaluations':evaluations,'seconds':time.perf_counter()-started,
            'optimizer':{'success':True,'message':str(result.message),'iterations':int(result.nit)},
            'fitted_parameter_count':len(result.x),'rough_prior_support':'retained proper Gaussian priors on log scale and, for OU, log decay; untruncated',
            'H_inference':'not fitted; marker identifies an exact covariance family'}


def configuration(theta):
    if theta[0]==0.:phi=0.;variance=math.exp(2*theta[2])
    elif theta[0]==.5:
        kappa=math.exp(theta[1]);phi=math.exp(-kappa);variance=math.exp(2*theta[2])
        if phi==1.:raise ArithmeticError('OU decay is not representable below one')
    elif theta[0]==-1.:return np.zeros(1),np.ones(1),np.zeros((1,1)),0.
    else:raise ValueError('unknown endpoint covariance marker')
    innovation=variance*(1-phi*phi)
    return np.array([phi]),np.ones(1),np.array([[innovation]]),variance
