"""Conditional Gaussian uncertainty of the stationary AR1 volatility level.

Given the retained persistence, innovation variance and fitted latent proxy,
flat prior density on the real level gives a proper Gaussian posterior.
This integrates one parameter conditional on plug-in estimates of the others;
it is not joint raw-return Bayesian SV inference.
"""
import numpy as np


def posterior_variance(fit,observations):
    rho=float(fit['coefficients'][0]);variance=float(fit['innovation_sd'])**2
    precision=(1.-rho)*(1.+rho)+(observations-1)*(1.-rho)**2
    return variance/precision


def response(rho,horizon):
    if rho==0.:return np.ones(horizon)
    return -np.expm1(np.arange(1,horizon+1)*np.log(rho))


def predictive_variance(process_variance,fit,observations):
    gain=response(float(fit['coefficients'][0]),len(process_variance))
    return process_variance+posterior_variance(fit,observations)*gain*gain
