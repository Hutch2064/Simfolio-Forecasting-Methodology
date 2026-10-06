"""Conditional scalar Gaussian state uncertainty, given plug-in parameters."""
import numpy as np


def variance_curve(conditional_variance,rho,initial_variance):
    if initial_variance==0.:return conditional_variance
    powers=np.power(rho,2*np.arange(1,len(conditional_variance)+1))
    return conditional_variance+initial_variance*powers


def initial_state(mean,variance,rng):
    if variance==0.:return mean
    return mean+np.sqrt(variance)*rng.standard_normal(size=mean.shape)
