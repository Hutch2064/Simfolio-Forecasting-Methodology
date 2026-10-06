"""Independent dense Gaussian checks of stationary AR1 level/variance profiling."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

p=Path(__file__).resolve().parents[1]/'tools/stationary_ar1_rough/state.py'
spec=importlib.util.spec_from_file_location('stationary_ar1_test_estimator',p)
state=importlib.util.module_from_spec(spec);spec.loader.exec_module(state)


@pytest.mark.parametrize('rho',[0.,.3,.9,.999])
def test_profile_matches_dense_covariance_and_independent_level_minimization(rho):
    h=np.random.default_rng(731).normal(size=31)+2.
    nll,level,variance=state.profile(h,rho)
    distance=np.abs(np.arange(len(h))[:,None]-np.arange(len(h)))
    covariance=variance/(1.-rho*rho)*rho**distance
    assert np.isclose(nll,-multivariate_normal.logpdf(h,mean=np.full(len(h),level),cov=covariance),rtol=1e-9,atol=1e-8)
    inverse=np.linalg.inv(rho**distance/(1.-rho*rho));one=np.ones(len(h))
    assert np.isclose(level,(one@inverse@h)/(one@inverse@one),rtol=1e-10,atol=1e-10)
    assert np.isclose(variance,((h-level)@inverse@(h-level))/len(h),rtol=1e-9)


def test_fit_matches_independent_joint_likelihood_optimizer():
    rng=np.random.default_rng(922);h=np.empty(400);h[0]=1.2
    for t in range(1,len(h)):h[t]=1.2+.92*(h[t-1]-1.2)+.1*rng.normal()
    fit=state.fit(h)
    def objective(z):
        rho=z[0];level=z[1];variance=np.exp(z[2]);e=h[1:]-level-rho*(h[:-1]-level)
        return .5*len(h)*np.log(2*np.pi*variance)-.5*np.log1p(-rho*rho)+((1-rho*rho)*(h[0]-level)**2+e@e)/(2*variance)
    result=minimize(objective,[.8,h.mean(),np.log(.01)],bounds=[(0.,.999999),(None,None),(None,None)],method='L-BFGS-B',options={'ftol':1e-12,'gtol':1e-8})
    assert result.success
    assert np.isclose(fit['negative_loglikelihood'],result.fun,rtol=1e-9,atol=1e-7)
    assert np.isclose(fit['coefficients'][0],result.x[0],atol=1e-6)
    assert np.isclose(fit['level'],result.x[1],atol=1e-6)


def test_zero_persistence_boundary_and_constant_history():
    fit=state.fit(np.array([1.,-1.]*50));assert fit['coefficients']==[0.]
    constant=state.fit(np.ones(10)*4.);assert constant['innovation_sd']==0.
    assert constant['level']==4. and constant['initial']==[0.]
