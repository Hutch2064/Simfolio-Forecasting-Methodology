"""Target, analytic-gradient, and defensive importance-density contracts."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit, logsumexp
from scipy.stats import multivariate_t

pytest.importorskip('numba')
pytest.importorskip('arviz')
TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS / 'mcmc_runtime'))
sys.path.insert(0, str(TOOLS / 'inference_candidates'))
import inference
from steady_kalman_target import target as original_target
from target import (
    batch_target,
    theta_target_gradient,
    unconstrained_target_gradient,
    variational_objective,
)


@pytest.mark.parametrize('phi,eta',[(.05,.03),(.5,.35),(.94,.1),(.998,1.)])
def test_target_and_gradient_match_incumbent(phi,eta):
    y=np.random.default_rng(813).normal(-7.,2.5,257)
    theta=np.array([-7.1,math.log(phi/(1.-phi)),math.log(eta)])
    value,grad=theta_target_gradient(y,theta)
    assert value == pytest.approx(original_target(y,*theta,y.mean()),abs=1e-9)
    numerical=approx_derivative(lambda t: original_target(y,*t,y.mean()),theta).ravel()
    assert np.allclose(grad,numerical,rtol=2e-4,atol=2e-5)


def test_support_transform_retains_original_target_and_jacobian():
    y=np.random.default_rng(82).normal(-7.,2.5,257)
    lower,upper=inference.support(y)
    draws=np.array([[-2.,1.,-.5],[0.,0.,0.],[3.,-3.,2.]])
    actual=batch_target(y,draws,lower,upper)
    for i,u in enumerate(draws):
        s=expit(u);theta=lower+(upper-lower)*s
        expected=original_target(y,*theta,y.mean())+np.log((upper-lower)*s*(1-s)).sum()
        value,grad=unconstrained_target_gradient(y,u,lower,upper)
        assert actual[i] == pytest.approx(expected,abs=1e-9)
        assert value == pytest.approx(expected,abs=1e-9)
        numerical=approx_derivative(lambda t: unconstrained_target_gradient(y,t,lower,upper)[0],u).ravel()
        assert np.allclose(grad,numerical,rtol=2e-4,atol=2e-5)


def test_full_rank_variational_gradient():
    rng=np.random.default_rng(113)
    y=rng.normal(-7.,2.5,257);lower,upper=inference.support(y)
    params=np.r_[np.zeros(3),math.log(.4),.1,math.log(.3),-.1,.2,math.log(.2)]
    z=rng.normal(size=(64,3))
    _,grad=variational_objective(y,params,z,lower,upper)
    numerical=approx_derivative(lambda p: variational_objective(y,p,z,lower,upper)[0],params).ravel()
    assert np.allclose(grad,numerical,rtol=2e-4,atol=2e-5)


@pytest.mark.parametrize('components',[
    [(np.zeros(3),np.eye(3))],
    [(np.zeros(3),np.eye(3)),(np.array([2.,-1.,.5]),np.diag([.5,2.,1.]))],
])
def test_importance_weights_use_complete_mixture_density(monkeypatch,components):
    def proposal_density(y,draws,lower,upper):
        parts=[multivariate_t.logpdf(draws,loc=mode,shape=L@L.T,df=5.)
               for mode,L in components]
        return logsumexp(np.asarray(parts),axis=0)-math.log(len(parts))
    monkeypatch.setattr(inference,'batch_target',proposal_density)
    nodes,diag=inference.mixture_importance(np.zeros(1),components,np.zeros(3),np.ones(3),
                                            np.random.default_rng(192),count=1024)
    # The target equals the complete proposal mixture: every importance weight
    # must be equal even when the generating mixture component differs.
    assert diag['importance_ess'] == pytest.approx(1024.,abs=1e-8)
    assert diag['max_normalized_weight'] == pytest.approx(1/1024.,abs=1e-12)
    assert nodes.shape == (16,3)
