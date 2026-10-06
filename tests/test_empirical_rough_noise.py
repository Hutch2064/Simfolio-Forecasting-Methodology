"""Empirical measurement noise preserves the intended Gaussian QL."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def model():
    path=Path(__file__).resolve().parents[1]/"tools/empirical_rough_noise/models.py"
    spec=importlib.util.spec_from_file_location("empirical_rough_noise_test",path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m

@pytest.mark.parametrize("noise_variance",[1.,4.934802200544679,9.])
def test_noise_scaling_matches_dense_gaussian_likelihood(model,noise_variance):
    from scipy.stats import multivariate_normal
    y=np.random.default_rng(23).normal(size=13)
    phi=np.array([.2,.7,.97]);w=np.ones(3);q=np.diag([.4,.2,.01]);level=-.3
    stationary=q/(1-phi[:,None]*phi[None,:])
    c=np.empty((len(y),len(y)))
    for i in range(len(y)):
        for j in range(len(y)):c[i,j]=w@(phi**abs(i-j)*stationary)@w
    c+=np.eye(len(y))*noise_variance
    expected=multivariate_normal.logpdf(y,np.full(len(y),level),c)
    scale=np.sqrt(4.934802200544679/noise_variance)
    actual=model.overlay.filter_rough(y*scale,phi,w,q*scale**2,level*scale)[0]+len(y)*np.log(scale)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)

def test_measurement_variance_comes_only_from_existing_innovation_pool(model):
    x=np.random.default_rng(8).standard_t(5,size=300)*.01
    scale,variance=model.measurement_scale(x.tobytes())
    pool=model.predecessor_fit(x.tobytes())["innovation_pool"]
    noise,_=model.shell.bd._sv_observed_log_variance(pool*.01,float(pool.mean())*.01)
    assert variance==float(np.var(noise,ddof=1))
    assert scale==np.sqrt(4.934802200544679/variance)
