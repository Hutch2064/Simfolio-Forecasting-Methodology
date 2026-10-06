"""Independent AR covariance reference, likelihood optimum, and mean preservation."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

@pytest.fixture(scope='module')
def model():
    p=Path(__file__).resolve().parents[1]/'tools/har_proxy_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_har_proxy_model',p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m


def test_ar_moments_match_independent_dense_state_recursion(model):
    ar=np.zeros(22);ar[0]=.4;ar[:5]+=.2/5;ar[:22]+=.2/22
    f={'ar':ar,'last':np.linspace(1,2,22),'intercept':.1,'innovation_variance':.3}
    mean,variance=model.har.moments(f,100)
    A=np.zeros((22,22));A[0]=ar;A[1:,:-1]=np.eye(21)
    state=f['last'].copy();cov=np.zeros((22,22));q=np.zeros_like(cov);q[0,0]=.3
    expected=[];v=[]
    for i in range(100):
        state=A@state;state[0]+=.1;cov=A@cov@A.T+q;expected.append(state[0]);v.append(cov[0,0])
    np.testing.assert_allclose(mean,expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(variance,v,rtol=1e-12,atol=1e-12)


def test_likelihood_recovers_interior_ordinary_least_squares(model):
    rng=np.random.default_rng(20);h=np.zeros(8000)
    for t in range(22,len(h)):h[t]=.1+.4*h[t-1]+.25*h[t-5:t].mean()+.15*h[t-22:t].mean()+rng.normal(0,.1)
    f=model.har.fit(h);t=np.arange(22,len(h));x=np.column_stack([np.ones(len(t)),h[t-1],np.array([h[j-5:j].mean() for j in t]),np.array([h[j-22:j].mean() for j in t])])
    reference=np.linalg.lstsq(x,h[t],rcond=None)[0]
    np.testing.assert_allclose(np.r_[f['intercept'],f['beta']],reference,rtol=1e-5,atol=1e-6)
    assert np.all(f['beta']>=0);assert f['beta'].sum()<1


def test_century_mean_unchanged_and_stationary_variance_finite(model):
    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import moment_return_curves
    data=np.random.default_rng(10).normal(.0005,.01,600).tobytes()
    reference,_=moment_return_curves(model.streamed.predecessor_fit(data),25200)
    mean,sd=model.curves(data,25200)
    assert mean.tobytes()==reference.tobytes();assert np.isfinite(sd).all();assert np.all(sd>0)
    fit=model.har_fit(data);assert fit['optimizer_success'];assert fit['beta'].sum()<1
