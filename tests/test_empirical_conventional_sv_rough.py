"""Explicit-noise Kalman likelihood and smoother against dense Gaussian conditioning."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

@pytest.fixture(scope='module')
def model():
    p=Path(__file__).resolve().parents[1]/'tools/empirical_conventional_sv_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_empirical_conventional_model',p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m

@pytest.mark.parametrize('noise',[.5,4.934802200544679,7.])
def test_likelihood_and_smoothing_match_dense_gaussian_conditioning(model,noise):
    y=np.array([1.,.8,1.4,1.3,.9,1.5,1.]);level=1.2;phi=.8;eta=.3
    indices=np.arange(len(y));K=eta*eta/(1-phi*phi)*phi**np.abs(indices[:,None]-indices)
    C=K+noise*np.eye(len(y));residual=y-level
    expected=-.5*(len(y)*np.log(2*np.pi)+np.linalg.slogdet(C)[1]+residual@np.linalg.solve(C,residual))
    likelihood,mean,var=model.gaussian_sv.smooth_states(y,level,phi,eta,noise)
    np.testing.assert_allclose(likelihood,expected,rtol=1e-13)
    np.testing.assert_allclose(mean,level+K@np.linalg.solve(C,residual),rtol=1e-13)
    np.testing.assert_allclose(var,np.diag(K-K@np.linalg.solve(C,K)),rtol=1e-13)


def test_refit_uses_same_empirical_noise_and_preserves_century_return_mean(model):
    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import moment_return_curves
    data=np.random.default_rng(10).normal(.0005,.01,600).tobytes()
    original=model.original_fit(data);_,R=model.original_noise(data)
    fit=model.refit(data);assert fit['empirical_conventional_fit']['success'];assert fit['empirical_conventional_fit']['noise_variance']==R
    expected,_=moment_return_curves(original,25200)
    mean,paths=model.asset_paths(data,np.full((2,25200),.5))
    assert mean.tobytes()==expected.tobytes();assert np.isfinite(paths).all()
    assert model.parent.parent.base.noise.measurement_scale(data)[1]==R
