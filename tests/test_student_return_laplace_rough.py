"""Independent Student likelihood/gradient and Gaussian endpoint references."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import toeplitz
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/student_return_laplace_rough'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/raw_return_laplace_rough'))
import laplace_sv
import student_laplace_sv as student


@pytest.mark.parametrize('u', [0., 1e-7, .05, .3, .49])
def test_student_normalizer_and_observation_derivatives(u):
    h=np.array([-.8, .2, 2.]); squared=np.array([0., .4, 50.])
    energy,g,curvature,_,energy_u,g_u,curvature_u=student.observation_terms(h,squared,u)
    logc,dc=student.normalizer(u)
    if u:
        expected=-t.logpdf(np.sqrt(squared)*np.exp(-.5*h)/np.sqrt(1-2*u), 1/u)+.5*np.log(1-2*u)+.5*h
        np.testing.assert_allclose(energy-logc,expected,atol=2e-9)
    numerical=approx_derivative(lambda x:student.observation_terms(x,squared,u)[0],h).diagonal()
    np.testing.assert_allclose(g,numerical,atol=2e-7)
    numerical=approx_derivative(lambda x:student.observation_terms(x,squared,u)[1],h).diagonal()
    np.testing.assert_allclose(curvature,numerical,atol=2e-7)
    for actual,index in [(energy_u,0),(g_u,1),(curvature_u,2)]:
        numerical=approx_derivative(lambda x,index=index:student.observation_terms(h,squared,x[0])[index], [u], bounds=(0., .5), abs_step=1e-7).ravel()
        np.testing.assert_allclose(actual,numerical,rtol=2e-5,atol=2e-5)
    if u:
        numerical=approx_derivative(lambda x:student.normalizer(x[0])[0],[u],abs_step=min(1e-7,u/4)).item()
        assert dc==pytest.approx(numerical,rel=2e-5,abs=2e-5)


@pytest.mark.parametrize('n,phi,u', [(1,.5,.05),(9,.8,.2),(101,.98,.1)])
def test_student_sparse_laplace_matches_dense_and_parameter_derivatives(n,phi,u):
    squared=np.random.default_rng(37).normal(size=n)**2
    squared[-1]=40.
    level,eta=.1,.3
    covariance=toeplitz(eta*eta/(1-phi*phi)*phi**np.arange(n));q=np.linalg.inv(covariance)
    def objective(h):
        return .5*((h-level)@q@(h-level))+student.observation_terms(h,squared,u)[0].sum()
    def gradient(h):return q@(h-level)+student.observation_terms(h,squared,u)[1]
    ref=minimize(objective,np.full(n,level),jac=gradient,method='BFGS',options={'gtol':1e-8})
    h=ref.x;hessian=q+np.diag(student.observation_terms(h,squared,u)[2])
    expected=n*student.normalizer(u)[0]-.5*2*np.log(np.diag(np.linalg.cholesky(covariance))).sum()-objective(h)-.5*2*np.log(np.diag(np.linalg.cholesky(hessian))).sum()
    result=student.likelihood_gradient(squared,level,phi,eta,u)
    assert result[4]
    assert result[0]==pytest.approx(expected,abs=2e-7)
    np.testing.assert_allclose(result[2],h,atol=1e-5,rtol=1e-5)
    def target(x):return student.likelihood_gradient(squared,x[0],x[1],np.exp(x[2]),x[3])[0]
    numerical=approx_derivative(target,[level,phi,np.log(eta),u]).ravel()
    np.testing.assert_allclose(result[1],numerical,atol=3e-5,rtol=3e-5)


def test_gaussian_endpoint_matches_existing_laplace_target():
    squared=np.random.default_rng(83).normal(size=101)**2
    expected=laplace_sv.likelihood_gradient(squared,.1,.9,.3)
    actual=student.likelihood_gradient(squared,.1,.9,.3,0.)
    np.testing.assert_allclose(actual[0],expected[0],atol=1e-12)
    np.testing.assert_allclose(actual[1][:3],expected[1],atol=1e-12)
    np.testing.assert_allclose(actual[2],expected[2],atol=1e-12)

def test_century_mean_curve_remains_the_original_predecessor():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / 'tools/student_return_laplace_rough/models.py'
    spec = importlib.util.spec_from_file_location('student_laplace_mean_guard', path)
    model = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = model
    spec.loader.exec_module(model)
    model.initialize()
    x = np.random.default_rng(492).normal(.0003, .01, 507)
    data = x.tobytes()
    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
        moment_return_curves,
    )

    expected, _ = moment_return_curves(model.original_fit(data), 25200)
    uniforms = np.full((8, 25200), .5)
    actual, paths = model.asset_paths(data, uniforms)
    assert expected.tobytes() == actual.tobytes()
    assert np.isfinite(paths).all()
    assert model.refit(data)['student_return_laplace_fit']['success']


def test_map_fit_does_not_accept_stalled_gaussian_endpoint():
    rng=np.random.default_rng(910)
    h=np.empty(1000);h[0]=-.4
    for i in range(1,len(h)):
        h[i]=-.4+.95*(h[i-1]+.4)+.2*rng.normal()
    eps=np.exp(.5*h)*rng.standard_t(5,len(h))*np.sqrt(3/5)
    squared=eps*eps;proxy=np.log(np.maximum(squared,np.finfo(float).tiny))+1.2703628454614782
    theta,_,_,diagnostics=student.fit(squared,proxy,(-.8,.3,1.33))
    assert diagnostics['success']
    assert diagnostics['max_projected_mean_gradient']<=1e-6
    initial_gradient=student.likelihood_gradient(squared,-.8,.3,1.33,0.)[1]
    assert np.max(np.abs(initial_gradient))/len(squared)>1e-3
    assert 0<=theta[3]<.5
