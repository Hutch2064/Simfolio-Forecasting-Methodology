"""Independent density, gradient, identification, and recovery contracts."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.special import ndtri, stdtr, stdtrit
from scipy.stats import multivariate_normal, multivariate_t, t

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('joint_copula_test',ROOT/'tools/joint_innovation_student_copula/joint_fit.py')
fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(fit)


def sample(n=3000,nu=5,seed=172):
    rng=np.random.default_rng(seed)
    c=np.array([[1.,.65,-.2],[.65,1.,.15],[-.2,.15,1.]])
    z=rng.normal(size=(n,3))@np.linalg.cholesky(c).T
    z/=np.sqrt(rng.chisquare(nu,size=(n,1))/nu)
    return stdtr(nu,z),c


@pytest.mark.parametrize('nu',[.8,5.,50.,np.inf])
def test_density_matches_independent_scipy_multivariate_minus_marginals(nu):
    u,c=sample(n=150)
    likelihood=fit.JointLikelihood(u)
    if np.isinf(nu):
        z=ndtri(u)
        expected=np.mean(multivariate_normal.logpdf(z,cov=c)-np.sum(-.5*z*z-.5*np.log(2*np.pi),axis=1))
    else:
        z=stdtrit(nu,u)
        expected=np.mean(multivariate_t.logpdf(z,shape=c,df=nu)-t.logpdf(z,df=nu).sum(axis=1))
    np.testing.assert_allclose(likelihood.density(c,nu),expected,atol=2e-13)


@pytest.mark.parametrize('student',[False,True])
def test_coordinate_gradient_matches_independent_finite_differences(student):
    u,c=sample(n=500);likelihood=fit.JointLikelihood(u)
    x=fit.correlation_coordinates(c)
    if student:x=np.r_[x,np.log(4.)]
    actual=likelihood.objective(x,student)[1]
    expected=approx_derivative(lambda a:likelihood.objective(a,student)[0],x,method='3-point').ravel()
    np.testing.assert_allclose(actual,expected,atol=2e-8,rtol=2e-6)


def test_parameterization_covers_pd_correlation_and_preserves_unit_diagonal():
    rng=np.random.default_rng(154)
    a=rng.normal(size=(7,7));c=a@a.T;c/=np.sqrt(np.diag(c))[:,None]*np.sqrt(np.diag(c))[None,:]
    reconstructed,_,_=fit.correlation_from_coordinates(fit.correlation_coordinates(c),7)
    np.testing.assert_allclose(reconstructed,c,atol=4e-15)
    np.linalg.cholesky(reconstructed)


def test_joint_fit_recovers_known_student_dependence_and_improves_start_likelihood():
    u,c=sample(n=6000)
    estimated,root,nu,diagnostics=fit.fit_joint_copula(u,np.eye(3),10.)
    np.testing.assert_allclose(estimated,c,atol=.035)
    np.testing.assert_allclose(root@root.T,estimated,atol=2e-15)
    assert 3.5<nu<7
    assert diagnostics['log_likelihood']>diagnostics['initial_conditional_log_likelihood']
    assert diagnostics['log_likelihood']>diagnostics['gaussian_log_likelihood']
    assert diagnostics['student_maximum_gradient']<2e-6


def test_large_df_density_approaches_gaussian_without_gamma_cancellation():
    u,c=sample(n=200);likelihood=fit.JointLikelihood(u)
    np.testing.assert_allclose(likelihood.density(c,1e9),likelihood.density(c,np.inf),atol=2e-8)


@pytest.mark.parametrize('kind',['constant','duplicate','reflected'])
def test_unidentified_or_singular_data_fails_without_rho_cap_or_jitter(kind):
    u=np.arange(1,201)/201
    second=np.full_like(u,.5) if kind=='constant' else u if kind=='duplicate' else 1-u
    with pytest.raises(ValueError):fit.JointLikelihood(np.column_stack((u,second)))


def test_disabled_joint_fit_restores_M251_paths_after_enabled_cache(monkeypatch):
    spec=importlib.util.spec_from_file_location('joint_copula_test_models',ROOT/'tools/joint_innovation_student_copula/models.py')
    models=importlib.util.module_from_spec(spec);sys.modules[spec.name]=models;spec.loader.exec_module(models)
    pools,_=sample(n=700);assets=pools*np.linspace(.1,3,len(pools))[:,None]
    lookup={x.tobytes():pools[:,i] for i,x in enumerate(assets.T)}
    monkeypatch.setattr(models.parent.parent.parent.body,'refit',lambda data:({'innovation_pool':lookup[data]},{}))
    args=(assets.shape,assets.tobytes(),8,20,74)
    models.select_copula(False);baseline=models.parent.parent.student_uniforms(*args).copy()
    models.select_copula(True);candidate=models.parent.parent.student_uniforms(*args).copy()
    models.select_copula(False);again=models.parent.parent.student_uniforms(*args).copy()
    assert not np.array_equal(baseline,candidate)
    np.testing.assert_array_equal(baseline,again)
    models.select_copula(True);models.clear_path_cache()
