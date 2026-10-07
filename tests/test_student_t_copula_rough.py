"""Independent density, fitting and simulation contracts for tail dependence."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr, stdtr, stdtrit
from scipy.stats import multivariate_t, t

path=Path(__file__).resolve().parents[1]/'tools/student_t_copula_rough/models.py'
spec=importlib.util.spec_from_file_location('student_copula_test_models',path)
models=importlib.util.module_from_spec(spec);sys.modules[spec.name]=models;spec.loader.exec_module(models)
from tail_fit import CopulaLikelihood, fit_degrees_of_freedom


def test_density_matches_independent_scipy_joint_over_marginals():
    u=np.random.default_rng(41).uniform(.001,.999,size=(80,4))
    c=np.full((4,4),.25);np.fill_diagonal(c,1.)
    likelihood=CopulaLikelihood(u,c)
    for nu in (.6,1.,4.,23.,1000.):
        z=stdtrit(nu,u)
        expected=np.sum(multivariate_t.logpdf(z,shape=c,df=nu)-t.logpdf(z,df=nu).sum(1))
        np.testing.assert_allclose(likelihood(nu),expected,rtol=2e-11,atol=2e-10)
    assert likelihood(0.)==-np.inf


def test_fit_recovers_low_df_without_a_df_two_floor():
    rng=np.random.default_rng(91);c=np.array([[1.,.6],[.6,1.]])
    z=rng.normal(size=(12000,2))@np.linalg.cholesky(c).T
    u=stdtr(.8,z/np.sqrt(rng.gamma(.4,2.5,size=(len(z),1))))
    nu,diagnostics=fit_degrees_of_freedom(u,c)
    assert .7<nu<.9
    likelihood=CopulaLikelihood(u,c)
    assert diagnostics['log_likelihood']>=max(likelihood(v) for v in np.geomspace(.1,1000.,40))


def test_gaussian_limit_matches_gaussian_density():
    u=np.random.default_rng(73).uniform(.01,.99,size=(200,3))
    c=np.array([[1.,.3,.1],[.3,1.,.2],[.1,.2,1.]])
    likelihood=CopulaLikelihood(u,c)
    np.testing.assert_allclose(likelihood(1e6),likelihood(np.inf),atol=.001)
    nu,diagnostics=fit_degrees_of_freedom(u[:,:1],np.ones((1,1)))
    assert np.isinf(nu) and diagnostics['log_likelihood']==0.


def test_gaussian_endpoint_preserves_m248_every_draw(monkeypatch):
    x=np.random.default_rng(3).normal(size=(120,4));shape=x.shape;data=x.tobytes()
    c,root,shrink=models.parent.configuration(shape,data)
    monkeypatch.setattr(models,'configuration',lambda *args:(c,root,shrink,np.inf,{}))
    models.student_uniforms.cache_clear()
    actual=models.student_uniforms(shape,data,3,1030,82)
    expected=models.parent.gaussian_uniforms(shape,data,3,1030,82)
    np.testing.assert_array_equal(actual,expected)
    models.student_uniforms.cache_clear()


def test_uniform_marginals_and_student_joint_tails(monkeypatch):
    x=np.random.default_rng(13).normal(size=(80,2));c=np.array([[1.,.5],[.5,1.]])
    root=np.linalg.cholesky(c);nu=4.
    monkeypatch.setattr(models,'configuration',lambda *args:(c,root,0.,nu,{}))
    models.student_uniforms.cache_clear()
    u=models.student_uniforms(x.shape,x.tobytes(),400,400,97).reshape(-1,2)
    np.testing.assert_allclose(u.mean(0),.5,atol=.003)
    np.testing.assert_allclose(u.var(0),1/12,atol=.001)
    for q in (.01,.05,.5,.95,.99):
        np.testing.assert_allclose((u<q).mean(0),q,atol=.003)
    rng=np.random.default_rng(97);g=ndtr(rng.normal(size=u.shape)@root.T)
    assert np.mean(np.all(u<.01,axis=1))>1.5*np.mean(np.all(g<.01,axis=1))
    models.student_uniforms.cache_clear()
