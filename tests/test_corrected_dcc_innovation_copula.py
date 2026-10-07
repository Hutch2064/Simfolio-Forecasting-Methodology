"""Independent conditional likelihood, recursion and marginal-law references."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import kstest, multivariate_normal, norm

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('corrected_dcc_test_models',ROOT/'tools/corrected_dcc_innovation_copula/models.py')
models=importlib.util.module_from_spec(spec);sys.modules[spec.name]=models;spec.loader.exec_module(models)


def dense_likelihood(z,s,a,b):
    q=s.copy();total=0.
    for row in z:
        sd=np.sqrt(np.diag(q));r=q/sd[:,None]/sd[None,:]
        total-=multivariate_normal.logpdf(row,cov=r)-norm.logpdf(row).sum()
        w=sd*row;q=(1-a-b)*s+a*np.outer(w,w)+b*q
    return total,q


def test_compiled_likelihood_and_analytic_recursive_gradient_match_independent_references():
    z=np.random.default_rng(128).normal(size=(37,3));s=np.array([[1,.3,-.2],[.3,1,.1],[-.2,.1,1.]])
    for theta in (np.array([.1,.7]),np.array([.02,.97]),np.array([.3,0.])):
        value,gradient,q=models.state.likelihood(z,s,*theta,s)
        reference,reference_q=dense_likelihood(z,s,*theta)
        np.testing.assert_allclose(value,reference,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(q,reference_q,rtol=1e-13,atol=1e-13)
        fd=[]
        for j in range(2):
            delta=np.zeros(2);delta[j]=1e-6
            fd.append((models.state.likelihood(z,s,*(theta+delta),s)[0]-models.state.likelihood(z,s,*(theta-delta),s)[0])/2e-6)
        np.testing.assert_allclose(gradient,fd,rtol=2e-6,atol=1e-7)


def test_no_shock_recursion_is_constant_for_every_persistence():
    z=np.random.default_rng(82).normal(size=(50,2));s=np.array([[1,.6],[.6,1.]])
    base=dense_likelihood(z,s,0.,0.)[0]
    for b in (0.,.5,.99):
        loss,_,q=models.state.likelihood(z,s,0.,b,s)
        np.testing.assert_allclose(loss,base,atol=1e-12)
        np.testing.assert_allclose(q,s,atol=1e-14)


def test_native_paths_match_scalar_cholesky_reference_and_exact_blocking():
    rng=np.random.default_rng(90);normal=rng.normal(size=(1030,3,3));original=normal.copy()
    s=np.array([[1,.3,-.2],[.3,1,.1],[-.2,.1,1.]])
    initial=2*s;states=np.repeat(initial[None,:,:],3,axis=0);reference_states=states.copy()
    expected=normal.copy();a,b=.04,.94
    for t in range(len(normal)):
        for k in range(3):
            q=reference_states[k];w=np.linalg.cholesky(q)@original[t,k]
            expected[t,k]=w/np.sqrt(np.diag(q))
            reference_states[k]=(1-a-b)*s+a*np.outer(w,w)+b*q
    backend=models.native.load_paths();backend.scores(normal,s,states,a,b)
    np.testing.assert_allclose(normal,expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(states,reference_states,rtol=1e-12,atol=1e-12)
    blocked=original.copy();blocked_states=np.repeat(initial[None,:,:],3,axis=0)
    backend.scores(blocked[:1024],s,blocked_states,a,b)
    backend.scores(blocked[1024:],s,blocked_states,a,b)
    np.testing.assert_array_equal(normal,blocked);np.testing.assert_array_equal(states,blocked_states)


def test_conditional_margins_are_standard_normal_and_Q_correction_has_expected_update():
    # Q need not have unit diagonal; conditional z always does. Under that
    # law E[w w'|Q]=Q, giving E[Q_next|Q]=(1-a-b)S+(a+b)Q.
    s=np.array([[1,.4],[.4,1.]]);q=np.array([[2.,.7],[.7,.5]])
    sims=120000;states=np.repeat(q[None,:,:],sims,axis=0)
    z=np.random.default_rng(42).normal(size=(1,sims,2));a,b=.1,.8
    models.native.load_paths().scores(z,s,states,a,b)
    for margin in z[0].T:assert kstest(margin,'norm').statistic<.007
    correlation=q/np.sqrt(np.diag(q))[:,None]/np.sqrt(np.diag(q))[None,:]
    np.testing.assert_allclose(np.corrcoef(z[0].T),correlation,atol=.007)
    np.testing.assert_allclose(states.mean(axis=0),(1-a-b)*s+(a+b)*q,atol=.005)


def test_conditional_parameter_fit_recovers_dynamic_process_and_keeps_static_endpoint():
    s=np.array([[1,.45],[.45,1.]]);a,b=.06,.9
    normals=np.random.default_rng(156).normal(size=(18000,1,2));states=s[None,:,:].copy()
    models.native.load_paths().scores(normals,s,states,a,b)
    theta,q,diagnostics=models.state.fit(normals[:,0,:],s)
    assert abs(theta[0]-a)<.025 and abs(theta[1]-b)<.06
    assert theta.sum()<1. and np.min(np.linalg.eigvalsh(q))>0.
    assert diagnostics['training_nll']<=diagnostics['static_training_nll']


def test_uniform_stream_preserves_asset_order_and_horizon_prefix(monkeypatch):
    s=np.array([[1,.5],[.5,1.]]);q=2*s
    monkeypatch.setattr(models,'configuration',lambda *args:(s,q,.1,.8,None,None))
    models.dcc_uniforms.cache_clear()
    args=((100,2),b'',4,1030,73);actual=models.dcc_uniforms(*args)
    z=np.random.default_rng(73).normal(size=(1030,4,2));states=np.repeat(q[None,:,:],4,axis=0)
    models.native.load_paths().scores(z,s,states,.1,.8)
    expected=ndtr(z).transpose(1,0,2);np.clip(expected,1e-8,1-1e-8,out=expected)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(actual[:,:100],models.dcc_uniforms(args[0],b'',4,100,73))
    models.dcc_uniforms.cache_clear()


def test_disabled_candidate_restores_M251_uniforms_after_dynamic_cache(monkeypatch):
    rng=np.random.default_rng(61);pools=rng.normal(size=(600,3));pools[:,1]+=.7*pools[:,0]
    assets=pools*np.linspace(.2,2,len(pools))[:,None]
    lookup={x.tobytes():pools[:,i] for i,x in enumerate(assets.T)}
    monkeypatch.setattr(models.body,'refit',lambda data:({'innovation_pool':lookup[data]},{}))
    models.configuration.cache_clear();models.parent.aligned_configuration.cache_clear()
    args=(assets.shape,assets.tobytes(),7,20,53)
    models.select_copula(False);baseline=models.shell.controls.gaussian_uniforms(*args).copy()
    models.select_copula(True);candidate=models.shell.controls.gaussian_uniforms(*args).copy()
    models.select_copula(False);again=models.shell.controls.gaussian_uniforms(*args).copy()
    assert not np.array_equal(baseline,candidate)
    np.testing.assert_array_equal(baseline,again)
    models.select_copula(True);models.clear_path_cache()
    models.configuration.cache_clear();models.parent.aligned_configuration.cache_clear()
