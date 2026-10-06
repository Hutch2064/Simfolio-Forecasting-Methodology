"""MAP rough inference and exact predecessor-backbone correctness."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]/'tools/dynamic_rough_inference'
sys.path.insert(0,str(HERE))
from posterior import fit_map,InferenceLimit


def test_map_recovers_known_correlated_mode_and_reports_failure():
    mean=np.array([.3,-.4]);inverse=np.linalg.inv(np.array([[.5,.2],[.2,.7]]))
    result=fit_map(lambda x:-.5*((x-mean)@inverse@(x-mean)),[0.,0.],[(-10,10)]*2)
    np.testing.assert_allclose(result['map'],mean,atol=1e-5)
    assert result['optimizer']['success'] and not result['posterior_uncertainty']
    with pytest.raises(InferenceLimit):fit_map(lambda x:-x@x,[.5],[(-10,10)],max_seconds=0.)


def test_failed_line_search_retries_same_target(monkeypatch):
    import posterior
    from scipy.optimize import OptimizeResult
    original=posterior.minimize
    calls=[]
    def minimize(target,start,**kwargs):
        calls.append(kwargs)
        if len(calls)==1:
            return OptimizeResult(success=False,fun=target(start),x=np.asarray(start),nit=1,message='line search interrupted')
        return original(target,start,**kwargs)
    monkeypatch.setattr(posterior,'minimize',minimize)
    result=posterior.fit_map(lambda x:-(x[0]-.3)**2,[0.],[(-1.,1.)])
    np.testing.assert_allclose(result['map'],[.3],atol=1e-5)
    assert len(result['optimizer']['attempts'])==2
    assert calls[0]['options']['ftol']==calls[1]['options']['ftol']
    assert calls[0]['options']['gtol']==calls[1]['options']['gtol']


@pytest.fixture(scope='module')
def forecast_models():
    spec=importlib.util.spec_from_file_location('single_sv_rough_test',HERE/'models.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def test_rough_filter_likelihood_matches_dense_gaussian(forecast_models):
    from scipy.stats import multivariate_normal
    y=np.random.default_rng(23).normal(size=13)
    phi=np.array([.2,.7,.97]);w=np.ones(3);q=np.diag([.4,.2,.01]);level=-.3
    stationary=q/(1-phi[:,None]*phi[None,:])
    covariance=np.empty((len(y),len(y)))
    for i in range(len(y)):
        for j in range(len(y)):
            covariance[i,j]=w@(phi**abs(i-j)*stationary)@w
    covariance+=np.eye(len(y))*4.934802200544679
    expected=multivariate_normal.logpdf(y,np.full(len(y),level),covariance)
    actual=forecast_models.overlay.filter_rough(y,phi,w,q,level)[0]
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)


@pytest.mark.parametrize('scale',[.012,.8])
def test_predecessor_backbone_is_exact(forecast_models,scale):
    from simfolio_forecasting_methodology.models.asset_level.filtered_innovation_moment_sv import sorted_moment_marginals
    from simfolio_forecasting_methodology.models.numerical.dynamic_gaussian import map_uniforms_to_marginal_paths
    x=np.random.default_rng(37).normal(.0002,scale,300)
    u=np.random.default_rng(39).random((24,17))
    fit=forecast_models.shell.bd.fit_bdes_fastmap(x,filtered_innovations=True,fixed_mean=True)
    reference=map_uniforms_to_marginal_paths(sorted_moment_marginals(fit,24,17)[:,:,None],u[:,:,None])[:,:,0]
    _,actual=forecast_models.predecessor_asset_paths(x.tobytes(),u)
    np.testing.assert_array_equal(actual,reference)


def test_normalized_overlay_preserves_conditional_second_moment():
    # Exact Gaussian integration: E[exp(X-v/2)] = 1 for X~N(0,v).
    from scipy.special import roots_hermitenorm
    nodes,mass=roots_hermitenorm(32);mass/=mass.sum()
    for variance in (.01,.5,2.):
        multiplier=np.exp(.5*np.sqrt(variance)*nodes-.25*variance)
        np.testing.assert_allclose(mass@(multiplier**2),1.,rtol=1e-13)


def test_dynamic_kernel_accuracy_over_every_requested_daily_lag(forecast_models):
    from dynamic import selected_kernel
    lags=np.arange(401)
    for hurst,kappa in [(.03,1/2520),(.1,1/63),(.49,.5)]:
        phi,mass,record=selected_kernel(hurst,kappa,400)
        actual=(phi[None,:]**lags[:,None])@mass
        expected=forecast_models.overlay.exact_covariance(hurst,kappa,lags)
        assert np.max(abs(actual-expected))<=record['autocorrelation_error_upper_bound']
        assert record['autocorrelation_error_upper_bound']<=.001


def test_only_requested_asset_candidate_is_enabled(forecast_models):
    assert len(forecast_models.CANDIDATES)==1
    assert isinstance(forecast_models.CANDIDATES[0],forecast_models.PredecessorRoughMAP)


@pytest.mark.parametrize('horizon',[31,1100])
def test_compiled_policy_rejoin_matches_reference_exactly(forecast_models,horizon):
    from predictive_sv_candidates import fast_rebalanced
    paths=np.random.default_rng(52).normal(0,.015,(9,horizon,6))
    weights=np.array([.1,.2,.05,.35,.15,.15]);mask=np.arange(horizon)%21==0
    expected=fast_rebalanced(paths,weights,mask)
    actual=forecast_models.shell.rejoin(paths,weights,mask)
    assert actual.tobytes()==expected.tobytes()
