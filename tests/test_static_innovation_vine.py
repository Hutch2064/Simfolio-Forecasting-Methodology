"""Independent copula-law and asset-order/stream wiring checks."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyvinecopulib as pv
from scipy.stats import kstest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('static_vine_test_models',ROOT/'tools/static_innovation_vine/models.py')
models=importlib.util.module_from_spec(spec);sys.modules[spec.name]=models;spec.loader.exec_module(models)


def clayton_vine(order=(1,2)):
    return pv.Vinecop.from_structure(pv.RVineStructure.from_order(order),
        pair_copulas=[[pv.Bicop(pv.BicopFamily.clayton,parameters=np.array([[2.]]))]])


def test_inverse_mapping_has_uniform_margins_and_correct_asymmetric_joint_cdf(monkeypatch):
    vine=clayton_vine()
    monkeypatch.setattr(models,'configuration',lambda *args:(vine,None,None))
    models.vine_uniforms.cache_clear()
    u=models.vine_uniforms((100,2),b'',200,500,191).reshape(-1,2)
    for margin in u.T:assert kstest(margin,'uniform').statistic<.008
    for a,b in ((.1,.1),(.5,.5),(.9,.9),(.1,.8)):
        expected=(a**-2+b**-2-1)**-.5
        actual=np.mean((u[:,0]<=a)&(u[:,1]<=b))
        assert abs(actual-expected)<.004
    # Clayton has lower-tail dependence, distinct from its upper tail.
    lower=np.mean((u[:,0]<.05)&(u[:,1]<.05))
    upper=np.mean((u[:,0]>.95)&(u[:,1]>.95))
    assert lower>3*upper
    assert abs(np.corrcoef(u[::2,0],u[1::2,0])[0,1])<.015
    models.vine_uniforms.cache_clear()


def test_blocked_inverse_rosenblatt_preserves_original_asset_order_and_rng_prefix(monkeypatch):
    vine=clayton_vine(order=(2,1))
    monkeypatch.setattr(models,'configuration',lambda *args:(vine,None,None))
    models.vine_uniforms.cache_clear()
    # This crosses the 1024-day memory boundary; directly transform the same
    # flattened shocks in one batch as an independent mapping reference.
    from scipy.special import ndtr
    args=((100,2),b'',3,1030,61)
    actual=models.vine_uniforms(*args)
    shocks=ndtr(np.random.default_rng(61).normal(size=(1030,3,2)))
    expected=vine.inverse_rosenblatt(np.asfortranarray(shocks.reshape(-1,2))).reshape(1030,3,2).transpose(1,0,2)
    np.clip(expected,1e-8,1-1e-8,out=expected)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(actual[:,:100],models.vine_uniforms(args[0],b'',3,100,61))
    models.vine_uniforms.cache_clear()


def test_training_selection_uses_full_standardized_innovations_and_records_native_bounds(monkeypatch):
    rng=np.random.default_rng(187);g=rng.gamma(.5,1.,size=(600,1))
    u=(1+rng.exponential(size=(600,2))/g)**-.5
    assets=rng.normal(size=(600,2))
    lookup={x.tobytes():u[:,i] for i,x in enumerate(assets.T)}
    monkeypatch.setattr(models.body,'refit',lambda data:({'innovation_pool':lookup[data]},{}))
    models.configuration.cache_clear()
    vine,digest,diagnostics=models.configuration(assets.shape,assets.tobytes())
    import hashlib
    assert digest==hashlib.sha256(u.tobytes()).hexdigest()
    assert vine.dim==2 and len(vine.pair_copulas)==1
    assert 'tll' not in diagnostics['family_set'] and len(diagnostics['family_set'])==12
    assert diagnostics['preselect_families'] is False
    assert diagnostics['library_parameter_bounds']['student']['lower'][1]==[2.]
    assert diagnostics['library_parameter_bounds']['student']['upper'][1]==[50.]
    assert np.isfinite(diagnostics['log_likelihood']) and np.isfinite(diagnostics['bic'])
    models.configuration.cache_clear()


def test_disabled_vine_restores_M251_paths_after_enabled_cache(monkeypatch):
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
