"""Exact beta-mixture CDF, tie margins, chronological order and stream tests."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.special import betainc
from scipy.stats import kstest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('empirical_beta_test_models',ROOT/'tools/empirical_beta_innovation_copula/models.py')
models=importlib.util.module_from_spec(spec);sys.modules[spec.name]=models;spec.loader.exec_module(models)
from beta_copula import rank_intervals, sample_copula


def test_generated_joint_cdf_matches_independent_exact_beta_mixture():
    x=np.array([[1,3],[2,7],[3,1],[4,6],[5,5],[6,2],[7,4]],float)
    low,high=rank_intervals(x);n=len(x)
    u=sample_copula(low,high,150000,np.random.default_rng(91),np.random.default_rng(92),np.random.default_rng(93))
    for a,b in ((.1,.1),(.5,.5),(.8,.8),(.1,.8)):
        expected=np.mean(betainc(low[:,0],n+1-low[:,0],a)*betainc(low[:,1],n+1-low[:,1],b))
        actual=np.mean((u[:,0]<=a)&(u[:,1]<=b))
        assert abs(actual-expected)<.004
    for margin in u.T:assert kstest(margin,'uniform').statistic<.006


def test_tie_averaged_integer_ranks_preserve_exact_uniform_marginal_cdf():
    x=np.array([[0,0],[0,0],[0,1],[2,1],[3,1],[3,1],[3,2]],float)
    low,high=rank_intervals(x);n=len(x)
    for j in range(2):
        for u in (.01,.1,.5,.9,.99):
            expected=np.mean([np.mean(betainc(np.arange(a,b+1),n+1-np.arange(a,b+1),u)) for a,b in zip(low[:,j],high[:,j])])
            np.testing.assert_allclose(expected,u,atol=2e-15)
    sampled=sample_copula(low,high,150000,np.random.default_rng(4),np.random.default_rng(5),np.random.default_rng(6))
    for margin in sampled.T:assert kstest(margin,'uniform').statistic<.006


def test_constant_margin_has_no_spurious_dependency_and_remains_uniform():
    x=np.column_stack((np.arange(20),np.ones(20)))
    low,high=rank_intervals(x)
    sampled=sample_copula(low,high,100000,np.random.default_rng(84),np.random.default_rng(85),np.random.default_rng(86))
    assert abs(np.corrcoef(sampled.T)[0,1])<.012
    assert kstest(sampled[:,1],'uniform').statistic<.008


def test_ranks_invariant_to_increasing_marginal_transforms_and_no_bandwidth():
    x=np.random.default_rng(121).normal(size=(500,3))
    for a,b in zip(rank_intervals(x),rank_intervals(np.exp(x))):np.testing.assert_array_equal(a,b)


def test_memory_blocks_and_horizon_prefix_reproduce_identical_streams(monkeypatch):
    low,high=rank_intervals(np.array([[0,2],[0,1],[1,1],[2,0],[3,3]],float))
    monkeypatch.setattr(models,'configuration',lambda *args:(low,high,None,None))
    models.beta_uniforms.cache_clear()
    actual=models.beta_uniforms((5,2),b'',3,1030,43)
    expected=sample_copula(low,high,1030*3,np.random.default_rng(43),
        np.random.default_rng(np.random.SeedSequence([43,0x42455441])),
        np.random.default_rng(np.random.SeedSequence([43,0x54494553])))
    np.clip(expected,1e-8,1-1e-8,out=expected)
    expected=expected.reshape(1030,3,2).transpose(1,0,2)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(actual[:,:100],models.beta_uniforms((5,2),b'',3,100,43))
    models.beta_uniforms.cache_clear()


def test_disabled_beta_copula_restores_M251_paths_after_enabled_cache(monkeypatch):
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
