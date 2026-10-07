"""Copula input alignment contracts without changing asset fit algorithms."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtri

ROOT=Path(__file__).resolve().parents[1]


def load(name,directory):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/directory/'models.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


gaussian=load('innovation_gaussian_test_models','innovation_shrinkage_copula_rough')
student=load('innovation_student_test_models','innovation_student_copula_rough')


def histories():
    rng=np.random.default_rng(83)
    z=rng.normal(size=(2500,3));z[:,1]=.65*z[:,0]+np.sqrt(1-.65**2)*z[:,1]
    scales=np.ones_like(z);scales[:,1]=np.where(np.arange(len(z))%100<50,.05,8.)
    return z*scales,z


def fitter(assets,pools):
    lookup={x.tobytes():pools[:,i] for i,x in enumerate(assets.T)}
    def refit(data):return {'innovation_pool':lookup[data]},{}
    return refit


def test_alignment_preserves_full_chronology_and_asset_order():
    assets,pools=histories()
    result=gaussian.aligned_pools(assets,fitter(assets,pools))
    np.testing.assert_array_equal(result,pools)
    with pytest.raises(ArithmeticError,match='chronological'):
        gaussian.aligned_pools(assets,lambda data:({'innovation_pool':pools[:-1,0]},{}))


def test_gaussian_scatter_uses_standardized_innovations_not_raw_returns(monkeypatch):
    assets,pools=histories();monkeypatch.setattr(gaussian.parent.body,'refit',fitter(assets,pools))
    gaussian.aligned_configuration.cache_clear()
    c,root,weight,_=gaussian.aligned_configuration(assets.shape,assets.tobytes())
    u=gaussian.parent.body.shell.controls.dg.pseudo_observations(pools)
    expected=gaussian.parent.shrunk_correlation(ndtri(u))
    for actual,wanted in zip((c,root,weight),expected):np.testing.assert_array_equal(actual,wanted)
    raw=gaussian.original_configuration(assets.shape,assets.tobytes())[0]
    assert abs(c[0,1]-.65)<.035 and abs(raw[0,1]-c[0,1])>.07
    gaussian.aligned_configuration.cache_clear()


def test_student_scatter_and_tail_likelihood_use_the_same_aligned_input(monkeypatch):
    assets,pools=histories();monkeypatch.setattr(student.parent.parent.body,'refit',fitter(assets,pools))
    student.aligned_configuration.cache_clear()
    c,root,weight,nu,diagnostics,_=student.aligned_configuration(assets.shape,assets.tobytes())
    u=student.parent.parent.body.shell.controls.dg.pseudo_observations(pools)
    expected=student.parent.parent.shrunk_correlation(ndtri(u))
    for actual,wanted in zip((c,root,weight),expected):np.testing.assert_array_equal(actual,wanted)
    expected_nu,expected_diagnostics=student.parent.fit_degrees_of_freedom(u,c)
    assert nu==expected_nu and diagnostics==expected_diagnostics
    student.aligned_configuration.cache_clear()


def test_switching_alignment_does_not_reuse_wrong_gaussian_paths(monkeypatch):
    assets,pools=histories();monkeypatch.setattr(gaussian.parent.body,'refit',fitter(assets,pools))
    gaussian.aligned_configuration.cache_clear();args=(assets.shape,assets.tobytes(),10,30,57)
    gaussian.select_copula(False);baseline=gaussian.parent.gaussian_uniforms(*args).copy()
    gaussian.select_copula(True);aligned=gaussian.parent.gaussian_uniforms(*args).copy()
    gaussian.select_copula(False);again=gaussian.parent.gaussian_uniforms(*args).copy()
    assert not np.array_equal(baseline,aligned)
    np.testing.assert_array_equal(baseline,again)
    gaussian.select_copula(True);gaussian.clear_path_cache();gaussian.aligned_configuration.cache_clear()


def test_switching_alignment_does_not_reuse_wrong_student_paths(monkeypatch):
    assets,pools=histories();monkeypatch.setattr(student.parent.parent.body,'refit',fitter(assets,pools))
    student.aligned_configuration.cache_clear();args=(assets.shape,assets.tobytes(),10,30,57)
    student.select_copula(False);baseline=student.parent.student_uniforms(*args).copy()
    student.select_copula(True);aligned=student.parent.student_uniforms(*args).copy()
    student.select_copula(False);again=student.parent.student_uniforms(*args).copy()
    assert not np.array_equal(baseline,aligned)
    np.testing.assert_array_equal(baseline,again)
    student.select_copula(True);student.clear_path_cache();student.aligned_configuration.cache_clear()
