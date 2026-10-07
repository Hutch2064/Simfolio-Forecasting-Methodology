"""Distribution and numerical contracts for randomized return-shock strata."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import kstest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stratified_copula_tests',ROOT/'tools/stratified_innovation_copula/models.py')
models=importlib.util.module_from_spec(spec);sys.modules[spec.name]=models;spec.loader.exec_module(models)
from sampling import copula_uniforms, stratified_uniforms


def test_every_primitive_has_all_strata_with_independent_permutations():
    u=stratified_uniforms(np.random.default_rng(74),20,240,4)
    indices=np.floor(u*240).astype(int)
    np.testing.assert_array_equal(np.sort(indices,axis=1),np.broadcast_to(np.arange(240)[None,:,None],u.shape))
    assert not np.array_equal(indices[0,:,0],indices[0,:,1])
    assert not np.array_equal(indices[0,:,0],indices[1,:,0])
    assert (u>0).all() and (u<1).all()


def test_fixed_path_gaussian_law_is_preserved_across_independent_days():
    c=np.array([[1.,.6,-.2],[.6,1.,.1],[-.2,.1,1.]])
    u=copula_uniforms(np.linalg.cholesky(c),np.inf,12,20000,917)
    z=ndtri(u[0])
    np.testing.assert_allclose(np.cov(z,rowvar=False),c,atol=.035)
    np.testing.assert_allclose(z.mean(axis=0),0.,atol=.025)
    np.testing.assert_allclose(np.diag(np.cov(z,rowvar=False)),1.,atol=.035)
    assert abs(np.corrcoef(z[:-1,0],z[1:,0])[0,1])<.025


def test_fixed_path_student_uniform_marginals_and_tail_dependence():
    c=np.array([[1.,.6],[.6,1.]])
    u=copula_uniforms(np.linalg.cholesky(c),4.,12,20000,918)[0]
    for column in u.T:assert kstest(column,'uniform').statistic<.018
    # Independent reference uses standard normal / chi-square construction.
    rng=np.random.default_rng(414)
    z=rng.normal(size=(100000,2))@np.linalg.cholesky(c).T
    z/=np.sqrt(rng.chisquare(4,size=(len(z),1))/4)
    from scipy.special import stdtr
    reference=stdtr(4,z)
    candidate=np.mean(np.all(u>.95,axis=1))
    expected=np.mean(np.all(reference>.95,axis=1))
    assert abs(candidate-expected)<.004


def test_sampling_reproducible_and_single_requested_path_valid():
    for nu in (np.inf,4.):
        a=copula_uniforms(np.eye(3),nu,1,40,17)
        b=copula_uniforms(np.eye(3),nu,1,40,17)
        np.testing.assert_array_equal(a,b)
        assert a.shape==(1,40,3) and np.isfinite(a).all()


def test_disabling_restores_both_baseline_uniform_generators(monkeypatch):
    assets=np.random.default_rng(54).normal(size=(100,3))
    args=(assets.shape,assets.tobytes(),12,40,26)
    for family,branch in models.branches.items():
        parent=branch.parent
        configuration=(np.eye(3),np.eye(3),0.)
        if family=='student':configuration+= (4.,{})
        monkeypatch.setattr(parent,'configuration',lambda *args,c=configuration:c)
    models.select_copula(False)
    originals={k:b.shell.controls.gaussian_uniforms(*args).copy() for k,b in models.branches.items()}
    models.select_copula(True)
    for k,b in models.branches.items():assert not np.array_equal(originals[k],b.shell.controls.gaussian_uniforms(*args))
    models.select_copula(False)
    for k,b in models.branches.items():np.testing.assert_array_equal(originals[k],b.shell.controls.gaussian_uniforms(*args))
    models.select_copula(True);models.clear_path_cache()
