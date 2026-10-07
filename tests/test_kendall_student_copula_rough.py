"""Independent sign-statistic, elliptical-law and execution contracts."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('kendall_copula_test_models',ROOT/'tools/kendall_student_copula_rough/models.py')
models=importlib.util.module_from_spec(spec);sys.modules[spec.name]=models;spec.loader.exec_module(models)
from scatter import kendall_correlation, positive_correlation


def test_tied_sign_statistic_matches_explicit_all_pairs_without_jitter():
    x=np.array([[0,2,0],[0,1,0],[1,1,0],[2,0,0],[3,3,0],[3,1,0]],float)
    c,root,repair=kendall_correlation(x)
    tau=np.mean([np.sign(x[i,0]-x[j,0])*np.sign(x[i,1]-x[j,1]) for i in range(len(x)) for j in range(i)])
    np.testing.assert_allclose(c[0,1],np.sin(np.pi*tau/2),atol=1e-15)
    np.testing.assert_array_equal(c[2],[0.,0.,1.])
    np.testing.assert_allclose(root@root.T,c,atol=1e-14)
    assert not repair['eigenvalue_repair_applied']


def test_estimator_is_invariant_to_strictly_increasing_marginal_transforms():
    x=np.random.default_rng(9).normal(size=(1000,4))
    first=kendall_correlation(x)
    second=kendall_correlation(np.exp(x))
    for a,b in zip(first[:2],second[:2]):np.testing.assert_array_equal(a,b)


def test_student_elliptical_scatter_recovery_does_not_assume_gaussian_ranks():
    rng=np.random.default_rng(765);c=np.array([[1.,.7,-.2],[.7,1.,.1],[-.2,.1,1.]])
    x=rng.normal(size=(30000,3))@np.linalg.cholesky(c).T
    x/=np.sqrt(rng.chisquare(4,size=(len(x),1))/4)
    estimate,_,repair=kendall_correlation(x)
    np.testing.assert_allclose(estimate,c,atol=.02)
    assert not repair['eigenvalue_repair_applied']


def test_eigenvalue_repair_preserves_unit_diagonal_and_root():
    c,root,repair=positive_correlation(np.array([[1.,.9,.9],[.9,1.,-.9],[.9,-.9,1.]]))
    assert repair['eigenvalue_repair_applied']
    np.linalg.cholesky(c)
    np.testing.assert_array_equal(np.diag(c),np.ones(3))
    np.testing.assert_allclose(root@root.T,c,atol=1e-14)


def test_duplicate_reflected_and_constant_histories_do_not_create_spurious_df():
    u=np.arange(1,501.)/501
    uniforms=np.column_stack((u,u,1-u,np.full_like(u,.5)))
    c,_,_=kendall_correlation(uniforms)
    nu,diagnostics=models.fit_informative_degrees(uniforms,c)
    assert np.isinf(nu)
    assert diagnostics['informative_dimensions']==1
    assert diagnostics['excluded_constant_columns']==[3]
    assert diagnostics['excluded_duplicate_or_reflected_columns']==[1,2]


def test_disabled_scatter_restores_M251_paths_after_enabled_cache(monkeypatch):
    rng=np.random.default_rng(64);pools=rng.normal(size=(900,3));pools[:,1]+=.6*pools[:,0]
    assets=pools*np.linspace(.1,3,len(pools))[:,None]
    lookup={x.tobytes():pools[:,i] for i,x in enumerate(assets.T)}
    monkeypatch.setattr(models.parent.parent.parent.body,'refit',lambda data:({'innovation_pool':lookup[data]},{}))
    models.configuration.cache_clear();models.original_configuration.cache_clear()
    args=(assets.shape,assets.tobytes(),8,20,74)
    models.select_copula(False);baseline=models.parent.parent.student_uniforms(*args).copy()
    models.select_copula(True);candidate=models.parent.parent.student_uniforms(*args).copy()
    models.select_copula(False);again=models.parent.parent.student_uniforms(*args).copy()
    assert not np.array_equal(baseline,candidate)
    np.testing.assert_array_equal(baseline,again)
    models.select_copula(True);models.clear_path_cache()
    models.configuration.cache_clear();models.original_configuration.cache_clear()
