"""Full-matrix static copula contracts, independent of forecast scoring."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtri

path=Path(__file__).resolve().parents[1]/'tools/static_shrinkage_copula_rough/models.py'
spec=importlib.util.spec_from_file_location('static_copula_test_models',path)
models=importlib.util.module_from_spec(spec);sys.modules[spec.name]=models;spec.loader.exec_module(models)


def test_full_covariance_has_no_three_factor_cap_and_is_psd():
    x=np.random.default_rng(73).normal(size=(300,11))
    correlation,root,weight=models.configuration(x.shape,x.tobytes())
    assert root.shape==(11,11) and np.linalg.matrix_rank(root)==11
    assert 0<=weight<=1
    np.testing.assert_allclose(root@root.T,correlation,atol=2e-14)
    np.testing.assert_array_equal(np.diag(correlation),np.ones(11))


def test_forecast_has_standard_gaussian_marginals_and_fitted_dependence():
    rng=np.random.default_rng(59);x=rng.normal(size=(500,4));x[:,1]+=.7*x[:,0]
    c,_,_=models.configuration(x.shape,x.tobytes())
    u=models.gaussian_uniforms(x.shape,x.tobytes(),300,400,46)
    z=ndtri(u).reshape(-1,4)
    np.testing.assert_allclose(z.mean(0),0,atol=.012)
    np.testing.assert_allclose(z.var(0),1,atol=.018)
    np.testing.assert_allclose(np.corrcoef(z.T),c,atol=.012)
    assert abs(np.corrcoef(ndtri(u)[:,:-1,0].ravel(),ndtri(u)[:,1:,0].ravel())[0,1])<.012


def test_blocking_preserves_every_rng_draw_and_asset_mapping():
    x=np.random.default_rng(31).normal(size=(120,5));_,root,_=models.configuration(x.shape,x.tobytes())
    u=models.gaussian_uniforms(x.shape,x.tobytes(),3,1030,92)
    from scipy.special import ndtr
    z=np.random.default_rng(92).normal(size=(1030,3,5))@root.T
    expected=np.clip(ndtr(z),1e-8,1-1e-8).transpose(1,0,2)
    np.testing.assert_array_equal(u,expected)


def test_shrinkage_matches_outer_product_definition():
    rng=np.random.default_rng(57);x=rng.normal(size=(75,4));x[:,1]+=.8*x[:,0]
    z=x-x.mean(0);z/=np.sqrt(np.mean(z*z,0));n,p=z.shape
    sample=sum(np.outer(row,row) for row in z)/n;target=np.trace(sample)/p
    beta=sum(np.sum((np.outer(row,row)-sample)**2) for row in z)/(p*n*n)
    delta=np.sum((sample-target*np.eye(p))**2)/p;weight=np.clip(beta/delta,0,1)
    covariance=(1-weight)*sample+weight*target*np.eye(p)
    expected=covariance/np.sqrt(np.diag(covariance))[:,None]/np.sqrt(np.diag(covariance))[None,:]
    correlation,root,shrinkage=models.shrunk_correlation(x)
    np.testing.assert_allclose(shrinkage,weight,atol=1e-14)
    np.testing.assert_allclose(correlation,expected,atol=1e-14)
    np.testing.assert_allclose(root@root.T,correlation,atol=2e-14)


def test_scale_and_permutation_invariance():
    x=np.random.default_rng(4).normal(size=(120,5));x[:,2]+=.9*x[:,1]
    correlation,root,weight=models.shrunk_correlation(x);order=[3,1,4,0,2]
    c,r,w=models.shrunk_correlation(x[:,order]*np.array([1.,3.,2.,7.,.1])+20.)
    np.testing.assert_allclose(c,correlation[np.ix_(order,order)],atol=1e-14)
    np.testing.assert_allclose(r,root[np.ix_(order,order)],atol=1e-14)
    np.testing.assert_allclose(w,weight,atol=1e-14)


