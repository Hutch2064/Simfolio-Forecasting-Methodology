"""Independent covariance checks for training-only volatility dependence."""
import importlib.util
from pathlib import Path

import numpy as np

spec=importlib.util.spec_from_file_location('volatility_dependence',Path(__file__).parents[1]/'tools/correlated_volatility_rough/dependence.py')
dep=importlib.util.module_from_spec(spec);spec.loader.exec_module(dep)


def test_shrinkage_matches_outer_product_definition():
    rng=np.random.default_rng(57);x=rng.normal(size=(75,4));x[:,1]+=.8*x[:,0]
    z=x-x.mean(0);z/=np.sqrt(np.mean(z*z,0));n,p=z.shape
    sample=sum(np.outer(row,row) for row in z)/n;target=np.trace(sample)/p
    beta=sum(np.sum((np.outer(row,row)-sample)**2) for row in z)/(p*n*n)
    delta=np.sum((sample-target*np.eye(p))**2)/p;weight=np.clip(beta/delta,0,1)
    covariance=(1-weight)*sample+weight*target*np.eye(p)
    expected=covariance/np.sqrt(np.diag(covariance))[:,None]/np.sqrt(np.diag(covariance))[None,:]
    correlation,root,shrinkage=dep.shrunk_correlation(x)
    np.testing.assert_allclose(shrinkage,weight,atol=1e-14)
    np.testing.assert_allclose(correlation,expected,atol=1e-14)
    np.testing.assert_allclose(root@root.T,correlation,atol=2e-14)


def test_scale_and_permutation_invariance():
    x=np.random.default_rng(4).normal(size=(120,5));x[:,2]+=.9*x[:,1]
    correlation,root,weight=dep.shrunk_correlation(x);order=[3,1,4,0,2]
    c,r,w=dep.shrunk_correlation(x[:,order]*np.array([1.,3.,2.,7.,.1])+20.)
    np.testing.assert_allclose(c,correlation[np.ix_(order,order)],atol=1e-14)
    np.testing.assert_allclose(r,root[np.ix_(order,order)],atol=1e-14)
    np.testing.assert_allclose(w,weight,atol=1e-14)


def test_constant_assets_and_single_asset_identity():
    for x in (np.zeros((4,3)),np.arange(20.)[:,None]):
        correlation,root,_=dep.shrunk_correlation(x)
        np.testing.assert_array_equal(correlation,np.eye(x.shape[1]))
        np.testing.assert_array_equal(root,np.eye(x.shape[1]))
    x=np.random.default_rng(1).normal(size=(40,3));x[:,1]=3.
    correlation,root,_=dep.shrunk_correlation(x)
    np.testing.assert_array_equal(correlation[1],np.array([0.,1.,0.]))
    np.testing.assert_allclose(root@root.T,correlation,atol=1e-14)


def test_conditional_ar1_cross_covariance():
    correlation=np.array([[1.,.6],[.6,1.]])
    rho=np.array([.4,.9]);sd=np.array([.2,.1]);horizon=7
    loading=np.zeros((2,horizon*2))
    eigenvalues,v=np.linalg.eigh(correlation);root=(v*np.sqrt(eigenvalues))@v.T
    for a in range(2):
        for t in range(horizon):loading[a,t*2:(t+1)*2]=rho[a]**(horizon-1-t)*sd[a]*root[a]
    covariance=loading@loading.T
    expected=np.outer(sd,sd)*correlation*np.array([[sum((r*s)**j for j in range(horizon)) for s in rho] for r in rho])
    np.testing.assert_allclose(covariance,expected,atol=1e-15)


def test_native_precomputed_innovations_match_scalar_forecast():
    import math
    import sys
    spec=importlib.util.spec_from_file_location('correlated_vol_native_test',Path(__file__).parents[1]/'tools/correlated_volatility_rough/correlated_native.py')
    native_loader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native_loader;spec.loader.exec_module(native_loader)
    native,dot=native_loader.load_paths();days=31;rho=.9;sd=.2;initial=np.array([.3]);mu=-1.
    mean=np.full(days,.0003);base=np.linspace(-.01,.01,days)
    mm=mu+rho**np.arange(1,days+1)*initial[0];mv=sd**2*(1-rho**(2*np.arange(1,days+1)))/(1-rho*rho)
    shocks=np.random.default_rng(73).standard_normal(days);current=initial[0];expected=np.empty(days)
    for t in range(days):
        current=rho*current+sd*shocks[t]
        expected[t]=np.clip(mean[t]+(base[t]-mean[t])*math.exp(.5*(mu+current-mm[t])-.25*mv[t]),-1.,1.)
    rough_rng=np.random.default_rng(72);memory_rng=np.random.default_rng(73);state=memory_rng.bit_generator.state;output=np.empty(days)
    native.map_path(np.array([.9]),np.zeros(1),np.zeros((1,1)),np.zeros(1),np.zeros(days),np.zeros(days),rough_rng.bit_generator.capsule,dot,mean,base,output,np.empty(0),np.array([rho]),sd,initial,mu,mm,mv,memory_rng.bit_generator.capsule,True,shocks)
    assert output.tobytes()==expected.tobytes()
    assert memory_rng.bit_generator.state==state
