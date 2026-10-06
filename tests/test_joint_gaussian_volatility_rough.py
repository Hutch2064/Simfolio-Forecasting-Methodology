"""Independent finite-sample covariance, conditioning and streamed-state references."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def model():
    path=Path(__file__).resolve().parents[1]/'tools/joint_gaussian_volatility_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_joint_gaussian_model',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize('n',[9,10])
def test_multiscale_spectrum_matches_dense_differenced_covariance(model,n):
    from scipy.linalg import toeplitz
    phi=np.array([.3,.9,-.2]);w=np.array([.3,.5,1.])
    common=np.array([.1,.2,.12]);independent=np.array([0.,0.,.15])
    q=np.outer(common,common)+np.outer(independent,independent)
    stationary=q/(1-phi[:,None]*phi[None,:])
    c=np.array([w@(np.diag(phi**lag)@stationary)@w for lag in range(n+1)])
    d=np.diff(np.eye(n+1),axis=0);dense=d@toeplitz(c)@d.T
    f=np.exp(-2j*np.pi*np.outer(np.arange(n//2+1),np.arange(n))/n)
    expected=np.real(np.einsum('ij,jk,ik->i',f,dense,f.conj()))/n
    actual=model.expected_periodogram(phi,w,q,n)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)


def test_joint_terminal_filter_matches_dense_gaussian_conditioning(model):
    phi=np.array([.8,.3,.9]);w=np.array([.4,1.,.6])
    q=np.array([[.1,.02,0.],[.02,.2,0.],[0.,0.,.3]])
    p=q/(1-phi[:,None]*phi[None,:]);n=12;noise=5.5;level=-1.
    y=np.random.default_rng(39).normal(-1.,2.,n)
    c=np.empty((n,n))
    for i in range(n):
        for j in range(n):c[i,j]=w@(np.diag(phi**abs(i-j))@p)@w
    c+=np.eye(n)*noise
    cross=np.column_stack([np.diag(phi**(n-1-t))@p@w for t in range(n)])
    expected_state=cross@np.linalg.solve(c,y-level)
    expected_cov=p-cross@np.linalg.solve(c,cross.T)
    scale=np.sqrt(4.934802200544679/noise)
    state,cov=model.overlay.terminal_filter(y*scale,phi,w,q*scale**2,level*scale)
    np.testing.assert_allclose(state/scale,expected_state,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(cov/scale**2,expected_cov,rtol=1e-12,atol=1e-12)
    assert abs(expected_cov[0,2])>1e-6


@pytest.mark.parametrize('normalize',[False,True])
def test_native_joint_state_forecast_matches_scalar_reference(model,normalize):
    native,_=model.load_paths()
    phi=np.array([.8,-.3,.9,.2]);w=np.array([.2,1.,.6,.8])
    common=np.array([.1,.15]);independent=np.array([0.,.12]);rough_sd=np.array([.2,.15])
    initial=np.array([.1,-.2,.15,.3]);level=-2.;days=31
    mean=np.full(days,.0005);u=np.linspace(.03,.99,days);nodes=np.array([-1.5,-.2,.3,1.4])
    baseline=np.linspace(.01,.02,days);norm_mean=np.linspace(-1.,.1,days);norm_variance=np.linspace(.2,.7,days)
    expected=np.empty(days);state=initial.copy();rng=np.random.default_rng(1703)
    for t in range(days):
        a,b=rng.standard_normal(2)
        for i in range(2):state[i]=phi[i]*state[i]+common[i]*a+independent[i]*b
        for i in range(2,4):state[i]=phi[i]*state[i]+rough_sd[i-2]*rng.standard_normal()
        h=level
        for i in range(4):h+=w[i]*state[i]
        sd=baseline[t]*math.exp(.5*(h-norm_mean[t])-.25*norm_variance[t]) if normalize else math.exp(.5*h)/100
        pos=u[t]*(len(nodes)-1);lo=math.floor(pos);hi=min(lo+1,len(nodes)-1);fraction=pos-lo
        left=np.clip(mean[t]+sd*nodes[lo],-1,1);right=np.clip(mean[t]+sd*nodes[hi],-1,1)
        expected[t]=left*(1-fraction)+right*fraction
    storage=np.empty((days,2));out=storage[:,0]
    generator=np.random.default_rng(1703)
    native.map_path(phi,w,rough_sd,initial,common,independent,level,mean,u,nodes,generator.bit_generator.capsule,out,baseline if normalize else np.empty(0),norm_mean if normalize else np.empty(0),norm_variance if normalize else np.empty(0),normalize)
    assert out.tobytes()==expected.tobytes()
    assert rng.bit_generator.state==generator.bit_generator.state


def test_native_joint_volatility_rejects_nonfinite_scale(model):
    native,_=model.load_paths();rng=np.random.default_rng(4)
    with pytest.raises(RuntimeError,match='nonfinite joint volatility forecast scale'):
        native.map_path(np.array([0.]),np.ones(1),np.zeros(1),np.zeros(1),np.empty(0),np.empty(0),2000.,np.zeros(1),np.ones(1)*.4,np.array([-1.,1.]),rng.bit_generator.capsule,np.empty(1),np.empty(0),np.empty(0),np.empty(0),False)


def test_joint_conditional_normalizers_match_analytical_future_covariance(model):
    phi=np.array([.7,-.2,.9]);w=np.array([.2,1.,.6])
    root=np.array([[.1,0.,0.],[.05,.2,0.],[0.,0.,.3]])
    p=np.array([[.3,-.02,-.03],[-.02,.4,-.01],[-.03,-.01,.5]])
    state=np.array([.1,-.3,.4]);q=root@root.T;days=30
    next_cov=p*phi[:,None]*phi[None,:]+q
    means,variances=model.overlay.path_normalizers(phi,w,root,next_cov,phi*state,days)
    for t in range(1,days+1):
        powers=phi**t
        expected=p*powers[:,None]*powers[None,:]+q*(1-(phi[:,None]*phi[None,:])**t)/(1-phi[:,None]*phi[None,:])
        assert means[t-1]==pytest.approx(w@(powers*state),rel=1e-12,abs=1e-12)
        assert variances[t-1]==pytest.approx(w@expected@w,rel=1e-12,abs=1e-12)


def test_gaussian_forecast_multiplier_second_moment_is_one():
    from scipy.integrate import quad
    from scipy.stats import norm
    mean=.4;variance=.7
    def integrand(z):
        state=mean+math.sqrt(variance)*z
        multiplier=math.exp(.5*(state-mean)-.25*variance)
        return multiplier*multiplier*norm.pdf(z)
    integral,_=quad(integrand,-12.,12.,epsabs=1e-12)
    assert integral==pytest.approx(1.,rel=1e-12)
