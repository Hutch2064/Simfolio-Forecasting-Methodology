"""Shared training-only fits for a frozen broad feature screen."""
import hashlib
import numpy as np
from numba import njit
from scipy.optimize import minimize_scalar,nnls
from scipy.special import gammaln
from candidates.asset_prior_best import fit,mean
from candidates.asset_fading_vol import fade
from candidates.asset_gjr import fit as gjr
from candidates.asset_AR1_mean import fit_ar

FAMILIES=('constant','EWMA_reverting','GARCH','GJR','HARCH','SV_path','SV_RMS','AR1_GARCH','EWMA_recursive')

def normalized(z):
    z=z-z.mean(axis=0);sd=np.sqrt(np.mean(z*z,axis=0))
    if np.any(sd<=0):raise ArithmeticError('Positive innovation variation required')
    return z/sd

def correlation_root(z):
    z=normalized(z);cov=z.T@z/len(z);e,v=np.linalg.eigh(cov)
    if e.min()< -1e-10*max(e.max(),np.finfo(float).tiny):raise ArithmeticError('Invalid covariance')
    return v*np.sqrt(np.maximum(e,0.))

def student_inverse_df(z):
    squares=np.asarray(z).ravel()**2
    normal=.5*float(np.mean(np.log(2*np.pi)+squares))
    def objective(u):
        if u<=1e-7:return normal
        nu=1/u
        return float(-gammaln((nu+1)/2)+gammaln(nu/2)+.5*np.log(np.pi*(nu-2))+(nu+1)/2*np.mean(np.log1p(squares/(nu-2))))
    upper=.5-1/(len(squares)+4)
    answer=minimize_scalar(objective,bounds=(0.,upper),method='bounded',options={'xatol':1e-8})
    if not answer.success or not np.isfinite(answer.fun):raise ArithmeticError('Student training fit failed')
    return float(answer.x) if answer.fun<normal else 0.

@njit(cache=True,fastmath=False)
def ewma_states(y,decay):
    out=np.empty(len(y));v=1.
    for i in range(len(y)):
        out[i]=v;v=decay*v+(1-decay)*y[i]*y[i]
    return out,v

@njit(cache=True,fastmath=False)
def har_states(y,weights):
    ring=np.ones(22);pointer=0;omega=1.-weights.sum();out=np.empty(len(y))
    for i in range(len(y)):
        recent=0.
        for j in range(1,6):recent+=ring[(pointer-j)%22]
        out[i]=omega+weights[0]*ring[(pointer-1)%22]+weights[1]*recent/5+weights[2]*ring.sum()/22
        ring[pointer]=y[i]*y[i];pointer=(pointer+1)%22
    return out

def fit_sv(y):
    # Soft log-square proxy is finite at genuine zero returns; no observations are deleted.
    n=len(y);proxy=np.log(y*y+1/n);z=proxy-proxy.mean();noise=np.pi*np.pi/2
    V=max(0.,float(z@z/n)-noise)
    phi=0. if V==0 else float(np.clip((z[1:]@z[:-1])/(n-1)/V,0.,1.-1/n))
    q=V*(1-phi*phi);level=-V/2;m=level;P=V;past=np.empty(n)
    for t,value in enumerate(z):
        predicted=level+phi*(m-level);variance=phi*phi*P+q
        gain=variance/(variance+noise);m=predicted+gain*(value+level-predicted);P=(1-gain)*variance
        past[t]=m+.5*P
    return {'level':level,'sv_phi':phi,'q':q,'m':m,'P':P,'V':V,'pool':y/np.exp(.5*past),'proxy':proxy}

def base_fit(history):
    x=np.asarray(history,dtype=float)
    return {'fits':[fit.fit_variance_target(x[:,a]) for a in range(x.shape[1])],'mean':mean.estimate(x)}

def prepare(history,family,base=None):
    x=np.asarray(history,dtype=float);n,p=x.shape
    base=base_fit(x) if base is None else base
    raw=base['fits'];batch=base['mean']
    raw_params={k:np.array([f[k] for f in raw]) for k in ('mean','target_variance','alpha','beta','next_variance_normalized')}
    result={k:np.zeros(p) for k in ('alpha','beta','positive','negative','omega','initial','decay','phi','level','sv_phi','q','m','P','V')}
    result['target']=raw_params['target_variance'].copy();result['weights']=np.zeros((p,3));result['ring']=np.empty((p,22));pools=[];proxies=[]
    for a in range(p):
        f=raw[a];y=(x[:,a]-f['mean'])/np.sqrt(f['target_variance']);result['initial'][a]=1.;result['omega'][a]=1.;result['ring'][a]=y[-22:]**2
        if family in ('GARCH','AR1_GARCH'):
            if family=='AR1_GARCH':result['phi'][a],f=fit_ar.innovation_fit(x[:,a],f)
            result['target'][a]=f['target_variance'];result['alpha'][a]=f['alpha'];result['beta'][a]=f['beta'];result['initial'][a]=f['next_variance_normalized'];result['omega'][a]=1-f['alpha']-f['beta'];pool=f['pool']
        elif family=='GJR':
            f=gjr.fit(x[:,a]);pool=f['pool']
            for out,key in [('positive','alpha_positive'),('negative','alpha_negative'),('beta','beta'),('initial','next_variance_normalized'),('omega','omega_forecast')]:result[out][a]=f[key]
        elif family in ('EWMA_reverting','EWMA_recursive'):
            f=fade.fit_ewma(x[:,a]);d=f['decay'];v,next_v=ewma_states(y,d);pool=y/np.sqrt(v);result['initial'][a]=next_v;result['decay'][a]=d
        elif family=='HARCH':
            square=y*y;features=np.array([[square[t-1],square[t-5:t].mean(),square[t-22:t].mean()] for t in range(22,n)])
            coefficients=nnls(np.column_stack((np.ones(n-22),features)),square[22:])[0][1:]
            if coefficients.sum()>=1:coefficients*= (1.-1/n)/coefficients.sum()
            result['weights'][a]=coefficients;v=har_states(y,coefficients);pool=y/np.sqrt(v)
            ring=result['ring'][a];result['initial'][a]=1-coefficients.sum()+coefficients[0]*ring[-1]+coefficients[1]*ring[-5:].mean()+coefficients[2]*ring.mean()
        elif family in ('SV_path','SV_RMS'):
            f=fit_sv(y);pool=f['pool'];proxies.append(f['proxy'])
            for key in ('level','sv_phi','q','m','P','V'):result[key][a]=f[key]
        elif family=='constant':pool=y
        else:raise ValueError(family)
        pools.append(pool)
    result['pool']=np.stack(pools,axis=1)
    # Existing GARCH/GJR/AR pools are already normalized by their exact parent fitter.
    if family not in ('GARCH','GJR','AR1_GARCH'):result['pool']=normalized(result['pool'])
    result['root']=correlation_root(result['pool']);result['inverse_df']=student_inverse_df(result['pool'])
    result['vol_root']=np.eye(p)
    if proxies and any(result['V']):
        changes=np.diff(np.stack(proxies,axis=1),axis=0);active=np.flatnonzero(np.var(changes,axis=0)>0)
        if len(active):result['vol_root'][np.ix_(active,active)]=correlation_root(changes[:,active])
    return {'raw':raw_params,'mean':batch,'filter':result,'family':family,'history_sha256':hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()}
