"""Same financial fitting laws; vectorized HARCH reductions and strict scalar SV loop."""
import hashlib
import numpy as np
from numba import njit
from . import gjr_with_base
from candidates.asset_feature_batch import fits as reference
fit=reference.fit;mean=reference.mean;fade=reference.fade;gjr=reference.gjr;fit_ar=reference.fit_ar
FAMILIES=reference.FAMILIES;base_fit=reference.base_fit;normalized=reference.normalized
correlation_root=reference.correlation_root;student_inverse_df=reference.student_inverse_df
ewma_states=reference.ewma_states;har_states=reference.har_states;nnls=reference.nnls

def harch_features(square):
    # Same contiguous5/22-element NumPy reductions for every native training row.
    five=np.lib.stride_tricks.sliding_window_view(square,5)[17:-1].mean(axis=1)
    twenty_two=np.lib.stride_tricks.sliding_window_view(square,22)[:-1].mean(axis=1)
    return np.column_stack((square[21:-1],five,twenty_two))

@njit(cache=False,fastmath=False)
def sv_recursion(z,level,phi,q,V,noise):
    m=level;P=V;past=np.empty(len(z))
    for t,value in enumerate(z):
        predicted=level+phi*(m-level);variance=phi*phi*P+q
        gain=variance/(variance+noise);m=predicted+gain*(value+level-predicted);P=(1-gain)*variance
        past[t]=m+.5*P
    return m,P,past

def fit_sv(y):
    # Soft log-square proxy is finite at genuine zero returns; no observations are deleted.
    n=len(y);proxy=np.log(y*y+1/n);z=proxy-proxy.mean();noise=np.pi*np.pi/2
    V=max(0.,float(z@z/n)-noise)
    phi=0. if V==0 else float(np.clip((z[1:]@z[:-1])/(n-1)/V,0.,1.-1/n))
    q=V*(1-phi*phi);level=-V/2;m,P,past=sv_recursion(z,level,phi,q,V,noise)
    return {'level':level,'sv_phi':phi,'q':q,'m':m,'P':P,'V':V,'pool':y/np.exp(.5*past),'proxy':proxy}

def prepare(history,family,base=None,*,prefix_work=None):
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
            f=gjr_with_base.fit(x[:,a],base_parent=raw[a]);pool=f['pool']
            for out,key in [('positive','alpha_positive'),('negative','alpha_negative'),('beta','beta'),('initial','next_variance_normalized'),('omega','omega_forecast')]:result[out][a]=f[key]
        elif family in ('EWMA_reverting','EWMA_recursive'):
            f=ewma_fit(x,a,prefix_work);d=f['decay'];v,next_v=ewma_states(y,d);pool=y/np.sqrt(v);result['initial'][a]=next_v;result['decay'][a]=d
        elif family=='HARCH':
            square=y*y;features=harch_features(square)
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

def pure_prepare(history,family,base=None,*,prefix_work=None):
    result=prepare(history,family,base,prefix_work=prefix_work)
    result['pool_fit']={'raw_fraction':0.,'ewma_fit':[]}
    return result

def leg_parent_prepare(history,family,raw_fade,base,bundle,*,prefix_work=None):
    """Only states consumed by ParentLeg166/167; no learned parent mixture.

    Legacy unused-calibration failure guard is deliberately NOT preserved.
    This mode requires separate invalid-mask and loss qualification.
    """
    x=np.asarray(history,dtype=float)
    if family!='SV_RMS' or raw_fade!='EWMA':raise ValueError('Leg-only preparation is frozen to SV_RMS/EWMA')
    if bundle['family']!=family or bundle['history_sha256']!=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest():raise ValueError('Exact same-prefix SV_RMS state required')
    def finite(value):
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        if isinstance(value,np.ndarray):return np.isfinite(value).all()
        if isinstance(value,(float,np.floating)):return np.isfinite(value)
        return True
    if not finite(bundle):raise ArithmeticError('Nonfinite retained parent-leg financial state')
    result=dict(bundle)
    result['pool_fit']={'raw_fraction':0.,'ewma_fit':[ewma_fit(x,a,prefix_work) for a in range(x.shape[1])]}
    return result


def ewma_fit(history,asset,prefix_work=None):
    """Reuse the exact EWMA result only inside GroupWork's ONE-prefix call."""
    if prefix_work is None:return fade.fit_ewma(history[:,asset])
    key=('EWMA',asset)
    if key not in prefix_work:prefix_work[key]=fade.fit_ewma(history[:,asset])
    return prefix_work[key]
