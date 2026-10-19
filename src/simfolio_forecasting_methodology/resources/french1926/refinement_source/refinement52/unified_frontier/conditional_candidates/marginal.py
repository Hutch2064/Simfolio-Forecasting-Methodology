"""Location-only correction using the actual existing clipped-knot mapper."""
import numpy as np


def compensation(point_mean, forecast_sd, nodes, state_variance):
    mean=np.asarray(point_mean,dtype=np.float64);sd=np.asarray(forecast_sd,dtype=np.float64)
    v=np.asarray(state_variance,dtype=np.float64);x=np.asarray(nodes,dtype=np.float64)
    if mean.ndim!=1 or sd.shape!=mean.shape or v.shape!=mean.shape or x.ndim!=1 or len(x)<2:
        raise ValueError('mean/SD/state curve and node dimensions')
    if not all(np.isfinite(a).all() for a in (mean,sd,v,x)) or np.any(sd<0) or np.any(v<0):
        raise ValueError('finite nonnegative fitted SD/state variance required')
    delta=np.empty_like(mean)
    # Existing 1024-row numerical block size only; no statistical truncation.
    for start in range(0,len(mean),1024):
        stop=min(start+1024,len(mean))
        knots=np.clip(mean[start:stop,None]+sd[start:stop,None]*x[None,:],-1.,1.)
        input_deviation=np.mean((knots[:,:-1]+knots[:,1:])*.5,axis=1)-mean[start:stop]
        delta[start:stop]=input_deviation*np.expm1(-v[start:stop]/8.)
    return delta


def finish(mapped_base,point_mean,delta=None):
    """Preserve original RMS rounding; correction follows mapper, then old clip."""
    base=np.asarray(mapped_base,dtype=np.float64).copy()
    base[:]=point_mean+(base-point_mean)*1.
    if delta is not None:base+=delta
    return np.clip(base,-1.,1.)
