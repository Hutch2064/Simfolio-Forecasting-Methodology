"""Training marginal CRPS pooling with exact leave-one-out empirical terms."""
import math
import numpy as np
from numba import njit
from candidates.asset_feature_batch import fits
from candidates.asset_prior_best import fit
from candidates.asset_fading_vol import fade


@njit(cache=True,fastmath=False)
def normal_absolute(x,scale):
    z=x/scale
    return scale*math.sqrt(2/math.pi)*math.exp(-.5*z*z)+x*math.erf(z/math.sqrt(2.))


@njit(cache=True,fastmath=False)
def cross_values(pool,scales):
    out=np.empty(len(scales))
    for i,scale in enumerate(scales):
        total=0.
        for x in pool:total+=normal_absolute(x,scale)
        out[i]=total/len(pool)
    return out


def cross_interpolated(pool,scales,relative_tolerance=1e-6):
    scales=np.asarray(scales,dtype=float);lower=float(scales.min());upper=float(scales.max())
    if lower<=0 or not np.isfinite(scales).all()or relative_tolerance<=0:raise ValueError('Finite positive scales and precision required')
    if lower==upper:return np.full(len(scales),cross_values(pool,np.array([lower]))[0]),0.,1
    # g''(s) <= sqrt(2/pi)*2/(e*s), g(s) >= sqrt(2/pi)*s.
    # Linear chord error / g(s) <= (upper/lower-1)^2/(4e).
    intervals=int(np.ceil(np.log(upper/lower)/np.log1p(np.sqrt(4*np.e*relative_tolerance))))
    nodes=np.geomspace(lower,upper,intervals+1);values=cross_values(pool,nodes)
    actual=np.interp(scales,nodes,values)
    return actual,float(relative_tolerance*actual.max()),len(nodes)


def empirical_absolute(sorted_pool,points):
    n=len(sorted_pool);prefix=np.concatenate(([0.],np.cumsum(sorted_pool)));k=np.searchsorted(sorted_pool,points,side='right')
    return ((2*k-n)*points+prefix[-1]-2*prefix[k])/n


def coefficients(pool,observed,raw_scale,filtered_scale,relative_tolerance=1e-6):
    y=np.asarray(pool,dtype=float);observed=np.asarray(observed,dtype=float);n=len(y)
    if n<3 or any(a.shape!=y.shape for a in(observed,raw_scale,filtered_scale)):raise ValueError('Matching training arrays required')
    sorted_y=np.sort(y);pair_half=float(np.dot(2*np.arange(n)-n+1,sorted_y))/n**2
    removed_distance=n*empirical_absolute(sorted_y,y)
    self_raw=raw_scale*(n*n*pair_half-removed_distance)/(n-1)**2
    observed_raw=raw_scale*(n*empirical_absolute(sorted_y,observed/raw_scale)-np.abs(y-observed/raw_scale))/(n-1)
    self_filtered=filtered_scale/np.sqrt(np.pi)
    observed_filtered=np.array([normal_absolute(v,s)for v,s in zip(observed,filtered_scale)])
    cross,error,nodes=cross_interpolated(y,filtered_scale/raw_scale,relative_tolerance)
    cross=raw_scale*(n*cross-np.array([normal_absolute(v,s)for v,s in zip(y,filtered_scale/raw_scale)]))/(n-1)
    constant=observed_filtered-self_filtered
    linear=observed_raw-observed_filtered+2*self_filtered-cross
    quadratic=cross-self_raw-self_filtered
    scale=max(1.,float(np.max(np.abs(cross))))
    if np.min(quadratic)<-100*np.finfo(float).eps*scale:raise ArithmeticError('Negative energy-distance curvature')
    return float(constant.sum()),float(linear.sum()),float(quadratic.sum()),float(np.max(raw_scale)*n/(n-1)*error),nodes


def training_scales(history,bundle,base,raw_fade):
    x=np.asarray(history,dtype=float);n,p=x.shape;raw=bundle['raw'];f=bundle['filter'];raw_scales=np.ones_like(x);filtered=np.ones_like(x);ewma=[]
    for a in range(p):
        y=(x[:,a]-raw['mean'][a])/np.sqrt(raw['target_variance'][a]);g=base['fits'][a]
        garch=fit.filter_loss(y,g['alpha'],g['beta'])[2]
        if raw_fade=='GARCH':raw_scales[:,a]=np.sqrt(garch)
        elif raw_fade=='EWMA':
            estimated=fade.fit_ewma(x[:,a]);past,_=fits.ewma_states(y,estimated['decay']);raw_scales[:,a]=np.sqrt(past);ewma.append(estimated)
        if bundle['family']=='GARCH':filtered[:,a]=np.sqrt(garch)
        elif bundle['family']=='SV_RMS':
            proxy=np.log(y*y+1/n);z=proxy-proxy.mean();level=f['level'][a];phi=f['sv_phi'][a];q=f['q'][a];m=level;P=f['V'][a]
            for t,value in enumerate(z):
                predicted=level+phi*(m-level);variance=phi*phi*P+q
                filtered[t,a]=np.exp(.5*predicted+.25*variance)
                gain=variance/(variance+np.pi*np.pi/2);m=predicted+gain*(value+level-predicted);P=(1-gain)*variance
        else:raise ValueError('Unknown frozen marginal filter')
    return raw_scales,filtered,ewma


def estimate(history,bundle,base,raw_fade,relative_tolerance=1e-6):
    x=np.asarray(history,dtype=float);raw_scale,filtered_scale,ewma=training_scales(x,bundle,base,raw_fade)
    total=np.zeros(3);error=0.;nodes=0
    for a in range(x.shape[1]):
        mu=bundle['raw']['mean'][a];sd=np.sqrt(bundle['raw']['target_variance'][a]);pool=(x[:,a]-mu)/sd
        observed=(x[:,a]-bundle['mean']['center'][a])/sd
        c,l,q,bound,count=coefficients(pool,observed,raw_scale[:,a],filtered_scale[:,a],relative_tolerance)
        total+=np.array([c,l,q]);error+=bound*len(x);nodes+=count
    constant,linear,quadratic=total
    if quadratic>0:weight=float(np.clip(-linear/(2*quadratic),0.,1.))
    else:weight=1. if constant+linear<=constant else 0.
    return {'raw_fraction':weight,'training_objective':float(constant+linear*weight+quadratic*weight*weight),'quadratic_coefficients':total,'training_rows':len(x),'assets':x.shape[1],'cross_interpolation_sum_absolute_error_bound':error,'cross_interpolation_nodes':nodes,'relative_interpolation_tolerance':relative_tolerance,'ewma_fit':ewma}
