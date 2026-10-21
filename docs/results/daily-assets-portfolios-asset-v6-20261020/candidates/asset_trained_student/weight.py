"""Analytic Student CRPS pooling and a derived convex interpolation bound."""
import numpy as np
from scipy.special import stdtr,betaln
from candidates.asset_trained_pool import weight as parent


def absolute(x,scale,nu):
    if not np.isfinite(nu)or nu<=2:raise ValueError('Finite Student df greater than two required')
    x,scale=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(scale,dtype=float))
    if np.any(scale<=0):raise ValueError('Positive conditional standard deviations required')
    s=scale*np.sqrt((nu-2)/nu);z=x/s
    pdf=np.exp(-.5*np.log(nu)-betaln(.5,nu/2)-.5*(nu+1)*np.log1p(z*z/nu))
    return x*(2*stdtr(nu,z)-1)+2*s*(nu+z*z)/(nu-1)*pdf


def self_half(nu):
    return float(2*np.sqrt(nu-2)/(nu-1)*np.exp(betaln(.5,nu-.5)-2*betaln(.5,nu/2)))


def cross_interpolated(pool,scales,nu,relative_tolerance=1e-6):
    scales=np.asarray(scales,dtype=float);lower=float(scales.min());upper=float(scales.max())
    if lower<=0 or not np.isfinite(scales).all()or relative_tolerance<=0:raise ValueError('Finite positive scales and precision required')
    if lower==upper:return np.full(len(scales),np.mean(absolute(pool,lower,nu))),0.,1
    # g''(s)=2 E[(X/s)^2 f_T(X/s)]/s <= K/s, g(s)>=s E|T|.
    # u^2 f_T(u) is maximized at u^2=2(nu-2)/(nu-1).
    u2=2*(nu-2)/(nu-1)
    pdf=np.exp(-.5*np.log(nu-2)-betaln(.5,nu/2)-.5*(nu+1)*np.log1p(u2/(nu-2)))
    K=2*u2*pdf;lower_absolute=float(absolute(0.,1.,nu))
    ratio=np.sqrt(8*lower_absolute*relative_tolerance/K)
    intervals=int(np.ceil(np.log(upper/lower)/np.log1p(ratio)));nodes=np.geomspace(lower,upper,intervals+1)
    values=np.array([np.mean(absolute(pool,s,nu))for s in nodes]);actual=np.interp(scales,nodes,values)
    return actual,float(relative_tolerance*actual.max()),len(nodes)


def coefficients(pool,observed,raw_scale,filtered_scale,nu,relative_tolerance=1e-6):
    y=np.asarray(pool,dtype=float);observed=np.asarray(observed,dtype=float);n=len(y)
    if n<3 or any(a.shape!=y.shape for a in(observed,raw_scale,filtered_scale)):raise ValueError('Matching training arrays required')
    sorted_y=np.sort(y);pair_half=float(np.dot(2*np.arange(n)-n+1,sorted_y))/n**2
    removed_distance=n*parent.empirical_absolute(sorted_y,y)
    self_raw=raw_scale*(n*n*pair_half-removed_distance)/(n-1)**2
    observed_raw=raw_scale*(n*parent.empirical_absolute(sorted_y,observed/raw_scale)-np.abs(y-observed/raw_scale))/(n-1)
    self_filtered=filtered_scale*self_half(nu);observed_filtered=absolute(observed,filtered_scale,nu)
    cross,error,nodes=cross_interpolated(y,filtered_scale/raw_scale,nu,relative_tolerance)
    cross=raw_scale*(n*cross-absolute(y,filtered_scale/raw_scale,nu))/(n-1)
    constant=observed_filtered-self_filtered;linear=observed_raw-observed_filtered+2*self_filtered-cross;quadratic=cross-self_raw-self_filtered
    if np.min(quadratic)<-100*np.finfo(float).eps*max(1.,float(np.max(np.abs(cross)))):raise ArithmeticError('Negative energy-distance curvature')
    return float(constant.sum()),float(linear.sum()),float(quadratic.sum()),float(np.max(raw_scale)*n/(n-1)*error),nodes


def estimate(history,bundle,base,raw_fade,relative_tolerance=1e-6):
    if bundle['filter']['inverse_df']==0:
        result=parent.estimate(history,bundle,base,raw_fade,relative_tolerance);result['Student_df']=None;result['Gaussian_endpoint']=True;return result
    nu=1/float(bundle['filter']['inverse_df']);x=np.asarray(history,dtype=float)
    raw_scale,filtered_scale,ewma=parent.training_scales(x,bundle,base,raw_fade);total=np.zeros(3);error=0.;nodes=0
    for a in range(x.shape[1]):
        mu=bundle['raw']['mean'][a];sd=np.sqrt(bundle['raw']['target_variance'][a]);pool=(x[:,a]-mu)/sd;observed=(x[:,a]-bundle['mean']['center'][a])/sd
        c,l,q,bound,count=coefficients(pool,observed,raw_scale[:,a],filtered_scale[:,a],nu,relative_tolerance)
        total+=np.array([c,l,q]);error+=bound*len(x);nodes+=count
    constant,linear,quadratic=total
    fraction=float(np.clip(-linear/(2*quadratic),0.,1.))if quadratic>0 else float(constant+linear<=constant)
    return {'raw_fraction':fraction,'training_objective':float(constant+linear*fraction+quadratic*fraction*fraction),'quadratic_coefficients':total,'training_rows':len(x),'assets':x.shape[1],'cross_interpolation_sum_absolute_error_bound':error,'cross_interpolation_nodes':nodes,'relative_interpolation_tolerance':relative_tolerance,'ewma_fit':ewma,'Student_df':nu,'Gaussian_endpoint':False}
