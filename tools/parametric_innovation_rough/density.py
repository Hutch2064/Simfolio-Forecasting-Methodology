"""Hansen (1994) standardized skew-t, including the normal tail limit.

Fit inverse degrees of freedom on [0, 1/2): no imposed maximum df.
"""
import math
import numpy as np
from scipy.special import betaln,ndtri
from scipy.stats import t
from scipy.optimize import minimize,minimize_scalar


def constants(inverse_df,skew):
    u=float(inverse_df)
    logc=-.5*math.log(2*math.pi) if u==0 else -betaln(.5,.5/u)-.5*(math.log1p(-2*u)-math.log(u))
    a=4*skew*math.exp(logc)*(1-2*u)/(1-u)
    b=math.sqrt(1+3*skew*skew-a*a)
    return logc,a,b


def logpdf(values,inverse_df,skew):
    u=float(inverse_df);logc,a,b=constants(u,skew)
    z=np.asarray(values,float)
    scaled=(b*z+a)/np.where(z<-a/b,1-skew,1+skew)
    term=.5*scaled*scaled if u==0 else .5*(1+u)*np.log1p(scaled*scaled*u/(1-2*u))/u
    return math.log(b)+logc-term


def ppf(probability,inverse_df,skew):
    p=np.asarray(probability,float);u=float(inverse_df);_,a,b=constants(u,skew)
    left=p<(1-skew)/2
    adjusted=np.where(left,p/(1-skew),(p+skew)/(1+skew))
    quantile=ndtri(adjusted) if u==0 else t.ppf(adjusted,1/u)*math.sqrt(1-2*u)
    return (np.where(left,1-skew,1+skew)*quantile-a)/b


def fit(values,kind):
    z=np.asarray(values,float)
    if kind=='gaussian':
        return {'inverse_df':0.,'skew':0.,'degrees_of_freedom':None,'normal_tail_limit':True,'estimator':'standard_normal_no_shape_parameter_fit','success':True,'evaluations':0,'negative_mean_log_likelihood':-float(logpdf(z,0.,0.).mean())}
    eps=np.sqrt(np.finfo(float).eps);upper=.5-eps
    def target(u,skew):return -float(logpdf(z,u,skew).mean())
    symmetric=minimize_scalar(lambda u:target(u,0.),bounds=(0.,upper),method='bounded',options={'xatol':1e-8})
    if not symmetric.success:raise ArithmeticError('Student tail MLE did not converge')
    u=float(symmetric.x) if symmetric.fun<target(0.,0.) else 0.
    skew=0.;success=True;iterations=int(symmetric.nfev)
    if kind=='skew_student':
        result=minimize(lambda x:target(x[0],x[1]),[u,0.],method='L-BFGS-B',bounds=[(0.,upper),(-1+eps,1-eps)],options={'ftol':1e-10,'gtol':1e-5,'maxiter':200})
        if not result.success:
            result=minimize(lambda x:target(x[0],x[1]),result.x,method='Powell',bounds=[(0.,upper),(-1+eps,1-eps)],options={'ftol':1e-10,'xtol':1e-6,'maxiter':200})
        if not result.success:raise ArithmeticError('Hansen skew-t MLE did not converge')
        if result.fun<target(u,0.):u,skew=map(float,result.x)
        iterations+=int(result.nfev)
    return {'inverse_df':u,'skew':skew,'degrees_of_freedom':None if u==0 else 1/u,'normal_tail_limit':u==0,'estimator':'conditional_MLE_on_original_filtered_innovation_pool','success':success,'evaluations':iterations,'negative_mean_log_likelihood':target(u,skew)}
