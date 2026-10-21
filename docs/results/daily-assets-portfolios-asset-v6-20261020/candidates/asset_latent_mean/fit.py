"""A latent AR mean predictor; state strength and persistence fit to training."""
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit


@njit(cache=True,fastmath=False)
def kalman(y,observation_variance,phi,share):
    n=len(y);m=0.;P=share;dm=np.zeros(2);dP=np.array([0.,1.]);gradient=np.zeros(2);loss=0.;predictions=np.empty(n)
    q=share*(1-phi*phi)
    for t in range(n):
        predicted=phi*m;variance=phi*phi*P+q;predictions[t]=predicted
        dmean=phi*dm;dmean[0]+=m
        dvariance=phi*phi*dP;dvariance[0]+=2*phi*(P-share);dvariance[1]+=1-phi*phi
        noise=(1-share)*observation_variance[t];dnoise=np.array([0.,-observation_variance[t]])
        s=variance+noise;ds=dvariance+dnoise;error=y[t]-predicted
        loss+=.5*(np.log(s)+error*error/s)
        gradient+=.5*(1/s-error*error/(s*s))*ds-error/s*dmean
        gain=variance/s;dgain=(dvariance*s-variance*ds)/(s*s)
        m=predicted+gain*error;dm=(1-gain)*dmean+dgain*error
        P=noise*variance/s;dP=(dnoise*variance+noise*dvariance)/s-noise*variance*ds/(s*s)
    return loss/n,gradient/n,m,P,predictions


def objective(theta,y,variance):
    phi=np.tanh(theta[0]);share=expit(theta[1]);value,gradient,_,_,_=kalman(y,variance,phi,share)
    return value,gradient*np.array([1-phi*phi,share*(1-share)])


def estimate(y,observation_variance):
    y=np.asarray(y,dtype=float);variance=np.asarray(observation_variance,dtype=float);n=len(y)
    if n<24 or y.shape!=variance.shape or not np.isfinite(y).all()or not np.isfinite(variance).all()or np.any(variance<=0):raise ValueError('Valid training observations and positive noise curve required')
    lag=float(y[1:]@y[:-1]/max(float(y[:-1]@y[:-1]),np.finfo(float).tiny));sign=-1. if lag<0 else 1.;solutions=[];reports=[]
    for initial_phi in(.5,.95):
        initial_share=float(np.clip(abs(lag)/initial_phi,1/n,1-1/n));theta=np.array([np.arctanh(sign*initial_phi),np.log(initial_share/(1-initial_share))])
        result=minimize(objective,theta,args=(y,variance),jac=True,method='L-BFGS-B',bounds=[(-17.,17.),(-35.,35.)],options={'ftol':1e-12,'gtol':1e-7,'maxiter':600,'maxls':40})
        value,gradient=objective(result.x,y,variance);eligible=bool(result.success and np.isfinite(value)and np.isfinite(gradient).all()and np.max(np.abs(gradient))<=1e-4)
        reports.append({'optimizer_success':bool(result.success),'eligible':eligible,'objective':float(value),'gradient_max':float(np.max(np.abs(gradient)))})
        if eligible:solutions.append((float(value),float(np.tanh(result.x[0])),float(expit(result.x[1]))))
    if not solutions:raise ArithmeticError('Both latent-mean likelihood fits failed')
    value,phi,share=min(solutions);null=kalman(y,variance,0.,0.)[0]
    if null<=value:value,phi,share=null,0.,0.
    _,_,m,P,predictions=kalman(y,variance,phi,share)
    return {'phi':phi,'share':share,'final_mean':m,'final_variance':P,'predictions':predictions,'objective':value,'null_objective':null,'BIC_selects_latent':bool(2*n*(null-value)>2*np.log(n)),'reports':reports}


def expected_curve(fitted,start,count):
    return fitted['final_mean']*fitted['phi']**np.arange(start+1,start+count+1)
