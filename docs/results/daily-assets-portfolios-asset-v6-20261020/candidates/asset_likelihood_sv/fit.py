"""Gaussian log-square measurement QMLE with causal prediction residuals."""
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit

NOISE=np.pi*np.pi/2


@njit(cache=True,fastmath=False)
def filter_loss(z,phi,V):
    n=len(z);m=0.;P=V;dm=np.zeros(2);dP=np.array([0.,1.]);gradient=np.zeros(2);loss=0.
    predicted_logscale=np.empty(n);q=V*(1-phi*phi)
    for t in range(n):
        predicted=phi*m;variance=phi*phi*P+q
        dmean=phi*dm;dmean[0]+=m
        dvariance=phi*phi*dP;dvariance[0]+=2*phi*(P-V);dvariance[1]+=1-phi*phi
        s=variance+NOISE;error=z[t]-predicted
        loss+=.5*(np.log(s)+error*error/s)
        gradient+=.5*(1/s-error*error/(s*s))*dvariance-error/s*dmean
        # Standardize historical innovations by information preceding this row.
        predicted_logscale[t]=predicted-V/2+variance/2
        gain=variance/s;dgain=NOISE/(s*s)*dvariance
        m=predicted+gain*error;dm=(1-gain)*dmean+dgain*error
        P=NOISE*variance/s;dP=NOISE*NOISE/(s*s)*dvariance
    return loss/n,gradient/n,m,P,predicted_logscale


def objective(theta,z):
    maximum=1-1/len(z);s=expit(theta[0]);phi=maximum*s;V=np.exp(theta[1])
    if not np.isfinite(V):return np.inf,np.zeros(2)
    loss,gradient,_,_,_=filter_loss(z,phi,V)
    return loss,gradient*np.array([maximum*s*(1-s),V])


def estimate(y):
    y=np.asarray(y,dtype=float);n=len(y)
    if n<24 or not np.isfinite(y).all():raise ValueError('Finite standardized training history required')
    # Preserve the earlier finite-at-zero proxy; change estimation and causal timing only.
    proxy=np.log(y*y+1/n);z=proxy-proxy.mean();variance=float(z@z/n);initial_V=max(variance-NOISE,np.finfo(float).eps)
    maximum=1-1/n;answers=[];attempts=[]
    for initial_phi,start_V in ((.5,initial_V),(.9,max(variance,np.finfo(float).eps))):
        ratio=initial_phi/maximum;theta=np.array([np.log(ratio/(1-ratio)),np.log(start_V)])
        result=minimize(objective,theta,args=(z,),jac=True,method='L-BFGS-B',options={'ftol':1e-12,'gtol':1e-7,'maxiter':500})
        loss,gradient=objective(result.x,z);valid=bool(result.success and np.isfinite(loss)and np.isfinite(gradient).all()and np.max(np.abs(gradient))<1e-4)
        attempts.append({'success':bool(result.success),'eligible':valid,'objective':float(loss),'gradient_max':float(np.max(np.abs(gradient)))})
        if valid:answers.append((float(loss),maximum*expit(result.x[0]),float(np.exp(result.x[1]))))
    if not answers:raise ArithmeticError('Both training log-volatility likelihood fits failed')
    null=filter_loss(z,0.,0.)[0];loss,phi,V=min(answers)
    if null<=loss:loss,phi,V=null,0.,0.
    _,_,m,P,past=filter_loss(z,phi,V)
    return {'level':-V/2,'sv_phi':phi,'V':V,'q':V*(1-phi*phi),'m':m-V/2,'P':P,'pool':y/np.exp(.5*past),'proxy':proxy,'objective':loss,'null_objective':null,'attempts':attempts}
