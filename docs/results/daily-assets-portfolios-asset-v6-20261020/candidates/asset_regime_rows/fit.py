"""Two-state common-noise Gaussian HMM on an aggregate log-volatility signal."""
import numpy as np
from numba import njit
from scipy.special import expit
from scipy.optimize import minimize


@njit(cache=True,fastmath=False)
def smooth(y,means,variance,transition,initial):
    n=len(y);emission=np.empty((n,2));alpha=np.empty((n,2));scales=np.empty(n);ll=0.
    for t in range(n):
        log_density=-.5*(np.log(2*np.pi*variance)+(y[t]-means)**2/variance);offset=max(log_density[0],log_density[1])
        for j in range(2):
            emission[t,j]=np.exp(log_density[j]-offset)
            prediction=initial[j]if t==0 else alpha[t-1,0]*transition[0,j]+alpha[t-1,1]*transition[1,j]
            alpha[t,j]=prediction*emission[t,j]
        scales[t]=alpha[t].sum();alpha[t]/=scales[t];ll+=np.log(scales[t])+offset
    gamma=np.empty_like(alpha);counts=np.zeros((2,2));beta=np.ones(2)
    for t in range(n-1,-1,-1):
        gamma[t]=alpha[t]*beta;gamma[t]/=gamma[t].sum()
        if t:
            joint=np.empty((2,2))
            for i in range(2):
                for j in range(2):joint[i,j]=alpha[t-1,i]*transition[i,j]*emission[t,j]*beta[j]
            counts+=joint/joint.sum();previous=np.empty(2)
            for i in range(2):previous[i]=(transition[i,0]*emission[t,0]*beta[0]+transition[i,1]*emission[t,1]*beta[1])/scales[t]
            beta=previous
    return ll,gamma,counts,alpha[-1].copy()


def unpack(theta):
    probabilities=expit(theta[3:5]);transition=np.column_stack((1-probabilities,probabilities));p=expit(theta[5])
    return theta[:2],float(np.exp(theta[2])),transition,np.array([1-p,p])


def objective(theta,y):
    means,variance,transition,initial=unpack(theta)
    if not np.isfinite(variance)or variance<=0 or np.any(transition<=0)or np.any(initial<=0):return np.inf,np.zeros(6)
    ll,gamma,counts,_=smooth(y,means,variance,transition,initial)
    residual=y[:,None]-means;gradient_means=np.sum(gamma*residual,axis=0)/variance
    gradient_variance=.5*(np.sum(gamma*residual*residual)/variance-len(y))
    gradient_transition=counts[:,1]-transition[:,1]*counts.sum(1);gradient_initial=gamma[0,1]-initial[1]
    return -ll/len(y),-np.r_[gradient_means,gradient_variance,gradient_transition,gradient_initial]/len(y)


def prepare(history):
    x=np.asarray(history,dtype=float);n,p=x.shape;mu=x.mean(0);sd=np.sqrt(np.mean((x-mu)**2,axis=0))
    if n<24 or not np.isfinite(x).all()or np.any(sd<=0):raise ValueError('Finite varying joint training history required')
    signal=np.log(np.mean(((x-mu)/sd)**2,axis=1)+1/n);scale=float(signal.std())
    if np.ptp(signal)==0:return {'one_state':True,'gamma':np.ones((n,1)),'transition':np.ones((1,1)),'filtered':np.ones(1),'one_state_BIC':0.,'two_state_BIC':np.inf,'reports':[]}
    y=(signal-signal.mean())/scale;label=y>np.median(y)
    if not label.any()or label.all():label=y>(y.min()+y.max())/2
    means=np.array([y[~label].mean(),y[label].mean()]);variance=float(np.mean((y-means[label.astype(int)])**2))
    if variance<=0:raise ArithmeticError('Zero common emission variance; no financial floor')
    mass=np.bincount(label.astype(int),minlength=2).astype(float);mass/=mass.sum();counts=np.ones((2,2))*np.finfo(float).eps
    for a,b in zip(label[:-1],label[1:]):counts[int(a),int(b)]+=1
    data_transition=counts/counts.sum(1,keepdims=True);solutions=[];reports=[]
    for transition in(data_transition,np.tile(mass,(2,1))):
        theta=np.r_[means,np.log(variance),np.log(transition[:,1]/transition[:,0]),np.log(mass[1]/mass[0])]
        result=minimize(objective,theta,args=(y,),jac=True,method='L-BFGS-B',bounds=[(None,None),(None,None),(np.log(np.finfo(float).eps),None),(-35,35),(-35,35),(-35,35)],options={'ftol':1e-12,'gtol':1e-7,'maxiter':1000,'maxls':40})
        value,gradient=objective(result.x,y);eligible=bool(result.success and np.isfinite(value)and np.isfinite(gradient).all()and np.max(np.abs(gradient))<=1e-4)
        reports.append({'optimizer_success':bool(result.success),'eligible':eligible,'objective':float(value),'gradient_max':float(np.max(np.abs(gradient))),'iterations':int(result.nit)})
        if eligible:solutions.append((float(value),result.x))
    if not solutions:raise ArithmeticError('Both regime likelihood fits failed: '+str(reports))
    value,theta=min(solutions,key=lambda r:r[0]);means,variance,transition,initial=unpack(theta);ll,gamma,counts,filtered=smooth(y,means,variance,transition,initial);order=np.argsort(means)
    null_ll=-.5*n*(np.log(2*np.pi)+1.)
    return {'one_state':False,'gamma':gamma[:,order],'transition':transition[np.ix_(order,order)],'filtered':filtered[order],'means':means[order],'common_variance':variance,'initial':initial[order],'one_state_BIC':-2*null_ll+2*np.log(n),'two_state_BIC':-2*ll+6*np.log(n),'reports':reports,'signal_mean':float(signal.mean()),'signal_sd':scale}
