"""Dirichlet restart probabilities for a stationary joint historical-row chain."""
import numpy as np
from numba import njit
from candidates.asset_training_blocks.dependence import select_length,mean_covariance


def conditional_means(history,length):
    x=np.asarray(history,dtype=float);n=len(x);r=1-1/length;p=1/length
    first=(p*r**np.arange(n))@x/(1-r**n);out=np.empty_like(x);following=first
    for i in range(n-1,-1,-1):out[i]=p*x[i]+r*following;following=out[i]
    return out


def prepare(history):
    x=np.asarray(history,dtype=float);n,p=x.shape
    selected=select_length(x);length=selected['length'];block_mean=conditional_means(x,length);centered=block_mean-block_mean.mean(0);covariance=centered.T@centered/n
    target=mean_covariance(x,length);daily_var=np.mean((x-x.mean(0))**2,axis=0)
    target_trace=float(np.sum(np.diag(target)/daily_var));source_trace=float(np.sum(np.diag(covariance)/daily_var))
    if not np.isfinite([target_trace,source_trace]).all()or target_trace<=0 or source_trace<=0:raise ArithmeticError('Positive training posterior-mean variation required')
    concentration=max(0.,source_trace/target_trace-1)
    return {'selected_length':length,'marginal_lengths':selected['marginal_lengths'],'block_mean_covariance':covariance,'target_mean_covariance':target,'matched_concentration':concentration,'target_standardized_trace':target_trace,'source_standardized_trace':source_trace}


@njit(cache=True,fastmath=False)
def initial_rows(probabilities,length,uniforms):
    paths,n=probabilities.shape;r=1-1/length;p=1/length;out=np.empty(paths,dtype=np.int64)
    for a in range(paths):
        previous=0.;power=1.
        for j in range(n-1,-1,-1):previous+=p*power*probabilities[a,j];power*=r
        previous/=1-power;cumulative=0.;selected=n-1
        for j in range(n):
            current=r*previous+p*probabilities[a,j];cumulative+=current
            if uniforms[a]<cumulative:selected=j;break
            previous=current
        out[a]=selected
    return out


@njit(cache=True,fastmath=False)
def take(cdf,current,restarts,rows,length,position):
    width,paths=restarts.shape;n=cdf.shape[1];out=np.empty((paths,width),dtype=np.int64)
    for t in range(width):
        if position+t:
            for a in range(paths):
                if restarts[t,a]<1/length:current[a]=np.searchsorted(cdf[a],rows[t,a],side='right')
                else:current[a]=(current[a]+1)%n
        for a in range(paths):out[a,t]=current[a]
    return out


class RandomRows:
    def __init__(self,n,paths,seed,length,concentration):
        if n<2 or paths<1 or length<1 or concentration<0 or not np.isfinite([length,concentration]).all():raise ValueError('Valid history, paths and statistical parameters required')
        children=np.random.SeedSequence(int(seed)).spawn(4);rng=np.random.default_rng(children[0]);self.length=length;self.paths=paths;self.position=0
        if concentration==0:
            probabilities=np.zeros((paths,n));probabilities[np.arange(paths),rng.integers(0,n,size=paths)]=1.
        else:
            probabilities=rng.gamma(concentration/n,1.,size=(paths,n));totals=probabilities.sum(1)
            if np.any(totals<=0)or not np.isfinite(totals).all():raise ArithmeticError('Invalid Dirichlet draw; no path deletion')
            probabilities/=totals[:,None]
        self.current=initial_rows(probabilities,length,np.random.default_rng(children[1]).random(paths))
        np.cumsum(probabilities,axis=1,out=probabilities);probabilities[:,-1]=1.;self.cdf=probabilities
        self.restart_rng=np.random.default_rng(children[2]);self.row_rng=np.random.default_rng(children[3])
    def take(self,width):
        out=take(self.cdf,self.current,self.restart_rng.random((width,self.paths)),self.row_rng.random((width,self.paths)),self.length,self.position);self.position+=width;return out
