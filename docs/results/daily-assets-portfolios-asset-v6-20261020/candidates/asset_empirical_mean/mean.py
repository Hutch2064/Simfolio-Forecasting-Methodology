"""Bounded joint batch-mean uncertainty with the incumbent's population moments."""
import numpy as np
from candidates.asset_prior_best import mean as parent

def estimate(history):
    x=np.asarray(history,dtype=float);f=parent.estimate(x)
    groups=np.array_split(x,f['batch_count'])
    f['support']=np.stack([g.mean(axis=0)-f['sample_mean'] for g in groups])/np.sqrt(f['df']-2.)
    f['probabilities']=np.asarray([len(g)/len(x) for g in groups])
    return f

def draws(f,paths,seed,*,symmetric):
    if paths<2 or paths%2:raise ValueError('Even ensemble required')
    rng=np.random.default_rng(seed);count=paths//2 if symmetric else paths
    positions=(np.arange(count)+rng.random(count))/count
    cdf=np.cumsum(f['probabilities']);cdf[-1]=1.
    indices=np.searchsorted(cdf,positions,side='left')
    values=f['support'][indices]
    if symmetric:values=np.concatenate((values,-values),axis=0)
    else:values=values-values.mean(axis=0)
    values=values[rng.permutation(paths)]
    return values+(f['center']-f['sample_mean'])
