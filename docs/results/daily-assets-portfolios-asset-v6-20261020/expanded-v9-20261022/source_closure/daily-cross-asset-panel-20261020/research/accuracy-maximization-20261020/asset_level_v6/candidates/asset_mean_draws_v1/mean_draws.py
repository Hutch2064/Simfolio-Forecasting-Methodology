"""Literal original draws, replacing the generic norm.ppf wrapper by ndtri."""
import numpy as np
from scipy.special import ndtri


def draws(fit,paths,seed):
    if not isinstance(paths,(int,np.integer)) or paths<2 or paths%2:
        raise ValueError('Even ensemble required for antithetic mean uncertainty')
    sample=np.asarray(fit['sample_mean']);center=np.asarray(fit['center']);root=np.asarray(fit['root'])
    if (sample.ndim!=1 or not len(sample) or center.shape!=sample.shape or root.shape!=(len(sample),len(sample))
            or not np.isfinite(sample).all() or not np.isfinite(center).all() or not np.isfinite(root).all()):
        raise ValueError('Finite aligned mean vectors and square root required')
    rng=np.random.default_rng(seed)
    z=np.empty((paths,len(fit['sample_mean'])))
    for a in range(z.shape[1]):
        probabilities=(np.arange(paths//2)+1.-rng.random(paths//2))/paths
        # random lies in[0,1), so every constructed probability lies in(0,.5].
        lower=ndtri(probabilities)
        z[:,a]=np.concatenate((lower,-lower))[rng.permutation(paths)]
    return z@fit['root'].T+(fit['center']-fit['sample_mean'])
