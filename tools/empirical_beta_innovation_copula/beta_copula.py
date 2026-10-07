"""Empirical beta copula, Segers, Sibuya and Tsukahara (2017), eq. (2.1).

Ties average independent uniform rank assignments within each tie block.
This extension preserves exact uniform marginal mixtures without jittering
observations or substituting fractional midranks as beta shapes.
"""
import numpy as np
from scipy.stats import rankdata


def rank_intervals(innovations):
    x=np.asarray(innovations,dtype=np.float64)
    if x.ndim!=2 or min(x.shape)<1 or not np.isfinite(x).all():
        raise ValueError('finite full aligned innovation history required')
    low=rankdata(x,method='min',axis=0).astype(np.int64)
    high=rankdata(x,method='max',axis=0).astype(np.int64)
    return low,high


def sample_ranks(low,high,rows,tie_rng):
    selected=low[rows]
    if np.array_equal(low,high):return selected
    # Selecting one tied observation uniformly and then its rank uniformly
    # gives weight 1/n to each integer rank, just as an untied permutation.
    return tie_rng.integers(selected,high[rows]+1)


def sample_copula(low,high,count,row_rng,beta_rng,tie_rng):
    n=len(low)
    rows=row_rng.integers(n,size=count)
    ranks=sample_ranks(low,high,rows,tie_rng)
    return beta_rng.beta(ranks,n+1-ranks)
