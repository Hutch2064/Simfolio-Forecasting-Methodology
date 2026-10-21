"""Matched-covariance posterior mean draws with joint antithetic vectors."""
import numpy as np
from scipy.special import ndtri
from scipy.stats import chi2
from candidates.asset_prior_best import mean as parent

def draws(fit,paths,seed,law):
    if law=='parent':return parent.draws(fit,paths,seed)
    if law not in ('Gaussian_joint','Student_joint')or paths<2 or paths%2:raise ValueError('Even ensemble and declared law required')
    rng=np.random.default_rng(seed);half=paths//2;assets=len(fit['sample_mean']);z=np.empty((half,assets))
    def stratified():
        u=(np.arange(half)+1.-rng.random(half))/half
        return u[rng.permutation(half)]
    for a in range(assets):z[:,a]=ndtri(stratified())
    u=stratified();radial=np.ones(half)
    if law=='Student_joint' and np.isfinite(fit['df']):
        df=float(fit['df'])
        if df<=2:raise ValueError('Student mean law needs finite covariance')
        radial=np.sqrt((df-2)/chi2.ppf(u,df))
    vectors=(z*radial[:,None])@fit['root'].T
    paired=np.concatenate((vectors,-vectors),axis=0);paired=paired[rng.permutation(paths)]
    return paired+(fit['center']-fit['sample_mean'])
