"""Training plug-in normal empirical Bayes for correlated mean uncertainty."""
import numpy as np


def estimate(parent, posterior_covariance):
    mu=np.asarray(parent['sample_mean'],dtype=float)
    covariance=np.asarray(parent['covariance'],dtype=float)
    eigenvalues,vectors=np.linalg.eigh(covariance)
    tolerance=np.finfo(float).eps*len(mu)*max(float(eigenvalues.max()),np.finfo(float).tiny)
    if eigenvalues.min() < -100*tolerance:raise ArithmeticError('Invalid mean covariance')
    active=eigenvalues>tolerance
    rank=int(active.sum())
    coordinates=vectors[:,active].T@mu
    energy=float(np.sum(coordinates**2/eigenvalues[active]))
    fraction=0. if energy<=rank else 1.-rank/energy
    # Zero-noise directions have known means and must not be shrunk.
    center=mu-(1.-fraction)*(vectors[:,active]@coordinates)
    result=dict(parent)
    result['center']=center
    if posterior_covariance:
        result['covariance']=fraction*covariance
        result['root']=np.sqrt(fraction)*parent['root']
    result['empirical_bayes']={'rank':rank,'mahalanobis_energy':energy,
        'retained_mean_fraction':fraction,'prior_variance_ratio':max(0.,energy/rank-1.)if rank else 0.,
        'posterior_covariance':bool(posterior_covariance),'numerical_rank_tolerance':tolerance}
    return result
