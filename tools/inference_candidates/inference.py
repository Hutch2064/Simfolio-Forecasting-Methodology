"""Alternative inference for the same bounded three-parameter SV quasi-posterior."""
import hashlib
import math
from functools import lru_cache

import arviz as az
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln, ndtri
from scipy.stats import qmc
from target import batch_target, hmc_chain, unconstrained_target_gradient, variational_objective

from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd

RECORDS = []
TRACES = {}


def support(y):
    lo, hi = np.quantile(y, [.01, .99])
    return np.array([lo - 4., -7., math.log(.02)]), np.array([
        hi + 4., min(7., math.log(.999 / .001)), math.log(2.5)])


def physical(u, lower, upper):
    return lower + (upper - lower) * expit(u)


def laplace(y, center, lower, upper):
    fraction = np.clip((center - lower) / (upper - lower), 1e-7, 1. - 1e-7)
    start = np.log(fraction / (1. - fraction))
    def objective(u):
        value, grad = unconstrained_target_gradient(y, u, lower, upper)
        return -value, -grad
    result = minimize(objective, start, jac=True, method='L-BFGS-B',
                      bounds=[(-18., 18.)] * 3,
                      options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-5})
    mode = result.x
    hessian = np.empty((3, 3))
    for j in range(3):
        delta = np.zeros(3); delta[j] = 1e-4
        hessian[:, j] = (objective(mode + delta)[1] - objective(mode - delta)[1]) / 2e-4
    eigenvalues, vectors = np.linalg.eigh(.5 * (hessian + hessian.T))
    covariance = (vectors * (1. / np.maximum(eigenvalues, 1e-5))) @ vectors.T
    return mode, np.linalg.cholesky(covariance), {
        'laplace_optimizer_success': bool(result.success),
        'laplace_iterations': int(result.nit),
        'laplace_gradient_max': float(np.max(np.abs(objective(mode)[1]))),
        'laplace_min_hessian_eigenvalue': float(eigenvalues.min()),
    }


def importance(y, mode, L, lower, upper, rng, count, *, student):
    normals = rng.normal(size=(count, 3))
    if student:
        z = normals / np.sqrt(rng.chisquare(5., size=count)[:, None] / 5.)
        q = (gammaln(4.) - gammaln(2.5) - 1.5 * math.log(5. * math.pi)
             - 4. * np.log1p(np.sum(z * z, axis=1) / 5.))
    else:
        z = normals
        q = -1.5 * math.log(2. * math.pi) - .5 * np.sum(z * z, axis=1)
    draws = mode + z @ L.T
    q -= np.log(np.diag(L)).sum()
    weights = batch_target(y, draws, lower, upper) - q
    weights = np.exp(weights - np.max(weights))
    weights /= weights.sum()
    _, khat = az.psislw(np.log(np.maximum(weights, 1e-300)))
    diag = {'importance_draws': count, 'importance_ess': float(1. / np.dot(weights, weights)),
            'max_normalized_weight': float(weights.max()), 'pareto_k': float(khat)}
    positions = (np.arange(16) + rng.random()) / 16.
    nodes = draws[np.searchsorted(np.cumsum(weights), positions)]
    return nodes, diag


def full_rank_vi(y, mode, L, lower, upper, seed):
    # Frozen randomized quasi-Monte Carlo nodes make the ELBO optimizer deterministic.
    z = ndtri(np.clip(qmc.Sobol(3, scramble=True, seed=seed).random_base2(6), 1e-10, 1-1e-10))
    params = np.r_[mode, math.log(L[0, 0]), L[1, 0], math.log(L[1, 1]),
                   L[2, 0], L[2, 1], math.log(L[2, 2])]
    result = minimize(lambda p: variational_objective(y, p, z, lower, upper),
                      params, jac=True, method='L-BFGS-B',
                      bounds=[(-18., 18.)]*3 + [(-8., 3.), (-20., 20.), (-8., 3.),
                                               (-20., 20.), (-20., 20.), (-8., 3.)],
                      options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-4})
    p = result.x
    fitted = np.array([[math.exp(p[3]), 0., 0.], [p[4], math.exp(p[5]), 0.],
                       [p[6], p[7], math.exp(p[8])]])
    return p[:3], fitted, {'vi_optimizer_success': bool(result.success),
                          'vi_iterations': int(result.nit),
                          'vi_gradient_max': float(np.max(np.abs(result.jac)))}


def diagnose_chains(traces):
    data = {name: traces[:, :, j] for j, name in enumerate(('level', 'phi_logit', 'log_eta'))}
    idata = az.from_dict(posterior=data)
    return {'rhat': {k: float(v) for k, v in az.rhat(idata, method='rank').data_vars.items()},
            'bulk_ess': {k: float(v) for k, v in az.ess(idata, method='bulk').data_vars.items()},
            'tail_ess': {k: float(v) for k, v in az.ess(idata, method='tail').data_vars.items()}}


@lru_cache(maxsize=4096)
def parameter_fit(data, method):
    x = np.frombuffer(data, np.float64)
    fit = bd.fit_bdes_fastmap(x, filtered_innovations=True, fixed_mean=True)
    y, eps = bd._sv_observed_log_variance(x, x.mean())
    center = fit['posterior_center']
    center = np.array([center[0], bd._logit(center[1]), math.log(center[2])])
    lower, upper = support(y)
    mode, L, diag = laplace(y, center, lower, upper)
    seed = int.from_bytes(hashlib.sha256(data).digest()[:4], 'little')
    rng = np.random.default_rng(seed)
    if method == 'laplace_mixture_is':
        components, reports = multistart_components(y, center, lower, upper)
        nodes, more = mixture_importance(y, components, lower, upper, rng)
        more['mode_search'] = reports
    elif method == 'laplace_is':
        nodes, more = importance(y, mode, L, lower, upper, rng, 2048, student=True)
    elif method in ('full_rank_vi', 'vi_importance'):
        mode, L, more = full_rank_vi(y, mode, L, lower, upper, seed)
        weighted, checks = importance(y, mode, L, lower, upper, rng, 2048, student=False)
        more.update(checks)
        nodes = weighted if method == 'vi_importance' else mode + rng.normal(size=(16, 3)) @ L.T
    elif method in ('hmc', 'hmc_long'):
        burn, kept = (1024, 2048) if method == 'hmc_long' else (384, 512)
        chains = [hmc_chain(y, mode, L, lower, upper, burn, kept,
                            (seed + j * 99173) % 2**32) for j in range(4)]
        traces = np.stack([c[0] for c in chains])
        more = diagnose_chains(physical(traces, lower, upper))
        more.update({'acceptance': [c[1] for c in chains],
                     'energy_error_count': sum(c[2] for c in chains),
                     'step_size': [c[3] for c in chains], 'burn': burn, 'kept': kept, 'chains': 4})
        pooled = traces.reshape(-1, 3)
        nodes = pooled[np.linspace(0, len(pooled)-1, 16, dtype=int)]
        TRACES[(hashlib.sha256(data).hexdigest(), method)] = traces
    else:
        raise ValueError(method)
    diag.update(more)
    theta = physical(nodes, lower, upper)
    children = []
    for t in theta:
        _, path, _ = bd._sv_kalman_rts_smoother_mean(y, t[0], float(expit(t[1])), math.exp(t[2]))
        child = dict(fit)
        child['bdes_multiscale_vol'] = bd._bdes_multiscale_components(path, 4, bd.BDES_MULTISCALE_GRID_FIXED)
        child['innovation_pool'] = bd._standardized_empirical_innovation_pool(
            eps * np.exp(-.5*path), clip=None, method='mean_std')
        children.append(child)
    out = dict(fit); out['parameter_posterior_children'] = children
    RECORDS.append({'method': method, 'training_sha256': hashlib.sha256(data).hexdigest(),
                    'observations': len(x), **diag})
    return out


def mixture_importance(y, components, lower, upper, rng, count=4096):
    """Defensive Student-t mixture; weights use the full mixture density."""
    from scipy.special import logsumexp
    choices = rng.integers(len(components), size=count)
    draws = np.empty((count, 3))
    for j, (mode, L) in enumerate(components):
        selected = np.flatnonzero(choices == j)
        z = rng.normal(size=(len(selected), 3)) / np.sqrt(rng.chisquare(5., len(selected))[:, None]/5.)
        draws[selected] = mode + z @ L.T
    densities = []
    constant = gammaln(4.) - gammaln(2.5) - 1.5 * math.log(5.*math.pi)
    for mode, L in components:
        z = np.linalg.solve(L, (draws-mode).T).T
        densities.append(constant - np.log(np.diag(L)).sum()
                         - 4.*np.log1p(np.sum(z*z,axis=1)/5.))
    logq = logsumexp(np.array(densities),axis=0)-math.log(len(components))
    logw = batch_target(y,draws,lower,upper)-logq
    _, khat = az.psislw(logw)
    weights = np.exp(logw-logw.max());weights/=weights.sum()
    positions=(np.arange(16)+rng.random())/16.
    nodes=draws[np.searchsorted(np.cumsum(weights),positions)]
    return nodes, {'importance_draws':count,'importance_ess':float(1/np.dot(weights,weights)),
                  'max_normalized_weight':float(weights.max()),'pareto_k':float(khat),
                  'proposal_components':len(components)}


def multistart_components(y, center, lower, upper):
    components=[];reports=[]
    starts=[center.copy()]
    for phi in (.10,.50,.95,.995):
        t=center.copy();t[1]=bd._logit(phi);t[2]=math.log(.35);starts.append(t)
    for start in starts:
        mode,L,diag=laplace(y,start,lower,upper)
        # Exclude duplicated stationary points, preserving distinct modes.
        duplicate=any(np.linalg.norm(np.linalg.solve(old_L,mode-old_mode))<.25 for old_mode,old_L in components)
        if not duplicate:
            components.append((mode,L));reports.append(diag)
    # A second scale covers curvature error without imposing a new target.
    components += [(mode,2.*L) for mode,L in list(components)]
    return components,reports
