"""Training-only importance quadrature for stationary GARCH coefficient uncertainty."""
from pathlib import Path
import sys
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import qmc, norm, chi2, multivariate_t
ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'daily-cross-asset-panel-20261020')
CODE = ROOT.parent/'french551-full-20261019/code'
for path in (ROOT, CODE/'src', CODE/'scripts'):
    sys.path.insert(0, str(path))
from expanded_public_smooth_mean_live import module as parent
gaussian = parent.retained.FHS.base
SCALE = 1.-gaussian.STATIONARITY_MARGIN
PROPOSAL_DF = 5.
PROPOSAL_SD_MULTIPLIER = 2.


@njit(cache=True, fastmath=False)
def log_targets(points, y):
    out = np.empty(len(points))
    for i in range(len(points)):
        x, z = points[i, 0], points[i, 1]
        top = max(0., x, z)
        den = np.exp(-top)+np.exp(x-top)+np.exp(z-top)
        logden = top+np.log(den)
        a = SCALE*np.exp(x-logden)
        b = SCALE*np.exp(z-logden)
        value = 2*np.log(SCALE)+x+z-3*logden
        variance = 1.
        for t in range(len(y)):
            if t:
                variance = 1.-a-b+a*y[t-1]**2+b*variance
            value -= .5*(np.log(variance)+y[t]**2/variance)
        out[i] = value
    return out


def mode_objective(parameters, y):
    loss, gradient = gaussian.objective(parameters, y)
    _, _, z = gaussian.parameters(parameters)
    logjac = 2*np.log(SCALE)+np.log(z).sum()
    return len(y)*loss-logjac, len(y)*gradient-(1.-3*z[1:])


def fit_proposal(y):
    eligible = []
    for a, b in ((.05, .90), (.10, .50), (1/3, 1/3)):
        initial = np.log(np.array([a, b])/(1.-a-b))
        fit = minimize(mode_objective, initial, args=(y,), jac=True, method='L-BFGS-B',
            options={'ftol': 1e-12, 'gtol': 1e-7, 'maxiter': 500})
        value, gradient = mode_objective(fit.x, y)
        if fit.success and np.isfinite(value) and np.max(np.abs(gradient)) <= 1e-3:
            eligible.append((value, fit.x))
    if not eligible:
        raise ArithmeticError('No converged stationary coefficient posterior mode')
    _, mode = min(eligible, key=lambda x: x[0])
    columns = []
    for j in range(2):
        step = np.cbrt(np.finfo(float).eps)*(1.+abs(mode[j]))
        shift = np.zeros(2)
        shift[j] = step
        columns.append((mode_objective(mode+shift, y)[1]-mode_objective(mode-shift, y)[1])/(2*step))
    hessian = np.column_stack(columns)
    hessian = (hessian+hessian.T)/2.
    if not np.isfinite(hessian).all() or np.linalg.eigvalsh(hessian).min() <= 0:
        raise ArithmeticError('Posterior mode lacks positive definite local information')
    return mode, np.linalg.inv(hessian)


def importance_nodes(y, mode, covariance, seed, power):
    u = qmc.Sobol(3, scramble=True, seed=seed).random_base2(power)
    # Quantile endpoint protection is numerical, not truncation of a financial law.
    u = np.clip(u, np.nextafter(0., 1.), np.nextafter(1., 0.))
    shape = covariance*PROPOSAL_SD_MULTIPLIER**2
    normal = norm.ppf(u[:, :2])@np.linalg.cholesky(shape).T
    points = mode+normal/np.sqrt(chi2.ppf(u[:, 2], PROPOSAL_DF)/PROPOSAL_DF)[:, None]
    logweight = log_targets(points, y)-multivariate_t.logpdf(points, loc=mode, shape=shape, df=PROPOSAL_DF)
    weights = np.exp(logweight-logsumexp(logweight))
    top = np.maximum(0., points.max(axis=1))
    exp = np.exp(points-top[:, None])
    den = np.exp(-top)+exp.sum(axis=1)
    ab = SCALE*exp/den[:, None]
    if not np.isfinite(weights).all() or not np.isfinite(ab).all() or not np.isclose(weights.sum(), 1.):
        raise ArithmeticError('Invalid posterior importance weights')
    moments = weights@ab
    centered = ab-moments
    cov = (centered*weights[:, None]).T@centered
    return {'points': points, 'coefficients': ab, 'weights': weights,
            'mean': moments, 'covariance': cov, 'ESS': float(1./np.sum(weights*weights)),
            'largest_normalized_weight': float(weights.max()), 'nodes': len(weights)}
