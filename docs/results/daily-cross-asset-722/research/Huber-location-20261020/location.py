"""Fixed Huber location contrast; no truncation or deletion of return observations."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtri
C = 1.345
NORMAL_MAD = float(ndtri(.75))


def estimate(history, c=C):
    history = np.asarray(history, dtype=float)
    if history.ndim != 1 or not len(history) or not np.isfinite(history).all() or c <= 0:
        raise ValueError('Finite one-dimensional nonempty history and positive Huber constant required')
    if np.isposinf(c):
        return float(history.mean())
    if not np.isfinite(c):
        raise ValueError('Finite positive Huber constant required')
    median = float(np.median(history))
    scale = float(np.median(np.abs(history-median))/NORMAL_MAD)
    if scale == 0:
        scale = float(np.std(history, ddof=0))
    if scale == 0:
        return median
    # Solving the monotone bounded score avoids dependence on an IRLS start.
    score = lambda center: float(np.sum(np.clip((history-center)/scale, -c, c)))
    return float(brentq(score, float(history.min()), float(history.max()),
                       xtol=np.nextafter(0., 1.), rtol=4*np.finfo(float).eps, maxiter=500))
