"""Request-local CDF-mixture CRPS gradient aggregation, research only."""
import numpy as np


def learn(A, B, represented):
    """Chronological weighted-fold BOA; return every pre/post-update simplex.

    A[t], B[t] define L_t(w)=w@A[t]-.5*w@B[t]@w. Each representative
    is ONE weighted loss m_t*L_t, not m_t fictitious independent observations.
    Uses Fig.2 excess-loss sign and previous-rate correction. Dyadic ranges
    start in first nonzero excess units; see DESIGN.md for this adaptation.
    """
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)
    mass = np.asarray(represented, dtype=float)
    if (A.ndim != 2 or A.shape[1] < 1 or B.shape != (len(A), A.shape[1], A.shape[1])
            or mass.shape != (len(A),) or not np.isfinite(A).all()
            or not np.isfinite(B).all() or not np.isfinite(mass).all()
            or np.any(mass <= 0) or np.any(mass != np.floor(mass))):
        raise ValueError('Finite ordered coefficients and positive integer representation counts required')
    k = A.shape[1]; prior = np.full(k, 1. / k); w = prior.copy()
    trajectory = [w.copy()]; variance = np.zeros(k); correction = np.zeros(k)
    rate = np.zeros(k); maxima = np.zeros(k); anchor = None
    for a, b, m in zip(A, B, mass):
        # Centering eliminates the irrelevant constant in a simplex gradient.
        gradient = a - b @ w
        excess = m * (gradient - w @ gradient)
        # Identical experts / singleton have exactly zero excess analytically.
        # Centering roundoff is removed only when all gradient entries coincide.
        if np.all(gradient == gradient[0]):
            excess = np.zeros(k)
        if not np.isfinite(excess).all():
            raise ArithmeticError('Unrepresentable weighted CRPS gradient')
        largest = float(np.max(np.abs(excess)))
        if anchor is None and largest == 0:
            trajectory.append(w.copy()); continue
        if anchor is None:
            anchor = largest
        maxima = np.maximum(maxima, np.abs(excess))
        # Conservative dyadic range >= 2*max|excess|. Shared initial scale
        # avoids the undefined infinite rate for a coordinate with zero regret.
        with np.errstate(over='raise', under='raise', invalid='raise', divide='raise'):
            try:
                bounds = 2. * anchor * np.exp2(np.ceil(np.log2(np.maximum(1., maxima / anchor))))
                correction += .5 * (excess * (1. + rate * excess)
                                     + bounds * (rate * excess > .5))
                variance += excess * excess
                next_rate = 1. / bounds
                np.minimum(next_rate, np.sqrt(np.divide(np.log(k), variance,
                    out=np.full(k, np.inf), where=variance > 0)), out=next_rate)
                logits = np.log(prior) + np.log(next_rate) - next_rate * correction
                probability = np.exp(logits - logits.max())
                w = probability / probability.sum()
            except FloatingPointError as error:
                raise ArithmeticError('BOA state exceeds float64 range; no clipped/fallback weights') from error
        if not np.isfinite(w).all() or not np.isfinite(variance).all():
            raise ArithmeticError('Nonfinite BOA state; no fallback')
        rate = next_rate
        trajectory.append(w.copy())
    trajectory = np.asarray(trajectory)
    return {'weights': w, 'progressive_weights': trajectory.mean(axis=0),
            'trajectory': trajectory, 'variance': variance, 'correction': correction,
            'rate': rate, 'initial_excess_scale': anchor,
            'round_semantics': 'one represented-count-weighted loss per dated representative'}


def pack_B(B):
    """Lossless float64 upper triangles; no reduced-precision cache."""
    B = np.asarray(B, dtype=float)
    return B[:, np.triu_indices(B.shape[1])[0], np.triu_indices(B.shape[1])[1]]


def unpack_B(packed, k):
    packed = np.asarray(packed, dtype=float)
    out = np.zeros((len(packed), k, k)); i, j = np.triu_indices(k)
    out[:, i, j] = packed; out[:, j, i] = packed
    return out
