"""Compile scalar recurrences without fastmath; only spell out finite guard."""
import inspect

from numba import njit

from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd


def make():
    source=inspect.getsource(bd._sv_kalman_rts_smoother_mean)
    guard='not all(np.isfinite(v) for v in (level, phi, eta))'
    assert source.count(guard)==1
    source=source.replace(guard,'not (np.isfinite(level) and np.isfinite(phi) and np.isfinite(eta))')
    namespace=dict(bd.__dict__);exec(compile(source,__file__,'exec'),namespace)  # noqa: S102 - compiles trusted repository source, never user input.
    ewma_source=inspect.getsource(bd._bdes_ewma).replace('dtype=float','dtype=np.float64')
    exec(compile(ewma_source,__file__,'exec'),namespace)  # noqa: S102 - compiles trusted repository source, never user input.
    return njit(cache=False,fastmath=False,nogil=True)(namespace['_sv_kalman_rts_smoother_mean']),njit(cache=False,fastmath=False,nogil=True)(namespace['_bdes_ewma'])
