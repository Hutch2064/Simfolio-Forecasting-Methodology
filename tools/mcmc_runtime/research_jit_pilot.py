"""Compile public scalar recurrences with the validated arithmetic order."""
import inspect

import numpy as np
from numba import njit

from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd


def make_accelerators():
    original_dlm = bd._dlm_ar1_loglik
    original_sv = bd._sv_kalman_filter
    fast_dlm = njit(cache=False, fastmath=False)(original_dlm)
    source = inspect.getsource(original_sv)
    guard = "not all(np.isfinite(v) for v in (level, phi, eta))"
    assert source.count(guard) == 1
    fast_source = source.replace(guard, "not (np.isfinite(level) and np.isfinite(phi) and np.isfinite(eta))")
    namespace = dict(bd.__dict__)
    exec(compile(fast_source, __file__, "exec"), namespace)  # noqa: S102 - compiles trusted repository source, never user input.
    fast_sv = njit(cache=False, fastmath=False)(namespace[original_sv.__name__])
    x = np.asarray([0.1, 0.2, -0.3], dtype=np.float64)
    fast_dlm(x, 1.0, 0.9, 0.01)
    fast_sv(x, 0.0, 0.9, 0.2, return_path=True)
    def fast_sv_wrapper(y, level, phi, eta, *, return_path=False):
        return fast_sv(y, level, phi, eta, return_path=return_path)
    return original_dlm, original_sv, fast_dlm, fast_sv_wrapper
