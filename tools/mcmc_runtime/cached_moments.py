"""Reuse unchanged power arrays and standardized nodes for repeated laws."""
import inspect
from functools import lru_cache

import numpy as np

from simfolio_forecasting_methodology.models.asset_level import sv_moment_functions as mf


@lru_cache(maxsize=128)
def power(value,horizon):
    return value**np.arange(1,horizon+1,dtype=np.float64)

def make():
    namespace=dict(mf.__dict__);namespace['power']=power
    source=inspect.getsource(mf.predictive_state_moments)
    source=source.replace('product ** days','power(product, horizon)').replace('drift_phi ** days','power(drift_phi, horizon)')
    source=source.replace('powers = phi[:, None] ** days[None, :]','powers = np.vstack([power(value, horizon) for value in phi])')
    exec(compile(source,__file__,'exec'),namespace)  # noqa: S102 - compiles trusted repository source, never user input.
    source=inspect.getsource(mf.moment_return_curves)
    exec(compile(source,__file__,'exec'),namespace)  # noqa: S102 - compiles trusted repository source, never user input.
    return namespace['moment_return_curves']
