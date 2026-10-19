"""Avoid redundant finite-input cleanup and repeated exact reductions."""
import inspect

import numpy as np
from numba import njit

from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd

# Runtime installation replaces public functions; compile their retained source.
_SOURCE_COMPONENTS = bd._bdes_multiscale_components


@njit(cache=False,fastmath=False,nogil=True)
def ewma_initial_level(x):
    values=np.asarray(x,dtype=np.float64);finite=values[np.isfinite(values)]
    return float(np.nanmedian(finite)) if finite.size else 0.0

@njit(cache=False,fastmath=False,nogil=True)
def ewma_from_initial(values,alpha,level):
    out=np.empty_like(values)
    for idx,value in enumerate(values):
        if np.isfinite(value):level=alpha*float(value)+(1.0-alpha)*level
        out[idx]=level
    return out

def make():
    original=_SOURCE_COMPONENTS;source=inspect.getsource(original)
    for old,new in [('np.nanmean(','np.mean('),('np.nanvar(','np.var('),('np.nanstd(','np.std(')]:source=source.replace(old,new)
    cleanup='h = np.asarray(h_path, dtype=float); finite = h[np.isfinite(h)]; fill = float(np.nanmedian(finite)) if finite.size else 0.0\n    h = np.clip(np.nan_to_num(h, nan=fill, posinf=fill, neginf=fill), -18.0, 18.0)'
    assert source.count(cleanup)==1
    source=source.replace(cleanup,'h = np.clip(np.asarray(h_path, dtype=float), -18.0, 18.0)')
    source=source.replace('signal = np.abs(b) * np.std(q, axis=0)','q_std = np.std(q, axis=0)\n    signal = np.abs(b) * q_std',1)
    source=source.replace('post_signal = np.abs(b) * np.std(q, axis=0)','post_signal = np.abs(b) * q_std')
    source=source.replace('"hbar": float(np.mean(h))','"hbar": ell')
    source=source.replace('for k, phi in enumerate(phis):','ewma_level = ewma_initial_level(centered)\n    for k, phi in enumerate(phis):',1)
    source=source.replace('_bdes_ewma(centered, 1.0 - phi)','ewma_from_initial(centered, 1.0 - phi, ewma_level)',1)
    source=source.replace('h_q005, h_q25, h_q75, h_q995 =','component_q01, component_q99 = np.quantile(component, [0.01, 0.99])\n    h_q005, h_q25, h_q75, h_q995 =',1)
    source=source.replace('float(np.quantile(component, 0.01))','float(component_q01)').replace('float(np.quantile(component, 0.99))','float(component_q99)')
    namespace=dict(bd.__dict__);namespace.update(ewma_initial_level=ewma_initial_level,ewma_from_initial=ewma_from_initial)
    exec(compile(source,__file__,'exec'),namespace)  # noqa: S102 - compiles trusted repository source, never user input.
    finite=namespace['_bdes_multiscale_components']
    def result(path,*args,**kwargs):
        if not np.all(np.isfinite(path)):return original(path,*args,**kwargs)
        return finite(path,*args,**kwargs)
    return result
