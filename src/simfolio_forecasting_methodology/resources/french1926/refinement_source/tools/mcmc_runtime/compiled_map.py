"""Same MAP optimizer/objective, accelerated exact scalar likelihood."""
import inspect

from steady_kalman_target import target

from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd

# Runtime installation replaces public functions; compile their retained source.
_SOURCE_MAP = bd._fit_sv_map_state_space_params


def make():
    source=inspect.getsource(_SOURCE_MAP)
    source=source.replace('    y = y[np.isfinite(y)]','    y = y[np.isfinite(y)]\n    ymean = float(np.mean(y))')
    old='_sv_transformed_log_posterior(y, float(params[0]), float(params[1]), float(params[2]))'
    assert source.count(old)==1
    source=source.replace(old,'target(y, float(params[0]), float(params[1]), float(params[2]), ymean)')
    namespace=dict(bd.__dict__);namespace['target']=target
    exec(compile(source,__file__,'exec'),namespace)  # noqa: S102 - compiles trusted repository source, never user input.
    return namespace['_fit_sv_map_state_space_params']
