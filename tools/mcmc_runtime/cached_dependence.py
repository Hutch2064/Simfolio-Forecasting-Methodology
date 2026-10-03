"""Reuse inverse matrices only for byte-identical predicted covariances."""
import inspect
from collections import OrderedDict

from simfolio_forecasting_methodology.models.numerical import dynamic_gaussian as dg


def make():
    source=inspect.getsource(dg.kalman_terminal_posterior)
    source=source.replace('    for index, row in enumerate(observations):','    inverses = OrderedDict()\n    for index, row in enumerate(observations):')
    old='''        prior_precision = np.linalg.inv(covariance)
        posterior_precision = prior_precision + observation_information
        covariance = np.linalg.inv(posterior_precision)'''
    new='''        key = covariance.tobytes()
        cached = inverses.get(key)
        if cached is None:
            prior_precision = np.linalg.inv(covariance)
            posterior_precision = prior_precision + observation_information
            covariance = np.linalg.inv(posterior_precision)
            inverses[key] = (prior_precision, covariance)
            if len(inverses) > 16:
                inverses.popitem(last=False)
        else:
            prior_precision, covariance = cached
            inverses.move_to_end(key)'''
    assert source.count(old)==1;source=source.replace(old,new)
    namespace=dict(dg.__dict__);namespace['OrderedDict']=OrderedDict
    exec(compile(source,__file__,'exec'),namespace)  # noqa: S102 - compiles trusted repository source, never user input.
    return namespace['kalman_terminal_posterior']
