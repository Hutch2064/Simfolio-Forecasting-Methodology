"""Private factorial adapter must not mutate the recorded candidate arms."""
import importlib.util
import sys
from pathlib import Path

import numpy as np


def test_combined_uses_residual_observations_empirical_noise_and_shared_cache():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("combined_rough_test",root/"tools/conditional_empirical_rough/models.py")
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    assert m.noise.shell is m.base.shell
    assert m.noise.observed is m.base.observed
    assert m.base.rough_fit is m.noise.rough_fit
    assert m.base.prepared is m.prepared
    x=np.random.default_rng(9).normal(0,.01,300).tobytes()
    theta=np.array([.1,np.log(1/63),np.log(.7)])
    phi,w,noise_root,state,initial_root,means,variances=m.prepared(x,theta.tobytes(),330,31)
    np.testing.assert_array_equal(means,np.zeros(31))
    np.testing.assert_allclose(variances,.7**2,atol=1e-15,rtol=0)
    y,_=m.base.observed(x);scale,_=m.noise.measurement_scale(x)
    q=noise_root@noise_root.T
    expected_state,p=m.overlay.terminal_filter(y*scale,phi,w,q*scale**2,float(y.mean())*scale)
    np.testing.assert_allclose(state,expected_state/scale,atol=1e-15,rtol=0)
    np.testing.assert_allclose(initial_root@initial_root.T,p/scale**2,atol=1e-14,rtol=0)
    # A new load of the empirical-only arm still observes the original proxy.
    spec=importlib.util.spec_from_file_location("unmodified_empirical_test",root/"tools/empirical_rough_noise/models.py")
    other=importlib.util.module_from_spec(spec);sys.modules[spec.name]=other;spec.loader.exec_module(other)
    values=np.frombuffer(x,np.float64)
    original,_=other.shell.bd._sv_observed_log_variance(values,float(values.mean()))
    np.testing.assert_array_equal(other.observed(x)[0],original)
