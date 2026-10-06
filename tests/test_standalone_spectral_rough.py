"""Standalone marginal variance targeting and retained century-long mean."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def model():
    p=Path(__file__).resolve().parents[1]/'tools/standalone_spectral_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_standalone_spectral',p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m


def test_base_nodes_target_historical_variance_and_mean_curve_is_unchanged(model):
    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
        moment_return_curves,
    )
    d=np.random.default_rng(10).normal(.0005,.01,600).tobytes()
    fit=model.streamed.predecessor_fit(d);reference,_=moment_return_curves(fit,25200)
    u=np.broadcast_to(np.linspace(0.,1.,240)[:,None],(240,5))
    _mean,paths=model.asset_paths(d,u)
    np.testing.assert_allclose(paths.mean(axis=0),reference[:5],atol=1e-16,rtol=1e-12)
    np.testing.assert_allclose(paths.var(axis=0),fit['base_fit']['sigma']**2,rtol=1e-12)
    long_mean,_=model.asset_paths(d,np.full((2,25200),.5))
    assert long_mean.tobytes()==reference.tobytes()


def test_rough_likelihood_receives_raw_proxy_without_conventional_offset(model):
    d=np.random.default_rng(11).normal(.0005,.01,600).tobytes();x=np.frombuffer(d,np.float64)
    expected,_=model.shell.bd._sv_observed_log_variance(x,float(x.mean()))
    actual,_=model.parent.parent.base.base.observed(d)
    assert actual.tobytes()==expected.tobytes()
    assert model.parent.parent.base.noise.observed(d)[0].tobytes()==expected.tobytes()
