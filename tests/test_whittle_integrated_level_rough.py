import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import toeplitz


@pytest.fixture(scope='module')
def model():
    path=Path(__file__).resolve().parents[1]/'tools/whittle_integrated_level_rough/models.py'
    spec=importlib.util.spec_from_file_location('integrated_whittle_test_model',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.initialize()
    return module


@pytest.mark.parametrize('n,R',[(31,.7),(32,4.934802200544679),(181,9.)])
def test_prepared_state_matches_dense_integrated_level_posterior(model,monkeypatch,n,R):
    phi=np.array([0.,.3,.95]);mass=np.array([.1,.2,.4]);w=np.ones(3)
    q=np.diag(mass*(1-phi*phi))
    y=np.random.default_rng(186).normal(size=n)+2.
    monkeypatch.setattr(model.streamed.base,'observed',lambda data:(y,0.))
    monkeypatch.setattr(model.streamed.parent.parent.base.noise,'measurement_scale',lambda data:(1.,R))
    monkeypatch.setattr(model.overlay,'configuration',lambda theta,lag:(phi,w,q))
    model.prepared.cache_clear()
    actual=model.prepared(y.tobytes(),np.array([.1,-4.,-.2]).tobytes(),100,20)
    covariance=toeplitz((phi[None,:]**np.arange(n)[:,None])@mass)+R*np.eye(n)
    inverse_one=np.linalg.solve(covariance,np.ones(n));precision=inverse_one.sum()
    level=inverse_one@y/precision
    cross=mass[:,None]*phi[:,None]**np.arange(n-1,-1,-1)[None,:]
    state=cross@np.linalg.solve(covariance,y-level)
    response=cross@inverse_one
    p=np.diag(mass)-cross@np.linalg.solve(covariance,cross.T)+np.outer(response,response)/precision
    np.testing.assert_allclose(actual[3],state,atol=2e-14,rtol=2e-14)
    np.testing.assert_allclose(actual[4]@actual[4].T,p,atol=2e-14,rtol=2e-14)
    np.testing.assert_array_equal(actual[5],np.zeros(20))
    np.testing.assert_allclose(actual[6],mass.sum())


def test_covariance_target_and_return_backbone_remain_M200(model):
    assert model.streamed.rough_fit is model.streamed.parent.parent.rough_fit
    assert model.streamed.predecessor_fit is model.streamed.base.predecessor_fit
    assert model.streamed.predecessor_asset_paths is model.streamed.base.predecessor_asset_paths
    assert model.streamed.prepared is model.prepared
