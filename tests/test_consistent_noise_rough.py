"""Independent filtering recursion and exact unchanged-noise reference."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

@pytest.fixture(scope='module')
def model():
    p=Path(__file__).resolve().parents[1]/'tools/consistent_noise_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_consistent_noise_model',p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m

@pytest.mark.parametrize('noise',[.5,2.,8.])
def test_causal_offset_matches_independent_filter(model,noise):
    y=np.random.default_rng(57).normal(size=100);level,phi,eta=.2,.95,.3
    mean=level;variance=eta**2/(1-phi**2);expected=[]
    for observation in y:
        expected.append(mean)
        filtered_mean=mean+variance/(variance+noise)*(observation-mean)
        filtered_variance=variance*noise/(variance+noise)
        mean=level+phi*(filtered_mean-level);variance=phi**2*filtered_variance+eta**2
    np.testing.assert_allclose(model.causal_predictor(y,level,phi,eta,noise),expected,rtol=1e-13,atol=1e-13)


def test_original_noise_is_exact_and_current_observation_cannot_change_prediction(model):
    old=model.parent.base.base.causal_log_variance_predictor
    y=np.random.default_rng(42).normal(size=100)
    assert old(y,.2,.95,.3).tobytes()==model.causal_predictor(y,.2,.95,.3,4.934802200544679).tobytes()
    changed=y.copy();changed[50]+=100
    a=model.causal_predictor(y,.2,.95,.3,2.);b=model.causal_predictor(changed,.2,.95,.3,2.)
    assert np.array_equal(a[:51],b[:51]);assert a[51]!=b[51]
    assert model.parent.streamed.predecessor_asset_paths is model.parent.streamed.base.predecessor_asset_paths
