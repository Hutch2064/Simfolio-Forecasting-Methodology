import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def model():
    path=Path(__file__).resolve().parents[1]/'tools/streamed_rough_paths/models.py'
    spec=importlib.util.spec_from_file_location('streamed_test_model',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.initialize()
    return module


@pytest.mark.parametrize('n',[1,4,7,14,37])
@pytest.mark.parametrize('horizon',[1,29,8766])
def test_entire_multiplier_and_rng_state_byte_exact(model,n,horizon):
    phi=np.linspace(0,.95,n);w=np.linspace(.3,1,n)
    root=np.diag(np.linspace(0,.3,n));initial=np.linspace(-.2,.4,n)
    means=np.linspace(0,.1,horizon);variance=np.linspace(1,2,horizon)
    for seed in [1,982,739]:
        reference_rng=np.random.default_rng(seed);stream_rng=np.random.default_rng(seed)
        normals=np.empty((horizon+1,n));normals[0]=0
        normals[1:]=reference_rng.normal(size=(horizon,n))
        reference=model.overlay.multiplier_independent_prepared(phi,w,root,initial,normals,means,variance)
        actual=model.stream_multiplier(phi,w,root,initial,means,variance,stream_rng.bit_generator.capsule)
        assert actual.tobytes()==reference.tobytes()
        assert stream_rng.bit_generator.state==reference_rng.bit_generator.state
        mapped_rng=np.random.default_rng(seed)
        return_mean=np.linspace(-.1,.1,horizon)
        base_path=np.linspace(-1,1,horizon)
        destination=np.zeros((horizon,3))
        module,dot=model.load_paths()
        module.map_path(phi,w,root,initial,means,variance,mapped_rng.bit_generator.capsule,
                        dot,return_mean,base_path,destination[:,1])
        expected=np.clip(return_mean+(base_path-return_mean)*reference,-1,1)
        assert destination[:,1].tobytes()==expected.tobytes()
        assert mapped_rng.bit_generator.state==reference_rng.bit_generator.state
        assert not destination[:,0].any() and not destination[:,2].any()


def test_dense_root_rejected_without_consuming_rng(model):
    rng=np.random.default_rng(186);before=rng.bit_generator.state
    with pytest.raises(ValueError,match='diagonal'):
        model.stream_multiplier(np.array([.8,.9]),np.ones(2),np.ones((2,2)),
                                np.zeros(2),np.zeros(1),np.ones(1),rng.bit_generator.capsule)
    assert before==rng.bit_generator.state


def test_inference_prediction_parameters_and_identity_are_inherited(model):
    assert model.rough_fit is model.base.rough_fit
    assert model.prepared is model.base.prepared
    assert model.predecessor_fit is model.base.predecessor_fit
    assert model.CANDIDATES[0].model_id==model.parent.CANDIDATES[0].model_id
