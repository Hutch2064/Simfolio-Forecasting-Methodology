"""Noise estimation requires the transformation Jacobian in the likelihood."""
import importlib.util
import sys
from pathlib import Path

import numpy as np


def test_joint_noise_likelihood_has_correct_gaussian_variance_optimum():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("profile_noise_test",root/"tools/profile_noise_rough/models.py")
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    y=np.random.default_rng(25).normal(.2,2.,300)
    level=float(y.mean());phi=np.array([.9]);w=np.ones(1);q=np.zeros((1,1))
    def target(theta):
        scale=np.sqrt(4.934802200544679/np.exp(theta[0]))
        return m.overlay.filter_rough(y*scale,phi,w,q,level*scale)[0]+len(y)*np.log(scale)
    fit=m.base.fit_map(target,[np.log(1.)],[(-20.,20.)])
    np.testing.assert_allclose(np.exp(fit["map"][0]),np.mean((y-level)**2),rtol=1e-5)
    assert m.base.rough_fit is m.rough_fit
    assert m.base.prepared is m.prepared
