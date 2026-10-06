"""Verify this ablation changes H support without changing its interior target."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('student_full_hurst_test_candidate',ROOT/'tools/student_pathwise_full_hurst_rough/models.py')
model=importlib.util.module_from_spec(spec);sys.modules[spec.name]=model;spec.loader.exec_module(model)


@pytest.mark.parametrize('hurst',[.02,.1,.3,.48])
def test_interior_likelihood_and_prior_are_byte_identical_to_m221(hurst):
    theta=np.array([hurst,math.log(1/63),math.log(.7)])
    n=101;periodogram=np.abs(np.fft.rfft(np.random.default_rng(78).normal(size=n)))**2/n
    a=model.whittle_target(theta,periodogram,n,5.5)
    b=model.backend.whittle_target(theta,periodogram,n,5.5)
    assert np.float64(a).tobytes()==np.float64(b).tobytes()


@pytest.mark.parametrize('hurst',[1e-6,.001,.499,.499999])
def test_full_rough_domain_is_accepted_without_new_interior_cutoffs(hurst):
    theta=np.array([hurst,math.log(1/63),math.log(.7)])
    assert np.isfinite(model.log_prior(theta))
    assert not np.isfinite(model.backend.log_prior(theta))
    assert np.isfinite(model.whittle_target(theta,np.ones(51),101,5.5))


@pytest.mark.parametrize('hurst',[-.01,0.,.5,.51])
def test_mathematical_domain_endpoints_are_excluded(hurst):
    assert model.log_prior(np.array([hurst,math.log(1/63),math.log(.7)]))==-np.inf
