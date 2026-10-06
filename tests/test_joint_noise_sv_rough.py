"""Joint observation variance inference and unchanged long-horizon return mean."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def model():
    p=Path(__file__).resolve().parents[1]/'tools/joint_noise_sv_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_joint_noise_model',p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m


def test_joint_target_improves_fixed_noise_solution_and_shared_noise_is_identical(model):
    d=np.random.default_rng(10).normal(.0005,.01,600).tobytes();x=np.frombuffer(d,np.float64)
    original=model.original_fit(d);fit=model.refit(d);diagnostic=fit['joint_noise_fit'];R=diagnostic['noise_variance']
    assert diagnostic['success'];assert np.isfinite(R) and R>0
    y,_=model.shell.bd._sv_observed_log_variance(x,float(x.mean()))
    l,p,e=fit['posterior_center'][:3]
    prior=-.5*((l-y.mean())/4)**2-.5*((p-.94)/.20)**2-.5*(math.log(e)-math.log(.35))**2+math.log(p*(1-p))+math.log(e)
    likelihood=model.gaussian_sv.filter_states(y,l,p,e,R)[0]
    assert diagnostic['objective']==-(likelihood+prior)
    baseline=model.shell.bd._sv_transformed_log_posterior(y,original['posterior_center'][0],model.shell.bd._logit(original['posterior_center'][1]),math.log(original['posterior_center'][2]))
    assert diagnostic['objective']<=-baseline+1e-6
    assert model.parent.parent.base.noise.measurement_scale(d)[1]==R
    observed,_=model.observed(d)
    expected=y-model.parent.causal_predictor(y,l,p,e,R)
    assert observed.tobytes()==expected.tobytes()


def test_century_mean_exactly_preserved(model):
    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
        moment_return_curves,
    )
    d=np.random.default_rng(21).normal(.0005,.01,600).tobytes()
    reference,_=moment_return_curves(model.original_fit(d),25200)
    mean,paths=model.asset_paths(d,np.full((2,25200),.5))
    assert mean.tobytes()==reference.tobytes();assert np.isfinite(paths).all()
