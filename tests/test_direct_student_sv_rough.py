"""Direct Student parameter inheritance and dense conditional state covariance."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def model():
    path=Path(__file__).resolve().parents[1]/'tools/direct_student_sv_rough/models.py'
    spec=importlib.util.spec_from_file_location('direct_student_sv_test',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    m.initialize(None,1,1)
    return m


def test_direct_fit_preserves_raw_MAP_parameters_and_unclipped_terminal_state(model,monkeypatch):
    original={'posterior_center':(.5,.8,.2,-22.,-.1),'state_path_variance_last':.3}
    monkeypatch.setattr(model.learned,'original_student_fit',lambda _:original)
    model.refit.cache_clear()
    result,fit=model.refit(np.arange(97.,dtype=float).tobytes())
    assert result is original
    assert fit['level']==.5 and fit['coefficients']==[.8] and fit['innovation_sd']==.2
    assert fit['initial']==[-22.5] and fit['terminal_laplace_variance']==.3
    assert fit['estimator']=='direct_raw_Student_return_Laplace_MAP_parameters'
    model.refit.cache_clear()


@pytest.mark.parametrize('initial_variance',[0.,.3])
def test_state_configuration_matches_full_gaussian_innovation_map(model,monkeypatch,initial_variance):
    from simfolio_forecasting_methodology.models.asset_level import sv_moment_functions as moments
    rho=.8;sd=.2;n=31;level=.5;last=.8
    fit={'phis':[],'coefficients':[rho],'initial':[last-level],'innovation_sd':sd,'level':level,'terminal_laplace_variance':initial_variance}
    original={'base_fit':{'sigma':.02}}
    monkeypatch.setattr(model,'refit',lambda _: (original,fit))
    monkeypatch.setattr(moments,'predictive_state_moments',lambda *_:(None,None,np.zeros(n),np.zeros(n)))
    model.configuration.cache_clear()
    *_,mean,variance,return_sd=model.configuration(b'dense_state_reference',n)
    basis=np.zeros((n,n+1))
    for t in range(n):
        basis[t,0]=np.sqrt(initial_variance)*rho**(t+1)
        for j in range(t+1):basis[t,j+1]=sd*rho**(t-j)
    np.testing.assert_allclose(mean,level+(last-level)*rho**np.arange(1,n+1),rtol=1e-12)
    np.testing.assert_allclose(variance,np.diag(basis@basis.T),rtol=1e-12)
    np.testing.assert_allclose(return_sd,np.exp(.5*mean+.25*variance)/100.,rtol=1e-12)
    model.configuration.cache_clear()
