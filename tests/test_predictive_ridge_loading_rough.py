"""Independent ridge, identifiability, forecast-target and state construction tests."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def model():
    path=Path(__file__).resolve().parents[1]/'tools/predictive_ridge_loading_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_predictive_ridge',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize('ridge',[.01,500.])
def test_svd_ridge_matches_regularized_normal_equations(model,ridge):
    h=np.random.default_rng(52).normal(size=200);h-=h.mean();phis=np.array([.7,.97])
    q,b=model.ridge_fit.projection(h,phis,ridge)
    expected=np.linalg.solve(q.T@q+ridge*max(h.var(),1e-8)*np.eye(2),q.T@h)
    np.testing.assert_allclose(b,expected,rtol=1e-12,atol=1e-12)


def test_zero_ridge_handles_rank_deficient_columns(model):
    h=np.random.default_rng(53).normal(size=200);h-=h.mean()
    q,b=model.ridge_fit.projection(h,np.array([.8,.8]),0.)
    expected=np.linalg.lstsq(q,h,rcond=np.finfo(float).eps*max(q.shape))[0]
    np.testing.assert_allclose(b,expected,rtol=1e-12,atol=1e-12)


def test_one_column_ridge_is_absorbed_by_contraction(model):
    h=np.random.default_rng(54).normal(size=200);h-=h.mean();phis=np.array([.9]);scale=.3
    q,zero=model.ridge_fit.projection(h,phis,0.)
    _,penalized=model.ridge_fit.projection(h,phis,.05)
    equivalent=scale*(q.T@q).item()/((q.T@q).item()+.05*max(h.var(),1e-8))
    assert 0<equivalent<scale<1
    np.testing.assert_allclose(zero*equivalent,penalized*scale,rtol=1e-12)


def test_predictive_errors_and_state_construction_agree(model):
    h=np.random.default_rng(55).normal(-1.,.3,300);centered=h-h.mean()
    phis=np.array([.7,.98]);scale=.4;ridge=3.
    q,raw=model.ridge_fit.projection(centered,phis,ridge);b=raw*scale
    residual=centered-q@b;residual-=residual.mean()
    rho=np.clip(residual[:-1]@residual[1:]/(residual[:-1]@residual[:-1]),-.999999,.999999)
    expected=centered[1:]-(q[:-1]*phis)@b-rho*residual[:-1]
    np.testing.assert_allclose(model.ridge_fit.errors(centered,phis,scale,ridge),expected,rtol=1e-12,atol=1e-12)
    state=model.learned.components(h,phis,loading_scale=scale,loading_coefficients=raw)
    np.testing.assert_allclose(state['b'],b,rtol=1e-12)
    assert state['residual_phi']==pytest.approx(rho,rel=1e-12)


def test_adaptive_fit_counts_only_identifiable_ridge_parameter(model):
    h=np.random.default_rng(56).normal(size=300)
    rates,record=model.ridge_fit.fit_adaptive(h,np.array([.9]))
    assert record['success'] and 0<record['loading_scale']<1
    assert record['ridge_parameter_count']==int(len(rates)>1)
    assert record['bic_parameter_count']==2*len(rates)+4+int(len(rates)>1)
    assert record['selected_bic']==min(x['bic'] for x in record['tested_orders'])
    assert len(rates)>1 or record['ridge_scale']==0.
