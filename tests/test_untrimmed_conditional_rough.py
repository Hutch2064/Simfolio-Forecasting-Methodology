"""Untrimmed observations affect only the rough quasi likelihood."""
import importlib.util
import sys
from pathlib import Path

import numpy as np


def test_untrimmed_proxy_keeps_tails_and_preserves_conventional_offset():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("untrimmed_rough_test",root/"tools/untrimmed_conditional_rough/models.py")
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    x=np.random.default_rng(39).normal(0,.01,300);x[-1]=.5
    raw=m.log_square(x)
    expected=np.log(((x-x.mean())*100)**2)-m.shell.bd.SV_LOG_CHI_SQUARE_MEAN
    np.testing.assert_array_equal(raw,expected)
    old,_=m.shell.bd._sv_observed_log_variance(x,float(x.mean()))
    assert raw[-1]>old[-1]
    fit=m.base.predecessor_fit(x.tobytes())
    level,phi,eta,_,_=fit["posterior_center"]
    actual,_=m.observed(x.tobytes())
    np.testing.assert_array_equal(actual,raw-m.base.causal_log_variance_predictor(old,level,phi,eta))
    reference=m.shell.bd.fit_bdes_fastmap(x,filtered_innovations=True,fixed_mean=True)
    np.testing.assert_array_equal(fit["innovation_pool"],reference["innovation_pool"])
    _scale,variance=m.measurement_scale(x.tobytes())
    assert variance==np.var(m.log_square(fit["innovation_pool"]*.01),ddof=1)
    assert np.isfinite(m.log_square(np.zeros(10))).all()
