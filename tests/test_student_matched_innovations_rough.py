"""Independent Student(3) quantile reference and finite-grid moment checks."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('student_matched_nodes_test',ROOT/'tools/student_matched_innovations_rough/models.py')
model=importlib.util.module_from_spec(spec);sys.modules[spec.name]=model;spec.loader.exec_module(model)


def test_student_three_nodes_match_closed_form_cdf_inversion():
    n=31;probabilities=np.linspace(.5/n,1-.5/n,n)
    def cdf(x):
        z=x/math.sqrt(3)
        return .5+(math.atan(z)+z/(1+z*z))/math.pi
    reference=np.array([brentq(lambda x,p=p:cdf(x)-p,-100.,100.,xtol=1e-13) for p in probabilities])
    reference-=reference.mean();reference/=np.sqrt(np.mean(reference*reference))
    np.testing.assert_allclose(model.normalized_student_nodes(1/3,n),reference,rtol=3e-12,atol=3e-12)


@pytest.mark.parametrize('inverse_df',[0.,1e-6,.01,.1,.3,.499999])
def test_all_tail_shapes_preserve_grid_mean_variance_and_order(inverse_df):
    nodes=model.normalized_student_nodes(inverse_df,240)
    assert np.isfinite(nodes).all()
    assert np.all(np.diff(nodes)>0)
    assert abs(nodes.mean())<2e-15
    assert np.mean(nodes*nodes)==pytest.approx(1.,abs=3e-15)
    np.testing.assert_allclose(nodes,-nodes[::-1],atol=2e-12)
