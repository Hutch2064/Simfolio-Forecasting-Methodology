import numpy as np
import pytest
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
import ma1


def test_filter_matches_independent_library():
    y = np.random.default_rng(17).normal(size=100)
    for theta in (-.8, -.1, 0., .2, .9):
        errors, variances = ma1.innovations(y, theta)
        independent, factors = arma_innovations(y, ma_params=[theta], sigma2=1.)
        np.testing.assert_allclose(errors, independent, atol=2e-14, rtol=1e-13)
        np.testing.assert_allclose(variances, factors, atol=2e-14, rtol=1e-13)


def test_profile_likelihood_matches_dense_covariance():
    y = np.random.default_rng(9).normal(size=75)
    for theta in (-.7, 0., .4):
        matrix = np.diag(np.full(len(y), 1.+theta*theta))
        matrix += np.diag(np.full(len(y)-1, theta), 1)+np.diag(np.full(len(y)-1, theta), -1)
        chol = np.linalg.cholesky(matrix)
        solved = np.linalg.solve(chol, y)
        concentrated_variance = float(solved@solved/len(y))
        dense = .5*(np.log(concentrated_variance)+2*np.sum(np.log(np.diag(chol)))/len(y))
        assert ma1.concentrated_loss(theta, y) == pytest.approx(dense, abs=2e-13)


def test_fit_equivariance_and_known_MA_signal():
    eps = np.random.default_rng(25).normal(size=12001)
    for theta in (-.5, .2, .7):
        history = eps[1:]+theta*eps[:-1]
        fit = ma1.fit(history)
        transformed = ma1.fit(2*history+.02)
        assert abs(fit['theta']-theta) < .025
        assert transformed['theta'] == pytest.approx(fit['theta'], abs=1e-7)
        assert transformed['innovation_variance'] == pytest.approx(4*fit['innovation_variance'], rel=1e-7)
        assert fit['profile_objective_standardized'] <= fit['null_objective_standardized']


def test_zero_coefficient_is_exact_IID_boundary():
    y = np.random.default_rng(7).normal(size=60)
    errors, factors = ma1.innovations(y, 0.)
    np.testing.assert_array_equal(errors, y)
    np.testing.assert_array_equal(factors, np.ones(len(y)))
