"""Numerical gates for the research-only five-model panel."""
from __future__ import annotations

import importlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from numba import njit
from scipy.stats import poisson

pytest.importorskip('pyvinecopulib')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/mcmc_runtime'))
sys.path.insert(0, str(ROOT/'tools/rough_jump_vine'))
models = importlib.import_module('models')


def test_five_distinct_asset_level_ablations():
    assert len(models.CANDIDATES) == len(set(models.MODEL_IDS)) == 5
    assert [(m.rough, m.jumps, m.vine) for m in models.CANDIDATES] == [
        (False, False, False), (True, False, False), (False, True, False),
        (False, False, True), (True, True, True),
    ]


def test_exact_compound_poisson_count_inverse():
    uniforms = np.random.default_rng(91).random((1000, 5))
    rates = np.array([0., .005, .03, 1.2, 20.])
    actual = models.poisson_inverse(uniforms, rates)
    assert np.array_equal(actual, poisson.ppf(uniforms, rates[None, :]).astype(np.int64))


def test_fused_jump_mapping_preserves_every_value():
    rng = np.random.default_rng(31)
    original = rng.normal(size=(240, 126))
    counts = rng.poisson(.1, size=original.shape)
    normals = rng.normal(size=original.shape)
    rates = np.linspace(.01, .2, original.shape[1])
    location, variance = -.8, 3.7
    expected = original.copy()
    expected += counts*location+np.sqrt(counts*variance)*normals-rates[None, :]*location
    normalizer = np.sqrt(1+rates*(variance+location**2))
    expected /= normalizer[None, :]
    actual = original.copy()
    models.apply_jumps(actual, counts, normals, location, variance, rates*location, normalizer)
    assert actual.tobytes() == expected.tobytes()


def test_rough_gaussian_transition_and_optimization_are_exact():
    phi, weights, covariance = models.rough_parameters([.1, np.log(1/63), np.log(.7)])
    stationary = covariance/(1-phi[:, None]*phi[None, :])
    assert np.allclose(stationary*phi[:, None]*phi[None, :]+covariance, stationary)
    assert np.linalg.eigvalsh(covariance).min() > -1e-14
    rng = np.random.default_rng(191)
    initial = rng.normal(size=(240, 8))
    normals = rng.normal(size=(240, 126, 8))
    chol = np.linalg.cholesky(covariance+np.eye(8)*1e-14)
    slow = models.rough_paths_reference(phi, weights, chol, initial, normals)
    fast = models.rough_paths(phi, weights, chol, initial, normals)
    assert slow.tobytes() == fast.tobytes()
    assert fast.shape == (240, 126)


def test_rough_filter_fixed_covariance_reuse_preserves_the_original_recursion():
    @njit
    def original(y, phi, weights, covariance, level):
        state = np.zeros(len(phi))
        posterior = covariance/(1-phi[:, None]*phi[None, :])
        path = np.empty(len(y))
        likelihood = 0.
        for t in range(len(y)):
            predicted = posterior*phi[:, None]*phi[None, :]+covariance
            predicted_state = phi*state
            pw = predicted@weights
            variance = 4.934802200544679+np.dot(weights, pw)
            innovation = y[t]-level-np.dot(weights, predicted_state)
            state = predicted_state+(pw/variance)*innovation
            posterior = predicted-np.outer(pw, pw)/variance
            likelihood -= .5*(np.log(2*np.pi*variance)+innovation**2/variance)
            path[t] = level+np.dot(weights, state)
        return likelihood, state, posterior, path
    y = np.random.default_rng(191).normal(size=6000)
    phi, weights, covariance = models.rough_parameters([.1, np.log(1/63), np.log(.7)])
    slow = original(y, phi, weights, covariance, -.1)
    fast = models.filter_rough(y, phi, weights, covariance, -.1)
    assert slow[0] == fast[0]
    assert all(a.tobytes() == b.tobytes() for a, b in zip(slow[1:], fast[1:]))


def test_hmm_smoothing_matches_exhaustive_state_enumeration():
    transition = np.array([[.8, .1, .1], [.1, .7, .2], [.2, .1, .7]])
    initial = np.array([.2, .5, .3])
    emissions = np.array([[.8, .4, .5], [.2, .9, .4], [.7, .3, .8], [.5, .6, .3]])
    likelihood, posterior, counts, terminal = models.hmm_forward_backward(
        np.log(emissions), transition, initial)
    expected = np.zeros_like(posterior)
    expected_counts = np.zeros_like(counts)
    total = 0.
    for states in itertools.product(range(3), repeat=4):
        mass = initial[states[0]]*emissions[0, states[0]]
        for t in range(1, 4):
            mass *= transition[states[t-1], states[t]]*emissions[t, states[t]]
        total += mass
        for t, state in enumerate(states):
            expected[t, state] += mass
        for t in range(1, 4):
            expected_counts[states[t-1], states[t]] += mass
    assert np.allclose(posterior, expected/total, rtol=1e-13, atol=1e-13)
    assert np.allclose(counts, expected_counts/total, rtol=1e-13, atol=1e-13)
    assert np.allclose(terminal, posterior[-1])
    assert np.isclose(likelihood, np.log(total))


def test_scoring_retains_the_empirical_n_squared_denominator():
    from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
    samples = np.random.default_rng(2).normal(size=(240, 8))
    target = np.linspace(-1., 1., 8)
    expected = np.abs(samples-target[None, :]).mean(axis=0)
    expected -= .5*np.abs(samples[:, None, :]-samples[None, :, :]).mean(axis=(0, 1))
    assert np.allclose(empirical_crps_by_horizon(samples, target), expected, rtol=1e-13)
