"""Exact scalar/path guards and lifecycle of the isolated latent filter adapter."""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numba")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/catalogue52"))
import latent_exact_filter as adapter


@pytest.fixture(scope="module", autouse=True)
def prepared():
    adapter.prepare()


def assert_filter_bytes(actual, reference):
    assert struct.pack("<d", actual[0]) == struct.pack("<d", reference[0])
    for left, right in zip(actual[1:], reference[1:]):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert left.tobytes() == right.tobytes()


@pytest.mark.parametrize("return_path", [False, True])
@pytest.mark.parametrize("n", [1, 507, 1536])
@pytest.mark.parametrize("phi,eta", [(0.0, 0.35), (0.94, 0.35), (0.9989, 0.02)])
def test_complete_likelihood_and_path_bytes(return_path, n, phi, eta):
    values = np.random.default_rng(83).normal(-8.0, 2.0, n)
    assert_filter_bytes(adapter._filter(values, -8.0, phi, eta, return_path=return_path),
                        adapter._SOURCE_FILTER(values, -8.0, phi, eta, return_path=return_path))


@pytest.mark.parametrize("values", [np.array([]), np.array([np.nan, np.inf]),
    np.array([np.nan, -7.0, np.inf, -9.0]), np.arange(12, dtype=np.float64)[::2],
    np.frombuffer(np.array([-7.0, -8.0]).tobytes(), dtype=np.float64)])
@pytest.mark.parametrize("level,phi,eta", [(-8.0, 0.94, 0.35), (np.nan, 0.94, 0.35),
    (-8.0, np.inf, 0.35), (-8.0, -0.001, 0.35), (-8.0, 0.999, 0.35),
    (-8.0, 0.94, 0.0), (-8.0, 0.94, 1e-6), (-8.0, 0.94, np.nan)])
def test_original_guards_and_array_layouts(values, level, phi, eta):
    for return_path in (False, True):
        assert_filter_bytes(adapter._filter(values, level, phi, eta, return_path=return_path),
                            adapter._SOURCE_FILTER(values, level, phi, eta,
                                                   return_path=return_path))


def test_omitted_keyword_default_and_retained_source():
    values = np.array([-8.0, -7.0, -9.0])
    assert_filter_bytes(adapter._filter(values, -8.0, 0.94, 0.35),
                        adapter._SOURCE_FILTER(values, -8.0, 0.94, 0.35))
    assert adapter._COMPILED_FILTER.py_func is adapter._SOURCE_FILTER
    assert adapter._COMPILED_FILTER.targetoptions["fastmath"] is False


def test_nested_error_restoration_and_only_one_global_replacement():
    original = adapter.mc._sv_kalman_filter
    untouched = {name: getattr(adapter.mc, name) for name in
                 ("_dlm_ar1_loglik", "_ffbs_sv_log_variance_path", "fit_full_mcmc_sv")}
    with adapter.accelerated():
        assert adapter.mc._sv_kalman_filter is adapter._filter
        with pytest.raises(RuntimeError, match="intentional"), adapter.accelerated():
            raise RuntimeError("intentional")
        assert adapter.mc._sv_kalman_filter is adapter._filter
        assert all(getattr(adapter.mc, name) is value for name, value in untouched.items())
    assert adapter.mc._sv_kalman_filter is original


def test_model_error_restores_filter():
    adapter.initialize(None, 1, 1)
    original = adapter.mc._sv_kalman_filter
    with pytest.raises(AttributeError):
        adapter.CANDIDATES[0].simulate_daily_log_returns(None, None)
    assert adapter.mc._sv_kalman_filter is original


def test_repeat_initialize_preserves_all_original_forty_factories():
    original = adapter.mc._sv_kalman_filter
    compiled = adapter._COMPILED_FILTER
    adapter.initialize(None, 1, 1)
    with adapter.accelerated():
        adapter.initialize(None, 1, 1)
        assert adapter.mc._sv_kalman_filter is adapter._filter
        assert adapter._COMPILED_FILTER is compiled
    adapter.initialize(None, 1, 1)
    assert adapter.mc._sv_kalman_filter is original
    assert adapter._COMPILED_FILTER is compiled
    assert tuple(model.model_id for model in adapter.CANDIDATES) == (
        adapter.mc.canonical_full_mcmc_sv_ids())
    assert len(adapter.CANDIDATES) == 40
    assert all(type(model.reference) is adapter.full_mcmc.FullMCMCSVModel
               for model in adapter.CANDIDATES)
    assert all(len(value) == 64 for value in adapter.source_version()["files"].values())
