"""Parity tests for the source-derived canonical base closure."""

from __future__ import annotations

from importlib import resources

import numpy as np
import pytest

from simfolio_forecasting_methodology.models.numerical.base_models import (
    CanonicalBaseModel,
    build_canonical_base_model,
    fit_base_model,
    fit_mean,
    historical_base_seed,
    simulate_base_paths,
)
from simfolio_forecasting_methodology.runner import ForecastContext, TrainingData
from simfolio_forecasting_methodology.specifications import parse_compositional_spec

_VOL_MODELS = (
    "constant_sample_volatility",
    "garch_1_1_volatility",
    "gjr_tarch_1_1_volatility",
    "egarch_1_1_volatility",
)
_PARAMETRIC = (
    "gaussian_iid_standardized_innovations",
    "student_t_standardized_innovations",
    "skew_t_standardized_innovations",
)
_EMPIRICAL = (
    ("stationary_bootstrap", "filtered_empirical_tail"),
    ("stationary_bootstrap", "automated_evt_pot_gpd_tail"),
    ("iid", "filtered_empirical_tail"),
    ("iid", "automated_evt_pot_gpd_tail"),
)
_PARAM_KEYS = ("sigma_x", "omega", "alpha[1]", "gamma[1]", "beta[1]", "nu", "eta", "lambda")


def _ids_for_means(means: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for mean in means:
        for volatility in _VOL_MODELS:
            values.extend(
                f"{mean}|{volatility}|{innovation}|parametric"
                for innovation in _PARAMETRIC
            )
            values.extend(
                f"{mean}|{volatility}|empirical|{resampling}|{tail}"
                for resampling, tail in _EMPIRICAL
            )
    return values


def _ids() -> list[str]:
    return _ids_for_means(("expanding_sample_mean",))


def _load(name: str = "source_reference.npz") -> dict[str, np.ndarray]:
    fixture = (
        resources.files("simfolio_forecasting_methodology")
        .joinpath("resources", "test_fixtures", "base", name)
    )
    with fixture.open("rb") as handle, np.load(handle, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def test_canonical_base_domain_has_exactly_84_ids() -> None:
    ids = _ids()
    assert len(ids) == 28
    all_ids = [
        model_id.replace("expanding_sample_mean", mean)
        for mean in (
            "expanding_sample_mean",
            "bic_auto_arma_mean",
            "factor_premium_near_zero_alpha_shrinkage",
        )
        for model_id in ids
    ]
    assert len(all_ids) == 84
    assert len(set(all_ids)) == 84
    for model_id in all_ids:
        assert build_canonical_base_model(model_id).model_id == model_id


def test_expanding_source_fixture_matches_fit_state_and_paths() -> None:
    fixture = _load()
    ids = [str(value) for value in fixture["model_ids"]]
    assert ids == _ids()
    values = fixture["training_log_returns"]
    for row, model_id in enumerate(ids):
        fit = fit_base_model(values, parse_compositional_spec(model_id))
        np.testing.assert_allclose(fit["mu"], fixture["fit_mu"][row], rtol=0.0, atol=2e-12)
        np.testing.assert_allclose(
            fit["residuals"], fixture["fit_residuals"][row], rtol=0.0, atol=2e-12
        )
        np.testing.assert_allclose(
            fit["standardized_residuals"],
            fixture["fit_standardized_residuals"][row],
            rtol=0.0,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            fit["vol_fit"]["sigma_x"], fixture["fit_sigma_x"][row], rtol=0.0, atol=2e-12
        )
        params = fit["vol_fit"].get("params", {})
        observed = np.asarray([float(params.get(key, np.nan)) for key in _PARAM_KEYS])
        expected = fixture["fit_params"][row]
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2e-12, equal_nan=True)
        seed = historical_base_seed(model_id, "2026-05-13", 16, 16)
        assert seed == int(fixture["seed"][row])
        paths = simulate_base_paths(
            fit,
            horizon_days=16,
            simulations=16,
            rng=np.random.default_rng(seed),
        )
        np.testing.assert_allclose(paths, fixture["paths"][row], rtol=0.0, atol=2e-12)


def test_all_84_mean_and_volatility_branches_match_source_fixture() -> None:
    fixture = _load("source_reference_all_means.npz")
    ids = [str(value) for value in fixture["model_ids"]]
    assert ids == _ids_for_means(
        (
            "expanding_sample_mean",
            "bic_auto_arma_mean",
            "factor_premium_near_zero_alpha_shrinkage",
        )
    )
    values = fixture["training_log_returns"]
    for row, model_id in enumerate(ids):
        fit = fit_base_model(values, parse_compositional_spec(model_id))
        np.testing.assert_allclose(fit["mu"], fixture["fit_mu"][row], rtol=0.0, atol=2e-12)
        np.testing.assert_allclose(
            fit["residuals"], fixture["fit_residuals"][row], rtol=0.0, atol=2e-12
        )
        np.testing.assert_allclose(
            fit["standardized_residuals"],
            fixture["fit_standardized_residuals"][row],
            rtol=0.0,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            fit["vol_fit"]["sigma_x"], fixture["fit_sigma_x"][row], rtol=0.0, atol=2e-12
        )
        params = fit["vol_fit"].get("params", {})
        observed = np.asarray([float(params.get(key, np.nan)) for key in _PARAM_KEYS])
        np.testing.assert_allclose(
            observed, fixture["fit_params"][row], rtol=0.0, atol=2e-12, equal_nan=True
        )
        seed = historical_base_seed(model_id, "2026-05-13", 16, 16)
        paths = simulate_base_paths(
            fit,
            horizon_days=16,
            simulations=16,
            rng=np.random.default_rng(seed),
        )
        np.testing.assert_allclose(paths, fixture["paths"][row], rtol=0.0, atol=2e-12)


def test_public_adapter_matches_source_paths_without_private_imports() -> None:
    fixture = _load()
    model_id = str(fixture["model_ids"][0])
    model = CanonicalBaseModel(model_id)
    training = TrainingData(portfolio_log_returns=fixture["training_log_returns"])
    context = ForecastContext(
        model_id=model_id,
        portfolio_id="fixture",
        origin_label="2026-05-13",
        origin_date="2026-05-13",
        horizon_days=16,
        simulations=16,
        seed=999,
    )
    output = model.simulate_daily_log_returns(training, context)
    np.testing.assert_allclose(output, fixture["paths"][0], rtol=0.0, atol=2e-12)


def test_unknown_base_components_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown volatility model"):
        build_canonical_base_model(
            "expanding_sample_mean|estimated_decay_ewma_volatility|gaussian_iid_standardized_innovations|parametric"
        )
    with pytest.raises(ValueError, match="unknown canonical base mean model"):
        fit_mean(np.ones(40, dtype=np.float64), "unknown_mean_model")
