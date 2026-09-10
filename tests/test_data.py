import json
import shutil
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from simfolio_forecasting_methodology import data
from simfolio_forecasting_methodology.data import (
    CANONICAL_COMMON_RETURN_COUNT,
    CANONICAL_NORMALIZED_RETURN_SHA256,
    CanonicalDataUnavailable,
    _array_digest,
    _frozen_derived_snapshot_manifest,
    _load_frozen_derived_snapshot,
    _validate_frozen_derived_arrays,
    canonical_calendar,
    canonical_snapshot_manifest,
    load_canonical_engine_inputs,
    verify_canonical_snapshot,
    verify_prepared_canonical_data,
    write_canonical_data,
)


@pytest.fixture(scope="module")
def prepared_cache(tmp_path_factory) -> Path:
    cache = tmp_path_factory.mktemp("canonical-data") / "prepared"
    write_canonical_data(cache)
    return cache


def test_manifest_and_calendar_are_value_free_identities():
    manifest = canonical_snapshot_manifest()
    assert len(manifest["series"]) == 52
    assert manifest["common_date_count"] == CANONICAL_COMMON_RETURN_COUNT
    assert manifest["normalization"]["normalized_return_matrix_sha256"] == CANONICAL_NORMALIZED_RETURN_SHA256
    calendar = canonical_calendar()
    assert len(calendar) == CANONICAL_COMMON_RETURN_COUNT
    assert calendar[0].date().isoformat() == "1979-12-31"
    assert calendar[-1].date().isoformat() == "2026-05-13"


def test_snapshot_verifier_fails_closed_before_any_source_read(tmp_path):
    with pytest.raises(CanonicalDataUnavailable, match="snapshot series"):
        verify_canonical_snapshot(tmp_path)


def test_packaged_snapshot_verifies_without_external_source():
    result = verify_canonical_snapshot()
    assert result["rights_status"] == "caller_authorized_local_use"
    assert result["redistribution_status"] == "user_authorized_exact_snapshot"
    assert result["series_count"] == 52
    assert result["supporting_series"] == ["BNDSIM", "CASHX", "EFFRX", "KMLMSIM", "TIPSIM", "UUPSIM"]
    assert result["factor_inputs"] == ["french_daily", "q5_daily"]


def test_packaged_derived_snapshot_has_exact_schema_and_fingerprints():
    dates, tickers, portfolio_ids, assets, portfolios = _load_frozen_derived_snapshot()
    assert dates.dtype.str == "<M8[D]"
    assert assets.dtype.str == "<f8"
    assert portfolios.dtype.str == "<f8"
    assert assets.shape == (CANONICAL_COMMON_RETURN_COUNT, 52)
    assert portfolios.shape == (CANONICAL_COMMON_RETURN_COUNT, 80)
    assert tickers[0] == "EFASIM"
    assert tickers[-1] == "ZROZSIM"
    assert portfolio_ids[0] == "equal_class_history_001"
    assert portfolio_ids[-1] == "equal_class_history_080"


def test_packaged_snapshot_tamper_is_rejected(tmp_path):
    resource = resources.files("simfolio_forecasting_methodology").joinpath(
        "resources", "data", "canonical_snapshot"
    )
    with resources.as_file(resource) as source_root:
        copied = tmp_path / "snapshot"
        shutil.copytree(source_root, copied)
    target = copied / "app" / "simulated_data" / "series" / "EFASIM.csv.gz"
    original = target.read_bytes()
    target.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(CanonicalDataUnavailable, match="source hash mismatch"):
        verify_canonical_snapshot(copied, rights_confirmed=True)


def test_default_preparation_is_offline_and_reconstructs_fingerprints(prepared_cache):
    cache = prepared_cache
    manifest = cache / "canonical_data_manifest.json"
    assert manifest == cache / "canonical_data_manifest.json"
    result = verify_prepared_canonical_data(cache)
    assert result["source_series_count"] == 52
    assert result["supporting_series_count"] == 6
    assert result["factor_input_count"] == 2
    assert result["portfolio_series_count"] == 80
    assert result["asset_log_matrix_sha256"] == (
        "d0198f779d1e60146df4e0c5761ca2716abad8a140119222cada0c894354a460"
    )
    assert result["portfolio_log_matrix_sha256"] == (
        "cc685b4f6f4570b48bbac1a92be9e91659e14cf84dd2d6703efaa84530e2927f"
    )


def test_prepared_engine_load_uses_frozen_matrices(prepared_cache, monkeypatch):
    def fail_recompute(*args, **kwargs):
        raise AssertionError("prepared loading must not recompute source-derived values")

    monkeypatch.setattr(data, "_price_simple_returns", fail_recompute)
    monkeypatch.setattr(data, "_source_portfolio_log_returns", fail_recompute)
    assets, portfolios = load_canonical_engine_inputs(prepared_cache)
    _, tickers, portfolio_ids, expected_assets, expected_portfolios = (
        _load_frozen_derived_snapshot()
    )
    assert tuple(assets.columns) == tickers
    assert np.array_equal(assets.to_numpy(dtype="<f8"), expected_assets)
    assert tuple(portfolios) == portfolio_ids
    assert all(
        np.array_equal(portfolios[portfolio_id].to_numpy(dtype="<f8"), expected_portfolios[:, index])
        for index, portfolio_id in enumerate(portfolio_ids)
    )


def test_prepared_derived_snapshot_corruption_is_rejected(prepared_cache, tmp_path):
    copied = tmp_path / "corrupt"
    shutil.copytree(prepared_cache, copied)
    target = copied / "canonical_derived_matrices.npz"
    original = target.read_bytes()
    target.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(CanonicalDataUnavailable, match="derived snapshot hash mismatch"):
        verify_prepared_canonical_data(copied)


def test_derived_snapshot_rejects_column_order_and_matrix_mutation():
    dates, tickers, portfolio_ids, assets, portfolios = _load_frozen_derived_snapshot()
    expected = json.loads(json.dumps(_frozen_derived_snapshot_manifest()))

    reordered = {
        "dates": dates.copy(),
        "tickers": np.asarray((tickers[1], tickers[0], *tickers[2:]), dtype="<U8"),
        "portfolio_ids": np.asarray(portfolio_ids, dtype="<U24"),
        "asset_log_returns": assets.copy(),
        "portfolio_log_returns": portfolios.copy(),
    }
    expected["array_schema"]["tickers"]["sha256"] = _array_digest(reordered["tickers"])
    with pytest.raises(CanonicalDataUnavailable, match="asset column order"):
        _validate_frozen_derived_arrays(reordered, expected_snapshot=expected)

    mutated = {
        "dates": dates.copy(),
        "tickers": np.asarray(tickers, dtype="<U8"),
        "portfolio_ids": np.asarray(portfolio_ids, dtype="<U24"),
        "asset_log_returns": assets.copy(),
        "portfolio_log_returns": portfolios.copy(),
    }
    expected["array_schema"]["tickers"]["sha256"] = _array_digest(mutated["tickers"])
    mutated["asset_log_returns"][0, 0] = np.nextafter(
        mutated["asset_log_returns"][0, 0], np.inf
    )
    expected["array_schema"]["asset_log_returns"]["sha256"] = _array_digest(
        mutated["asset_log_returns"]
    )
    with pytest.raises(CanonicalDataUnavailable, match="asset matrix fingerprint"):
        _validate_frozen_derived_arrays(mutated, expected_snapshot=expected)
