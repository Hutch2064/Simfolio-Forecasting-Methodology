import shutil
from importlib import resources

import pytest

from simfolio_forecasting_methodology.data import (
    CANONICAL_COMMON_RETURN_COUNT,
    CANONICAL_NORMALIZED_RETURN_SHA256,
    CanonicalDataUnavailable,
    canonical_calendar,
    canonical_snapshot_manifest,
    verify_canonical_snapshot,
    verify_prepared_canonical_data,
    write_canonical_data,
)


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


def test_default_preparation_is_offline_and_reconstructs_fingerprints(tmp_path):
    cache = tmp_path / "prepared"
    manifest = write_canonical_data(cache)
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
