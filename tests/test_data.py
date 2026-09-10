import pytest

from simfolio_forecasting_methodology.data import (
    CANONICAL_COMMON_RETURN_COUNT,
    CANONICAL_NORMALIZED_RETURN_SHA256,
    CanonicalDataUnavailable,
    canonical_calendar,
    canonical_snapshot_manifest,
    verify_canonical_snapshot,
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


def test_compatibility_builder_cannot_create_proxy_data(tmp_path):
    with pytest.raises(CanonicalDataUnavailable, match="proxy"):
        write_canonical_data(tmp_path)
