from pathlib import Path

from simfolio_forecasting_methodology.data import (
    CANONICAL_COMMON_RETURN_COUNT,
    CANONICAL_NORMALIZED_RETURN_SHA256,
    CANONICAL_TICKERS,
    canonical_snapshot_manifest,
)
from simfolio_forecasting_methodology.panel import (
    CANONICAL_PANEL_RESOURCE_SHA256,
    load_scored52_portfolio_panel,
)
from simfolio_forecasting_methodology.protocol import CANONICAL_SOURCE_REVISION


def test_source_comparison_fixture_covers_all_canonical_series():
    manifest = canonical_snapshot_manifest()
    series = {row["ticker"]: row for row in manifest["series"]}
    assert tuple(manifest["ticker_order"]) == tuple(sorted(CANONICAL_TICKERS))
    assert set(series) == set(CANONICAL_TICKERS)
    assert len(series) == 52
    for ticker in CANONICAL_TICKERS:
        row = series[ticker]
        assert row["path"] == f"app/simulated_data/series/{ticker}.csv.gz"
        assert row["schema"] == ["date", "daily_return", "price"]
        assert len(row["sha256"]) == 64
    assert manifest["common_date_count"] == CANONICAL_COMMON_RETURN_COUNT
    assert manifest["normalization"]["normalized_return_matrix_sha256"] == CANONICAL_NORMALIZED_RETURN_SHA256
    assert manifest["source_revision"] == CANONICAL_SOURCE_REVISION


def test_scored_panel_resource_hash_and_shape_are_frozen():
    assert len(load_scored52_portfolio_panel()) == 80
    resource = Path("src/simfolio_forecasting_methodology/resources/panels/scored52_portfolio_panel.csv")
    import hashlib

    assert hashlib.sha256(resource.read_bytes()).hexdigest() == CANONICAL_PANEL_RESOURCE_SHA256


def test_harness_identity_is_source_relative_and_pinned():
    path = Path("src/simfolio_forecasting_methodology/resources/protocols/source_harness_identity.json")
    identity = __import__("json").loads(path.read_text())
    assert identity["harness_path"] == "source-research/scripts/forecast_oos_research_gate.py"
    assert identity["harness_sha256"] == "e061aba8ed259339f75a98e9ea8e0a1a275c99980ed649b92efe652d7271f997"
    assert identity["status"] == "identity_only_historical_adapter_parity_unresolved"
