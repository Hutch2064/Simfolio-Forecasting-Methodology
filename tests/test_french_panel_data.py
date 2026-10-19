import zipfile

import numpy as np
import pytest

from simfolio_forecasting_methodology.french_panel import data


def zipped(tmp_path, text):
    path = tmp_path / "provider.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("returns.csv", text)
    return path


def test_provider_parser_keeps_both_weights_and_ignores_other_sections(tmp_path):
    path = zipped(tmp_path, "Provider metadata\nAverage Value Weighted Returns -- Daily\n,A,B\n19260701,1,2\n19260702,3,-99.99\n\nNumber of Firms\n,A,B\n19260701,999,999\n\nAverage Equal Weighted Returns -- Daily\n,A,B\n19260701,4,5\n19260702,6,7\n\nAverage Value Weighted Returns -- Monthly\n,A,B\n192607,99,99\n")
    blocks = data.parse_portfolio_zip(path)
    assert set(blocks) == {"value", "equal"}
    assert blocks["value"][0] == ["A", "B"]
    assert blocks["value"][1] == ["19260701", "19260702"]
    np.testing.assert_array_equal(blocks["equal"][2], [[4, 5], [6, 7]])
    assert not data._valid_percent_returns(blocks["value"][2][:, 1])


def test_market_reconstructs_simple_total_return_before_log_transform(tmp_path):
    path = zipped(tmp_path, "metadata\n,Mkt-RF,SMB,HML,RF\n19260630,99,0,0,0\n19260701,1.5,0,0,0.1\n19260702,-2,0,0,0.2\n\nAnnual\n1926,1,2,3,4\n")
    dates, market, cash = data.parse_market_zip(path)
    assert dates == ["19260701", "19260702"]
    np.testing.assert_allclose(market, [1.6, -1.8])
    np.testing.assert_array_equal(cash, [.1, .2])
    assert np.log1p(market[0] / 100) != np.log1p(.015) + np.log1p(.001)


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [-100], [-999], [-99.99]])
def test_no_filling_or_invalid_simple_returns(values):
    assert not data._valid_percent_returns(np.array(values))


def test_inventory_is_exact_unique_and_matches_source_audit():
    rows, audit = data.inventory(), data._audit()
    assert len(rows) == 550
    assert audit["daily_observations"] == 26317
    assert audit["unique_eligible_columns_both_weights"] == len(rows)
    assert sum(row["weighting"] == "value" for row in rows) == 275
    assert sum(row["weighting"] == "equal" for row in rows) == 275
    eligible = {(f["url"], s["weighting"], s["column"]): s
                for f in audit["files"] for s in f["series"]
                if s["eligible_without_filling"]}
    for row in rows:
        record = eligible[row["source_url"], row["weighting"], row["published_column"]]
        assert record["returns_sha256"] == row["return_series_sha256"]
        assert record["missing_observations"] == 0


def test_existing_source_archives_must_match_frozen_sha(tmp_path):
    path = tmp_path / "bad.zip"
    path.write_bytes(b"wrong vintage")
    with pytest.raises(ValueError, match="checksum"):
        data._download("https://example.invalid/no-network", path, "0" * 64)


def test_fresh_panel_uses_actual_historical_saturdays_and_no_old_cap():
    # The scheduler accepts source observations including Saturdays.
    dates = np.arange(np.datetime64("1926-07-01"), np.datetime64("1998-07-20"))
    schedule = data.origin_schedule(dates)
    assert len(schedule) == 51
    assert max(d.max_horizon for d in schedule) > 8766
    assert sum(d.evaluation_split == "rolling_origin" for d in schedule) == 48
