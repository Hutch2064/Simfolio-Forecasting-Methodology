"""Identity checks and preparation for the canonical 52-series snapshot.

The retained white-paper run used simulated return arrays from the pinned
source repository. Those arrays are not redistributed by this package. The
package carries only source-relative identities, schemas, hashes, and the
common calendar. A caller with the required rights must provide a frozen
source snapshot and explicitly authorize its use before preparation.

There is intentionally no network, proxy, refresh, or source-builder path in
this module. A missing or unauthorized snapshot fails closed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from collections.abc import Mapping
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_START = pd.Timestamp("1979-12-31")
CANONICAL_END = pd.Timestamp("2026-05-13")
CANONICAL_DATASET_ID = "simfolio-canonical-simulated-returns-2026-05-13"
CANONICAL_SOURCE_REVISION = "773bc1c325559e6bf57a567f1d8bf473a3427fbc"
CANONICAL_NORMALIZED_RETURN_SHA256 = (
    "52c5bdd96b39762183ef0c204fa8165c2dfd5a4864e7198662615daddb8d6a49"
)
CANONICAL_COMMON_RETURN_COUNT = 11_687
CANONICAL_CALENDAR_SHA256 = (
    "8204fbf08e1664a8f254d935052b34c2b4ec9f088ee91f414e7e72c95fb32e04"
)

# This is the lexicographic order used by the normalized matrix identity.
CANONICAL_TICKERS = (
    "EFASIM", "GLDSIM", "GSGSIM", "IEFSIM", "IEISIM", "IWCSIM", "IWMSIM",
    "REITSIM", "SDSSIM", "SHYSIM", "SLVSIM", "SPMOSIM", "SPXLSIM", "SPXUSIM",
    "SPYSIM", "SSOSIM", "TECLSIM", "TLTSIM", "TMFSIM", "TNASIM", "TYDSIM",
    "UBTSIM", "UGLSIM", "UPROSIM", "URTHSIM", "URTYSIM", "UWMSIM", "VBKSIM",
    "VBRSIM", "VBSIM", "VEASIM", "VOESIM", "VOOSIM", "VOSIM", "VOTSIM",
    "VTISIM", "VTSIM", "VTVSIM", "VUGSIM", "VVSIM", "VXUSSIM", "XLBSIM",
    "XLCSIM", "XLESIM", "XLFSIM", "XLISIM", "XLKSIM", "XLPSIM", "XLUSIM",
    "XLVSIM", "XLYSIM", "ZROZSIM",
)


class CanonicalDataUnavailable(RuntimeError):
    """Raised when canonical arrays are absent, unauthorized, or inconsistent."""


def _resource_path(*parts: str) -> resources.abc.Traversable:
    return resources.files("simfolio_forecasting_methodology").joinpath("resources", *parts)


def canonical_snapshot_manifest() -> dict[str, object]:
    """Return the published source identity manifest without loading arrays."""
    with _resource_path("data", "canonical_snapshot_manifest.json").open(
        "r", encoding="utf-8"
    ) as handle:
        return json.load(handle)


def canonical_calendar() -> pd.DatetimeIndex:
    """Load and verify the published common date calendar."""
    path = _resource_path("data", "canonical_calendar.csv")
    with path.open("rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    if digest != CANONICAL_CALENDAR_SHA256:
        raise CanonicalDataUnavailable("packaged canonical calendar hash drifted")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["date"]:
        raise CanonicalDataUnavailable("packaged canonical calendar schema drifted")
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
    if (
        len(dates) != CANONICAL_COMMON_RETURN_COUNT
        or dates.has_duplicates
        or not dates.is_monotonic_increasing
        or dates[0] != CANONICAL_START
        or dates[-1] != CANONICAL_END
    ):
        raise CanonicalDataUnavailable("packaged canonical calendar identity drifted")
    return dates


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_series_root(snapshot_root: Path) -> Path:
    root = Path(snapshot_root)
    candidates = (
        root,
        root / "series",
        root / "app" / "simulated_data" / "series",
    )
    for candidate in candidates:
        if all((candidate / f"{ticker}.csv.gz").is_file() for ticker in CANONICAL_TICKERS):
            return candidate
    raise CanonicalDataUnavailable(
        "canonical snapshot series were not found below the supplied snapshot root"
    )


def _snapshot_factor_path(snapshot_root: Path, relative_path: str) -> Path:
    root = Path(snapshot_root)
    relative = Path(relative_path)
    candidates = [root / relative]
    if relative.parts[:1] == ("app",):
        candidates.append(root / relative.parts[0] / Path(*relative.parts[1:]))
        if root.name == "simulated_data":
            candidates.append(root.parent / Path(*relative.parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise CanonicalDataUnavailable(f"missing required factor input {relative_path}")


def _read_snapshot_series(path: Path, expected: Mapping[str, object]) -> pd.DataFrame:
    ticker = str(expected["ticker"])
    if _sha256(path) != str(expected["sha256"]):
        raise CanonicalDataUnavailable(f"source hash mismatch for {ticker}")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        frame = pd.read_csv(handle, float_precision="round_trip")
    expected_schema = ["date", "daily_return", "price"]
    if list(frame.columns) != expected_schema:
        raise CanonicalDataUnavailable(f"schema mismatch for {ticker}")
    dates = pd.DatetimeIndex(pd.to_datetime(frame.pop("date"), errors="raise"))
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise CanonicalDataUnavailable(f"date ordering mismatch for {ticker}")
    if len(frame) != int(expected["row_count"]):
        raise CanonicalDataUnavailable(f"row count mismatch for {ticker}")
    if str(dates[0].date()) != str(expected["first_date"]):
        raise CanonicalDataUnavailable(f"first date mismatch for {ticker}")
    if str(dates[-1].date()) != str(expected["last_date"]):
        raise CanonicalDataUnavailable(f"last date mismatch for {ticker}")
    frame.index = dates
    frame = frame.apply(pd.to_numeric, errors="raise")
    if not np.all(np.isfinite(frame.to_numpy(dtype=np.float64))):
        raise CanonicalDataUnavailable(f"nonfinite values in {ticker}")
    return frame


def _normalized_return_hash(series: Mapping[str, pd.Series], dates: pd.DatetimeIndex) -> str:
    ordered = sorted(series)
    lines = ["date," + ",".join(ordered)]
    for date in dates:
        values = [format(float(series[ticker].loc[date]), ".17g") for ticker in ordered]
        lines.append(f"{pd.Timestamp(date).date().isoformat()}," + ",".join(values))
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _factor_identity(path: Path, expected: Mapping[str, object]) -> None:
    factor_id = str(expected["id"])
    if _sha256(path) != str(expected["sha256"]):
        raise CanonicalDataUnavailable(f"factor input hash mismatch for {factor_id}")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        frame = pd.read_csv(handle, float_precision="round_trip")
    if list(frame.columns) != list(expected["schema"]):
        raise CanonicalDataUnavailable(f"factor input schema mismatch for {factor_id}")


def verify_canonical_snapshot(
    snapshot_root: Path,
    *,
    rights_confirmed: bool = False,
) -> dict[str, object]:
    """Verify a caller-provided exact snapshot against the published identities."""
    if not rights_confirmed:
        raise CanonicalDataUnavailable(
            "canonical arrays are identified but use and redistribution rights are not confirmed; "
            "provide an authorized snapshot and pass rights_confirmed=True"
        )
    manifest = canonical_snapshot_manifest()
    root = _snapshot_series_root(Path(snapshot_root))
    expected_series = {str(item["ticker"]): item for item in manifest["series"]}
    frames: dict[str, pd.DataFrame] = {}
    for ticker in CANONICAL_TICKERS:
        expected = expected_series.get(ticker)
        if expected is None:
            raise CanonicalDataUnavailable(f"published identity is missing ticker {ticker}")
        frames[ticker] = _read_snapshot_series(root / f"{ticker}.csv.gz", expected)

    common: pd.DatetimeIndex | None = None
    for frame in frames.values():
        common = frame.index if common is None else common.intersection(frame.index)
    if common is None:
        raise CanonicalDataUnavailable("canonical snapshot has no common dates")
    common = common[(common >= CANONICAL_START) & (common <= CANONICAL_END)]
    calendar = canonical_calendar()
    if not common.equals(calendar):
        raise CanonicalDataUnavailable("canonical source dates do not match the frozen calendar")
    returns = {ticker: frames[ticker].loc[common, "daily_return"] for ticker in CANONICAL_TICKERS}
    normalized_hash = _normalized_return_hash(returns, common)
    if normalized_hash != CANONICAL_NORMALIZED_RETURN_SHA256:
        raise CanonicalDataUnavailable("normalized canonical return matrix hash mismatch")

    supporting_verified: list[str] = []
    supporting = {str(item["ticker"]): item for item in manifest.get("supporting_series", [])}
    for ticker, expected in supporting.items():
        path = root / f"{ticker}.csv.gz"
        if not path.is_file():
            raise CanonicalDataUnavailable(f"missing required supporting series {ticker}")
        _read_snapshot_series(path, expected)
        supporting_verified.append(ticker)

    factor_verified: list[str] = []
    for factor in manifest.get("factor_inputs", []):
        _factor_identity(_snapshot_factor_path(Path(snapshot_root), str(factor["path"])), factor)
        factor_verified.append(str(factor["id"]))

    return {
        "verified": True,
        "rights_status": "caller_authorized",
        "dataset_id": str(manifest["dataset_id"]),
        "source_revision": str(manifest["source_revision"]),
        "series_count": len(frames),
        "common_date_count": len(common),
        "common_start": str(common.min().date()),
        "common_end": str(common.max().date()),
        "normalized_return_matrix_sha256": normalized_hash,
        "supporting_series": supporting_verified,
        "factor_inputs": factor_verified,
    }


def _price_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Match the engine price-to-return adapter used by the retained panel."""
    frame = prices.astype(float).replace([np.inf, -np.inf], np.nan).ffill(limit=1)
    returns = (frame / frame.shift(1)) - 1.0
    if len(returns):
        returns.iloc[0] = 0.0
    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")


def _rebalance_dates(dates: pd.DatetimeIndex, frequency: str) -> set[pd.Timestamp]:
    normalized = str(frequency).lower()
    if normalized in {"none", ""} or len(dates) == 0:
        return set()
    aliases = {"monthly": "ME", "quarterly": "QE", "annually": "YE", "annual": "YE"}
    alias = aliases.get(normalized)
    if alias is None:
        raise CanonicalDataUnavailable(f"unsupported canonical rebalance frequency {frequency!r}")
    targets = pd.date_range(start=dates.min(), end=dates.max(), freq=alias)
    positions = np.unique(dates.searchsorted(targets, side="right") - 1)
    positions = positions[positions >= 0]
    return {pd.Timestamp(value) for value in dates.take(positions)}


def _source_portfolio_log_returns(prices: pd.DataFrame, spec: object) -> pd.Series:
    """Reproduce the no-flow engine portfolio construction from source prices."""
    tickers = tuple(str(value) for value in spec.tickers)
    weights = np.asarray(spec.weights, dtype=np.float64)
    selected = prices.reindex(columns=list(tickers)).dropna(how="any")
    if selected.empty:
        raise CanonicalDataUnavailable(f"no common price history for {spec.name}")
    returns = _price_simple_returns(selected)
    if len(returns) < 1:
        raise CanonicalDataUnavailable(f"no price returns for {spec.name}")
    holdings = weights * 10_000.0
    rebalance_dates = _rebalance_dates(pd.DatetimeIndex(returns.index), spec.rebalance)
    equity = np.empty(len(returns), dtype=np.float64)
    cost_rate = 15.0 / 10_000.0
    for row_index, (date, row) in enumerate(returns.iterrows()):
        holdings *= 1.0 + row.to_numpy(dtype=np.float64)
        current = float(np.sum(holdings))
        if row_index > 0 and pd.Timestamp(date) in rebalance_dates and current > 0.0:
            drifted = holdings / current
            gross = np.abs(weights - drifted)
            trade_cost = 0.5 * current * float(np.sum(gross * cost_rate))
            current = max(current - trade_cost, 0.0)
            holdings = current * weights
        equity[row_index] = current
    simple = pd.Series(equity, index=returns.index).pct_change().dropna()
    values = np.clip(simple.to_numpy(dtype=np.float64), -0.999999, None)
    return pd.Series(np.log1p(values), index=simple.index, dtype=float)


def _load_verified_source_frames(snapshot_root: Path) -> dict[str, pd.DataFrame]:
    manifest = canonical_snapshot_manifest()
    root = _snapshot_series_root(Path(snapshot_root))
    expected = {str(item["ticker"]): item for item in manifest["series"]}
    return {
        ticker: _read_snapshot_series(root / f"{ticker}.csv.gz", expected[ticker])
        for ticker in CANONICAL_TICKERS
    }


def prepare_canonical_data(
    snapshot_root: Path,
    destination: Path,
    *,
    rights_confirmed: bool = False,
) -> Path:
    """Write a deterministic local cache from a verified caller-owned snapshot.

    The source snapshot is retained only in the caller's cache.  Portfolio logs
    are constructed from the full source price histories before the canonical
    window is trimmed, matching the historical ``simulate_portfolio`` closure.
    """
    verification = verify_canonical_snapshot(snapshot_root, rights_confirmed=rights_confirmed)
    source_root = _snapshot_series_root(Path(snapshot_root))
    frames = _load_verified_source_frames(Path(snapshot_root))
    common = canonical_calendar()
    destination = Path(destination)
    source_series_root = destination / "source_series"
    series_root = destination / "canonical_series"
    portfolio_root = destination / "canonical_portfolios"
    source_series_root.mkdir(parents=True, exist_ok=True)
    series_root.mkdir(parents=True, exist_ok=True)
    portfolio_root.mkdir(parents=True, exist_ok=True)

    # Keep an authorized exact copy outside the package so source-price-based
    # portfolio construction can be audited without a live source dependency.
    source_records: list[dict[str, object]] = []
    for ticker in CANONICAL_TICKERS:
        source_path = source_root / f"{ticker}.csv.gz"
        destination_path = source_series_root / source_path.name
        shutil.copyfile(source_path, destination_path)
        source_records.append({
            "ticker": ticker,
            "path": f"source_series/{destination_path.name}",
            "sha256": _sha256(destination_path),
        })

    generated: list[dict[str, object]] = []
    for ticker in CANONICAL_TICKERS:
        values = frames[ticker].loc[common, "daily_return"]
        lines = ["date,daily_return"]
        lines.extend(
            f"{date.date().isoformat()},{format(float(value), '.17g')}"
            for date, value in values.items()
        )
        path = series_root / f"{ticker}.csv.gz"
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
                handle.write(("\n".join(lines) + "\n").encode("utf-8"))
        generated.append(
            {
                "ticker": ticker,
                "rows": len(values),
                "first_date": str(common.min().date()),
                "last_date": str(common.max().date()),
                "sha256": _sha256(path),
            }
        )

    from .panel import load_scored52_portfolio_panel

    portfolio_records: list[dict[str, object]] = []
    for spec in load_scored52_portfolio_panel():
        logs = _source_portfolio_log_returns(
            pd.concat({ticker: frame["price"] for ticker, frame in frames.items()}, axis=1),
            spec,
        )
        path = portfolio_root / f"{spec.name}.csv.gz"
        frame = pd.DataFrame({"date": logs.index.strftime("%Y-%m-%d"), "portfolio_log_return": logs.to_numpy()})
        csv_bytes = frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
                handle.write(csv_bytes)
        portfolio_records.append({
            "portfolio_id": spec.name,
            "path": f"canonical_portfolios/{path.name}",
            "rows": len(logs),
            "first_date": str(logs.index.min().date()),
            "last_date": str(logs.index.max().date()),
            "sha256": _sha256(path),
        })

    output = {
        "dataset_id": CANONICAL_DATASET_ID,
        "source_revision": CANONICAL_SOURCE_REVISION,
        "rights_status": verification["rights_status"],
        "canonical_window": {"start": str(CANONICAL_START.date()), "end": str(CANONICAL_END.date())},
        "common_date_count": len(common),
        "normalized_return_matrix_sha256": CANONICAL_NORMALIZED_RETURN_SHA256,
        "portfolio_construction": {
            "method": "source_price_simulate_portfolio_no_flows",
            "initial_capital": 10000.0,
            "turnover_cost_bps": 15.0,
            "constructed_before_common_window_trim": True,
        },
        "source_series": source_records,
        "series": generated,
        "portfolio_series": portfolio_records,
    }
    manifest_path = destination / "canonical_data_manifest.json"
    manifest_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def verify_prepared_canonical_data(cache: Path) -> dict[str, object]:
    """Verify a prepared cache without accessing source files or live services."""
    root = Path(cache)
    manifest_path = root / "canonical_data_manifest.json"
    if not manifest_path.is_file():
        raise CanonicalDataUnavailable("prepared canonical manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != CANONICAL_DATASET_ID:
        raise CanonicalDataUnavailable("prepared cache is not the canonical dataset identity")
    if manifest.get("rights_status") != "caller_authorized":
        raise CanonicalDataUnavailable("prepared cache does not record caller authorization")
    calendar = canonical_calendar()
    columns: dict[str, pd.Series] = {}
    for ticker in CANONICAL_TICKERS:
        path = root / "canonical_series" / f"{ticker}.csv.gz"
        if not path.is_file():
            raise CanonicalDataUnavailable(f"missing prepared canonical series {ticker}")
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            frame = pd.read_csv(handle, float_precision="round_trip")
        if list(frame.columns) != ["date", "daily_return"]:
            raise CanonicalDataUnavailable(f"prepared schema mismatch for {ticker}")
        dates = pd.DatetimeIndex(pd.to_datetime(frame.pop("date"), errors="raise"))
        values = pd.to_numeric(frame["daily_return"], errors="raise").to_numpy(dtype=np.float64)
        if not dates.equals(calendar) or not np.all(np.isfinite(values)):
            raise CanonicalDataUnavailable(f"prepared dates or values mismatch for {ticker}")
        columns[ticker] = pd.Series(values, index=dates)
    normalized_hash = _normalized_return_hash(columns, calendar)
    if normalized_hash != CANONICAL_NORMALIZED_RETURN_SHA256:
        raise CanonicalDataUnavailable("prepared canonical return hash mismatch")

    # Exact portfolio construction requires the authorized source-price copy;
    # a raw return-only cache cannot claim parity with the retained panel.
    source_records = {str(item["ticker"]): item for item in manifest.get("source_series", [])}
    expected_source = {str(item["ticker"]): item for item in canonical_snapshot_manifest()["series"]}
    if set(source_records) != set(CANONICAL_TICKERS):
        raise CanonicalDataUnavailable("prepared cache is missing exact source series")
    for ticker in CANONICAL_TICKERS:
        relative = Path(str(source_records[ticker]["path"]))
        path = root / relative
        if not path.is_file() or _sha256(path) != str(expected_source[ticker]["sha256"]):
            raise CanonicalDataUnavailable(f"prepared source series hash mismatch for {ticker}")

    from .panel import load_scored52_portfolio_panel

    expected_portfolios = {item.name for item in load_scored52_portfolio_panel()}
    portfolio_records = {
        str(item["portfolio_id"]): item for item in manifest.get("portfolio_series", [])
    }
    if set(portfolio_records) != expected_portfolios:
        raise CanonicalDataUnavailable("prepared cache is missing exact portfolio series")
    for portfolio_id, record in portfolio_records.items():
        path = root / Path(str(record["path"]))
        if not path.is_file() or _sha256(path) != str(record["sha256"]):
            raise CanonicalDataUnavailable(f"prepared portfolio series hash mismatch for {portfolio_id}")
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            frame = pd.read_csv(handle, float_precision="round_trip")
        if list(frame.columns) != ["date", "portfolio_log_return"]:
            raise CanonicalDataUnavailable(f"prepared portfolio schema mismatch for {portfolio_id}")
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
        values = pd.to_numeric(frame["portfolio_log_return"], errors="raise").to_numpy(dtype=np.float64)
        if dates.has_duplicates or not dates.is_monotonic_increasing or not np.all(np.isfinite(values)):
            raise CanonicalDataUnavailable(f"prepared portfolio values mismatch for {portfolio_id}")
        if len(dates) < CANONICAL_COMMON_RETURN_COUNT or not set(calendar).issubset(set(dates)):
            raise CanonicalDataUnavailable(f"prepared portfolio does not cover canonical calendar for {portfolio_id}")

    construction = manifest.get("portfolio_construction") or {}
    if construction.get("constructed_before_common_window_trim") is not True:
        raise CanonicalDataUnavailable("portfolio construction provenance is incomplete")
    if float(construction.get("turnover_cost_bps", -1.0)) != 15.0:
        raise CanonicalDataUnavailable("portfolio turnover cost provenance drifted")
    return {
        "verified": True,
        "rights_status": "caller_authorized",
        "dataset_id": CANONICAL_DATASET_ID,
        "common_date_count": CANONICAL_COMMON_RETURN_COUNT,
        "common_start": str(CANONICAL_START.date()),
        "common_end": str(CANONICAL_END.date()),
        "normalized_return_matrix_sha256": normalized_hash,
        "source_series_count": len(source_records),
        "portfolio_series_count": len(portfolio_records),
        "portfolio_construction": construction,
    }


def load_canonical_engine_inputs(
    cache: Path,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Load exact source-price-derived assets and portfolio logs from a cache."""
    verify_prepared_canonical_data(Path(cache))
    root = Path(cache)
    calendar = canonical_calendar()
    source_records = {
        str(item["ticker"]): item
        for item in json.loads((root / "canonical_data_manifest.json").read_text(encoding="utf-8"))["source_series"]
    }
    asset_logs: dict[str, pd.Series] = {}
    expected_source = {str(item["ticker"]): item for item in canonical_snapshot_manifest()["series"]}
    for ticker in CANONICAL_TICKERS:
        frame = _read_snapshot_series(
            root / Path(str(source_records[ticker]["path"])), expected_source[ticker]
        )
        simple = _price_simple_returns(frame[["price"]])["price"]
        values = np.log1p(np.clip(simple.to_numpy(dtype=np.float64), -0.999999, None))
        series = pd.Series(values, index=simple.index).reindex(calendar)
        if series.isna().any():
            raise CanonicalDataUnavailable(f"prepared source price history misses calendar for {ticker}")
        asset_logs[ticker] = series.astype(float)
    portfolio_logs: dict[str, pd.Series] = {}
    manifest = json.loads((root / "canonical_data_manifest.json").read_text(encoding="utf-8"))
    for record in manifest["portfolio_series"]:
        path = root / Path(str(record["path"]))
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            frame = pd.read_csv(handle, float_precision="round_trip")
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
        values = pd.to_numeric(frame["portfolio_log_return"], errors="raise").to_numpy(dtype=np.float64)
        series = pd.Series(values, index=dates).reindex(calendar)
        if series.isna().any():
            raise CanonicalDataUnavailable(f"prepared portfolio history misses calendar for {record['portfolio_id']}")
        portfolio_logs[str(record["portfolio_id"])] = series.astype(float)
    return pd.DataFrame(asset_logs, index=calendar), portfolio_logs


def load_canonical_returns(cache: Path) -> pd.DataFrame:
    """Load only a verified prepared canonical cache."""
    verify_prepared_canonical_data(Path(cache))
    root = Path(cache) / "canonical_series"
    columns: dict[str, pd.Series] = {}
    for ticker in CANONICAL_TICKERS:
        with gzip.open(root / f"{ticker}.csv.gz", "rt", encoding="utf-8", newline="") as handle:
            frame = pd.read_csv(handle, float_precision="round_trip")
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
        columns[ticker] = pd.Series(
            pd.to_numeric(frame["daily_return"], errors="raise").to_numpy(dtype=np.float64),
            index=dates,
        )
    return pd.concat(columns, axis=1).sort_index()


def write_canonical_data(cache: Path, *, refresh: bool = False) -> Path:
    """Compatibility entry point that deliberately refuses live/proxy builds."""
    del cache, refresh
    raise CanonicalDataUnavailable(
        "live and public-proxy canonical builders were removed; prepare an authorized "
        "frozen snapshot with prepare_canonical_data"
    )
