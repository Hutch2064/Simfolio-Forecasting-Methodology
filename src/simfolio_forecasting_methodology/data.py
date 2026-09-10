"""Identity checks and offline preparation for the canonical 52-series snapshot.

The retained white-paper run used simulated return arrays from the pinned
source repository. This package carries the exact, user-authorized snapshot
files identified by the manifest, together with their source attribution and
hashes, plus the verified source-derived canonical matrices. Verification and
preparation are local operations; an explicitly supplied caller snapshot
remains supported for independent re-verification.

There is intentionally no network, proxy, refresh, or source-builder path in
this module. A missing or inconsistent snapshot fails closed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from collections.abc import Mapping
from contextlib import contextmanager
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


@contextmanager
def _snapshot_root_context(snapshot_root: Path | None):
    """Yield a filesystem path for an external or packaged snapshot.

    ``importlib.resources.as_file`` also supports a wheel imported from a
    zipped loader, so the verifier never assumes that package resources are
    ordinary checkout paths.
    """
    if snapshot_root is not None:
        yield Path(snapshot_root)
        return
    with resources.as_file(_resource_path("data", "canonical_snapshot")) as packaged:
        yield Path(packaged)


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
        root / "source_series",
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
        # The pinned source registry uses pandas' default CSV float parser.
        # Keep that parser for source-price parity; the return-matrix identity
        # is independently stabilized by the .17g normalized hash.
        frame = pd.read_csv(handle)
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


def _derived_fingerprint_manifest() -> dict[str, object]:
    with _resource_path("data", "canonical_derived_fingerprints.json").open(
        "r", encoding="utf-8"
    ) as handle:
        return json.load(handle)


def _frozen_derived_snapshot_manifest() -> Mapping[str, object]:
    manifest = _derived_fingerprint_manifest().get("frozen_snapshot")
    if not isinstance(manifest, Mapping):
        raise CanonicalDataUnavailable("frozen canonical derived snapshot metadata is missing")
    return manifest


@contextmanager
def _frozen_derived_snapshot_context(snapshot_path: Path | None = None):
    """Yield the packaged or explicitly supplied frozen derived snapshot."""
    if snapshot_path is not None:
        yield Path(snapshot_path)
        return
    with resources.as_file(
        _resource_path("data", "canonical_derived_matrices.npz")
    ) as packaged:
        yield Path(packaged)


def _matrix_fingerprint(
    series: Mapping[str, pd.Series],
    dates: pd.DatetimeIndex,
) -> str:
    """Hash a deterministic date/column matrix using the published formatter."""
    return _normalized_return_hash(series, dates)


def _array_digest(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _array_matrix_fingerprint(
    values: np.ndarray,
    labels: tuple[str, ...],
    dates: np.ndarray,
) -> str:
    """Apply the published ``.17g`` matrix identity to a frozen ndarray."""
    order = sorted(range(len(labels)), key=lambda index: labels[index])
    ordered = [labels[index] for index in order]
    lines = ["date," + ",".join(ordered)]
    for date, row in zip(dates, values):
        rendered = [format(float(row[index]), ".17g") for index in order]
        lines.append(
            f"{pd.Timestamp(date).date().isoformat()}," + ",".join(rendered)
        )
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _validate_frozen_derived_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_snapshot: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray]:
    """Validate the complete frozen derived matrix schema and identities."""
    if expected_snapshot is None:
        expected_snapshot = _frozen_derived_snapshot_manifest()
    if expected_snapshot.get("calendar_sha256") != CANONICAL_CALENDAR_SHA256:
        raise CanonicalDataUnavailable("frozen derived calendar identity is not canonical")
    expected_schema = expected_snapshot.get("array_schema")
    if not isinstance(expected_schema, Mapping):
        raise CanonicalDataUnavailable("frozen derived array schema is missing")
    expected_keys = {
        "dates", "tickers", "portfolio_ids", "asset_log_returns", "portfolio_log_returns"
    }
    if set(arrays) != expected_keys:
        raise CanonicalDataUnavailable("frozen derived snapshot array keys drifted")

    for key in expected_keys:
        array = arrays[key]
        expected = expected_schema.get(key)
        if not isinstance(expected, Mapping):
            raise CanonicalDataUnavailable(f"frozen derived schema is missing {key}")
        if array.dtype.str != str(expected.get("dtype")):
            raise CanonicalDataUnavailable(f"frozen derived dtype mismatch for {key}")
        if tuple(array.shape) != tuple(int(value) for value in expected.get("shape", ())):
            raise CanonicalDataUnavailable(f"frozen derived shape mismatch for {key}")
        if not array.flags.c_contiguous:
            raise CanonicalDataUnavailable(f"frozen derived layout mismatch for {key}")
        if _array_digest(array) != str(expected.get("sha256")):
            raise CanonicalDataUnavailable(f"frozen derived array identity mismatch for {key}")

    dates = arrays["dates"]
    calendar = canonical_calendar().to_numpy(dtype="<M8[D]")
    if not np.array_equal(dates, calendar):
        raise CanonicalDataUnavailable("frozen derived date order does not match canonical calendar")

    tickers = tuple(str(value) for value in arrays["tickers"].tolist())
    if tickers != CANONICAL_TICKERS:
        raise CanonicalDataUnavailable("frozen derived asset column order drifted")

    from .panel import load_scored52_portfolio_panel

    portfolio_ids = tuple(spec.name for spec in load_scored52_portfolio_panel())
    stored_portfolio_ids = tuple(str(value) for value in arrays["portfolio_ids"].tolist())
    if stored_portfolio_ids != portfolio_ids:
        raise CanonicalDataUnavailable("frozen derived portfolio column order drifted")

    asset_values = arrays["asset_log_returns"]
    portfolio_values = arrays["portfolio_log_returns"]
    if not np.all(np.isfinite(asset_values)) or not np.all(np.isfinite(portfolio_values)):
        raise CanonicalDataUnavailable("frozen derived matrices contain nonfinite values")
    if _array_matrix_fingerprint(asset_values, tickers, dates) != str(
        expected_schema["asset_log_returns"]["matrix_fingerprint"]
    ):
        raise CanonicalDataUnavailable("frozen derived asset matrix fingerprint mismatch")
    if _array_matrix_fingerprint(portfolio_values, portfolio_ids, dates) != str(
        expected_schema["portfolio_log_returns"]["matrix_fingerprint"]
    ):
        raise CanonicalDataUnavailable("frozen derived portfolio matrix fingerprint mismatch")
    return dates, tickers, portfolio_ids, asset_values, portfolio_values


def _load_frozen_derived_snapshot(
    snapshot_path: Path | None = None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray]:
    """Load and validate the immutable source-derived matrices."""
    expected_snapshot = _frozen_derived_snapshot_manifest()
    path = Path(snapshot_path) if snapshot_path is not None else None
    with _frozen_derived_snapshot_context(path) as resolved:
        if not resolved.is_file():
            raise CanonicalDataUnavailable("frozen canonical derived snapshot is missing")
        if _sha256(resolved) != str(expected_snapshot.get("sha256")):
            raise CanonicalDataUnavailable("frozen canonical derived snapshot hash mismatch")
        try:
            with np.load(resolved, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
        except (OSError, ValueError, EOFError) as exc:
            raise CanonicalDataUnavailable(
                "frozen canonical derived snapshot cannot be read"
            ) from exc
    return _validate_frozen_derived_arrays(arrays)


def _factor_identity(path: Path, expected: Mapping[str, object]) -> None:
    factor_id = str(expected["id"])
    if _sha256(path) != str(expected["sha256"]):
        raise CanonicalDataUnavailable(f"factor input hash mismatch for {factor_id}")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        frame = pd.read_csv(handle)
    if list(frame.columns) != list(expected["schema"]):
        raise CanonicalDataUnavailable(f"factor input schema mismatch for {factor_id}")


def _verify_canonical_snapshot_path(
    snapshot_root: Path,
    *,
    rights_confirmed: bool = False,
) -> dict[str, object]:
    """Verify one exact snapshot path against the published identities."""
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
        "rights_status": (
            "caller_authorized_local_use" if rights_confirmed else "local_use_unconfirmed"
        ),
        "redistribution_status": (
            "user_authorized_exact_snapshot" if rights_confirmed
            else "not_granted_by_this_verifier"
        ),
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


def verify_canonical_snapshot(
    snapshot_root: Path | None = None,
    *,
    rights_confirmed: bool | None = None,
) -> dict[str, object]:
    """Verify the packaged snapshot or an explicitly supplied snapshot.

    Omitting ``snapshot_root`` selects the immutable package resource and
    records the user-authorized distribution status. An external snapshot is
    unconfirmed by default; callers may set ``rights_confirmed=True`` when
    they have separately established the required rights.
    """
    confirmed = snapshot_root is None if rights_confirmed is None else bool(rights_confirmed)
    with _snapshot_root_context(snapshot_root) as root:
        return _verify_canonical_snapshot_path(root, rights_confirmed=confirmed)


def _price_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Match the engine price-to-return adapter used by the retained panel."""
    frame = prices.astype(float).replace([np.inf, -np.inf], np.nan).ffill(limit=1)
    returns = (frame / frame.shift(1)) - 1.0
    if len(returns):
        returns.iloc[0] = 0.0
    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")


def _source_loader_price_frame(prices: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    """Match ``load_price_data_ohlc`` normalization before source simulation.

    The pinned loader normalizes simulated price frames to the union calendar,
    forward-fills at most one missing observation, and then drops rows that
    remain incomplete.  This matters at the pre-1979 boundary and when one
    simulated series ends one observation after its portfolio peers.
    """
    frame = prices.reindex(columns=list(tickers)).copy()
    if frame.empty:
        return frame
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None)
    frame = frame.sort_index()
    first_valid = [frame[ticker].first_valid_index() for ticker in tickers]
    first_valid = [value for value in first_valid if value is not None and not pd.isna(value)]
    if first_valid:
        frame = frame.loc[frame.index >= max(first_valid)]
    trading_days = frame[list(tickers)].dropna(how="all").index
    frame = frame.loc[trading_days]
    return frame.ffill(limit=1).dropna(how="any")


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
    # The source generator passes equal-class weights as ``1 / len(tickers)``
    # and then normalizes them in ``_construct_portfolio_returns``.  The
    # reviewed panel stores the post-normalization IEEE-754 value, so deriving
    # the raw equal-holding value here preserves source call semantics instead
    # of normalizing the serialized value a second time.
    weights = np.full(len(tickers), 1.0 / max(len(tickers), 1), dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise CanonicalDataUnavailable(f"invalid equal-class weights for {spec.name}")
    weights = weights / total_weight
    selected = _source_loader_price_frame(prices, tickers)
    if selected.empty:
        raise CanonicalDataUnavailable(f"no common price history for {spec.name}")
    returns = _price_simple_returns(selected)
    if len(returns) < 1:
        raise CanonicalDataUnavailable(f"no price returns for {spec.name}")
    holdings = {
        ticker: 10_000.0 * float(weight)
        for ticker, weight in zip(tickers, weights)
    }
    rebalance_dates = _rebalance_dates(pd.DatetimeIndex(returns.index), spec.rebalance)
    equity = np.empty(len(returns), dtype=np.float64)
    cost_rate = 15.0 / 10_000.0
    for row_index, (date, row) in enumerate(returns.iterrows()):
        current = 0.0
        daily_returns = row.to_numpy(dtype=np.float64)
        for asset_index, ticker in enumerate(tickers):
            holdings[ticker] *= 1.0 + daily_returns[asset_index]
            current += holdings[ticker]
        if row_index > 0 and pd.Timestamp(date) in rebalance_dates and current > 0.0:
            drifted = {
                ticker: holdings[ticker] / current
                for ticker in tickers
            }
            gross = {
                ticker: abs(float(weight) - drifted[ticker])
                for ticker, weight in zip(tickers, weights)
            }
            trade_cost = 0.5 * current * sum(
                gross[ticker] * cost_rate
                for ticker in tickers
            )
            current = max(current - trade_cost, 0.0)
            for ticker, weight in zip(tickers, weights):
                holdings[ticker] = current * float(weight)
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


def _prepare_canonical_data_path(
    snapshot_root: Path,
    destination: Path,
    *,
    rights_confirmed: bool = False,
) -> Path:
    """Write a deterministic local cache from one verified snapshot path.

    The source snapshot is retained in the caller's cache, while the exact
    source-derived canonical-window matrices are copied from the packaged
    frozen snapshot. This avoids platform-dependent re-computation.
    """
    verification = _verify_canonical_snapshot_path(
        snapshot_root, rights_confirmed=rights_confirmed
    )
    source_root = _snapshot_series_root(Path(snapshot_root))
    frames = _load_verified_source_frames(Path(snapshot_root))
    common = canonical_calendar()
    destination = Path(destination)
    source_series_root = destination / "source_series"
    supporting_root = destination / "supporting_series"
    factor_root = destination / "factor_inputs"
    series_root = destination / "canonical_series"
    portfolio_root = destination / "canonical_portfolios"
    source_series_root.mkdir(parents=True, exist_ok=True)
    supporting_root.mkdir(parents=True, exist_ok=True)
    factor_root.mkdir(parents=True, exist_ok=True)
    series_root.mkdir(parents=True, exist_ok=True)
    portfolio_root.mkdir(parents=True, exist_ok=True)

    expected_snapshot = _frozen_derived_snapshot_manifest()
    with _frozen_derived_snapshot_context() as packaged_snapshot:
        frozen_dates, frozen_tickers, frozen_portfolio_ids, frozen_assets, frozen_portfolios = (
            _load_frozen_derived_snapshot(packaged_snapshot)
        )
        derived_snapshot_path = destination / "canonical_derived_matrices.npz"
        shutil.copyfile(packaged_snapshot, derived_snapshot_path)
    if _sha256(derived_snapshot_path) != str(expected_snapshot["sha256"]):
        raise CanonicalDataUnavailable("prepared frozen derived snapshot copy drifted")

    # Keep an authorized exact source copy beside the frozen derived matrices
    # so source-price construction can still be audited without a live source.
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

    # Preserve the authorized local context inputs beside the exact source
    # series. These files are verified by hash and remain separate from the
    # canonical derived matrix consumed by the engine.
    manifest = canonical_snapshot_manifest()
    supporting_records: list[dict[str, object]] = []
    for expected in manifest.get("supporting_series", []):
        ticker = str(expected["ticker"])
        source_path = source_root / f"{ticker}.csv.gz"
        if not source_path.is_file():
            raise CanonicalDataUnavailable(f"missing required supporting series {ticker}")
        destination_path = supporting_root / source_path.name
        shutil.copyfile(source_path, destination_path)
        supporting_records.append({
            "ticker": ticker,
            "path": f"supporting_series/{destination_path.name}",
            "sha256": _sha256(destination_path),
        })

    factor_records: list[dict[str, object]] = []
    for expected in manifest.get("factor_inputs", []):
        source_path = _snapshot_factor_path(Path(snapshot_root), str(expected["path"]))
        destination_path = factor_root / Path(str(expected["path"])).name
        shutil.copyfile(source_path, destination_path)
        factor_records.append({
            "id": str(expected["id"]),
            "path": f"factor_inputs/{destination_path.name}",
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
        with path.open("wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0
        ) as handle:
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

    portfolio_records: list[dict[str, object]] = []
    for portfolio_index, portfolio_id in enumerate(frozen_portfolio_ids):
        logs = pd.Series(
            frozen_portfolios[:, portfolio_index].copy(),
            index=pd.DatetimeIndex(frozen_dates),
            dtype=float,
        )
        path = portfolio_root / f"{portfolio_id}.csv.gz"
        frame = pd.DataFrame(
            {
                "date": logs.index.strftime("%Y-%m-%d"),
                "portfolio_log_return": logs.to_numpy(),
            }
        )
        csv_bytes = frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")
        with path.open("wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0
        ) as handle:
            handle.write(csv_bytes)
        portfolio_records.append({
            "portfolio_id": portfolio_id,
            "path": f"canonical_portfolios/{path.name}",
            "rows": len(logs),
            "first_date": str(logs.index.min().date()),
            "last_date": str(logs.index.max().date()),
            "sha256": _sha256(path),
        })

    derived_fingerprints = _derived_fingerprint_manifest()
    if not np.array_equal(frozen_dates, common.to_numpy(dtype="datetime64[D]")):
        raise CanonicalDataUnavailable("frozen derived dates do not match canonical calendar")
    if frozen_tickers != CANONICAL_TICKERS:
        raise CanonicalDataUnavailable("frozen derived asset order does not match canonical order")
    asset_fingerprint = _array_matrix_fingerprint(
        frozen_assets, frozen_tickers, frozen_dates
    )
    portfolio_fingerprint = _array_matrix_fingerprint(
        frozen_portfolios, frozen_portfolio_ids, frozen_dates
    )
    if asset_fingerprint != str(derived_fingerprints["asset_log_matrix_sha256"]):
        raise CanonicalDataUnavailable("frozen canonical asset fingerprint does not match source evidence")
    if portfolio_fingerprint != str(derived_fingerprints["portfolio_log_matrix_sha256"]):
        raise CanonicalDataUnavailable("frozen canonical portfolio fingerprint does not match source evidence")

    output = {
        "dataset_id": CANONICAL_DATASET_ID,
        "source_revision": CANONICAL_SOURCE_REVISION,
        "rights_status": verification["rights_status"],
        "redistribution_status": verification["redistribution_status"],
        "canonical_window": {"start": str(CANONICAL_START.date()), "end": str(CANONICAL_END.date())},
        "common_date_count": len(common),
        "normalized_return_matrix_sha256": CANONICAL_NORMALIZED_RETURN_SHA256,
        "derived_fingerprints": {
            "asset_log_matrix_sha256": asset_fingerprint,
            "portfolio_log_matrix_sha256": portfolio_fingerprint,
            "source_loader_price_frame": dict(derived_fingerprints["source_loader_price_frame"]),
        },
        "derived_snapshot": {
            "schema_version": str(expected_snapshot["schema_version"]),
            "path": derived_snapshot_path.name,
            "sha256": str(expected_snapshot["sha256"]),
            "source_cache_manifest_sha256": str(expected_snapshot["source_cache_manifest_sha256"]),
            "source_snapshot_manifest_sha256": str(expected_snapshot["source_snapshot_manifest_sha256"]),
            "source_manifest_sha256": str(expected_snapshot["source_manifest_sha256"]),
            "source_revision": str(expected_snapshot["source_revision"]),
            "calendar_sha256": str(expected_snapshot["calendar_sha256"]),
            "generation": dict(expected_snapshot["generation"]),
            "array_schema": dict(expected_snapshot["array_schema"]),
        },
        "portfolio_construction": {
            "method": "source_price_simulate_portfolio_no_flows",
            "initial_capital": 10000.0,
            "turnover_cost_bps": 15.0,
            "constructed_before_common_window_trim": True,
            "frozen_common_window_matrix": True,
        },
        "source_series": source_records,
        "supporting_series": supporting_records,
        "factor_inputs": factor_records,
        "series": generated,
        "portfolio_series": portfolio_records,
    }
    manifest_path = destination / "canonical_data_manifest.json"
    manifest_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def prepare_canonical_data(
    snapshot_root: Path | None = None,
    destination: Path | None = None,
    *,
    rights_confirmed: bool | None = None,
) -> Path:
    """Prepare a deterministic local cache from the packaged snapshot by default.

    ``snapshot_root`` may point to an independently held source export. When
    omitted, the exact snapshot bundled under package resources is verified
    and used without network access. ``destination`` is always explicit so a
    caller controls where the generated cache is written.
    """
    if destination is None:
        raise TypeError("prepare_canonical_data requires a destination path")
    confirmed = snapshot_root is None if rights_confirmed is None else bool(rights_confirmed)
    with _snapshot_root_context(snapshot_root) as root:
        return _prepare_canonical_data_path(
            root, Path(destination), rights_confirmed=confirmed
        )


def verify_prepared_canonical_data(cache: Path) -> dict[str, object]:
    """Verify a prepared cache, including source-derived portfolio values."""
    root = Path(cache)
    manifest_path = root / "canonical_data_manifest.json"
    if not manifest_path.is_file():
        raise CanonicalDataUnavailable("prepared canonical manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != CANONICAL_DATASET_ID:
        raise CanonicalDataUnavailable("prepared cache is not the canonical dataset identity")
    if manifest.get("source_revision") != CANONICAL_SOURCE_REVISION:
        raise CanonicalDataUnavailable("prepared cache source revision drifted")
    if int(manifest.get("common_date_count", -1)) != CANONICAL_COMMON_RETURN_COUNT:
        raise CanonicalDataUnavailable("prepared cache common date count drifted")
    if manifest.get("normalized_return_matrix_sha256") != CANONICAL_NORMALIZED_RETURN_SHA256:
        raise CanonicalDataUnavailable("prepared cache declared return identity drifted")
    if manifest.get("rights_status") not in {
        "caller_authorized_local_use",
        "local_use_unconfirmed",
        # Accept the legacy spelling for caches prepared by f126ac7; source
        # hashes and derived values below still provide the integrity gate.
        "caller_authorized",
    }:
        raise CanonicalDataUnavailable("prepared cache has an unknown local-use status")

    calendar = canonical_calendar()
    expected_snapshot = _frozen_derived_snapshot_manifest()
    declared_snapshot = manifest.get("derived_snapshot")
    if not isinstance(declared_snapshot, Mapping):
        raise CanonicalDataUnavailable(
            "prepared cache frozen derived snapshot metadata is missing"
        )
    if declared_snapshot.get("path") != "canonical_derived_matrices.npz":
        raise CanonicalDataUnavailable("prepared cache frozen derived snapshot path drifted")
    for field in (
        "schema_version",
        "sha256",
        "source_cache_manifest_sha256",
        "source_snapshot_manifest_sha256",
        "source_manifest_sha256",
        "source_revision",
        "calendar_sha256",
        "generation",
        "array_schema",
    ):
        if declared_snapshot.get(field) != expected_snapshot.get(field):
            raise CanonicalDataUnavailable(
                f"prepared cache frozen derived snapshot metadata drifted: {field}"
            )
    frozen_snapshot_path = root / "canonical_derived_matrices.npz"
    frozen_dates, frozen_tickers, frozen_portfolio_ids, frozen_assets, frozen_portfolios = (
        _load_frozen_derived_snapshot(frozen_snapshot_path)
    )
    if not np.array_equal(frozen_dates, calendar.to_numpy(dtype="datetime64[D]")):
        raise CanonicalDataUnavailable(
            "prepared frozen derived dates do not match canonical calendar"
        )
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

    snapshot_manifest = canonical_snapshot_manifest()
    expected_source = {str(item["ticker"]): item for item in snapshot_manifest["series"]}
    source_records = {
        str(item["ticker"]): item for item in manifest.get("source_series", [])
    }
    if set(source_records) != set(CANONICAL_TICKERS):
        raise CanonicalDataUnavailable("prepared cache is missing exact source series")
    source_frames: dict[str, pd.DataFrame] = {}
    for ticker in CANONICAL_TICKERS:
        relative = Path(str(source_records[ticker]["path"]))
        path = root / relative
        expected = expected_source[ticker]
        if not path.is_file() or _sha256(path) != str(expected["sha256"]):
            raise CanonicalDataUnavailable(f"prepared source series hash mismatch for {ticker}")
        if _sha256(path) != str(source_records[ticker].get("sha256", "")):
            raise CanonicalDataUnavailable(f"prepared source record hash mismatch for {ticker}")
        source_frames[ticker] = _read_snapshot_series(path, expected)

    expected_supporting = {
        str(item["ticker"]): item
        for item in snapshot_manifest.get("supporting_series", [])
    }
    supporting_records = {
        str(item["ticker"]): item for item in manifest.get("supporting_series", [])
    }
    if set(supporting_records) != set(expected_supporting):
        raise CanonicalDataUnavailable("prepared cache is missing exact supporting series")
    for ticker, expected in expected_supporting.items():
        path = root / Path(str(supporting_records[ticker]["path"]))
        if not path.is_file() or _sha256(path) != str(expected["sha256"]):
            raise CanonicalDataUnavailable(f"prepared supporting series hash mismatch for {ticker}")
        if _sha256(path) != str(supporting_records[ticker].get("sha256", "")):
            raise CanonicalDataUnavailable(f"prepared supporting record hash mismatch for {ticker}")
        _read_snapshot_series(path, expected)

    expected_factors = {
        str(item["id"]): item for item in snapshot_manifest.get("factor_inputs", [])
    }
    factor_records = {
        str(item["id"]): item for item in manifest.get("factor_inputs", [])
    }
    if set(factor_records) != set(expected_factors):
        raise CanonicalDataUnavailable("prepared cache is missing exact factor inputs")
    for factor_id, expected in expected_factors.items():
        path = root / Path(str(factor_records[factor_id]["path"]))
        if not path.is_file() or _sha256(path) != str(expected["sha256"]):
            raise CanonicalDataUnavailable(f"prepared factor input hash mismatch for {factor_id}")
        if _sha256(path) != str(factor_records[factor_id].get("sha256", "")):
            raise CanonicalDataUnavailable(f"prepared factor record hash mismatch for {factor_id}")
        _factor_identity(path, expected)

    fingerprints = _derived_fingerprint_manifest()
    asset_fingerprint = _array_matrix_fingerprint(
        frozen_assets, frozen_tickers, frozen_dates
    )
    if asset_fingerprint != str(fingerprints["asset_log_matrix_sha256"]):
        raise CanonicalDataUnavailable("prepared frozen asset fingerprint mismatch")

    from .panel import load_scored52_portfolio_panel

    panel = load_scored52_portfolio_panel()
    expected_portfolios = {item.name: item for item in panel}
    portfolio_records = {
        str(item["portfolio_id"]): item
        for item in manifest.get("portfolio_series", [])
    }
    if set(portfolio_records) != set(expected_portfolios):
        raise CanonicalDataUnavailable("prepared cache is missing exact portfolio series")
    if tuple(expected_portfolios) != frozen_portfolio_ids:
        raise CanonicalDataUnavailable("prepared portfolio panel order differs from frozen matrix")
    for portfolio_index, portfolio_id in enumerate(frozen_portfolio_ids):
        record = portfolio_records[portfolio_id]
        path = root / Path(str(record["path"]))
        if not path.is_file() or _sha256(path) != str(record.get("sha256", "")):
            raise CanonicalDataUnavailable(f"prepared portfolio bytes mismatch for {portfolio_id}")
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            frame = pd.read_csv(handle, float_precision="round_trip")
        if list(frame.columns) != ["date", "portfolio_log_return"]:
            raise CanonicalDataUnavailable(f"prepared portfolio schema mismatch for {portfolio_id}")
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
        values = pd.to_numeric(frame["portfolio_log_return"], errors="raise").to_numpy(dtype=np.float64)
        if (
            dates.has_duplicates
            or not dates.is_monotonic_increasing
            or not np.all(np.isfinite(values))
            or len(values) != len(calendar)
            or not dates.equals(calendar)
        ):
            raise CanonicalDataUnavailable(f"prepared portfolio values mismatch for {portfolio_id}")
        if not np.array_equal(values, frozen_portfolios[:, portfolio_index]):
            raise CanonicalDataUnavailable(
                f"prepared portfolio values do not match frozen matrix for {portfolio_id}"
            )

    portfolio_fingerprint = _array_matrix_fingerprint(
        frozen_portfolios, frozen_portfolio_ids, frozen_dates
    )
    if portfolio_fingerprint != str(fingerprints["portfolio_log_matrix_sha256"]):
        raise CanonicalDataUnavailable("prepared frozen portfolio fingerprint mismatch")
    declared_derived = manifest.get("derived_fingerprints") or {}
    if declared_derived.get("asset_log_matrix_sha256") != asset_fingerprint:
        raise CanonicalDataUnavailable("prepared manifest asset fingerprint mismatch")
    if declared_derived.get("portfolio_log_matrix_sha256") != portfolio_fingerprint:
        raise CanonicalDataUnavailable("prepared manifest portfolio fingerprint mismatch")
    if declared_derived.get("source_loader_price_frame") != fingerprints.get(
        "source_loader_price_frame"
    ):
        raise CanonicalDataUnavailable("prepared source-loader provenance drifted")

    construction = manifest.get("portfolio_construction") or {}
    if construction.get("constructed_before_common_window_trim") is not True:
        raise CanonicalDataUnavailable("portfolio construction provenance is incomplete")
    if float(construction.get("turnover_cost_bps", -1.0)) != 15.0:
        raise CanonicalDataUnavailable("portfolio turnover cost provenance drifted")
    if construction.get("frozen_common_window_matrix") is not True:
        raise CanonicalDataUnavailable("prepared frozen portfolio provenance is incomplete")
    return {
        "verified": True,
        "rights_status": str(manifest.get("rights_status")),
        "redistribution_status": str(manifest.get("redistribution_status", "unknown")),
        "dataset_id": CANONICAL_DATASET_ID,
        "common_date_count": CANONICAL_COMMON_RETURN_COUNT,
        "common_start": str(CANONICAL_START.date()),
        "common_end": str(CANONICAL_END.date()),
        "normalized_return_matrix_sha256": normalized_hash,
        "asset_log_matrix_sha256": asset_fingerprint,
        "portfolio_log_matrix_sha256": portfolio_fingerprint,
        "source_series_count": len(source_records),
        "supporting_series_count": len(supporting_records),
        "factor_input_count": len(factor_records),
        "portfolio_series_count": len(portfolio_records),
        "portfolio_construction": construction,
    }

def load_canonical_engine_inputs(
    cache: Path,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Load the verified frozen source-derived matrices from a cache."""
    verify_prepared_canonical_data(Path(cache))
    root = Path(cache)
    calendar = canonical_calendar()
    frozen_dates, frozen_tickers, frozen_portfolio_ids, frozen_assets, frozen_portfolios = (
        _load_frozen_derived_snapshot(root / "canonical_derived_matrices.npz")
    )
    if not np.array_equal(frozen_dates, calendar.to_numpy(dtype="datetime64[D]")):
        raise CanonicalDataUnavailable("prepared frozen derived dates do not match canonical calendar")
    # Preserve the source loader's column-oriented frame layout as well as
    # its values; BLAS reductions can depend on array strides.
    asset_frame = pd.DataFrame(
        {ticker: frozen_assets[:, index].copy()
         for index, ticker in enumerate(frozen_tickers)},
        index=calendar,
    )
    portfolio_logs = {
        portfolio_id: pd.Series(
            frozen_portfolios[:, index].copy(), index=calendar, dtype=float
        )
        for index, portfolio_id in enumerate(frozen_portfolio_ids)
    }
    return asset_frame, portfolio_logs


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
    """Prepare the packaged frozen snapshot into ``cache`` offline.

    ``refresh`` is retained for compatibility and cannot select a network or
    proxy source; every invocation verifies the packaged bytes first.
    """
    del refresh
    return prepare_canonical_data(destination=Path(cache), rights_confirmed=True)
