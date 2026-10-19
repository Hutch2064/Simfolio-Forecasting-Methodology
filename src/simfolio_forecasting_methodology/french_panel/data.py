"""Freeze and validate the independently scored French daily return series.

Raw provider files are immutable inputs, not an automatically refreshing cache.
The audit fixes their byte hashes; a changed provider vintage requires a new panel.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

import numpy as np

from ..origins import origin_schedule

PANEL_ID = "french1926-single-series-552-v1"
PROTOCOL_ID = "french1926-whitepaper-dense-daily-240-v1"
SIMULATIONS = 240
RESOURCE = files("simfolio_forecasting_methodology").joinpath("resources", "french1926")


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _audit() -> dict:
    return json.loads(RESOURCE.joinpath("source_audit.json").read_text())


def inventory() -> list[dict]:
    with RESOURCE.joinpath("equity_series.csv").open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 550 or len({r["return_series_sha256"] for r in rows}) != 550:
        raise ValueError("frozen equity inventory drifted")
    return rows


def parse_portfolio_zip(path: Path) -> dict[str, tuple[list[str], list[str], np.ndarray]]:
    """Read only published daily value/equal-weighted RETURN blocks, in percent."""
    output = {}
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("expected one provider CSV")
        with archive.open(members[0]) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
            weight, headers, dates, values = None, None, [], []

            def finish():
                if dates:
                    if weight in output:
                        raise ValueError("duplicate daily return section")
                    output[weight] = (headers, dates, np.asarray(values, dtype="<f8"))

            for row in reader:
                title = " ".join(row).strip().lower()
                if "weighted returns" in title and "daily" in title:
                    finish()
                    weight = "equal" if "equal" in title else "value"
                    headers, dates, values = None, [], []
                elif weight and headers is None and len(row) > 1 and not row[0].strip():
                    headers = [v.strip() for v in row[1:]]
                elif headers is not None and row and re.fullmatch(r"\d{8}", row[0].strip()):
                    if len(row) != len(headers) + 1:
                        raise ValueError("provider row width drifted")
                    dates.append(row[0].strip())
                    values.append([float(v) for v in row[1:]])
                elif dates:
                    finish()
                    weight, headers, dates, values = None, None, [], []
            finish()
    if set(output) != {"value", "equal"}:
        raise ValueError("both published daily return weightings are required")
    return output


def parse_market_zip(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    audit = _audit()
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("expected one market CSV")
        dates, market, cash, headers = [], [], [], None
        with archive.open(members[0]) as raw:
            for row in csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig")):
                if row and "Mkt-RF" in row and "RF" in row:
                    if dates:
                        break
                    headers = [v.strip() for v in row]
                elif headers and row and re.fullmatch(r"\d{8}", row[0].strip()):
                    date = row[0].strip()
                    if audit["common_start"] <= date <= audit["common_end"]:
                        rf = float(row[headers.index("RF")])
                        dates.append(date)
                        market.append(float(row[headers.index("Mkt-RF")]) + rf)
                        cash.append(rf)
                elif dates:
                    break
    return dates, np.asarray(market, dtype="<f8"), np.asarray(cash, dtype="<f8")


def _valid_percent_returns(values: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(values)) and np.all(values > -100)
                and not np.any(values == -99.99) and not np.any(values == -999))


def _download(url: str, target: Path, expected: str) -> None:
    if not target.exists():
        temporary = target.with_suffix(".download")
        req = urllib.request.Request(url, headers={"User-Agent": "SimfolioResearch/1.0"})
        with urllib.request.urlopen(req, timeout=120) as source, temporary.open("wb") as dest:
            total = 0
            while block := source.read(1024 * 1024):
                total += len(block)
                if total > 64 * 1024 * 1024:
                    raise ValueError("provider archive exceeds bounded download size")
                dest.write(block)
        if digest_file(temporary) != expected:
            temporary.unlink()
            raise ValueError("provider vintage changed: source SHA256 differs from frozen audit")
        temporary.replace(target)
    if digest_file(target) != expected:
        raise ValueError("source archive checksum mismatch")


def build_panel(root: Path) -> dict:
    """Materialize data and a value-free task schedule. Never import or run a model."""
    root = Path(root)
    if (root / "panel.json").exists():
        return load_panel(root)
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "raw"
    raw.mkdir(exist_ok=True)
    audit, selected = _audit(), inventory()
    source_records = audit["files"] + [{
        "url": audit["market_source"]["url"],
        "source_sha256": audit["market_source"]["sha256"],
    }]
    for source in source_records:
        _download(source["url"], raw / source["url"].rsplit("/", 1)[-1], source["source_sha256"])
    dates, market, cash = parse_market_zip(raw / audit["market_source"]["url"].rsplit("/", 1)[-1])
    if len(dates) != audit["daily_observations"] or dates[0] != audit["common_start"] or dates[-1] != audit["common_end"]:
        raise ValueError("market calendar differs from frozen audit")
    date_array = np.array([f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates], dtype="datetime64[D]")
    np.save(root / "dates.npy", date_array, allow_pickle=False)
    matrix = np.lib.format.open_memmap(root / "daily_log_returns.npy", mode="w+", dtype="<f8", shape=(552, len(dates)))
    series, seen, cached_url, blocks = [], set(), None, None
    for index, row in enumerate(selected + [
        {"family": "broad_market", "dataset_family": "market", "weighting": "value", "published_column": "Mkt-RF + RF", "source_url": audit["market_source"]["url"], "source_sha256": audit["market_source"]["sha256"]},
        {"family": "cash", "dataset_family": "cash", "weighting": "cash_control", "published_column": "RF", "source_url": audit["market_source"]["url"], "source_sha256": audit["market_source"]["sha256"]},
    ]):
        if index < 550:
            if row["source_url"] != cached_url:
                cached_url = row["source_url"]
                blocks = parse_portfolio_zip(raw / cached_url.rsplit("/", 1)[-1])
            headers, all_dates, data = blocks[row["weighting"]]
            positions = [i for i, d in enumerate(all_dates) if dates[0] <= d <= dates[-1]]
            if [all_dates[i] for i in positions] != dates:
                raise ValueError("portfolio calendar mismatch; no filling allowed")
            values = np.asarray(data[positions, headers.index(row["published_column"])], dtype="<f8")
        else:
            values = market if index == 550 else cash
        value_hash = hashlib.sha256(values.tobytes()).hexdigest()
        if value_hash != row.get("return_series_sha256", value_hash) or value_hash in seen:
            raise ValueError("selected return history changed or duplicates another series")
        if not _valid_percent_returns(values) or np.ptp(values) == 0:
            raise ValueError("selected series has missing, invalid or constant observations")
        seen.add(value_hash)
        logs = np.log1p(values / 100.0)
        matrix[index] = logs
        series.append({**row, "series_id": f"FR{index + 1:04d}", "role": "cash_control" if index == 551 else "equity", "return_series_sha256": value_hash, "log_returns_sha256": hashlib.sha256(logs.tobytes()).hexdigest()})
    matrix.flush()
    del matrix
    descriptors = origin_schedule(date_array)
    origin_rows = [asdict(item) for item in descriptors]
    (root / "origins.json").write_text(json.dumps(origin_rows, indent=2) + "\n")
    panel = {
        "panel_id": PANEL_ID, "protocol_id": PROTOCOL_ID, "status": "ready_unscored",
        "common_start": str(date_array[0]), "common_end": str(date_array[-1]),
        "observations": len(dates), "series_count": 552, "equity_series_count": 551,
        "cash_controls": 1, "simulations_per_origin": SIMULATIONS,
        "origins_per_series": len(descriptors), "expected_tasks": 552 * len(descriptors),
        "max_horizon": max(d.max_horizon for d in descriptors),
        "expected_cells": 552 * max(d.max_horizon for d in descriptors),
        "registered_models": [], "results": [], "series": series,
        "score_target": "terminal_log_return", "score": "exact_empirical_crps_n_squared",
        "aggregation": "mean_origins_within_series_horizon_then_equal_all_cells",
        "cash_control_in_primary_aggregate": True,
        "calendar": "actual_provider_dates_including_historical_saturdays_no_fill",
        "future_policy_calendar": "synthetic_business_days_matching_existing_harness",
        "return_transform": "log1p(published_simple_percent_return / 100)",
        "source_audit_sha256": hashlib.sha256(RESOURCE.joinpath("source_audit.json").read_bytes()).hexdigest(),
        "inventory_sha256": hashlib.sha256(RESOURCE.joinpath("equity_series.csv").read_bytes()).hexdigest(),
        "files": {name: digest_file(root / name) for name in ("dates.npy", "daily_log_returns.npy", "origins.json")},
        "raw_sources": {s["url"].rsplit("/", 1)[-1]: s["source_sha256"] for s in source_records},
    }
    panel["panel_fingerprint"] = stable_digest(panel)
    (root / "panel.json").write_text(json.dumps(panel, indent=2, sort_keys=True) + "\n")
    return load_panel(root)


def load_dates(root: Path) -> np.ndarray:
    return np.load(Path(root) / "dates.npy", allow_pickle=False)


def load_returns(root: Path) -> np.memmap:
    return np.load(Path(root) / "daily_log_returns.npy", mmap_mode="r", allow_pickle=False)


def schedule(root: Path) -> tuple:
    return origin_schedule(load_dates(root))


def load_panel(root: Path) -> dict:
    root = Path(root)
    panel = json.loads((root / "panel.json").read_text())
    payload = {key: value for key, value in panel.items() if key != "panel_fingerprint"}
    if panel.get("panel_fingerprint") != stable_digest(payload):
        raise ValueError("panel manifest identity mismatch")
    frozen = json.loads(RESOURCE.joinpath("panel_manifest.json").read_text())
    if panel["panel_fingerprint"] != frozen["panel_fingerprint"]:
        raise ValueError("panel differs from the packaged frozen definition")
    if panel["panel_id"] != PANEL_ID or panel["protocol_id"] != PROTOCOL_ID:
        raise ValueError("wrong French panel identity")
    if panel["source_audit_sha256"] != hashlib.sha256(RESOURCE.joinpath("source_audit.json").read_bytes()).hexdigest() or panel["inventory_sha256"] != hashlib.sha256(RESOURCE.joinpath("equity_series.csv").read_bytes()).hexdigest():
        raise ValueError("frozen inventory provenance mismatch")
    for name, expected in panel["files"].items():
        if name not in {"dates.npy", "daily_log_returns.npy", "origins.json"} or digest_file(root / name) != expected:
            raise ValueError("panel input checksum mismatch")
    values, dates = load_returns(root), load_dates(root)
    if values.shape != (552, 26317) or values.dtype != np.dtype("<f8") or dates.dtype != np.dtype("datetime64[D]"):
        raise ValueError("frozen panel dimensions or precision drifted")
    if len(dates) != 26317 or np.any(np.isnat(dates)) or np.any(dates[1:] <= dates[:-1]):
        raise ValueError("invalid source calendar")
    if json.loads((root / "origins.json").read_text()) != [asdict(d) for d in schedule(root)]:
        raise ValueError("origin schedule differs from existing whitepaper rules")
    if panel["registered_models"] or panel["results"]:
        raise ValueError("input panel must remain separate from model runs")
    return panel
