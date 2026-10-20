"""Freeze the authorized native-calendar ragged panel; never import forecast models."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD_PANEL = ROOT.parent / "french551-full-20261019/inputs/french1926-equity551"
PINNED_DATA = ROOT.parent / "catalogue52-20261018/methodology/src/simfolio_forecasting_methodology/resources/data"
PANEL_ID = "daily-cross-asset-native-ragged-20261020-v1"
SIMULATIONS = 240
MAX_INPUT_BYTES = 512 * 1024**2


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _schedule(dates):
    try:
        from simfolio_forecasting_methodology.origins import origin_schedule
    except ModuleNotFoundError:
        sys.path.insert(0, str(ROOT.parent / "french551-full-20261019/code/src"))
        from simfolio_forecasting_methodology.origins import origin_schedule
    return [asdict(origin) for origin in origin_schedule(dates)]


def validate_arrays(dates, returns):
    dates, returns = np.asarray(dates), np.asarray(returns)
    if dates.ndim != 1 or returns.ndim != 1 or len(dates) != len(returns) or len(dates) < 81:
        raise ValueError("series must have aligned one-dimensional dates/returns and at least81 observations")
    if dates.dtype != np.dtype("datetime64[D]") or returns.dtype != np.dtype("<f8"):
        raise ValueError("dates require datetime64[D] and log returns require float64")
    if np.any(np.isnat(dates)) or np.any(dates[1:] <= dates[:-1]) or not np.all(np.isfinite(returns)):
        raise ValueError("series calendar must be unique/increasing and log returns finite")


def dated_digest(dates, returns):
    return hashlib.sha256(dates.astype("<i8").tobytes() + returns.astype("<f8").tobytes()).hexdigest()


def same_subset_history(first_dates, first_returns, second_dates, second_returns):
    """True only when the entire shorter dated history exactly equals a subset."""
    if len(first_dates) > len(second_dates):
        first_dates, second_dates = second_dates, first_dates
        first_returns, second_returns = second_returns, first_returns
    endpoints = np.searchsorted(second_dates, first_dates[[0,-1]])
    if np.any(endpoints >= len(second_dates)) or not np.array_equal(second_dates[endpoints], first_dates[[0,-1]]) or not np.array_equal(second_returns[endpoints], first_returns[[0,-1]]):
        return False
    positions = np.searchsorted(second_dates, first_dates)
    if np.any(positions >= len(second_dates)):
        return False
    return bool(np.array_equal(first_dates, second_dates[positions]) and np.array_equal(first_returns, second_returns[positions]))


def levels_to_log_returns(dates, prices, *, invert=False):
    dates = np.asarray(dates, dtype="datetime64[D]")
    prices = np.asarray(prices, dtype="<f8")
    if len(dates) != len(prices) or len(prices) < 2 or not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("observed price levels must be aligned, positive and finite")
    if np.any(dates[1:] <= dates[:-1]):
        raise ValueError("observed price dates must be unique/increasing")
    logs = np.log(prices[1:] / prices[:-1])
    return dates[1:], np.asarray(-logs if invert else logs, dtype="<f8")


def _percent_logs(values):
    values = np.asarray(values, dtype="<f8")
    if not np.all(np.isfinite(values)) or np.any(values <= -100) or np.any(values == -99.99):
        raise ValueError("invalid provider percent return; no filling allowed")
    return np.log1p(values / 100)


def _french_blocks(path):
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("expected one French CSV")
        text = archive.read(members[0]).decode("utf-8-sig")
    output = {}; weight = headers = None; dates = []; values = []
    def finish():
        nonlocal weight, headers, dates, values
        if dates:
            output[weight] = (headers, np.array([f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates], dtype="datetime64[D]"), np.asarray(values, dtype="<f8"))
        weight = headers = None; dates = []; values = []
    for row in csv.reader(io.StringIO(text)):
        title = " ".join(row).strip().lower()
        if "weighted returns" in title and ("daily" in title or "monthly" in title):
            finish(); weight = "equal" if "equal" in title else "value"
        elif row and "Mkt-RF" in row and "RF" in row:
            finish(); weight = "market"; headers = [value.strip() for value in row[1:]]
        elif weight and headers is None and len(row) > 1 and not row[0].strip():
            headers = [value.strip() for value in row[1:]]
        elif headers is not None and row and re.fullmatch(r"\d{8}", row[0].strip()):
            if len(row) != len(headers) + 1:
                raise ValueError("French provider row width changed")
            dates.append(row[0].strip()); values.append([float(value) for value in row[1:]])
        elif dates:
            finish()
    finish()
    if not output:
        raise ValueError("no native daily return blocks")
    return output


def _original_candidates(old):
    panel = _json(old / "panel.json")
    payload = {key: value for key, value in panel.items() if key != "panel_fingerprint"}
    if stable_digest(payload) != panel["panel_fingerprint"] or panel["series_count"] != 551:
        raise ValueError("original551 manifest mismatch")
    for name, expected in panel["files"].items():
        if digest_file(old / name) != expected:
            raise ValueError("original551 checksum mismatch")
    dates = np.load(old / "dates.npy", allow_pickle=False)
    matrix = np.load(old / "daily_log_returns.npy", mmap_mode="r", allow_pickle=False)
    raw_cache = {}
    for index, item in enumerate(panel["series"]):
        values = np.asarray(matrix[index], dtype="<f8")
        if hashlib.sha256(values.tobytes()).hexdigest() != item["log_returns_sha256"]:
            raise ValueError("original551 row hash mismatch")
        active_dates, active_values = dates, values
        extension = None
        if item["published_column"] != "Mkt-RF + RF":
            name = item["source_url"].rsplit("/",1)[-1]
            if name not in raw_cache:
                source = old/"raw"/name
                if digest_file(source) != item["source_sha256"]:
                    raise ValueError("original raw source checksum mismatch")
                raw_cache[name] = _french_blocks(source)
            headers, full_dates, full_data = raw_cache[name][item["weighting"]]
            full_values = full_data[:,headers.index(item["published_column"]) ]
            valid = np.isfinite(full_values) & (full_values > -100) & (full_values != -99.99)
            invalid = np.flatnonzero(~valid)
            start = int(invalid[-1]+1) if len(invalid) else 0
            candidate_dates = full_dates[start:]
            if len(candidate_dates)>len(dates):
                candidate_values = _percent_logs(full_values[start:])
                if not same_subset_history(dates, values, candidate_dates, candidate_values):
                    raise ValueError("extended original history changed retained observations")
                active_dates,active_values = candidate_dates,candidate_values
                extension = dict(original_first_date=str(dates[0]),extended_first_date=str(active_dates[0]),extra_native_observations=len(active_dates)-len(dates),overlapping_values_exact=True)
        real_estate = item["published_column"] == "RlEst"
        yield active_dates, active_values, dict(series_id=item["series_id"], asset_class="real_estate" if real_estate else "equities", family="us_listed_real_estate" if real_estate else "us_" + item["family"], region="US", display_name=f"US {item['dataset_family']} {item['weighting']} {item['published_column']}", preserve_original=True, provenance=dict(original_panel_fingerprint=panel["panel_fingerprint"], original_series=item, original_dates_sha256=panel["files"]["dates.npy"], original_origins_sha256=panel["files"]["origins.json"],history_extension=extension))


def _extra_equity_candidates(root):
    cache = {}
    for ordinal, item in enumerate(_json(root / "equities/primary_recommended.json"), start=1):
        source = root / "equities/raw" / item["source_file"]
        if source.name not in cache:
            if digest_file(source) != item["source_sha256"]:
                raise ValueError("French source checksum mismatch")
            cache[source.name] = _french_blocks(source)
        blocks = cache[source.name]
        if item["weighting"] == "market_total_return":
            headers, dates, data = blocks["market"]
            values = data[:, headers.index("Mkt-RF")] + data[:, headers.index("RF")]
        else:
            headers, dates, data = blocks[item["weighting"]]
            values = data[:, headers.index(item["published_column"])]
        start = np.datetime64(item["contiguous_admission_start"][:4] + "-" + item["contiguous_admission_start"][4:6] + "-" + item["contiguous_admission_start"][6:], "D")
        selected = dates >= start
        dates, values = dates[selected], values[selected]
        logs = _percent_logs(values)
        real_estate = item["published_column"] == "RlEst"
        family = "us_industries" if item["region"] == "US" and item["family"].startswith("industry_") else item["region"].lower() + "_" + item["family"]
        yield dates, logs, dict(series_id=f"EQX{ordinal:04d}", asset_class="real_estate" if real_estate else "equities", family="us_listed_real_estate" if real_estate else family, region=item["region"], display_name=f"{item['region']} {item['family']} {item['published_column']}", preserve_original=False, provenance=dict(source_profile=item, transform="log1p(published_simple_percent /100)", invalid_policy="explicit_latest_contiguous_valid_source_segment_no_fill"))


def _bond_candidates(root):
    audit = _json(root / "fixed_income/pinned_canonical_snapshot_audit.json")
    manifest = Path(audit["canonical_snapshot_manifest"])
    if digest_file(manifest) != audit["canonical_snapshot_manifest_sha256"]:
        raise ValueError("accepted canonical manifest checksum mismatch")
    metadata = _json(manifest)
    records = {item["ticker"]: item for item in metadata["series"] + metadata["supporting_series"]}
    families = {"SHYSIM":"nominal_treasury", "IEISIM":"nominal_treasury", "IEFSIM":"nominal_treasury", "TLTSIM":"nominal_treasury", "ZROZSIM":"nominal_treasury", "BNDSIM":"aggregate_bonds", "TIPSIM":"inflation_linked_bonds"}
    for ticker, family in families.items():
        item = records[ticker]
        source = manifest.parent / "canonical_snapshot" / item["path"]
        if digest_file(source) != item["sha256"]:
            raise ValueError("accepted SIM source checksum mismatch")
        with gzip.open(source, "rt", newline="") as handle:
            rows = list(csv.DictReader(handle))
        dates = np.asarray([row["date"] for row in rows], dtype="datetime64[D]")
        values = np.asarray([float(row["daily_return"]) for row in rows], dtype="<f8")
        if np.any(values <= -1) or not np.all(np.isfinite(values)):
            raise ValueError("invalid accepted SIM simple return")
        yield dates, np.log1p(values), dict(series_id=ticker,asset_class="fixed_income",family=family,region="US",display_name=ticker,preserve_original=False,provenance=dict(source_file=str(source),source_sha256=item["sha256"],source_revision=metadata["source_revision"],source_snapshot=metadata["dataset_id"],construction_status="accepted_by_user_not_reaudited",transform="log1p(daily_return)"))


def _other_candidates(root):
    catalogue = _json(root / "other_assets/source_catalogue.json")
    sources = catalogue["sources"]
    h10_path = root / "other_assets/downloads/FRB_H10.zip"
    h10_meta = {item["instrument"]: item for item in sources if item.get("asset_class") == "foreign_exchange_spot"}
    if digest_file(h10_path) != h10_meta["AUD"]["sha256"]:
        raise ValueError("H10 source checksum mismatch")
    with zipfile.ZipFile(h10_path) as archive:
        xml = ET.fromstring(archive.read("H10_data.xml"))
    for currency in ("AUD", "CAD", "GBP", "JPY", "CHF", "EUR"):
        meta = h10_meta[currency]
        source = next(element for element in xml.iter() if element.tag.endswith("}Series") and element.attrib.get("SERIES_NAME") == meta["series_id"])
        observed = [element.attrib for element in source if element.tag.endswith("}Obs") and element.attrib.get("OBS_STATUS") == "A"]
        price_dates = [item["TIME_PERIOD"] for item in observed]
        prices = [float(item["OBS_VALUE"]) for item in observed]
        dates, logs = levels_to_log_returns(price_dates, prices, invert=meta["quote"] == "foreign currency units per USD")
        yield dates, logs, dict(series_id=f"FX_{currency}_USD",asset_class="foreign_exchange",family="bilateral_spot_currency",region=currency,display_name=f"{currency} versus USD",preserve_original=False,provenance=dict(source_record=meta,price_first_date=price_dates[0],transform=meta["usd_holding_log_return_formula"],gap_policy="observed_endpoint_returns_no_quote_fill",maximum_observed_price_interval_days=int(np.max(np.diff(np.asarray(price_dates,dtype="datetime64[D]")).astype(int)))))
    for ticker in ("BTC", "ETH"):
        meta = next(item for item in sources if item.get("instrument") == ticker)
        source = root / "other_assets" / meta["download_path"]
        if digest_file(source) != meta["sha256"]:
            raise ValueError("crypto source checksum mismatch")
        with source.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        dates, logs = levels_to_log_returns([row["time"][:10] for row in rows], [float(row["PriceUSD"]) for row in rows])
        yield dates, logs, dict(series_id=ticker,asset_class="cryptocurrencies",family="digital_asset_spot",region="global",display_name=f"{ticker} USD spot",preserve_original=False,provenance=dict(source_record=meta,price_first_date=rows[0]["time"][:10],transform="log(observed_PriceUSD_t/observed_PriceUSD_previous)",gap_policy="native_daily_UTC_observations_no_fill"))


def _admitted_candidates(path):
    manifest = _json(path)
    for item in manifest["series"]:
        def source_path(name):
            value = Path(item[name])
            return value if value.is_absolute() else path.parent / value
        dates = np.load(source_path("dates_file"), allow_pickle=False).astype("datetime64[D]")
        values = np.load(source_path("returns_file"), allow_pickle=False).astype("<f8")
        if item.get("returns_kind", "log") != "log":
            raise ValueError("admitted source arrays must already be log returns")
        for key in ("dates_file", "returns_file"):
            expected = item.get("file_sha256", {}).get(item[key]) or item.get(key.replace("_file", "_sha256")) or (item.get("return_dates_sha256") if key == "dates_file" else None)
            if expected and digest_file(source_path(key)) != expected:
                raise ValueError("admitted source checksum mismatch")
        interval_metadata = {}
        if item.get("interval_days_file"):
            interval_path = source_path("interval_days_file")
            if digest_file(interval_path) != item["interval_days_sha256"]:
                raise ValueError("admitted interval-day checksum mismatch")
            interval_values = np.load(interval_path,allow_pickle=False)
            if interval_values.shape != dates.shape or np.any(interval_values<=0):
                raise ValueError("admitted interval-day alignment mismatch")
            interval_metadata = dict(_provider_interval_days=interval_values)
        yield dates, values, dict(**interval_metadata,series_id=item["series_id"],asset_class=item["asset_class"],family=item["family"],region=item.get("region","global"),display_name=item.get("display_name",item.get("name",item["series_id"])),preserve_original=False,provenance=dict(admission_manifest=str(path),admission_manifest_sha256=digest_file(path),source_record=item))


def freeze_series(output, candidates, *, lineage=None):
    """Write a new panel exactly once, preserving originals before extra candidates."""
    output = Path(output)
    if (output / "panel.json").exists():
        raise FileExistsError("frozen panel already exists; verify or use a new versioned directory")
    output.mkdir(parents=True, exist_ok=True)
    rows = []; duplicates = []; retained = []; origin_cache = {}; ids = set(); total_bytes = 0
    for dates, logs, metadata in candidates:
        dates = np.asarray(dates, dtype="datetime64[D]"); logs = np.asarray(logs, dtype="<f8")
        validate_arrays(dates, logs)
        duplicate = None
        if not metadata.get("preserve_original"):
            for existing_dates, existing_logs, existing_id in retained:
                if same_subset_history(dates, logs, existing_dates, existing_logs):
                    duplicate = existing_id; break
        if duplicate:
            duplicates.append(dict(series_id=metadata["series_id"],duplicate_of=duplicate,reason="entire_shorter_native_dated_history_exactly_matches",provenance=metadata["provenance"]))
            continue
        series_id = metadata["series_id"]
        if series_id in ids or not re.fullmatch(r"[A-Za-z0-9_-]+", series_id):
            raise ValueError("duplicate or unsafe series identity")
        ids.add(series_id)
        folder = output / "series" / series_id; folder.mkdir(parents=True, exist_ok=False)
        np.save(folder / "dates.npy", dates, allow_pickle=False)
        np.save(folder / "log_returns.npy", logs, allow_pickle=False)
        intervals = np.r_[0,np.diff(dates).astype(int)].astype("<i4")
        anchor = metadata["provenance"].get("price_first_date")
        if anchor:
            intervals[0] = int((dates[0]-np.datetime64(anchor,"D")).astype(int))
        np.save(folder / "interval_days.npy",intervals,allow_pickle=False)
        if "_provider_interval_days" in metadata:
            np.save(folder/"provider_interval_days.npy",metadata["_provider_interval_days"],allow_pickle=False)
        calendar_key = hashlib.sha256(dates.astype("<i8").tobytes()).hexdigest()
        if calendar_key not in origin_cache:
            origin_cache[calendar_key] = _schedule(dates)
        origins = origin_cache[calendar_key]
        _write_json(folder / "origins.json", origins)
        calendar = {origin["origin_label"]:dict(training_first_date=str(dates[0]),training_last_date=str(dates[origin["position"]]),future_first_date=str(dates[origin["position"]+1]),future_last_date=str(dates[origin["position"]+origin["max_horizon"]])) for origin in origins}
        _write_json(folder / "origin_calendar.json", calendar)
        files = {str(path.relative_to(output)):digest_file(path) for path in sorted(folder.iterdir())}
        total_bytes += sum(path.stat().st_size for path in folder.iterdir())
        if total_bytes > MAX_INPUT_BYTES:
            raise ValueError("frozen input budget exceeds512MiB")
        row = {key:value for key,value in metadata.items() if key != "preserve_original" and not key.startswith("_")}
        row.update(first_date=str(dates[0]),last_date=str(dates[-1]),observations=len(dates),dates_file=str((folder/"dates.npy").relative_to(output)),returns_file=str((folder/"log_returns.npy").relative_to(output)),origins_file=str((folder/"origins.json").relative_to(output)),origin_calendar_file=str((folder/"origin_calendar.json").relative_to(output)),interval_days_file=str((folder/"interval_days.npy").relative_to(output)),interval_days_kind="actual_calendar_days_between_native_return_endpoints_zero_means_initial_anchor_unknown",source_frequency="daily_source_native_observed_intervals",origins_count=len(origins),max_horizon=max(item["max_horizon"] for item in origins),native_dated_returns_sha256=dated_digest(dates,logs),files=files)
        rows.append(row); retained.append((dates,logs,series_id))
    _write_json(output / "duplicate_exclusions.json", duplicates)
    panel = dict(schema_version=1,panel_id=PANEL_ID,status="frozen_unscored",simulations_per_origin=SIMULATIONS,series=rows,series_count=len(rows),expected_tasks=sum(row["origins_count"] for row in rows),expected_cells=sum(row["max_horizon"] for row in rows),asset_class_counts=dict(sorted(Counter(row["asset_class"] for row in rows).items())),lineage=lineage or {},calendar_policy="actual_native_provider_observations_no_fill_no_interpolation",origin_policy="original48_rolling_plus3_temporal_available_native_calendar",return_type="one_dimensional_native_observation_log_returns_from_daily_sources",source_equity_rows_preserved=sum(row["series_id"].startswith("FR") for row in rows),files={str(path.relative_to(output)):digest_file(path) for path in sorted(output.rglob("*")) if path.is_file()},input_bytes=total_bytes,duplicate_candidates_excluded=len(duplicates))
    panel["panel_fingerprint"] = stable_digest(panel)
    _write_json(output / "panel.json", panel)
    return load_panel(output)


def build_panel(root=ROOT, output_dir=None, *, old_panel=OLD_PANEL, allow_missing_admissions=False):
    root = Path(root); output = Path(output_dir) if output_dir else root / "inputs"
    admissions = [root / "commodities/admitted_series.json", root / "additional_fixed_income/admitted_series.json"]
    if not allow_missing_admissions:
        for path in admissions:
            if not path.exists():
                raise FileNotFoundError(f"awaiting required admitted source manifest: {path}")
    def candidates():
        yield from _original_candidates(Path(old_panel))
        yield from _extra_equity_candidates(root)
        yield from _bond_candidates(root)
        yield from _other_candidates(root)
        for path in admissions:
            if path.exists():
                yield from _admitted_candidates(path)
    return freeze_series(output,candidates(),lineage=dict(original_panel_fingerprint=_json(Path(old_panel)/"panel.json")["panel_fingerprint"],equity_audit_manifest_sha256=digest_file(root/"equities/AUDIT_MANIFEST.json"),admission_manifests={str(path.relative_to(root)):digest_file(path) for path in admissions if path.exists()},builder_sha256=digest_file(Path(__file__))))


def _input_path(root, relative):
    root = Path(root).resolve(); path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("input path escapes frozen panel")
    return path


def load_series(root, row):
    dates = np.load(_input_path(root,row["dates_file"]),mmap_mode="r",allow_pickle=False)
    logs = np.load(_input_path(root,row["returns_file"]),mmap_mode="r",allow_pickle=False)
    origins = _json(_input_path(root,row["origins_file"]))
    validate_arrays(dates,logs)
    if len(dates) != row["observations"] or str(dates[0]) != row["first_date"] or str(dates[-1]) != row["last_date"]:
        raise ValueError("series dimensions/calendar metadata mismatch")
    labels = set()
    for origin in origins:
        position,horizon = origin["position"],origin["max_horizon"]
        if origin["origin_label"] in labels or not 0 <= position < len(dates)-1 or not 0 < horizon <= len(dates)-position-1 or str(dates[position]) != origin["origin_date"]:
            raise ValueError("origin calendar/horizon mismatch")
        labels.add(origin["origin_label"])
    if len(origins) != row["origins_count"]:
        raise ValueError("origin count mismatch")
    return dates,logs,origins


def load_panel(root):
    root = Path(root);panel = _json(root/"panel.json")
    if panel["panel_fingerprint"] != stable_digest({key:value for key,value in panel.items() if key != "panel_fingerprint"}):
        raise ValueError("panel manifest identity mismatch")
    if panel["panel_id"] != PANEL_ID or panel["simulations_per_origin"] != SIMULATIONS:
        raise ValueError("panel identity/protocol mismatch")
    for relative,expected in panel["files"].items():
        if digest_file(_input_path(root,relative)) != expected:
            raise ValueError("frozen input checksum mismatch")
    ids = set(); tasks = 0
    for row in panel["series"]:
        if row["series_id"] in ids:
            raise ValueError("duplicate frozen series identity")
        ids.add(row["series_id"])
        required_files = {row[name] for name in ("dates_file","returns_file","origins_file","origin_calendar_file","interval_days_file")}
        if not required_files.issubset(row["files"]):
            raise ValueError("required series input missing from file manifest")
        for relative,expected in row["files"].items():
            if panel["files"].get(relative) != expected:
                raise ValueError("series file not bound to root manifest")
        _,_,origins = load_series(root,row);tasks += len(origins)
    if len(ids) != panel["series_count"] or tasks != panel["expected_tasks"]:
        raise ValueError("panel series/task count mismatch")
    return panel


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=ROOT);parser.add_argument("--output",type=Path);parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    result=load_panel(args.output or args.root/"inputs") if args.verify else build_panel(args.root,args.output)
    print(json.dumps({key:result[key] for key in ("panel_id","panel_fingerprint","series_count","expected_tasks","asset_class_counts","duplicate_candidates_excluded")},indent=2))
