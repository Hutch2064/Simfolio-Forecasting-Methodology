"""Public, deterministic reconstruction of the canonical research return panel.

The original OOS study used long-history simulated/proxy series.  This module
reconstructs the 38 series needed by the canonical 80-portfolio panel from
public research sources.  Nothing here depends on the production service.

Upstream datasets are downloaded into a user-supplied cache.  Generated series
are truncated to the canonical study cutoff and written with a manifest that
records source URLs and content hashes so a completed run is auditable.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
import zipfile

import numpy as np
import pandas as pd
import requests

CANONICAL_START = pd.Timestamp("1979-12-31")
CANONICAL_END = pd.Timestamp("2026-05-13")
PUBLIC_SOURCE_VERSION = "canonical_public_proxy_v1"

FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
WORLD_BANK = (
    "https://thedocs.worldbank.org/en/doc/"
    "74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/"
    "CMO-Historical-Data-Monthly.xlsx"
)
SOURCE_URLS = {
    "us_factors": f"{FRENCH}/F-F_Research_Data_Factors_daily_CSV.zip",
    "us_size": f"{FRENCH}/Portfolios_Formed_on_ME_Daily_CSV.zip",
    "us_value": f"{FRENCH}/25_Portfolios_5x5_Daily_CSV.zip",
    "industry12": f"{FRENCH}/12_Industry_Portfolios_Daily_CSV.zip",
    "industry48": f"{FRENCH}/48_Industry_Portfolios_Daily_CSV.zip",
    "developed": f"{FRENCH}/Developed_3_Factors_Daily_CSV.zip",
    "developed_ex_us": f"{FRENCH}/Developed_ex_US_3_Factors_Daily_CSV.zip",
    "developed_ex_us_value": f"{FRENCH}/Developed_ex_US_6_Portfolios_ME_BE-ME_daily_CSV.zip",
    "world_bank": WORLD_BANK,
}
FRED_SERIES = ("DGS2", "DGS3", "DGS5", "DGS10", "DGS20", "DGS30")

CANONICAL_TICKERS = (
    "EFASIM", "IWCSIM", "IWMSIM", "SPYSIM", "URTHSIM", "VBSIM", "VBKSIM",
    "VBRSIM", "VEASIM", "VOESIM", "VOOSIM", "VOSIM", "VOTSIM", "VTSIM",
    "VTISIM", "VTVSIM", "VUGSIM", "VVSIM", "VXUSSIM", "XLBSIM", "XLCSIM",
    "XLESIM", "XLFSIM", "XLISIM", "XLKSIM", "XLPSIM", "XLUSIM", "XLVSIM",
    "XLYSIM", "IEFSIM", "IEISIM", "SHYSIM", "TLTSIM", "ZROZSIM", "GLDSIM",
    "GSGSIM", "REITSIM", "SLVSIM",
)


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(15, 120), headers={"User-Agent": "simfolio-oos/1"})
            response.raise_for_status()
            path.write_bytes(response.content)
            return
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def ensure_public_sources(cache: Path, *, refresh: bool = False) -> dict[str, Path]:
    cache = Path(cache)
    output: dict[str, Path] = {}
    for source, url in SOURCE_URLS.items():
        suffix = ".xlsx" if source == "world_bank" else ".zip"
        path = cache / "sources" / f"{source}{suffix}"
        if refresh or not path.exists():
            _download(url, path)
        output[source] = path
    for series in FRED_SERIES:
        path = cache / "sources" / f"fred_{series}.csv"
        if refresh or not path.exists():
            _download(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}", path)
        output[f"fred_{series}"] = path
    return output


def _read_french(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        text = archive.read(names[0]).decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    pattern = re.compile(r"^\s*\d{8}\s*,")
    header = None
    for index, line in enumerate(lines[:-1]):
        if not line.lstrip().startswith(","):
            continue
        following = next((candidate for candidate in lines[index + 1 :] if candidate.strip()), "")
        if pattern.match(following):
            header = index
            break
    if header is None:
        raise ValueError(f"daily return table not found in {path}")
    rows = [lines[header]]
    started = False
    for line in lines[header + 1 :]:
        if pattern.match(line):
            rows.append(line)
            started = True
        elif started:
            break
    parsed = list(csv.reader(rows))
    columns = ["date", *[item.strip() for item in parsed[0][1:]]]
    frame = pd.DataFrame(parsed[1:], columns=columns)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("date").str.strip(), format="%Y%m%d"))
    frame = frame.apply(pd.to_numeric, errors="coerce").mask(lambda x: x <= -99.0) / 100.0
    return frame.sort_index()


def _fred(path: Path, name: str) -> pd.Series:
    frame = pd.read_csv(path)
    date_column = "observation_date" if "observation_date" in frame.columns else frame.columns[0]
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop(date_column)))
    value = pd.to_numeric(frame[name], errors="coerce").dropna()
    return value.sort_index()


def _clean(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index().astype(float)
    return out.clip(lower=-0.999999)


def _drag(series: pd.Series, annual: float) -> pd.Series:
    return _clean(series - float(annual) / 252.0)


def _market(frame: pd.DataFrame) -> pd.Series:
    return _clean(frame["Mkt-RF"] + frame["RF"])


def _bond(yields: pd.Series, duration: float, annual_fee: float) -> pd.Series:
    dates = pd.bdate_range(yields.index.min(), yields.index.max())
    y = yields.reindex(dates).ffill() / 100.0
    delta = y.diff().fillna(0.0)
    convexity = duration * (duration + 1.0)
    returns = y.shift(1).fillna(y) / 252.0 - duration * delta + 0.5 * convexity * delta.pow(2)
    return _drag(returns.clip(-0.95, 0.95), annual_fee)


def _world_bank(path: Path) -> dict[str, pd.Series]:
    prices = pd.read_excel(path, sheet_name="Monthly Prices", header=None)
    header = [str(value).strip() for value in prices.iloc[4]]
    mask = prices.iloc[:, 0].astype(str).str.match(r"^\d{4}M\d{2}$")
    data = prices.loc[mask]
    dates = pd.PeriodIndex(data.iloc[:, 0].astype(str).str.replace("M", "-"), freq="M").to_timestamp("M")
    result: dict[str, pd.Series] = {}
    for label in ("Gold", "Silver"):
        values = pd.to_numeric(data.iloc[:, header.index(label)], errors="coerce")
        result[label.lower()] = pd.Series(values.to_numpy(), index=dates).dropna()
    indices = pd.read_excel(path, sheet_name="Monthly Indices", header=None)
    mask = indices.iloc[:, 0].astype(str).str.match(r"^\d{4}M\d{2}$")
    data = indices.loc[mask]
    dates = pd.PeriodIndex(data.iloc[:, 0].astype(str).str.replace("M", "-"), freq="M").to_timestamp("M")
    result["total"] = pd.Series(pd.to_numeric(data.iloc[:, 1], errors="coerce").to_numpy(), index=dates).dropna()
    return result


def _monthly_to_daily(monthly_price: pd.Series, shape: pd.Series, scale: float) -> pd.Series:
    monthly = monthly_price.dropna().sort_index()
    monthly.index = monthly.index.to_period("M").to_timestamp("M")
    pieces: list[pd.Series] = []
    for prior, current in zip(monthly.index[:-1], monthly.index[1:]):
        if current.to_period("M").ordinal - prior.to_period("M").ordinal != 1:
            continue
        dates = pd.bdate_range(current.to_period("M").start_time, current)
        if dates.empty:
            continue
        target = math.log(float(monthly.loc[current]) / float(monthly.loc[prior]))
        shaped = np.log1p(shape.reindex(dates).fillna(0.0).clip(lower=-0.95))
        centered = (shaped - float(shaped.mean())) * float(scale)
        pieces.append(pd.Series(np.expm1(centered + target / len(dates)), index=dates))
    return _clean(pd.concat(pieces))


def build_canonical_series(source_paths: dict[str, Path]) -> dict[str, pd.Series]:
    factors = _read_french(source_paths["us_factors"])
    size = _read_french(source_paths["us_size"])
    value = _read_french(source_paths["us_value"])
    industry12 = _read_french(source_paths["industry12"])
    industry48 = _read_french(source_paths["industry48"])
    developed = _read_french(source_paths["developed"])
    ex_us = _read_french(source_paths["developed_ex_us"])
    ex_us_value = _read_french(source_paths["developed_ex_us_value"])
    market = _market(factors)
    developed_market = _market(developed)
    ex_us_market = _market(ex_us)
    out: dict[str, pd.Series] = {}
    out["SPYSIM"] = market
    out["VTISIM"] = _drag(market, 0.0003)

    size_specs = {
        "IWCSIM": (("Lo 10",), 0.0060), "IWMSIM": (("Lo 20",), 0.0019),
        "VBSIM": (("Lo 20",), 0.0005), "VOOSIM": (("Hi 20",), 0.0003),
        "VVSIM": (("Hi 20",), 0.0004),
    }
    for ticker, (columns, fee) in size_specs.items():
        out[ticker] = _drag(size[list(columns)].mean(axis=1), fee)
    value_specs = {
        "VBKSIM": (("SMALL LoBM", "ME1 BM2"), 0.0007),
        "VBRSIM": (("ME1 BM4", "SMALL HiBM"), 0.0007),
        "VOTSIM": (("ME3 BM1", "ME3 BM2"), 0.0007),
        "VOSIM": (("ME3 BM1", "ME3 BM2", "ME3 BM3", "ME3 BM4", "ME3 BM5"), 0.0004),
        "VOESIM": (("ME3 BM4", "ME3 BM5"), 0.0007),
        "VUGSIM": (("BIG LoBM", "ME5 BM2"), 0.0004),
        "VTVSIM": (("ME5 BM4", "BIG HiBM"), 0.0004),
    }
    for ticker, (columns, fee) in value_specs.items():
        out[ticker] = _drag(value[list(columns)].mean(axis=1), fee)
    sectors = {
        "XLBSIM": ("Chems",), "XLCSIM": ("Telcm",), "XLESIM": ("Enrgy",),
        "XLFSIM": ("Money",), "XLISIM": ("Manuf",), "XLKSIM": ("BusEq",),
        "XLPSIM": ("NoDur",), "XLUSIM": ("Utils",), "XLVSIM": ("Hlth",),
        "XLYSIM": ("Durbl", "Shops"),
    }
    for ticker, columns in sectors.items():
        out[ticker] = _drag(industry12[list(columns)].mean(axis=1), 0.0009)
    ex_value = ex_us_value[["SMALL HiBM", "BIG HiBM"]].mean(axis=1)
    del ex_value  # retained in source methodology but not needed by canonical-38
    out["EFASIM"] = _drag(ex_us_market, 0.0032)
    out["VEASIM"] = _drag(ex_us_market, 0.0003)
    out["VXUSSIM"] = _drag(ex_us_market, 0.0005)
    out["URTHSIM"] = _drag(developed_market, 0.0024)
    out["VTSIM"] = _drag(developed_market, 0.0008)
    out["REITSIM"] = _drag(industry48["RlEst"].dropna(), 0.0013)

    dgs2 = _fred(source_paths["fred_DGS2"], "DGS2")
    dgs3 = _fred(source_paths["fred_DGS3"], "DGS3")
    dgs5 = _fred(source_paths["fred_DGS5"], "DGS5")
    dgs10 = _fred(source_paths["fred_DGS10"], "DGS10")
    dgs20 = _fred(source_paths["fred_DGS20"], "DGS20")
    dgs30 = _fred(source_paths["fred_DGS30"], "DGS30")
    out["SHYSIM"] = _bond(dgs2.combine_first(dgs3), 1.8, 0.0015)
    out["IEISIM"] = _bond(dgs5, 4.5, 0.0015)
    out["IEFSIM"] = _bond(dgs10, 8.0, 0.0015)
    long_yield = dgs30.combine_first(dgs20)
    out["TLTSIM"] = _bond(long_yield, 17.0, 0.0015)
    out["ZROZSIM"] = _bond(long_yield, 25.0, 0.0015)

    commodities = _world_bank(source_paths["world_bank"])
    gold = _monthly_to_daily(commodities["gold"], industry48["Gold"].dropna(), 0.45)
    silver = _monthly_to_daily(commodities["silver"], industry48["Mines"].dropna(), 0.80)
    broad = _monthly_to_daily(commodities["total"], market, 0.45)
    out["GLDSIM"] = _drag(gold, 0.0040)
    out["SLVSIM"] = _drag(silver, 0.0050)
    out["GSGSIM"] = _drag(broad, 0.0075)

    missing = sorted(set(CANONICAL_TICKERS) - set(out))
    if missing:
        raise AssertionError(f"canonical series builder is incomplete: {missing}")
    return {ticker: _clean(out[ticker]).loc[:CANONICAL_END] for ticker in CANONICAL_TICKERS}


def write_canonical_data(cache: Path, *, refresh: bool = False) -> Path:
    cache = Path(cache)
    sources = ensure_public_sources(cache, refresh=refresh)
    series = build_canonical_series(sources)
    destination = cache / "canonical_series"
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for ticker, values in series.items():
        path = destination / f"{ticker}.csv.gz"
        frame = pd.DataFrame({"date": values.index.strftime("%Y-%m-%d"), "daily_return": values.to_numpy()})
        csv_bytes = frame.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode()
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
                handle.write(csv_bytes)
        records.append({
            "ticker": ticker,
            "rows": int(len(values)),
            "first_date": str(values.index.min().date()),
            "last_date": str(values.index.max().date()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    manifest = {
        "version": PUBLIC_SOURCE_VERSION,
        "canonical_cutoff": str(CANONICAL_END.date()),
        "series": records,
        "sources": {
            name: {"url": SOURCE_URLS.get(name, f"https://fred.stlouisfed.org/series/{name.removeprefix('fred_')}"),
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in sources.items()
        },
    }
    manifest_path = cache / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def load_canonical_returns(cache: Path) -> pd.DataFrame:
    root = Path(cache) / "canonical_series"
    columns: dict[str, pd.Series] = {}
    for ticker in CANONICAL_TICKERS:
        path = root / f"{ticker}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run `simfolio-oos data build` first")
        frame = pd.read_csv(path, compression="gzip")
        index = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
        columns[ticker] = pd.Series(pd.to_numeric(frame["daily_return"], errors="raise").to_numpy(), index=index)
    return pd.concat(columns, axis=1).sort_index()
