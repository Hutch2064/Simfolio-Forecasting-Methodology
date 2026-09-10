"""Frozen factor-frame loading shared by source-backed numerical families.

The loader is deliberately offline: it reads the packaged canonical snapshot,
normalizes dates exactly as the retained research engine did, and fails closed
when the required schema is not present.  It does not refresh from a provider
or write a cache.
"""

from __future__ import annotations

from importlib import resources
from typing import Final

import numpy as np
import pandas as pd

_FACTOR_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "ff6": ("Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF", "UMD"),
    "q5": ("R_MKT", "R_ME", "R_IA", "R_ROE", "R_EG", "R_F"),
}
_FACTOR_FILES: Final[dict[str, str]] = {
    "ff6": "french_daily.csv.gz",
    "q5": "q5_daily.csv.gz",
}


def load_packaged_factor_frame(factor_model: str) -> pd.DataFrame:
    """Load one validated frozen FF6 or Q5 factor frame from package data."""

    model = str(factor_model).lower()
    if model not in _FACTOR_COLUMNS:
        raise ValueError(f"unsupported packaged factor model {factor_model!r}")
    resource = resources.files("simfolio_forecasting_methodology").joinpath(
        "resources",
        "data",
        "canonical_snapshot",
        "app",
        "factor_data",
        _FACTOR_FILES[model],
    )
    try:
        with resources.as_file(resource) as path:
            frame = pd.read_csv(path, compression="infer")
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError(f"packaged {model} factor snapshot is unavailable") from exc
    date_column = "Date" if "Date" in frame.columns else "DATE"
    if date_column not in frame.columns:
        raise ValueError(f"packaged {model} factor snapshot lacks Date")
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    if frame[date_column].isna().any():
        raise ValueError(f"packaged {model} factor snapshot contains invalid dates")
    frame = frame.set_index(date_column)
    columns = list(_FACTOR_COLUMNS[model])
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"packaged {model} factor snapshot lacks columns {missing}")
    frame = frame[columns].apply(pd.to_numeric, errors="coerce")
    values = frame.to_numpy(dtype=np.float64)
    if frame.isna().any().any() or not np.isfinite(values).all():
        raise ValueError(f"packaged {model} factor snapshot contains non-finite values")
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    if frame.index.duplicated().any():
        raise ValueError(f"packaged {model} factor snapshot contains duplicate dates")
    return frame.sort_index()


def factor_columns(factor_model: str) -> tuple[str, ...]:
    """Return the source column order for a supported factor model."""

    model = str(factor_model).lower()
    try:
        return _FACTOR_COLUMNS[model]
    except KeyError as exc:
        raise ValueError(f"unsupported factor model {factor_model!r}") from exc


__all__ = ["factor_columns", "load_packaged_factor_frame"]
