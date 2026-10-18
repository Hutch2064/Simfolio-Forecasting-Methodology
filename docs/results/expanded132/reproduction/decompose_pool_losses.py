#!/usr/bin/env python3
"""Reproduce the 80-portfolio / 52-singleton pool-vs-incumbent loss audit.

This reads only saved result arrays and their score/lineage manifests. The
matrix cells are already means across forecast origins; canonical CRPS then
equally weights portfolio-horizon cells. Origin counts verify complete support
and are deliberately not applied a second time as weights.

Run from any directory with:
  python3 /path/to/decompose_pool_losses.py
Optional --reference and --output arguments select the sparse reference tree
and the JSON receipt destination.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


AUDIT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = AUDIT_ROOT / "reference"
DEFAULT_OUTPUT = AUDIT_ROOT / "expanded" / "pool-loss-decomposition.json"
BANDS = ((1, 21), (22, 63), (64, 252), (253, 1260),
         (1261, 2520), (2521, 5040), (5041, 8766))
EXPECTED_DATA_MANIFEST = "3825258f753ba0a4ec653c56403ef7e163e80ea89eb1c06856b6415160672a76"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def load_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        names = saved["portfolio"].astype(str)
        losses = saved["losses"].astype(np.float64)
        counts = saved["counts"].astype(np.int64)
    if losses.ndim != 2 or losses.shape[1] != 8766:
        raise ValueError(f"unexpected loss matrix shape in {path}: {losses.shape}")
    if len(names) != losses.shape[0] or len(set(names.tolist())) != len(names):
        raise ValueError(f"invalid portfolio axis in {path}")
    if counts.shape != losses.shape or (counts <= 0).any():
        raise ValueError(f"invalid or incomplete cell counts in {path}")
    if not np.isfinite(losses).all():
        raise ValueError(f"nonfinite saved loss in {path}")
    if not np.array_equal(names, np.sort(names)):
        raise ValueError(f"portfolio axis is not sorted in {path}")
    return names, losses, counts


def load_incumbent80(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        names = saved["portfolio"].astype(str)
        horizons = saved["horizon"].astype(np.int64)
        flat = saved["model_0"].astype(np.float64)
    if flat.shape != (80 * 8766,) or names.shape != flat.shape or horizons.shape != flat.shape:
        raise ValueError("incumbent80 cells have an unexpected fixed denominator")
    if not np.isfinite(flat).all():
        raise ValueError("incumbent80 contains a nonfinite score cell")
    expected_names = np.repeat(np.unique(names), 8766)
    expected_horizons = np.tile(np.arange(1, 8767, dtype=np.int64), 80)
    if not np.array_equal(names, expected_names) or not np.array_equal(horizons, expected_horizons):
        raise ValueError("incumbent80 portfolio/horizon ordering is not canonical")
    return names.reshape(80, 8766)[:, 0], flat.reshape(80, 8766)


def canonical_matrix_mean(losses: np.ndarray) -> float:
    """Match persisted matrix score artifacts: mean each portfolio row, then rows."""
    return float(np.mean(np.mean(losses, axis=1)))


def top_entries(names: np.ndarray, improvements: np.ndarray, block_weight: float) -> dict:
    order = np.argsort(improvements)
    def entry(index: int) -> dict:
        value = float(improvements[index])
        return {
            "name": str(names[index]).replace("single_", ""),
            "mean_crps_gain_incumbent_minus_pool": value,
            "contribution_to_equal_50_50_score": float(block_weight * value / len(names)),
        }
    return {
        "pool_wins": int(np.count_nonzero(improvements > 0)),
        "pool_losses": int(np.count_nonzero(improvements < 0)),
        "ties": int(np.count_nonzero(improvements == 0)),
        "largest_gains": [entry(int(i)) for i in order[-5:][::-1]],
        "largest_losses": [entry(int(i)) for i in order[:5]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    reference = args.reference.resolve()
    output = args.output.resolve()
    results = reference / "docs" / "results"
    expanded = results / "expanded132"

    input_paths = {
        "incumbent80_cells": results / "rough-bayesian" / "timescale-only-existing_sv_rate-cells.npz",
        "incumbent80_score": results / "rough-bayesian" / "timescale-only-existing_sv_rate-panel.json",
        "incumbent52_cells": expanded / "incumbent-single52.npz",
        "incumbent52_score": expanded / "incumbent-single52.json",
        "incumbent52_manifest": expanded / "incumbent-single52-manifest.json.gz",
        "pool80_cells": expanded / "pool-original80.npz",
        "pool80_score": expanded / "pool-original80.json",
        "pool80_manifest": expanded / "pool-original80-manifest.json",
        "pool52_cells": expanded / "pool-single52.npz",
        "pool52_score": expanded / "pool-single52.json",
        "pool52_manifest": expanded / "pool-single52-manifest.json.gz",
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required saved evidence: " + ", ".join(missing))

    inc80_names, inc80_loss = load_incumbent80(input_paths["incumbent80_cells"])
    inc52_names, inc52_loss, inc52_counts = load_matrix(input_paths["incumbent52_cells"])
    pool80_names, pool80_loss, pool80_counts = load_matrix(input_paths["pool80_cells"])
    pool52_names, pool52_loss, pool52_counts = load_matrix(input_paths["pool52_cells"])
    if len(inc80_names) != 80 or len(inc52_names) != 52:
        raise ValueError("portfolio counts drifted from 80 + 52 protocol")
    if not np.array_equal(inc80_names, pool80_names):
        raise ValueError("original80 portfolio identities or order differ")
    if not np.array_equal(inc52_names, pool52_names):
        raise ValueError("single52 ticker identities or order differ")
    if not np.array_equal(inc52_counts, pool52_counts):
        raise ValueError("single52 origin counts differ between paired arms")

    score_artifacts = {
        key: read_json(input_paths[key])
        for key in ("incumbent80_score", "incumbent52_score", "pool80_score", "pool52_score")
    }
    inc80_score = float(score_artifacts["incumbent80_score"]["scores"][
        "asset_m256_data_timescales_existing_sv_rate"]["exact_empirical_crps"])
    inc52_score = canonical_matrix_mean(inc52_loss)
    pool80_score = canonical_matrix_mean(pool80_loss)
    pool52_score = canonical_matrix_mean(pool52_loss)
    computed_scores = {
        "incumbent80": inc80_score,
        "incumbent52": inc52_score,
        "pool80": pool80_score,
        "pool52": pool52_score,
    }
    report_scores = {
        "incumbent80": inc80_score,
        "incumbent52": float(score_artifacts["incumbent52_score"]["exact_empirical_crps"]),
        "pool80": float(score_artifacts["pool80_score"]["exact_empirical_crps"]),
        "pool52": float(score_artifacts["pool52_score"]["exact_empirical_crps"]),
    }
    for key in computed_scores:
        if computed_scores[key] != report_scores[key]:
            raise AssertionError(f"canonical NumPy mean differs from saved score for {key}")

    inc80_flat = float(np.mean(inc80_loss.reshape(-1)))
    inc80_rowwise = canonical_matrix_mean(inc80_loss)
    means = {
        "incumbent80": (inc80_flat, float(np.mean(inc80_loss)), inc80_rowwise),
        "incumbent52": (float(np.mean(inc52_loss)), float(np.mean(inc52_loss)), canonical_matrix_mean(inc52_loss)),
        "pool80": (float(np.mean(pool80_loss)), float(np.mean(pool80_loss)), canonical_matrix_mean(pool80_loss)),
        "pool52": (float(np.mean(pool52_loss)), float(np.mean(pool52_loss)), canonical_matrix_mean(pool52_loss)),
    }
    canonical_methods = {
        "incumbent80": "np.mean(model_0) on the source 1D (portfolio,horizon)-ordered array",
        "incumbent52": "np.mean(np.mean(losses, axis=1)) on saved portfolio-by-horizon cells",
        "pool80": "np.mean(np.mean(losses, axis=1)) on saved portfolio-by-horizon cells",
        "pool52": "np.mean(np.mean(losses, axis=1)) on saved portfolio-by-horizon cells",
    }
    score_comparison = {}
    for key, (flat, source_flat, rowwise) in means.items():
        selected = inc80_score if key == "incumbent80" else rowwise
        score_comparison[key] = {
            "saved_score": report_scores[key],
            "canonical_mean_recomputed": selected,
            "canonical_exact_match": bool(selected == report_scores[key]),
            "flat_matrix_np_mean": flat,
            "rowwise_np_mean": rowwise,
            "flat_minus_saved": float(flat - report_scores[key]),
            "rowwise_minus_saved": float(rowwise - report_scores[key]),
            "selected_method": canonical_methods[key],
        }

    def band_summary(lo: int, hi: int) -> dict:
        width = hi - lo + 1
        sl = slice(lo - 1, hi)
        i80 = float(np.mean(inc80_loss[:, sl]))
        p80 = float(np.mean(pool80_loss[:, sl]))
        i52 = float(np.mean(inc52_loss[:, sl]))
        p52 = float(np.mean(pool52_loss[:, sl]))
        weight = width / 8766.0
        contribution80 = 0.5 * weight * (i80 - p80)
        contribution52 = 0.5 * weight * (i52 - p52)
        return {
            "days_inclusive": [lo, hi],
            "cells": {"original80": 80 * width, "single52": 52 * width},
            "within_block_cell_weight": weight,
            "origin_count_support": {
                "single52_min": int(inc52_counts[:, sl].min()),
                "single52_max": int(inc52_counts[:, sl].max()),
                "pool_original80_min": int(pool80_counts[:, sl].min()),
                "pool_original80_max": int(pool80_counts[:, sl].max()),
                "note": "counts are support checks; cell means receive equal weight",
            },
            "original80": {"incumbent": i80, "pool": p80, "gain": i80 - p80,
                           "aggregate_score_contribution": contribution80},
            "single52": {"incumbent": i52, "pool": p52, "gain": i52 - p52,
                         "aggregate_score_contribution": contribution52},
            "total_50_50_aggregate_gain_contribution": contribution80 + contribution52,
        }

    band_rows = [band_summary(lo, hi) for lo, hi in BANDS]
    gain80 = inc80_loss.mean(axis=1) - pool80_loss.mean(axis=1)
    gain52 = inc52_loss.mean(axis=1) - pool52_loss.mean(axis=1)
    aggregate_inc = 0.5 * (inc80_score + inc52_score)
    aggregate_pool = 0.5 * (pool80_score + pool52_score)
    aggregate_gain = aggregate_inc - aggregate_pool

    single_loo = []
    for i, name in enumerate(inc52_names):
        keep = np.arange(len(inc52_names)) != i
        score_i = 0.5 * (inc80_score + float(np.mean(inc52_loss[keep].mean(axis=1))))
        score_p = 0.5 * (pool80_score + float(np.mean(pool52_loss[keep].mean(axis=1))))
        single_loo.append((float(score_i - score_p), str(name)))
    single_loo.sort()

    portfolio_loo = []
    for i, name in enumerate(inc80_names):
        keep = np.arange(len(inc80_names)) != i
        score_i = 0.5 * (float(np.mean(inc80_loss[keep].mean(axis=1))) + inc52_score)
        score_p = 0.5 * (float(np.mean(pool80_loss[keep].mean(axis=1))) + pool52_score)
        portfolio_loo.append((float(score_i - score_p), str(name)))
    portfolio_loo.sort()

    manifests = {
        "incumbent52": read_json(input_paths["incumbent52_manifest"]),
        "pool80": read_json(input_paths["pool80_manifest"]),
        "pool52": read_json(input_paths["pool52_manifest"]),
    }
    data_manifest_hashes = {value.get("data_manifest_sha256") for value in manifests.values()}
    if data_manifest_hashes != {EXPECTED_DATA_MANIFEST}:
        raise AssertionError(f"input data manifest mismatch: {data_manifest_hashes}")
    for key, artifact in score_artifacts.items():
        if key != "incumbent80_score":
            array_key = key.replace("_score", "_cells")
            expected_archive_sha = artifact.get("cells_sha256")
            if expected_archive_sha != sha256(input_paths[array_key]):
                raise AssertionError(f"score report cell checksum mismatch for {key}")

    asset_summary = top_entries(inc52_names, gain52, 0.5)
    asset_summary["SPYSIM"] = {
        "mean_crps_gain_incumbent_minus_pool": float(gain52[np.where(inc52_names == "single_SPYSIM")[0][0]])
    }
    portfolio_summary = top_entries(inc80_names, gain80, 0.5)
    weighted_band_sum = float(sum(row["total_50_50_aggregate_gain_contribution"] for row in band_rows))

    receipt = {
        "receipt_schema": "pool_loss_decomposition_v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "blocks": {"original80": 0.5, "single52": 0.5},
            "horizons": 8766,
            "cell_definition": "portfolio-horizon loss averaged across forecast origins",
            "within_block_weighting": "equal weight per complete portfolio-horizon cell",
            "single52_ticker_order_sha256": hashlib.sha256("\n".join(inc52_names).encode()).hexdigest(),
            "original80_portfolio_order_sha256": hashlib.sha256("\n".join(inc80_names).encode()).hexdigest(),
        },
        "inputs": {
            key: {"path": str(path.relative_to(reference)), "sha256": sha256(path)}
            for key, path in input_paths.items()
        },
        "lineage": {
            "data_manifest_sha256": EXPECTED_DATA_MANIFEST,
            "expanded_adapters": {
                "incumbent52": manifests["incumbent52"].get("adapter"),
                "pool80": manifests["pool80"].get("adapter"),
                "pool52": manifests["pool52"].get("adapter"),
            },
        },
        "canonical_score_checks": score_comparison,
        "overall": {
            "incumbent80": inc80_score,
            "pool80": pool80_score,
            "gain80": inc80_score - pool80_score,
            "incumbent52": inc52_score,
            "pool52": pool52_score,
            "gain52": inc52_score - pool52_score,
            "equal_50_50_incumbent": aggregate_inc,
            "equal_50_50_pool": aggregate_pool,
            "equal_50_50_gain": aggregate_gain,
        },
        "horizon_bands": band_rows,
        "asset_wins_losses_and_contributions": asset_summary,
        "original80_portfolio_wins_losses_and_contributions": portfolio_summary,
        "leave_one_out_sensitivity": {
            "single_asset_out_fixed_original80": {
                "minimum_gain": single_loo[0][0],
                "omitted_at_minimum": single_loo[0][1],
                "maximum_gain": single_loo[-1][0],
                "omitted_at_maximum": single_loo[-1][1],
                "sign_flip": single_loo[0][0] <= 0 <= single_loo[-1][0],
            },
            "original80_portfolio_out_fixed_single52": {
                "minimum_gain": portfolio_loo[0][0],
                "omitted_at_minimum": portfolio_loo[0][1],
                "maximum_gain": portfolio_loo[-1][0],
                "omitted_at_maximum": portfolio_loo[-1][1],
                "sign_flip": portfolio_loo[0][0] <= 0 <= portfolio_loo[-1][0],
            },
        },
        "rounding_and_orientation": {
            "explanation": "The persisted source score reductions use a 1D mean for the historical original80 archive and rowwise means followed by a mean for the expanded matrix reports. Flattening a matrix and averaging changes only reduction order and can differ by a few floating-point ulps.",
            "matrix_flattening_is_not_used_to_replace_saved_scores": True,
            "sum_of_band_contributions": weighted_band_sum,
            "aggregate_gain_minus_band_sum": aggregate_gain - weighted_band_sum,
        },
        "helper_sha256": sha256(Path(__file__).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "helper_sha256": receipt["helper_sha256"],
        "scores_match_saved_reports": all(v["canonical_exact_match"] for v in score_comparison.values()),
        "aggregate_gain": aggregate_gain,
        "band_sum_residual": receipt["rounding_and_orientation"]["aggregate_gain_minus_band_sum"],
        "asset_wins_losses": [asset_summary["pool_wins"], asset_summary["pool_losses"]],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
