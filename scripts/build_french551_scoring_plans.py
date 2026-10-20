"""Build predeclared French551 v2 scoring artifacts using training prefixes only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np

from simfolio_forecasting_methodology.french_panel import runner as original
from french_score_sidecar import SCHEMA, ScoringPlan, sha256, training_standard_deviations

PRIMARY = "french551-training-reference-scaled-loghorizon-crps-v2"
SENSITIVITY = "french551-training-volatility-scaled-loghorizon-crps-v2-sensitivity"


def build(panel_root, output_root):
    panel_root, output_root = Path(panel_root), Path(output_root)
    panel = original.load_panel(panel_root)
    origins = tuple(original.schedule(panel_root))
    values = original.load_returns(panel_root)
    counts = np.array([o.position + 1 for o in origins], dtype=np.int64)
    if np.any(counts < 2):
        raise ValueError("sample standard deviation requires at least two training observations")
    standard_deviations = training_standard_deviations(values, origins, ddof=1)
    capacity = max(o.max_horizon for o in origins)
    horizons = np.arange(1, capacity + 1, dtype=np.float64)
    weights = np.log1p(1.0 / horizons)
    expected_counts = np.zeros(capacity, dtype=np.int64)
    for o in origins:
        expected_counts[:o.max_horizon] += 1
    common_arrays = {
        "training_scale": standard_deviations,
        "horizon_weight": weights,
        "series_weight": np.ones(len(panel["series"]), dtype="<f8"),
    }
    definitions = {
        PRIMARY: {
            "role": "primary",
            "methodology": "French551 training-reference-scaled loghorizon CRPS v2",
            "normalized_origin_loss": "rawCRPS_ioh / (s_io * sqrt(h + h*h/n_io))",
            "horizon_scale": np.ones(capacity, dtype="<f8"),
            "origin_horizon_scale": np.sqrt(horizons[None, :] + horizons[None, :] ** 2 / counts[:, None]),
            "assumption": "Unknown-mean IID reference scale combines horizon noise h*s^2 and estimated-mean uncertainty h^2*s^2/n. It is a common training-only scoring reference, not a claim that any forecast process is IID.",
        },
        SENSITIVITY: {
            "role": "predeclared_sensitivity",
            "methodology": "French551 training-volatility-scaled loghorizon CRPS v2 sensitivity",
            "normalized_origin_loss": "rawCRPS_ioh / (s_io * sqrt(h))",
            "horizon_scale": np.sqrt(horizons),
            "assumption": "Known-mean IID diffusion reference scale h*s^2, retained as a predeclared sensitivity; not a forecast-process assertion.",
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    methodology_path = Path(__file__).with_name("METHODOLOGY.md")
    source_identity = {"methodology_document_sha256": sha256(methodology_path),
                       "reporter": sha256(Path(__file__).with_name("french_score_sidecar.py")),
                       "plan_builder": sha256(__file__),
                       "original_source_closure": original._source_closure()}
    manifests = {}
    for version, definition in definitions.items():
        root = output_root / version
        root.mkdir(exist_ok=True)
        arrays = {**common_arrays, "horizon_scale": definition["horizon_scale"]}
        if "origin_horizon_scale" in definition:
            arrays["origin_horizon_scale"] = definition["origin_horizon_scale"]
        artifact = root / "scales-and-weights.npz"
        declaration_path = root / "scoring_plan.json"
        if declaration_path.exists() or artifact.exists():
            raise ValueError("versioned plan outputs already exist; never silently replace a frozen protocol")
        np.savez(artifact, **arrays)
        declaration = {
            "schema": SCHEMA, "approval_status": "approved", "score_version": version,
            "role": definition["role"], "methodology": definition["methodology"],
            "panel_fingerprint": panel["panel_fingerprint"],
            "arrays_file": artifact.name, "arrays_file_sha256": sha256(artifact),
            "formula_origin": "Application-level benchmark choice declared before examining new-score ranks; the cited papers support principles and do not prescribe this combined formula.",
            "references": [
                "https://doi.org/10.1198/016214506000001437",
                "https://doi.org/10.1214/22-STS864",
                "https://arxiv.org/html/1912.05642",
                "https://doi.org/10.1016/j.econlet.2006.04.010",
                "https://doi.org/10.1016/j.ijforecast.2006.03.001",
                "https://ascmo.copernicus.org/articles/11/23/2025/index.html",
            ],
            "formula": {
                "origin_loss": definition["normalized_origin_loss"],
                "s_io": "sample standard deviation (ddof=1) of observed daily log returns through origin.position inclusive",
                "n_io": "actual observed training prefix length = origin.position + 1",
                "cell_mean": "arithmetic mean normalized loss over all original origins eligible for series/horizon cell",
                "horizon_weight": "w_h = log1p(1/h), h=1..H; normalize by sum(w_h)=log(H+1)",
                "series_mean": "equal arithmetic mean across the 551 independently scored return series",
                "completeness": "all 51 original origins per series before that series enters either aggregate or matching comparison",
            },
            "counts": {"series": len(panel["series"]), "origins_per_series": len(origins),
                       "expected_origins_per_model": len(panel["series"])*len(origins),
                       "maximum_horizon": capacity, "expected_cells": len(panel["series"])*capacity},
            "expected_origin_counts_by_horizon_sha256": hashlib.sha256(expected_counts.astype('<i8').tobytes()).hexdigest(),
            "horizon_weight_sum": float(np.sum(weights)),
            "horizon_weight_sum_identity": "log(H+1)",
            "horizon_weight_normalization": "log(H+1)",
            "scale_information_set": "training_observations_through_origin_only",
            "weight_selection": "fixed_ex_ante_independent_of_model_results",
            "zero_variance_policy": "fail closed on zero/nonfinite sample scale; no epsilon, dropping, clipping or model-specific adjustment",
            "proper_conditional_score_argument": "Conditional on training data, every normalizer is a fixed strictly positive number and horizon/series weights are fixed nonnegative values independent of forecasts and held-out observations. Positive rescaling and averaging preserve CRPS propriety conditional on that information set. This says nothing about power, dependence or an optimal decision-weighting scheme.",
            "reference_scale_assumption": definition["assumption"],
            "inputs": {"files": panel.get("files", {}), "origins": [asdict(o) for o in origins]},
            "source_identity": source_identity,
            "checkpoint_identity": "original execution fingerprint/task identity retained unchanged; derived score identity binds original execution fingerprint plus this plan fingerprint",
            "legacy_raw_crps": "unchanged diagnostic in original summaries; no old scores silently relabeled",
            "forecast_reruns_required": 0,
        }
        plan = ScoringPlan(declaration, arrays, origins, panel)
        original._atomic_json(declaration_path, declaration)
        manifests[version] = {"plan_path": str(declaration_path), "plan_fingerprint": plan.fingerprint,
                              "role": definition["role"]}
    original._atomic_json(output_root / "score_protocol.json", {
        "canonical_score_version": PRIMARY, "sensitivity_score_version": SENSITIVITY,
        "definitions": manifests, "panel_fingerprint": panel["panel_fingerprint"],
        "original_checkpoints_and_raw_scores": "read only and unchanged",
        "forecast_reruns": 0,
    })
    return manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    build(args.panel_root, args.output_root)


if __name__ == '__main__':
    main()
