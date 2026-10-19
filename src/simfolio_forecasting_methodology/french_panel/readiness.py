"""Bounded native readiness probes and research-compute extrapolation.

This command never launches the full panel or produces a panel ranking. Each
model is profiled in its own interpreter because native sources install hooks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from ..seeds import coherent_daily_seed
from .data import SIMULATIONS, load_dates, load_panel, load_returns, schedule
from .runner import _atomic_json, _forecast, dependency_identity
from .selected import FACTORIES, build_selected, implementation_digest


def probe_plan(origins) -> list[tuple[object, int]]:
    """Three rolling-history strata and every long temporal horizon, unmodified."""
    rolling = [origin for origin in origins if origin.evaluation_split == "rolling_origin"]
    temporal = [origin for origin in origins if origin.evaluation_split != "rolling_origin"]
    groups = np.array_split(np.arange(len(rolling)), 3)
    return [(rolling[int(group[len(group) // 2])], len(group)) for group in groups if len(group)] + [
        (origin, 1) for origin in temporal]


def project_runtime(records: list[dict], series_count: int) -> dict:
    measured = [row for row in records if row["role"] == "projection_probe"]
    if not measured or any(row["status"] != "passed" for row in measured):
        return {"status": "incomplete_no_runtime_estimate"}
    times = np.array([row["forecast_and_score_seconds"] for row in measured])
    weights = np.array([row["represented_origins"] for row in measured])
    return {
        "status": "bounded_extrapolation_not_full_panel_measurement",
        "worker_seconds": float(series_count * np.dot(times, weights)),
        "mean_origin_extrapolation_worker_seconds": float(series_count * weights.sum() * times.mean()),
        "median_origin_extrapolation_worker_seconds": float(series_count * weights.sum() * np.median(times)),
        "series_count": series_count, "origins_per_series": int(weights.sum()),
        "coverage": "one broad-market series; three rolling history strata and all temporal origins",
        "limitations": "asset-specific fit costs, contention and checkpoint IO remain unmeasured",
    }


def profile_model(model_id: str, panel_root: Path, output: Path) -> dict:
    """Warm-up, six actual origins and industry; all 240 paths and full horizons."""
    started = perf_counter()
    panel = load_panel(panel_root)
    values, dates, origins = load_returns(panel_root), load_dates(panel_root), schedule(panel_root)
    market = next(i for i, row in enumerate(panel["series"]) if row["family"] == "broad_market")
    industry = next(i for i, row in enumerate(panel["series"]) if row["family"] == "industries")
    plan = [(market, origins[0], 0, "native_warmup")]
    plan.extend((market, origin, count, "projection_probe") for origin, count in probe_plan(origins))
    for origin in (origins[0], max(origins, key=lambda origin: origin.max_horizon)):
        plan.append((industry, origin, 0, "industry_smoke"))
    identity = implementation_digest(model_id)
    model = build_selected(model_id)
    records = []
    payload = {
        "model_id": model_id, "implementation_digest": identity,
        "panel_fingerprint": panel["panel_fingerprint"], "simulations": SIMULATIONS,
        "full_sweep_started": False, "records": records,
        "python": sys.version, "platform": sys.platform,
        "runtime": dependency_identity(model),
    }
    for index, (series, origin, represented, role) in enumerate(plan):
        row = {"series_id": panel["series"][series]["series_id"],
               "origin_label": origin.origin_label, "training_observations": origin.position + 1,
               "horizon": origin.max_horizon, "represented_origins": represented, "role": role,
               "first_call_includes_lazy_native_install": index == 0}
        try:
            losses, diagnostics = _forecast(model, row["series_id"], values[series], dates, origin,
                coherent_daily_seed(origin.origin_date, origin.max_horizon, SIMULATIONS))
            row.update(diagnostics)
            row.update(status="passed", loss_sha256=hashlib.sha256(losses.tobytes()).hexdigest())
            # Loss vectors are readiness evidence only, never panel checkpoints.
            np.save(output.parent / f"{output.stem}-{index}.npy", losses, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001 - isolate native probe failures and retain evidence
            row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        records.append(row)
        payload["projection"] = project_runtime(records, panel["series_count"])
        payload["elapsed_seconds"] = perf_counter() - started
        payload["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024)
        _atomic_json(output, payload)
    payload["status"] = "passed" if all(row["status"] == "passed" for row in records) else "failed"
    _atomic_json(output, payload)
    return payload


def summarize(profiles: list[dict], workers: int = 12) -> dict:
    ids = [row["model_id"] for row in profiles]
    if len(ids) != len(set(ids)) or set(ids) != set(FACTORIES):
        raise ValueError("runtime estimate requires exactly the fifteen selected profiles")
    if any(row["projection"]["status"] == "incomplete_no_runtime_estimate" for row in profiles):
        return {"status": "blocked_by_failed_readiness", "full_sweep_started": False}
    slots = [0.0] * workers
    for row in sorted(profiles, key=lambda row: row["projection"]["worker_seconds"], reverse=True):
        slot = min(range(workers), key=slots.__getitem__)
        slots[slot] += row["projection"]["worker_seconds"]
    return {
        "status": "bounded_extrapolation", "full_sweep_started": False,
        "native_readiness_passed": all(row.get("status") == "passed" for row in profiles),
        "models_with_probe_failures": [row["model_id"] for row in profiles if row.get("status") != "passed"],
        "methods": len(profiles), "workers": workers,
        "total_worker_hours": sum(row["projection"]["worker_seconds"] for row in profiles) / 3600,
        "projected_elapsed_hours": max(slots) / 3600,
        "scheduling": "one native model per worker, longest estimated model first",
        "maximum_observed_worker_rss_bytes": max(row["peak_rss_bytes"] for row in profiles),
        "limitations": "single-series stratified extrapolation; no parallel contention or full-sweep timing",
        "models": {row["model_id"]: row["projection"] for row in profiles},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", choices=FACTORIES)
    parser.add_argument("panel_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = profile_model(args.model_id, args.panel_root, args.output)
    print(json.dumps({key: result[key] for key in ("model_id", "status", "projection")}, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
