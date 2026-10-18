"""Prospective 50/100-year CAGR coherence audit; never an OOS score."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import warnings

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "execution-ou-pool"
OUTPUT = BASE / "expanded/long-cagr-ou"
ADAPTERS = {
    "ou_pool": ("tools/expanded_ou_pool/models.py", "equal_path_pool_stationary_ou_and_bayesian_ewma_sbb"),
}


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_model(name):
    import numpy as np
    import pandas as pd
    from scipy.special import logsumexp

    sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "tools/mcmc_runtime")]
    from simfolio_forecasting_methodology.data import load_canonical_engine_inputs
    from simfolio_forecasting_methodology.panel import load_scored52_portfolio_panel
    from simfolio_forecasting_methodology.runner import ForecastContext, PortfolioPolicy, TrainingData
    from simfolio_forecasting_methodology.seeds import coherent_daily_seed

    OUTPUT.mkdir(parents=True, exist_ok=True)
    adapter, model_id = ADAPTERS[name]
    os.environ["SIMFOLIO_TIMESCALE_VARIANT"] = "existing_sv_rate"
    path = SOURCE / adapter
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("long_cagr_selected_adapter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.initialize(None, 1, 1)
    candidate = next(c for c in module.CANDIDATES if c.model_id == model_id)
    assets, portfolios = load_canonical_engine_inputs(BASE / "data")
    origin = assets.index[-1].date().isoformat()
    panel = load_scored52_portfolio_panel()
    selected = [next(p for p in panel if p.rebalance == frequency)
                for frequency in ("none", "monthly", "quarterly", "annually")]
    cases = []
    pilots = ["SPYSIM", "SHYSIM", "SPXUSIM", "GLDSIM"]
    for ticker in pilots + [t for t in assets.columns if t not in pilots]:
        cases.append(("single_" + ticker, "single_asset", (ticker,), (1.0,), "none", assets[ticker]))
    for p in selected:
        cases.append((p.name, "original_portfolio", p.tickers, p.weights, p.rebalance, portfolios[p.name]))
    assert len(cases) == 56 and len(assets) == 11687
    source_hashes = {str(p.relative_to(SOURCE)): digest(p)
                     for folder in (SOURCE / "src", SOURCE / "tools")
                     for p in sorted(folder.rglob("*")) if p.suffix in (".py", ".cpp")}
    manifest = {
        "scope": "prospective coherence audit beyond available OOS history; not a backtested score or fitted objective",
        "model": name, "model_id": model_id, "adapter": adapter,
        "source_revision": subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": source_hashes, "audit_script_sha256": digest(Path(__file__)),
        "data_manifest_sha256": digest(BASE / "data/canonical_data_manifest.json"),
        "python": platform.python_version(), "packages": {p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "numba", "llvmlite", "pandas", "pyvinecopulib")},
        "training_start": str(assets.index[0].date()), "training_end": origin,
        "training_rows": len(assets), "paths": 240, "horizon_days": [12600, 25200],
        "portfolio_cases": [{"name": p.name, "tickers": p.tickers, "weights": p.weights, "rebalance": p.rebalance}
                            for p in selected],
        "annualization_days": 252, "future_policy_calendar": "pd.bdate_range after last training date",
        "seed": "coherent_daily_seed(origin_date,horizon_days,240); unchanged candidate-specific RNG contracts",
        "retention": "only terminal daily-log-return sums, no full paths; active campaign owner retains for archival",
    }
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    atomic(OUTPUT / (name + "-manifest.json"), manifest)
    records = []
    for case, group, tickers, weights, frequency, historical in cases:
        training = TrainingData(
            np.ascontiguousarray(historical.to_numpy(dtype=np.float64)),
            np.ascontiguousarray(assets.loc[:, list(tickers)].to_numpy(dtype=np.float64)),
            PortfolioPolicy(tuple(tickers), tuple(weights), frequency),
            assets.index.to_numpy(dtype="datetime64[ns]"),
        )
        training.validate()
        hist_cagr = float(np.expm1(252.0 * training.portfolio_log_returns.mean()))
        for years in (50, 100):
            horizon = years * 252
            key = f"{name}--{case}--{years}y"
            future = pd.bdate_range(assets.index[-1] + pd.offsets.BDay(1), periods=horizon)
            seed = coherent_daily_seed(origin, horizon, 240)
            context = ForecastContext(model_id, case, "full_history_prospective", horizon, 240,
                                      seed, future.to_numpy(dtype="datetime64[ns]"), origin)
            module.clear_path_cache()
            module.FIT_DIAGNOSTICS.clear()
            module.TIMINGS.clear()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    paths = np.asarray(candidate.simulate_daily_log_returns(training, context), dtype=np.float64)
                overflow = [str(w.message) for w in caught if "overflow" in str(w.message).lower()]
                if overflow:
                    raise ArithmeticError("model numerical overflow: " + "; ".join(overflow))
                if paths.shape != (240, horizon) or not np.isfinite(paths).all():
                    raise ArithmeticError("malformed or nonfinite candidate daily paths")
                terminal = np.sum(paths, axis=1, dtype=np.float64)
                daily_range = [float(paths.min()), float(paths.max())]
                del paths
                with np.errstate(over="raise", invalid="raise"):
                    path_cagrs = np.expm1(252.0 * terminal / horizon)
                    log_mean_ending_nav = float(logsumexp(terminal) - np.log(240.0))
                    mean_nav_cagr = float(np.expm1(252.0 * log_mean_ending_nav / horizon))
                if not np.isfinite(terminal).all() or not np.isfinite(path_cagrs).all() or not np.isfinite(mean_nav_cagr):
                    raise ArithmeticError("nonfinite terminal CAGR statistic")
            except Exception as error:
                atomic(OUTPUT / (name + "-failure.json"), {
                    "case": case, "years": years, "error": type(error).__name__, "message": str(error),
                    "completed_cases": len(records), "stopped_without_extra_clipping": True,
                })
                raise
            values = {
                "median_path_cagr": float(np.median(path_cagrs)),
                "mean_path_cagr": float(path_cagrs.mean()),
                "cagr_of_mean_ending_nav": mean_nav_cagr,
            }
            record = {
                "case": case, "group": group, "model": name, "model_id": model_id,
                "years": years, "horizon_days": horizon, "paths": 240, "seed": seed,
                "policy": asdict(training.policy), "historical_cagr": hist_cagr,
                **values, "differences_from_historical_cagr": {k: v - hist_cagr for k, v in values.items()},
                "ratios_to_positive_historical_cagr": {k: v / hist_cagr for k, v in values.items()} if hist_cagr > 0 else None,
                "historical_sign": "positive" if hist_cagr > 0 else ("negative" if hist_cagr < 0 else "zero"),
                "terminal_log_nav_quantiles": np.quantile(terminal, [0, .05, .5, .95, 1]).tolist(),
                "log_mean_ending_nav": log_mean_ending_nav, "daily_log_return_range": daily_range,
                "warnings": [str(w.message) for w in caught], "manifest_sha256": manifest_hash,
            }
            archive = OUTPUT / (key + ".npz")
            np.savez_compressed(archive, terminal_log_nav=terminal)
            record["terminal_file_sha256"] = digest(archive)
            atomic(OUTPUT / (key + ".json"), record)
            records.append(record)
            atomic(OUTPUT / (name + "-progress.json"), {"completed": len(records), "required": 112,
                                                        "last_case": case, "last_years": years})
            print(json.dumps({"model": name, "completed": len(records), "case": case, "years": years,
                              "historical_cagr": hist_cagr, **values}), flush=True)
            module.clear_path_cache()
    counts = {}
    for group in ("single_asset", "original_portfolio"):
        for years in (50, 100):
            rows = [r for r in records if r["group"] == group and r["years"] == years]
            positive = [r for r in rows if r["historical_cagr"] > 0]
            negative = [r for r in rows if r["historical_cagr"] < 0]
            counts[f"{group}_{years}y"] = {
                "cases": len(rows), "positive_historical": len(positive), "negative_historical": len(negative),
                "positive_history_diagnostic_counts": {
                    stat: {"ratio_below_0.5": sum(r[stat] < .5 * r["historical_cagr"] for r in positive),
                           "ratio_below_0.75": sum(r[stat] < .75 * r["historical_cagr"] for r in positive),
                           "forecast_nonpositive": sum(r[stat] <= 0 for r in positive)}
                    for stat in values},
                "negative_history_cases": [{"case": r["case"], "historical_cagr": r["historical_cagr"],
                                            **{stat: r[stat] for stat in values}} for r in negative],
            }
    assert len(records) == 112
    assert all(digest(SOURCE / p) == h for p, h in source_hashes.items())
    summary = {"model": name, "model_id": model_id, "scope": manifest["scope"], "manifest_sha256": manifest_hash,
               "completed_cases": len(records), "all_paths_finite": True, "extra_clipping": False,
               "counts": counts, "records": records}
    atomic(OUTPUT / (name + "-summary.json"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=ADAPTERS)
    args = parser.parse_args()
    if args.model:
        run_model(args.model)
        return
    environment = dict(os.environ)
    environment.update({"NUMBA_CACHE_DIR": str(BASE / "cache312"), "TMPDIR": str(BASE / "native-tmp"),
                        "NUMBA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"})
    for name in ADAPTERS:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--model", name],
                       env=environment, cwd=SOURCE, check=True)
    summaries = {name: json.loads((OUTPUT / (name + "-summary.json")).read_text()) for name in ADAPTERS}
    files = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name != "audit-receipt.json"}
    atomic(OUTPUT / "audit-receipt.json", {"scope": "prospective diagnostic; no OOS scoring or runtime claim",
        "completed_cases": sum(s["completed_cases"] for s in summaries.values()),
        "models": {k: {"manifest_sha256": v["manifest_sha256"], "counts": v["counts"]} for k, v in summaries.items()},
        "retained_bytes": sum(v["bytes"] for v in files.values()), "files": files,
        "owner": "parent research campaign; preserve until archival and explicit release", "passed": True})


if __name__ == "__main__":
    main()
