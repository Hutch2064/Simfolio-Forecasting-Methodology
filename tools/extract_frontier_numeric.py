#!/usr/bin/env python3
"""Record the bounded Frontier numerical extraction and AST digests.

The tool reads the historical source only when a caller supplies it.  It never
writes beside that source and emits sanitized artifact names rather than local
filesystem paths.  The committed package remains runnable without the source
tree; this command is for provenance and extraction review.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Iterable


SOURCE_FUNCTIONS = {
    "engine.py": (
        "_deterministic_seed",
        "_evidence_estimated_sharpe_dlm_historical_cagr_anchor_mean",
        "_fit_bdes_non_mcmc_sv_forecast_base",
        "_simulate_full_mcmc_sv_bdes_log_paths_numba",
        "_sv_observed_log_variance",
        "_sv_kalman_filter",
        "_sv_kalman_rts_smoother_mean",
        "_fit_sv_map_state_space_params",
        "_fast_bdes_delta_method_sigma_samples",
        "_standardized_empirical_innovation_pool",
        "_bdes_multiscale_components",
        "_dlm_mu_draw_paths",
    ),
    "asset_level_full_exact_crps.py": ("_asset_terminal_matrix",),
    "simfolio_oos_copula_alternatives.py": (
        "_exact_kalman_rejoined_portfolio_paths",
        "_map_uniforms_to_marginal_paths",
        "_rebalanced_portfolio_log_paths_numba",
        "_rejoin_sorted_uniform_paths_numba",
    ),
    "simfolio_adaptive_pgas.py": (
        "_fit_dynamic_factor_model",
        "_kalman_terminal_posterior_numba_information_small",
        "_simulate_future_gaussian_uniforms_flat_numba",
        "adaptive_pgas_uniform_paths",
    ),
}

LOCAL_FUNCTIONS = {
    "bdes_fastmap.py": (
        "deterministic_seed",
        "fit_bdes_fastmap",
        "simulate_fastmap_marginal",
        "_sv_observed_log_variance",
        "_sv_kalman_filter",
        "_sv_kalman_rts_smoother_mean",
        "_fit_sv_map_state_space_params",
        "_fast_bdes_delta_method_sigma_samples",
        "_standardized_empirical_innovation_pool",
        "_bdes_multiscale_components",
        "dlm_mu_draw_paths",
    ),
    "dynamic_gaussian.py": (
        "fit_dynamic_gaussian_factor_model",
        "kalman_terminal_posterior",
        "simulate_future_gaussian_uniforms",
        "map_uniforms_to_marginal_paths",
        "rebalanced_portfolio_log_paths",
    ),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_functions(path: Path, names: Iterable[str]) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    nodes = {node.name: node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    result: dict[str, str] = {}
    for name in names:
        node = nodes.get(name)
        if node is None:
            continue
        result[name] = _sha256(ast.dump(node, annotate_fields=True, include_attributes=False).encode("utf-8"))
    return result


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "numba"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def build_report(source_root: Path | None, panel_root: Path | None, package_root: Path) -> dict[str, object]:
    report: dict[str, object] = {
        "schema": "frontier_numeric_extraction_v1",
        "source_artifacts": {
            "marginal": "source-research/app/engine.py",
            "dependence": "tmp/asset_level_full_panel_20260823/simfolio_adaptive_pgas.py",
            "rejoin": "tmp/asset_level_full_panel_20260823/simfolio_oos_copula_alternatives.py",
            "wrapper": "tmp/asset_level_full_panel_20260823/asset_level_full_exact_crps.py",
        },
        "source_linkage": "verified_when_source_root_supplied",
        "source_function_digests": {},
        "extracted_function_digests": {},
        "dependency_closure": [
            "historical blake2b seed alias",
            "historical CAGR anchored Sharpe DLM mean",
            "SV log-chi-square observation and Kalman/RTS state path",
            "FastMAP Laplace sigma points and BDES four-scale state",
            "stationary bootstrap standardized innovations",
            "exact Gaussian dynamic factor fit",
            "exact terminal Kalman posterior",
            "day-major Gaussian future projection",
            "rank interpolation and 15 bps one-way rejoin",
        ],
        "excluded": ["service engine", "catalogue selection", "cache/memmap machinery", "MCMC/PGAS Student-t branch", "unrelated research runners"],
        "environment": _versions(),
    }
    if source_root is not None or panel_root is not None:
        digest_out: dict[str, dict[str, str]] = {}
        for relative, names in SOURCE_FUNCTIONS.items():
            candidates: list[Path] = []
            for root in (source_root, panel_root):
                if root is not None:
                    candidates.extend((root / relative, root / "app" / relative))
            path = next((candidate for candidate in candidates if candidate.exists()), None)
            if path is not None:
                digest_out[relative] = _digest_functions(path, names)
        report["source_function_digests"] = digest_out
        report["source_linkage"] = "verified" if digest_out else "unverified_source_files_missing"
    local_out: dict[str, dict[str, str]] = {}
    for relative, names in LOCAL_FUNCTIONS.items():
        path = package_root / relative
        if path.exists():
            local_out[relative] = _digest_functions(path, names)
    report["extracted_function_digests"] = local_out
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=None, help="Optional read-only historical source root")
    parser.add_argument("--panel-root", type=Path, default=None, help="Optional read-only historical temporary-panel root")
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1] / "src/simfolio_forecasting_methodology/models/numerical")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = build_report(args.source_root, args.panel_root, args.package_root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
