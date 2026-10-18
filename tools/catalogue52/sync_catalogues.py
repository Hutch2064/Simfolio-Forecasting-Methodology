"""Mirror full single52 results without editing catalogue definitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from simfolio_forecasting_methodology.catalogue import load_canonical_models
from simfolio_forecasting_methodology.single_asset_results import (
    DENOMINATORS,
    INDEX_RESOURCE,
    PROTOCOL,
    validate_single52_index,
)


def make_index(report_bytes):
    """Keep completed rows and their separate archive state; omit progress."""
    report = json.loads(report_bytes)
    if (report.get("protocol") != PROTOCOL or report.get("partial_results_ranked") is not False
            or report.get("blended_score_used") is not False
            or report.get("ranking_metric") != "single52_crps_only_lower_is_better"):
        raise ValueError("unqualified single52 source report")
    catalogue = load_canonical_models()
    known = {row["public_model_id"]: row for row in catalogue}
    models, seen = {}, set()
    for row in report["full_scores"]:
        model_id = row["model_id"]
        if model_id in seen:
            raise ValueError("duplicate immutable model ID in single52 report")
        seen.add(model_id)
        if model_id not in known:
            raise ValueError("single52 report model is absent from canonical catalogue")
        if any(row.get(key) != value for key, value in DENOMINATORS.items()):
            raise ValueError("partial single52 result cannot enter the result index")
        models[model_id] = {**row,
                           "specification_fingerprint": known[model_id]["specification_fingerprint"]}
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ranking_metric": report["ranking_metric"],
        "historical80_role": report["historical80_role"],
        "blended_score_used": False,
        "partial_results_ranked": False,
        "source_report": "docs/results/catalogue52/scores.json",
        "source_repository": "https://github.com/Hutch2064/Simfolio-Forecasting-Methodology",
        "source_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "inventory_sha256": report["inventory_sha256"],
        "models": dict(sorted(models.items())),
    }
    validate_single52_index(payload, catalogue=catalogue)
    return payload


def sync_catalogues(report_path, master_root, *, methodology_root=ROOT):
    """Validate all rows and master membership before replacing either sidecar."""
    payload = make_index(Path(report_path).read_bytes())
    master_root = Path(master_root)
    # Read membership only: master statistical metadata and fingerprints are
    # deliberately never merged with result metadata.
    import csv

    known = {}
    for name in ("canonical_175_part1.csv", "canonical_175_part2.csv"):
        with (master_root / "catalogs" / name).open(encoding="utf-8", newline="") as handle:
            known.update({row["model_id"]: int(row["canonical_rank"]) for row in csv.DictReader(handle)})
    if any(known.get(model_id) != row["canonical_rank"] for model_id, row in payload["models"].items()):
        raise ValueError("master catalogue identity or canonical rank mismatch")
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    destinations = (
        Path(methodology_root) / "src/simfolio_forecasting_methodology" / INDEX_RESOURCE,
        master_root / "catalogs/single52_results.json",
    )
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=ROOT / "docs/results/catalogue52/scores.json")
    parser.add_argument("--master-root", type=Path, required=True)
    args = parser.parse_args()
    index = sync_catalogues(args.report, args.master_root)
    print(json.dumps({"completed_models": len(index["models"]),
                      "source_report_sha256": index["source_report_sha256"]}))
