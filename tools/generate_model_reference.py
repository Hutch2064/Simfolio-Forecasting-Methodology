#!/usr/bin/env python3
"""Generate the public canonical-175 reference from the authoritative ledger."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from simfolio_forecasting_methodology.catalogue import (
    load_canonical_ledger,
    validate_canonical_ledger,
)

OUTPUT_PATH = REPOSITORY_ROOT / "docs" / "canonical-model-reference.md"


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_reference(payload: dict) -> str:
    """Render only the 175 records present in the validated ledger."""

    validate_canonical_ledger(payload)
    membership = payload["membership"]
    precision = payload["score_evidence"]
    identity = payload["identity_policy"]
    models = payload["models"]

    lines = [
        "# Canonical 175 model reference",
        "",
        "This file is generated from `src/simfolio_forecasting_methodology/resources/canonical_175/ledger.json`.",
        "The ledger is the sole membership authority for this public set.",
        "",
        f"- Membership: **{membership['count']}** rows, canonical ranks **{membership['canonical_rank_range'][0]}–{membership['canonical_rank_range'][1]}**.",
        f"- Retained source ranks: **{membership['source_rank_range'][0]}–{membership['source_rank_range'][1]}**.",
        f"- Membership digest: `{membership['membership_digest']}`.",
        f"- Retained score token: `{precision['source_score_token_policy']}`.",
        f"- Publication score token: `{precision['publication_score_policy']}`.",
        f"- Confirmed full statistical specifications: **{identity['full_statistical_specifications_confirmed']}**.",
        "- Rows retain source score evidence while implementation, numerical defaults, dependency closure, and parity are tracked by per-row verification flags.",
        "- Protocol, frozen data, and scored-panel fingerprints are recorded in the ledger. Their exact linkage to retained historical scores remains qualified by the documented source discrepancies.",
        "",
        "| Canonical | Historical | Public model ID | Display name | Family | Retained score token | Publication score token | Specification recovered | Status |",
        "| ---: | ---: | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for model in models:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(model["canonical_rank"]),
                    str(model["historical_rank"]),
                    _cell(model["public_model_id"]),
                    _cell(model["display_name"]),
                    _cell(model["model_family"]),
                    model["historical_score"]["exact_empirical_crps"],
                    model["score_precision"]["publication_token"],
                    str(model["specification_recovered"]).lower(),
                    _cell(model["verification_status"]),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--check", action="store_true", help="fail if the generated file is not current")
    args = parser.parse_args(argv)

    rendered = render_reference(load_canonical_ledger())
    if args.check:
        try:
            current = args.output.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"reference is missing: {args.output}", file=sys.stderr)
            return 1
        if current != rendered:
            print(f"reference is stale: {args.output}", file=sys.stderr)
            return 1
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
