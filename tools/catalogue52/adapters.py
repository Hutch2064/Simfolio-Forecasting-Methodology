"""Resolve catalogue identities to their retained executable implementations.

This module changes no statistical specification or seed. Each adapter must be
loaded in its own process: historical experiments patch private module globals.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "src/simfolio_forecasting_methodology/resources/canonical_175/ledger.json"
EXCLUDED_RANKS = frozenset({257, 258, 259, 260})


def model_specs():
    rows = json.loads(LEDGER.read_text())["models"]
    result = []
    for row in rows:
        rank = row["canonical_rank"]
        if rank in EXCLUDED_RANKS:
            continue
        environment = {}
        if rank in {1, 8}:
            adapter = "tools/catalogue52/legacy_exact_kernels.py"
        elif rank <= 175 or rank == 177:
            adapter = None
        elif rank in {176, 178, 179}:
            # Recovered historical source is wired independently; do not
            # substitute another model when original implementation is absent.
            adapter = "tools/catalogue52/legacy_admission.py"
        elif 180 <= rank <= 183:
            adapter = "tools/catalogue52/single_asset_bridge.py"
            environment["SIMFOLIO_CATALOGUE_ADAPTER_SOURCE"] = "tools/rough_jump_vine/models.py"
        elif 184 <= rank <= 192:
            # overlay.py/dynamic.py define the original classes but have no
            # candidate registry. The original suite constructs those exact
            # classes with the retained hyperparameters here.
            adapter = "tools/catalogue52/single_asset_bridge.py"
            environment["SIMFOLIO_CATALOGUE_ADAPTER_SOURCE"] = "tools/rough_bayesian/models.py"
        else:
            adapter = row["source_location_reference"]["implementation"]
        if rank == 261:
            environment["SIMFOLIO_TIMESCALE_VARIANT"] = "existing_sv_rate"
        if adapter is not None and not (ROOT / adapter).is_file():
            available = False
        else:
            available = True
        result.append({
            "canonical_rank": rank,
            "model_id": row["public_model_id"],
            "adapter": adapter,
            "environment": environment,
            "adapter_file_available": available,
            "catalogue_specification_fingerprint": row["specification_fingerprint"],
            "historical_source_revision": row["source_revision"],
        })
    return result


if __name__ == "__main__":
    print(json.dumps(model_specs(), indent=2, sort_keys=True))
