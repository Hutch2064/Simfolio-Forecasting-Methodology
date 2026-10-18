"""Same evaluator with source-bound, shared exact-fit memoization.

Only completed deterministic fits are reused; simulation/scoring are unchanged.
The inherited cache provides per-key file locks and atomic writes. Namespace
includes the entire source closure, selected adapter and model. The campaign
owns this cache, bounds its disk use, and archives it when research closes.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import run_panel as runner


def initialize(adapter, model_id):
    source = runner.ROOT / adapter
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("expanded_selected_adapter", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    namespace = hashlib.sha256(json.dumps({
        "adapter": adapter, "model": model_id, "sources": runner.hashes(),
        "configuration": {k: v for k, v in os.environ.items()
                          if k.startswith("SIMFOLIO_") and k != "SIMFOLIO_RESEARCH_FIT_CACHE"},
    }, sort_keys=True).encode()).hexdigest()
    root = Path(os.environ["SIMFOLIO_RESEARCH_FIT_CACHE"]) / namespace
    module.initialize(root, 1, 1)
    selected = [c for c in module.CANDIDATES if not model_id or c.model_id == model_id]
    if len(selected) != 1:
        raise ValueError("select exactly one candidate per process")
    runner.MODELS = module
    runner.CANDIDATE = selected[0]


runner.initialize = initialize

if __name__ == "__main__":
    runner.main()
