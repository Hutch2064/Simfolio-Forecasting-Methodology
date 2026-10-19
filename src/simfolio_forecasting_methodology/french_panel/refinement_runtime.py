"""Load the sealed M261 research source used by the refinement experiments."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources/french1926/refinement_source"


@lru_cache(maxsize=1)
def verify_source_closure() -> dict:
    manifest = json.loads((SOURCE_ROOT / "manifest.json").read_text())
    package = Path(__file__).resolve().parents[1]
    for category in ("snapshot_tools", "supplements", "package_primitives"):
        root = package if category == "package_primitives" else SOURCE_ROOT
        for relative, expected in manifest[category].items():
            path = root / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"sealed refinement source drift: {relative}")
    return manifest


def load_supplement(relative: str):
    verify_source_closure()
    path = SOURCE_ROOT / "refinement52" / relative
    name = "french_refinement_" + relative.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_m261_source():
    """Install the exact frozen numerical stack once in a dedicated process."""
    verify_source_closure()
    # Cached Numba functions contain private import aliases. Keep this sealed
    # owner apart from current-tool methods with different module aliases.
    from numba import config

    cache_root = config.CACHE_DIR
    if cache_root:
        source_id = hashlib.sha256((SOURCE_ROOT / "manifest.json").read_bytes()).hexdigest()[:16]
        config.CACHE_DIR = str(Path(cache_root) / f"m261-refinement-{source_id}")
    path = SOURCE_ROOT / "tools/data_selected_timescales/models.py"
    sys.path[:0] = [str(path.parent), str(SOURCE_ROOT / "tools/mcmc_runtime")]
    name = "french_refinement_frozen_m261"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = os.environ.get("SIMFOLIO_TIMESCALE_VARIANT")
    os.environ["SIMFOLIO_TIMESCALE_VARIANT"] = "existing_sv_rate"
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            os.environ.pop("SIMFOLIO_TIMESCALE_VARIANT", None)
        else:
            os.environ["SIMFOLIO_TIMESCALE_VARIANT"] = previous
    module.initialize(cache_root=None, vine_threads=1, state_workers=1)
    return module
