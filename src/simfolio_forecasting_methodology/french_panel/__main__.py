"""Data preparation and readiness checks only: no model run command."""

import argparse
import json
from pathlib import Path

from .data import build_panel, load_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    panel = build_panel(args.root) if args.command == "prepare" else load_panel(args.root)
    keys = ("panel_id", "panel_fingerprint", "status", "common_start", "common_end",
            "series_count", "observations", "origins_per_series", "expected_tasks",
            "max_horizon", "expected_cells", "registered_models", "results")
    print(json.dumps({key: panel[key] for key in keys}, indent=2))


if __name__ == "__main__":
    main()
