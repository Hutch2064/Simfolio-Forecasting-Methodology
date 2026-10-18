"""Bounded, independent one-origin admissions; never produces panel rankings."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from adapters import ROOT, model_specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ranks", help="Comma-separated ranks; default all available implementations")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ranks = None if args.ranks is None else set(map(int, args.ranks.split(",")))
    rows = [row for row in model_specs() if row["adapter_file_available"]
            and (ranks is None or row["canonical_rank"] in ranks)]
    environment = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        environment[name] = "1"
    def run(row):
        rank = row["canonical_rank"]
        output = args.output / f"m{rank:03d}-0000.json"
        with (args.output / f"m{rank:03d}.log").open("w") as log:
            child = subprocess.Popen([sys.executable, str(ROOT / "tools/catalogue52/admission.py"),
                "--rank", str(rank), "--data", str(args.data.resolve()),
                "--output", str(output.resolve()), "--repeat"], env=environment,
                stdout=log, stderr=subprocess.STDOUT)
            # The receipt binds observation to an actual subprocess handle;
            # no speculative wall-time culling is imposed on difficult fits.
            (args.output / f"m{rank:03d}-process.json").write_text(
                json.dumps({"pid": child.pid, "rank": rank, "model_id": row["model_id"]}))
            code = child.wait()
        result = {"rank": rank, "model_id": row["model_id"], "returncode": code,
                  "passed": code == 0 and output.exists()}
        (args.output / f"m{rank:03d}-admission.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, row) for row in rows]):
            result = future.result()
            completed.append(result)
            print(json.dumps(result), flush=True)
            temporary = args.output / "summary.tmp"
            temporary.write_text(json.dumps({"required": len(rows), "completed": len(completed),
                "passed": sum(row["passed"] for row in completed), "full_panel_score": False,
                "models": sorted(completed, key=lambda row: row["rank"])}, indent=2) + "\n")
            temporary.replace(args.output / "summary.json")


if __name__ == "__main__":
    main()
