"""Versioned, read-only rescoring of existing and future French loss checkpoints.

No model factory is imported and no forecast is run. A separately approved plan
must declare every scale/weight before this reporter runs. The original execution
manifests, raw scores, checkpoints and scheduler status are never modified.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from simfolio_forecasting_methodology.french_panel import runner as original
from simfolio_forecasting_methodology.french_panel import selected
from simfolio_forecasting_methodology.french_panel.sweep import exclusive_lock

PUBLIC_REFERENCE = "model_testing_joint_stationary_bootstrap_v1"
SCHEMA = "french-checkpoint-rescore-v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ScoringPlan:
    """A fixed separable linear scoring plan; this class chooses no scale/weight.

    For each series and horizon, average loss divided by declared training,
    horizon and optional origin-horizon scales over the eligible origins. Average these cells with the declared
    horizon weights, then average series using their declared series weights.
    A series enters reporting only after ALL its original origin checkpoints
    are valid. This makes partial comparisons use identical complete cells.
    """
    def __init__(self, declaration, arrays, origins, panel):
        if declaration.get("schema") != SCHEMA or declaration.get("approval_status") != "approved":
            raise ValueError("an explicitly approved versioned score declaration is required")
        if not declaration.get("score_version") or not declaration.get("methodology"):
            raise ValueError("score_version and methodology are required")
        if declaration.get("panel_fingerprint") != panel["panel_fingerprint"]:
            raise ValueError("scoring plan panel identity differs")
        self.declaration = declaration
        self.origins, self.panel = origins, panel
        self.capacity = max(o.max_horizon for o in origins)
        expected_shapes = {
            "training_scale": (len(panel["series"]), len(origins)),
            "horizon_scale": (self.capacity,),
            "horizon_weight": (self.capacity,),
            "series_weight": (len(panel["series"]),),
        }
        if "origin_horizon_scale" in arrays:
            expected_shapes["origin_horizon_scale"] = (len(origins), self.capacity)
        if set(arrays) != set(expected_shapes):
            raise ValueError("scoring arrays differ from declared scale/weight schema")
        self.arrays = {}
        for name, shape in expected_shapes.items():
            values = np.array(arrays[name], dtype="<f8", copy=True)
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid shape or nonfinite {name}")
            if "scale" in name and np.any(values <= 0):
                raise ValueError("scales must be strictly positive; zero variance requires an explicit protocol decision")
            if "weight" in name and (np.any(values < 0) or not np.any(values > 0)):
                raise ValueError("weights must be nonnegative with positive total")
            values.setflags(write=False)
            self.arrays[name] = values
        self.expected_counts = np.zeros(self.capacity, dtype=np.int64)
        for origin in origins:
            self.expected_counts[:origin.max_horizon] += 1
        self.horizon_total = float(np.sum(self.arrays["horizon_weight"]))
        if declaration.get("horizon_weight_normalization") == "log(H+1)":
            expected_weights = np.log1p(1.0 / np.arange(1, self.capacity + 1))
            if not np.array_equal(self.arrays["horizon_weight"], expected_weights):
                raise ValueError("declared exact log horizon weights differ")
            self.horizon_total = float(np.log1p(self.capacity))
        if not np.isfinite(self.horizon_total) or not np.isfinite(np.sum(self.arrays["series_weight"])):
            raise ValueError("weight totals must be finite")
        self.array_hashes = {name: hashlib.sha256(a.tobytes()).hexdigest()
                             for name, a in self.arrays.items()}
        self.fingerprint = original.stable_digest({
            "declaration": declaration, "array_sha256": self.array_hashes,
            "origins": [asdict(o) for o in origins],
        })

    @classmethod
    def load(cls, declaration_path, origins, panel):
        path = Path(declaration_path)
        declaration = json.loads(path.read_text())
        artifact = path.parent / declaration["arrays_file"]
        if sha256(artifact) != declaration["arrays_file_sha256"]:
            raise ValueError("scoring arrays checksum mismatch")
        with np.load(artifact, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        expected_reporter = declaration.get("source_identity", {}).get("reporter")
        if expected_reporter and expected_reporter != sha256(__file__):
            raise ValueError("frozen scoring reporter source differs")
        return cls(declaration, arrays, origins, panel)

    def normalized_losses(self, i, j, losses):
        length = self.origins[j].max_horizon
        scale = self.arrays["training_scale"][i, j] * self.arrays["horizon_scale"][:length]
        if "origin_horizon_scale" in self.arrays:
            scale = scale * self.arrays["origin_horizon_scale"][j, :length]
        result = losses / scale
        if not np.all(np.isfinite(result)) or np.any(result < 0):
            raise ValueError("nonfinite normalized losses")
        return result

    def contribution(self, i, j, losses):
        length = self.origins[j].max_horizon
        if losses.shape != (length,):
            raise ValueError("origin loss length differs from plan")
        coefficients = self.arrays["horizon_weight"][:length] / self.expected_counts[:length]
        result = float(np.sum(self.normalized_losses(i, j, losses) * coefficients)) / self.horizon_total
        if not np.isfinite(result) or result < 0:
            raise ValueError("nonfinite scoring contribution")
        return result


def training_standard_deviations(returns, origins, *, ddof):
    """Optional scale construction primitive; callers must choose/declare ddof.

    Daily standard deviation uses observed training history only. The caller
    must declare ddof; the reporter selects no estimator or epsilon floor.
    """
    result = np.empty((len(returns), len(origins)), dtype="<f8")
    for i, series in enumerate(returns):
        for j, origin in enumerate(origins):
            result[i, j] = np.std(series[:origin.position + 1], ddof=ddof)
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("training scale is nonfinite or zero; no arbitrary epsilon is permitted")
    return result


class ScoreSidecar:
    def __init__(self, panel_root, source_root, report_root, plan, *, audit_seconds=300):
        self.panel_root, self.source_root = Path(panel_root), Path(source_root)
        self.report_root = Path(report_root)
        if self.report_root == self.source_root or self.source_root in self.report_root.parents:
            raise ValueError("report root must be outside the active result tree")
        self.report_root.mkdir(parents=True, exist_ok=True)
        self.panel = original.load_panel(self.panel_root)
        self.origins = tuple(original.schedule(self.panel_root))
        self.plan = plan
        if plan.panel["panel_fingerprint"] != self.panel["panel_fingerprint"]:
            raise ValueError("report panel differs from plan")
        self.source_closure = original._source_closure()
        self.implementation_sources = selected.source_identity()
        self.implementation_digests = {}
        self.audit_seconds = audit_seconds
        self.directories = {}
        self.models = {}
        self.db = sqlite3.connect(self.report_root / "score-index.sqlite3")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS provenance (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        binding = {"schema": SCHEMA, "plan_fingerprint": plan.fingerprint,
                   "panel_fingerprint": self.panel["panel_fingerprint"],
                   "source_root": str(self.source_root.resolve())}
        previous = self.db.execute("SELECT value FROM provenance WHERE key='binding'").fetchone()
        if previous and json.loads(previous[0]) != binding:
            raise ValueError("report cache identity differs; choose a new versioned report directory")
        self.db.execute("INSERT OR IGNORE INTO provenance VALUES ('binding', ?)", (json.dumps(binding, sort_keys=True),))
        self.db.execute("""CREATE TABLE IF NOT EXISTS origins (
            model TEXT, series INTEGER, origin INTEGER, execution TEXT,
            checkpoint_sha256 TEXT, size INTEGER, mtime INTEGER,
            PRIMARY KEY (model, series, origin))""")
        self.db.execute("CREATE TABLE IF NOT EXISTS errors (model TEXT PRIMARY KEY, message TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS complete_series (model TEXT, series INTEGER, score REAL, checkpoint_set_fingerprint TEXT, PRIMARY KEY(model,series))")
        self.db.commit()
        original._atomic_json(self.report_root / "provenance.json", {
            **binding, "declaration": plan.declaration, "array_sha256": plan.array_hashes,
            "reporter_sha256": sha256(__file__), "forecast_runs": 0,
            "original_raw_score_preserved": True,
            "checkpoint_validation": "original checksum/identity reader plus whole-file SHA256; changed directories scanned incrementally, full stat audit periodically",
            "complete_cell_gate": "all original origins of a series must be valid before that series enters either score or matched comparisons",
        })

    def close(self):
        self.db.close()

    def manifest(self, root):
        value = json.loads((root / "execution.json").read_text())
        identity = value["identity"]
        if original.stable_digest(identity) != value["fingerprint"]:
            raise ValueError("execution manifest checksum mismatch")
        expected = {
            "protocol_id": original.PROTOCOL_ID,
            "panel_fingerprint": self.panel["panel_fingerprint"],
            "series_ids": [s["series_id"] for s in self.panel["series"]],
            "source_closure": self.source_closure,
            "origins": [asdict(o) for o in self.origins],
            "simulations_per_origin": original.SIMULATIONS,
            "seed_contract": "coherent_daily_seed(origin_date,max_horizon,240)",
            "aggregation": "origin_mean_per_series_horizon_then_equal_cell_mean",
            "future_dates": "synthetic_weekdays_after_origin_actual_dates_for_train_and_targets",
        }
        if any(identity.get(k) != v for k, v in expected.items()):
            raise ValueError("execution panel/statistical source/scoring contract differs")
        mid = identity["model_id"]
        if mid not in self.implementation_digests:
            self.implementation_digests[mid] = selected.implementation_digest(mid, self.implementation_sources)
        if identity.get("implementation_digest") != self.implementation_digests[mid]:
            raise ValueError("execution native/package implementation digest differs")
        if root.name != hashlib.sha256(mid.encode()).hexdigest():
            raise ValueError("model directory identity differs")
        previous = self.models.get(mid)
        if previous and previous["fingerprint"] != value["fingerprint"]:
            raise ValueError("execution identity changed during reporting")
        self.models[mid] = {**value, "root": root}
        return mid, value["fingerprint"]

    def checkpoint(self, mid, fingerprint, i, j, path):
        cached = self.db.execute("SELECT execution,checkpoint_sha256,size,mtime FROM origins WHERE model=? AND series=? AND origin=?", (mid, i, j)).fetchone()
        stat = path.stat()
        if cached and cached[0] != fingerprint:
            raise ValueError("indexed execution identity differs")
        if cached and (stat.st_size, stat.st_mtime_ns) == cached[2:]:
            return False
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if cached and digest != cached[1]:
            raise ValueError(f"previously indexed checkpoint changed: {path}")
        origin = self.origins[j]
        identity = {
            "execution_fingerprint": fingerprint,
            "series_id": self.panel["series"][i]["series_id"], "origin": asdict(origin),
            "seed": original.coherent_daily_seed(origin.origin_date, origin.max_horizon, original.SIMULATIONS),
        }
        losses, _ = original._read_checkpoint(io.BytesIO(payload), identity)
        self.db.execute("INSERT OR REPLACE INTO origins VALUES (?,?,?,?,?,?,?)", (
            mid, i, j, fingerprint, digest, stat.st_size, stat.st_mtime_ns))
        return True

    def scan_model(self, mid, fingerprint, root):
        changed = 0
        now = time.monotonic()
        for directory in sorted((root / "tasks").glob("[0-9][0-9][0-9][0-9][0-9][0-9]")):
            i = int(directory.name)
            if i >= len(self.panel["series"]):
                raise ValueError("unknown checkpoint series directory")
            stamp = directory.stat().st_mtime_ns
            previous = self.directories.get(directory)
            if previous and previous[0] == stamp and now - previous[1] < self.audit_seconds:
                continue
            present = set()
            for path in sorted(directory.glob("*.npz")):
                if not path.stem.isdigit() or len(path.stem) != 3:
                    raise ValueError("unknown checkpoint origin filename")
                j = int(path.stem)
                if j >= len(self.origins):
                    raise ValueError("unknown checkpoint origin")
                present.add(j)
                changed += self.checkpoint(mid, fingerprint, i, j, path)
            indexed = {r[0] for r in self.db.execute("SELECT origin FROM origins WHERE model=? AND series=?", (mid, i))}
            if not indexed <= present:
                raise ValueError("previously indexed checkpoint disappeared")
            self.directories[directory] = stamp, now
        indexed_series = {r[0] for r in self.db.execute("SELECT DISTINCT series FROM origins WHERE model=?", (mid,))}
        if any(not (root / "tasks" / f"{i:06d}").is_dir() for i in indexed_series):
            raise ValueError("previously indexed checkpoint series disappeared")
        return changed

    def aggregate_complete_series(self, mid, fingerprint, root):
        stored = {i for (i,) in self.db.execute("SELECT series FROM complete_series WHERE model=?", (mid,))}
        complete = [i for i, count in self.db.execute(
            "SELECT series,COUNT(*) FROM origins WHERE model=? GROUP BY series", (mid,))
            if count == len(self.origins) and i not in stored]
        for i in complete:
            sums = np.zeros(self.plan.capacity, dtype=np.float64)
            counts = np.zeros(self.plan.capacity, dtype=np.int64)
            checkpoint_hashes = []
            for j, origin in enumerate(self.origins):
                path = root / "tasks" / f"{i:06d}/{j:03d}.npz"
                payload = path.read_bytes()
                digest = hashlib.sha256(payload).hexdigest()
                indexed_hash = self.db.execute("SELECT checkpoint_sha256 FROM origins WHERE model=? AND series=? AND origin=?", (mid, i, j)).fetchone()[0]
                if digest != indexed_hash:
                    raise ValueError("checkpoint changed before canonical series aggregation")
                identity = {"execution_fingerprint": fingerprint,
                    "series_id": self.panel["series"][i]["series_id"], "origin": asdict(origin),
                    "seed": original.coherent_daily_seed(origin.origin_date, origin.max_horizon, original.SIMULATIONS)}
                losses, _ = original._read_checkpoint(io.BytesIO(payload), identity)
                sums[:origin.max_horizon] += self.plan.normalized_losses(i, j, losses)
                counts[:origin.max_horizon] += 1
                checkpoint_hashes.append(digest)
            if not np.array_equal(counts, self.plan.expected_counts):
                raise ValueError("complete series does not have every expected horizon cell")
            score = float(np.sum((sums / counts) * self.plan.arrays["horizon_weight"])) / self.plan.horizon_total
            binding = original.stable_digest({"execution": fingerprint, "plan": self.plan.fingerprint,
                                               "ordered_checkpoint_sha256": checkpoint_hashes})
            self.db.execute("INSERT INTO complete_series VALUES (?,?,?,?)", (mid, i, score, binding))

    def poll(self):
        indexed_this_poll = 0
        known_models = {hashlib.sha256(mid.encode()).hexdigest(): mid
                        for (mid,) in self.db.execute("SELECT DISTINCT model FROM origins")}
        for mid in self.models:
            known_models[hashlib.sha256(mid.encode()).hexdigest()] = mid
        for root in sorted((self.source_root / "models").iterdir()):
            if not root.is_dir() or not (root / "execution.json").exists():
                continue
            mid = known_models.get(root.name, root.name)
            try:
                mid, fingerprint = self.manifest(root)
                if self.db.execute("SELECT 1 FROM errors WHERE model=?", (mid,)).fetchone():
                    continue  # Integrity errors require explicit repair/new cache, never silent substitution.
                indexed_this_poll += self.scan_model(mid, fingerprint, root)
                self.aggregate_complete_series(mid, fingerprint, root)
            except Exception as exc:
                self.db.execute("INSERT OR REPLACE INTO errors VALUES (?,?)", (mid, f"{type(exc).__name__}: {exc}"))
        for mid, manifest in self.models.items():
            if not (manifest["root"] / "execution.json").exists():
                self.db.execute("INSERT OR REPLACE INTO errors VALUES (?,?)", (mid, "previously indexed execution manifest disappeared"))
        self.db.commit()
        return self.publish(indexed_this_poll)

    def publish(self, indexed_this_poll):
        models, series_scores = {}, {}
        origin_count = len(self.origins)
        expected = len(self.panel["series"]) * origin_count
        errors = dict(self.db.execute("SELECT model,message FROM errors"))
        for mid, manifest in self.models.items():
            complete = dict(self.db.execute("SELECT series,score FROM complete_series WHERE model=? ORDER BY series", (mid,)))
            series_scores[mid] = complete
            count = self.db.execute("SELECT COUNT(*) FROM origins WHERE model=?", (mid,)).fetchone()[0]
            unfinished_errors = [p for p in (manifest["root"] / "errors").glob("*/*.json")
                                 if not (manifest["root"] / "tasks" / p.parent.name / (p.stem + ".npz")).exists()]
            final = count == expected and not unfinished_errors and mid not in errors
            score = self.weighted_series_score(complete, sorted(complete))
            raw_path = manifest["root"] / "summary.json"
            raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}
            models[mid] = {
                "model_id": mid, "execution_fingerprint": manifest["fingerprint"],
                "derived_score_fingerprint": original.stable_digest({
                    "execution_fingerprint": manifest["fingerprint"],
                    "plan_fingerprint": self.plan.fingerprint}),
                "status": "failed_integrity" if mid in errors else "complete" if final else "incomplete",
                "completed_origins": count, "expected_origins": expected,
                "complete_series_ids": [self.panel["series"][i]["series_id"] for i in sorted(complete)],
                "final_score": score if final else None,
                "partial_complete_series_score": score if not final and mid not in errors else None,
                "error": errors.get(mid), "source_failed_origins": len(unfinished_errors),
                "original_raw_crps": {"final_score": raw.get("final_score"),
                    "partial_complete_series_score": raw.get("partial_complete_series_score"),
                    "complete_series_ids": raw.get("complete_series", []),
                    "status": raw.get("status"), "source_path": str(raw_path)},
            }
        reference = series_scores.get(PUBLIC_REFERENCE, {})
        if PUBLIC_REFERENCE in errors:
            reference = {}
        for mid, row in models.items():
            common = sorted(set(series_scores[mid]) & set(reference)) if mid not in errors else []
            a = self.weighted_series_score(series_scores[mid], common)
            b = self.weighted_series_score(reference, common)
            row["public_reference_comparison"] = {
                "reference_model_id": PUBLIC_REFERENCE,
                "matched_complete_series_ids": [self.panel["series"][i]["series_id"] for i in common],
                "model_score": a, "reference_score": b,
                "percent_better": 100 * (1 - a / b) if b is not None and b > 0 else None,
            }
        status = {
            "schema": SCHEMA, "score_version": self.plan.declaration["score_version"],
            "plan_fingerprint": self.plan.fingerprint,
            "updated_utc": datetime.now(timezone.utc).isoformat(), "forecast_runs": 0,
            "indexed_new_or_changed_origins_this_poll": indexed_this_poll,
            "metric": self.plan.declaration["methodology"], "models": models,
            "integrity_errors": errors,
            "scope": "new versioned scores from immutable loss checkpoints; raw scores and scheduler status remain original",
        }
        original._atomic_json(self.report_root / "score_status.json", status)
        return status

    def weighted_series_score(self, scores, indices):
        numerator = denominator = 0.0
        for i in indices:
            weight = self.plan.arrays["series_weight"][i]
            numerator += weight * scores[i]
            denominator += weight
        return float(numerator / denominator) if denominator > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path, action="append")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("poll seconds must be at least 1")
    panel = original.load_panel(args.panel_root)
    plans = [ScoringPlan.load(path, tuple(original.schedule(args.panel_root)), panel) for path in args.plan]
    versions = [p.declaration["score_version"] for p in plans]
    if len(set(versions)) != len(versions) or any(Path(v).name != v for v in versions):
        parser.error("score versions must be unique plain directory names")
    with exclusive_lock(args.report_root / ".reporter.lock"):
        reporters = [ScoreSidecar(args.panel_root, args.source_root, args.report_root / version, plan)
                     for version, plan in zip(versions, plans, strict=True)]
        try:
            while True:
                snapshots = {version: reporter.poll() for version, reporter in zip(versions, reporters, strict=True)}
                original._atomic_json(args.report_root / "scoring_master.json", {
                    "canonical_score_version": versions[0],
                    "score_versions": {version: {
                        "role": "primary" if index == 0 else "predeclared_sensitivity",
                        "plan_fingerprint": snapshots[version]["plan_fingerprint"],
                        "status_path": str(args.report_root / version / "score_status.json"),
                    } for index, version in enumerate(versions)},
                    "public_reference_model_id": PUBLIC_REFERENCE,
                    "original_raw_score": "preserved separately in original model summaries",
                    "forecast_runs": 0, "updated_utc": datetime.now(timezone.utc).isoformat(),
                })
                if args.once:
                    return
                time.sleep(args.poll_seconds)
        finally:
            for reporter in reporters:
                reporter.close()


if __name__ == "__main__":
    main()
