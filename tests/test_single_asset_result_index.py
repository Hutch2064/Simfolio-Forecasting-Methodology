"""The independent single52 sidecar cannot overwrite retained80 catalogue evidence."""

import copy
import csv
import importlib.util
import json
from importlib.resources import files
from pathlib import Path

import pytest

from simfolio_forecasting_methodology.catalog import load_canonical_175
from simfolio_forecasting_methodology.catalogue import load_canonical_models
from simfolio_forecasting_methodology.single_asset_results import (
    CONTROL_ID,
    INDEX_RESOURCE,
    load_single52_results,
    validate_single52_index,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "catalogue52_sync", ROOT / "tools/catalogue52/sync_catalogues.py"
)
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


@pytest.fixture
def index():
    return json.loads(files("simfolio_forecasting_methodology").joinpath(INDEX_RESOURCE).read_text())


def report_bytes(index):
    return json.dumps({
        "protocol": index["protocol"], "ranking_metric": index["ranking_metric"],
        "partial_results_ranked": False, "blended_score_used": False,
        "historical80_role": index["historical80_role"],
        "inventory_sha256": index["inventory_sha256"],
        "full_scores": [{key: value for key, value in row.items()
                         if key != "specification_fingerprint"}
                        for row in index["models"].values()],
        "progress": [{"completed_tasks": 1, "single52_crps": 0.0}],
    }).encode()


def test_packaged_results_link_to_unchanged_catalogue_and_separate_scores(index):
    before = load_canonical_models()
    rows = load_single52_results()
    assert rows == index["models"] and CONTROL_ID in rows
    assert load_canonical_models() == before
    assert all(row.cells == 701280 for row in load_canonical_175())
    for record in before:
        row = rows.get(record["public_model_id"])
        assert "single52_crps" not in record
        if row is None:
            continue
        assert row["specification_fingerprint"] == record["specification_fingerprint"]
        value = record["historical_score"]["exact_empirical_crps"]
        assert row["historical80_crps"] == (None if value is None else float(value))
        assert row["cells"] == 455832


@pytest.mark.parametrize("field,value", [
    ("portfolios", 51), ("tasks", 2651), ("cells", 455831), ("simulations", 120),
    ("single52_crps", float("nan")), ("cells_sha256", "invalid"),
    ("canonical_rank", 1), ("specification_fingerprint", "0" * 64),
    ("historical80_crps", 0.0), ("historical80_evidence_verified", None),
])
def test_denominator_score_and_identity_drift_fail_closed(index, field, value):
    index["models"][CONTROL_ID][field] = value
    with pytest.raises(ValueError):
        validate_single52_index(index)


def test_sync_preserves_every_evidence_field_and_omits_progress(index):
    result = sync.make_index(report_bytes(index))
    assert result["models"] == index["models"]
    assert "progress" not in result
    assert not result["blended_score_used"]
    assert not result["partial_results_ranked"]


def test_resource_loader_is_independent_of_working_directory(index, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_single52_results() == index["models"]


def test_duplicate_or_unknown_report_identity_is_rejected(index):
    report = json.loads(report_bytes(index))
    report["full_scores"].append(copy.deepcopy(report["full_scores"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        sync.make_index(json.dumps(report).encode())
    report["full_scores"][-1]["model_id"] = "unregistered_model"
    with pytest.raises(ValueError, match="absent"):
        sync.make_index(json.dumps(report).encode())


def test_unarchived_completion_retains_transparent_pending_archive_status(index):
    report = json.loads(report_bytes(index))
    row = next(row for row in report["full_scores"] if row["model_id"] != CONTROL_ID)
    row["raw_evidence"]["archived_and_download_verified"] = False
    result = sync.make_index(json.dumps(report).encode())
    assert result["models"][row["model_id"]]["raw_evidence"]["archived_and_download_verified"] is False
    assert CONTROL_ID in result["models"]
    index["models"][row["model_id"]]["raw_evidence"]["archive_sha256"] = "invalid"
    with pytest.raises(ValueError, match="archive requires"):
        validate_single52_index(index)


def test_sync_writes_identical_sidecars_without_touching_definitions(index, tmp_path):
    report = tmp_path / "scores.json"
    report.write_bytes(report_bytes(index))
    master = tmp_path / "master"
    (master / "catalogs").mkdir(parents=True)
    for part in (1, 2):
        with (master / f"catalogs/canonical_175_part{part}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model_id", "canonical_rank"])
            writer.writeheader()
            if part == 1:
                writer.writerows({key: row[key] for key in writer.fieldnames}
                                 for row in index["models"].values())
    manifest = master / "catalogs/master_catalog_manifest.yaml"
    manifest.write_text("unchanged_statistical_metadata: true\n")
    originals = {path: path.read_bytes() for path in (master / "catalogs").iterdir()}
    methodology = tmp_path / "methodology"
    before = load_canonical_models()
    sync.sync_catalogues(report, master, methodology_root=methodology)
    left = methodology / "src/simfolio_forecasting_methodology" / INDEX_RESOURCE
    right = master / "catalogs/single52_results.json"
    assert left.read_bytes() == right.read_bytes()
    assert load_single52_results(left) == index["models"]
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert load_canonical_models() == before


def test_public_index_rejects_absolute_local_reference(index):
    index["source_report"] = "/" + "private/run/scores.json"
    with pytest.raises(ValueError, match="absolute local"):
        validate_single52_index(index)
