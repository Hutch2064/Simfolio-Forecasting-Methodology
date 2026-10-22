"""Selection identity and bounded timing arithmetic; never a panel sweep."""
import base64
import csv
import hashlib
import json
from dataclasses import replace
from importlib.metadata import FileHash, PackagePath
from types import SimpleNamespace

import pytest

from simfolio_forecasting_methodology.french_panel import readiness, selected
from simfolio_forecasting_methodology.french_panel import runner as panel_runner
from simfolio_forecasting_methodology.origins import OriginDescriptor


def test_approved_fifteen_and_paused_control(tmp_path):
    assert len(selected.FACTORIES) == 15
    assert selected.selection_matches_metadata()
    control = selected.prepare_control(tmp_path / "control.json")
    assert control["stop"] is True
    assert control["workers"] == 12
    assert set(control["models"]) == set(selected.FACTORIES)
    assert all(row["enabled"] for row in control["models"].values())
    assert control == json.loads((tmp_path / "control.json").read_text())
    assert all(len(row["implementation_digest"]) == 64 for row in control["models"].values())


def wheel_identity_fixture(tmp_path, monkeypatch):
    package = tmp_path / "installed/simfolio_forecasting_methodology"
    files = []
    for name in ["simfolio_forecasting_methodology/french_panel/selected.py",
                 "simfolio_forecasting_methodology/resources/example.json",
                 "simfolio_forecasting_methodology-0.3.0.dist-info/METADATA",
                 "simfolio_forecasting_methodology-0.3.0.dist-info/WHEEL"]:
        record = PackagePath(name)
        target = package.parent / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(name)
        digest = base64.urlsafe_b64encode(hashlib.sha256(target.read_bytes()).digest())
        record.hash = FileHash("sha256=" + digest.rstrip(b"=").decode())
        files.append(record)
    distribution = SimpleNamespace(metadata={"Name": "simfolio-forecasting-methodology"},
                                   version="0.3.0", files=files,
                                   locate_file=lambda record: package.parent / str(record))
    monkeypatch.setattr(selected, "__file__", str(package / "french_panel/selected.py"))
    monkeypatch.setattr(selected.importlib.metadata, "distributions", lambda **_: [distribution])
    return package, distribution


def test_installed_identity_has_no_parent_checkout_dependency(tmp_path, monkeypatch):
    package, _ = wheel_identity_fixture(tmp_path, monkeypatch)
    assert not (package.parent / "pyproject.toml").exists()
    identity = selected.source_identity()
    assert len(identity) == 4
    assert all(key.startswith("installed-wheel-v1/0.3.0/") for key in identity)
    (package / "__pycache__").mkdir()
    (package / "__pycache__/runtime.pyc").write_bytes(b"runtime")
    assert selected.source_identity() == identity


def test_installed_identity_ignores_global_checkout_metadata(tmp_path, monkeypatch):
    package = tmp_path / "installed/simfolio_forecasting_methodology"
    metadata = package.parent / "simfolio_forecasting_methodology-0.3.0.dist-info"
    metadata.mkdir(parents=True)
    module = package / "french_panel/selected.py"
    module.parent.mkdir(parents=True)
    module.write_text("# executing wheel source\n")
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: simfolio-forecasting-methodology\nVersion: 0.3.0\n")
    (metadata / "WHEEL").write_text("Wheel-Version: 1.0\nTag: py3-none-any\n")
    with (metadata / "RECORD").open("w", newline="") as stream:
        writer = csv.writer(stream)
        for path in [module, metadata / "METADATA", metadata / "WHEEL"]:
            digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
            writer.writerow([str(path.relative_to(package.parent)),
                             "sha256=" + digest.rstrip(b"=").decode(), path.stat().st_size])
    shadow = tmp_path / "checkout"
    egg = shadow / "simfolio_forecasting_methodology.egg-info"
    egg.mkdir(parents=True)
    (egg / "PKG-INFO").write_text(
        "Metadata-Version: 2.1\nName: simfolio-forecasting-methodology\nVersion: 9.9.9\n")
    monkeypatch.syspath_prepend(str(shadow))
    assert selected.importlib.metadata.distribution(
        "simfolio-forecasting-methodology").version == "9.9.9"
    monkeypatch.setattr(selected, "__file__", str(module))
    identity = selected.source_identity()
    assert len(identity) == 3
    assert all(key.startswith("installed-wheel-v1/0.3.0/") for key in identity)


def test_install_target_named_src_is_not_a_checkout(tmp_path, monkeypatch):
    package, distribution = wheel_identity_fixture(tmp_path, monkeypatch)
    target = tmp_path / "src"
    package.parent.rename(target)
    distribution.locate_file = lambda record: target / str(record)
    monkeypatch.setattr(selected, "__file__", str(target / package.name / "french_panel/selected.py"))
    assert all(key.startswith("installed-wheel-v1/") for key in selected.source_identity())


@pytest.mark.parametrize("failure", ["modified", "missing_hash", "missing_metadata",
                                     "missing_record", "wrong_root", "missing_distribution",
                                     "ambiguous_distribution"])
def test_installed_identity_rejects_unbound_execution(tmp_path, monkeypatch, failure):
    package, distribution = wheel_identity_fixture(tmp_path, monkeypatch)
    if failure == "modified":
        (package / "resources/example.json").write_text("changed")
    elif failure == "missing_hash":
        distribution.files[0].hash = None
    elif failure == "missing_metadata":
        distribution.files.pop()
    elif failure == "missing_record":
        distribution.files.pop(1)
    elif failure == "missing_distribution":
        monkeypatch.setattr(selected.importlib.metadata, "distributions", lambda **_: [])
    elif failure == "ambiguous_distribution":
        monkeypatch.setattr(selected.importlib.metadata, "distributions",
                            lambda **_: [distribution, distribution])
    else:
        monkeypatch.setattr(selected, "__file__", str(tmp_path / "unrelated/french_panel/selected.py"))
    with pytest.raises(ValueError):
        selected.source_identity()


def test_source_checkout_rejects_wrong_project_metadata(tmp_path, monkeypatch):
    module = tmp_path / "src/simfolio_forecasting_methodology/french_panel/selected.py"
    (tmp_path / "pyproject.toml").write_text('[project]\nname="unrelated"\n')
    monkeypatch.setattr(selected, "__file__", str(module))
    with pytest.raises(ValueError, match="project identity mismatch"):
        selected.source_identity()


def test_source_checkout_requires_executing_source_bytes(tmp_path, monkeypatch):
    module = tmp_path / "src/simfolio_forecasting_methodology/french_panel/selected.py"
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="simfolio-forecasting-methodology"\n')
    monkeypatch.setattr(selected, "__file__", str(module))
    with pytest.raises(ValueError, match="executing module"):
        selected.source_identity()


def test_unknown_selection_fails_closed():
    with pytest.raises(ValueError, match="unselected"):
        selected.build_selected("not-a-method")


def test_statistical_source_changes_bind_identity():
    model_id = next(iter(selected.FACTORIES))
    assert selected.implementation_digest(model_id, {"native.cpp": "a"}) != selected.implementation_digest(
        model_id, {"native.cpp": "b"})


def test_runtime_bytecode_cannot_change_checkpoint_source_identity(tmp_path, monkeypatch):
    package = tmp_path / "package"
    module = package / "french_panel/runner.py"
    module.parent.mkdir(parents=True)
    module.write_text("# fixture")
    frozen = package / "resources/french1926/frozen"
    frozen.mkdir(parents=True)
    (frozen / "model.py").write_text("# numerical source")
    monkeypatch.setattr(panel_runner, "__file__", str(module))
    before = panel_runner._source_closure()
    bytecode = frozen / "__pycache__/model.cpython-312.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"runtime bytecode")
    assert panel_runner._source_closure() == before
    (frozen / "model.py").write_text("# changed numerical source")
    assert panel_runner._source_closure() != before


def test_probe_plan_represents_all_origins_without_horizon_changes():
    first = OriginDescriptor("rolling_01", "rolling_origin", 525, "1928-03-31", 131)
    rolling = [replace(first, origin_label=f"rolling_{i}", position=525 + i * 500)
               for i in range(48)]
    temporal = [replace(first, origin_label=f"temporal_{i}", evaluation_split="temporal",
                        max_horizon=19738 // i) for i in (1, 2, 3)]
    plan = readiness.probe_plan(rolling + temporal)
    assert len(plan) == 6
    assert sum(count for _, count in plan) == 51
    assert [count for _, count in plan] == [16, 16, 16, 1, 1, 1]
    assert plan[-3:] == [(origin, 1) for origin in temporal]
    assert all(origin in rolling + temporal for origin, _ in plan)


def test_extrapolation_uses_stratum_weights_not_cold_start():
    records = [{"role": "native_warmup", "status": "passed", "forecast_and_score_seconds": 1000}]
    records.extend({"role": "projection_probe", "status": "passed", "represented_origins": count,
                    "forecast_and_score_seconds": elapsed}
                   for count, elapsed in [(16, 1), (16, 2), (16, 3), (1, 10), (1, 20), (1, 30)])
    result = readiness.project_runtime(records, 552)
    assert result["worker_seconds"] == 552 * (16 * 6 + 60)
    assert result["origins_per_series"] == 51
    assert result["mean_origin_extrapolation_worker_seconds"] != result["median_origin_extrapolation_worker_seconds"]
    records.append({"role": "cash_smoke", "status": "failed"})
    assert readiness.project_runtime(records, 552)["worker_seconds"] == result["worker_seconds"]
    records[1]["status"] = "failed"
    assert readiness.project_runtime(records, 552)["status"] == "incomplete_no_runtime_estimate"


def test_scheduler_keeps_long_model_bottleneck_and_all_fifteen():
    profiles = [{"model_id": name, "status": "passed", "peak_rss_bytes": 100,
                 "projection": {"status": "bounded_extrapolation", "worker_seconds": 3600 if i == 0 else 60}}
                for i, name in enumerate(selected.FACTORIES)]
    result = readiness.summarize(profiles)
    assert result["projected_elapsed_hours"] == 1
    assert result["total_worker_hours"] == pytest.approx(1 + 14 / 60)
    with pytest.raises(ValueError, match="fifteen"):
        readiness.summarize(profiles[:-1])
