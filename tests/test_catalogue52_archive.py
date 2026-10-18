"""Real archives and mock network: fail closed before deleting unique evidence."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/catalogue52"))
import archive_completed as archive
import campaign


class Remote:
    repo = "Hutch2064/Simfolio-Forecasting-Methodology"
    release = "test-only"

    def __init__(self, *, fail=None, corrupt=None):
        self.assets, self.events = {}, []
        self.fail, self.corrupt = fail, corrupt

    def url(self, name):
        return f"https://example.test/{name}"

    def put(self, path, recovery):
        self.events.append(("put", path.name))
        if self.fail and self.fail in path.name:
            raise OSError("mock upload failed")
        data = path.read_bytes()
        if path.name in self.assets and self.assets[path.name] != data:
            raise ValueError("existing remote asset differs")
        self.assets[path.name] = data
        return self.url(path.name)

    def download(self, name, directory):
        self.events.append(("download", name))
        data = self.assets[name]
        if self.corrupt and self.corrupt in name:
            data += b"corrupt mock transfer"
        path = directory / name
        assert not path.exists(), "downloads must always use fresh staging"
        path.write_bytes(data)
        return path


@pytest.fixture(scope="module")
def full_evidence(tmp_path_factory):
    """Full 2,652-task/455,832-cell denominator, no forecasting or provider data."""
    root = tmp_path_factory.mktemp("archive-canonical-denominators")
    source = root / "source"
    (source / "tools/catalogue52").mkdir(parents=True)
    (source / "src").mkdir()
    (source / "tools/catalogue52/worker.py").write_text("# immutable test fixture\n")
    snapshot, source_sha = campaign.freeze_source(source, root / "snapshots")
    config = {"protocol": campaign.PROTOCOL, "simulations": 240, "expected": campaign.EXPECTED,
              "budget_gib": 1, "min_free_gib": 0}
    output = root / "campaign/models/fixture"
    output.mkdir(parents=True)
    campaign.atomic(output.parent.parent / "campaign.json", config)
    spec = {"indices": None, "snapshot": str(snapshot), "source_sha256": source_sha,
            "campaign_sha256": campaign.digest(config)}
    identities = [{"task_id": str(i), "portfolio_id": f"single_{i // 51:02d}",
                   "horizon_days": 8766, "seed": i, "origin_date": f"origin-{i}"}
                  for i in range(2652)]
    manifest = {"protocol": campaign.PROTOCOL, "registration": spec,
                "campaign": config, "task_identities": identities}
    binding = campaign.digest(manifest)
    campaign.atomic(output / "registration.json", spec)
    campaign.atomic(output / "manifest.json", manifest)
    campaign.atomic(output / "progress.json", {"status": "complete", "completed_tasks": 2652,
                    "scheduled_tasks": 2652, "required_tasks": 2652,
                    "full_score_eligible": True, "manifest_sha256": binding})
    campaign.atomic(output / "process.json", {"pid": 999999999, "token": "0" * 32,
                    "started": "fixture", "worker": "fixture", "ready": "fixture"})
    # NPZs are losslessly compressed; this bounded fixture occupies only a few MB.
    for index, identity in enumerate(identities):
        path = output / f"task-{index:04d}.npz"
        np.savez_compressed(path, manifest=binding, losses=np.full(8766, .5))
        campaign.atomic(path.with_suffix(".json"), {"index": index, "manifest_sha256": binding,
                    "sha256": campaign.file_hash(path), "horizon": 8766,
                    "portfolio": identity["portfolio_id"], "seed": index,
                    "origin": identity["origin_date"], "seconds": 1., "diagnostics": ["retained"]})
    names = [f"single_{i:02d}" for i in range(52)]
    np.savez_compressed(output / "cells.npz", portfolio=names,
                       losses=np.full((52, 8766), .5), counts=np.full((52, 8766), 51))
    campaign.atomic(output / "results.json", {"block": "single52", "simulations": 240,
                    **campaign.EXPECTED, "all_denominator_gates_passed": True,
                    "exact_empirical_crps": .5, "manifest_sha256": binding,
                    "portfolio_scores": dict.fromkeys(names, .5), "summed_worker_seconds": 2652.,
                    "cells_sha256": campaign.file_hash(output / "cells.npz")})
    (output / "worker.log").write_text("complete model log\n")
    campaign.atomic(output / "error.json", {"prior_attempt": "retained diagnostic"})
    (output / "runtime/numba").mkdir(parents=True)
    (output / "runtime/numba/generated.nbc").write_bytes(b"regenerable runtime cache")
    return output.parent.parent


@pytest.fixture
def completed(tmp_path, full_evidence):
    root = tmp_path / "campaign"
    shutil.copytree(full_evidence, root)
    return root, root / "models/fixture"


def assert_payload_present(output):
    assert len(list(output.glob("task-*.npz"))) == 2652
    assert len(list(output.glob("task-*.json"))) == 2652
    assert (output / "cells.npz").exists()
    assert (output / "runtime/numba/generated.nbc").exists()


def test_full_archive_roundtrip_receipt_before_cleanup_and_explicit_restore(completed, monkeypatch):
    root, output = completed
    remote = Remote()
    original = (output / "task-0000.npz").read_bytes()
    cleanup = archive.cleanup

    def checked_cleanup(output, inventory):
        assert_payload_present(output)
        assert (output / "archive-receipt.json").exists()
        assert remote.events[-1][0] == "download"
        assert remote.events[-1][1].endswith(".receipt.json")
        cleanup(output, inventory)

    monkeypatch.setattr(archive, "cleanup", checked_cleanup)
    local = archive.archive_model(root, "fixture", remote)
    assert local["status"] == "archived"
    assert local["storage"]["free_before_bytes"] > 0
    assert local["storage"]["free_after_bytes"] > 0
    assert not list(output.glob("task-*.npz")) and not (output / "cells.npz").exists()
    assert not (output / "runtime").exists()
    assert (output / "results.json").exists() and (output / "manifest.json").exists()
    assert not list(root.glob(".archive-one-*"))
    inventory = json.loads(next(value for key, value in remote.assets.items() if key.endswith(".manifest.json")))
    assert "sourcesnapshot.json" in inventory["members"]
    assert "model/error.json" in inventory["members"]
    assert "model/worker.log" in inventory["members"]
    assert len([name for name in inventory["members"] if name.startswith("model/task-")]) == 5304
    assert not any(name.startswith("source/") or "runtime" in name for name in inventory["members"])
    archive.restore_model(root, "fixture", remote)
    assert (output / "task-0000.npz").read_bytes() == original
    assert len(list(output.glob("task-*.npz"))) == 2652
    assert not (output / "archive-receipt.json").exists()
    assert archive.validate_complete(output)


@pytest.mark.parametrize("options", [{"fail": ".tar.gz"}, {"fail": ".manifest.json"},
    {"fail": ".receipt.json"}, {"corrupt": ".tar.gz"}, {"corrupt": ".manifest.json"},
    {"corrupt": ".receipt.json"}])
def test_transfer_failure_never_cleans_unique_evidence(completed, options):
    root, output = completed
    remote = Remote(**options)
    with pytest.raises((OSError, ValueError)):
        archive.archive_model(root, "fixture", remote)
    assert_payload_present(output)
    assert not (output / "archive-receipt.json").exists()
    assert not list(root.glob(".archive-one-*"))


def test_partial_cleanup_crash_reproves_remote_and_finishes(completed, monkeypatch):
    root, output = completed
    remote = Remote()
    cleanup = archive.cleanup

    def interrupted(output, inventory):
        (output / "task-0000.npz").unlink()
        raise OSError("fixture interruption after durable remote proof")

    monkeypatch.setattr(archive, "cleanup", interrupted)
    with pytest.raises(OSError, match="fixture interruption"):
        archive.archive_model(root, "fixture", remote)
    assert (output / "archive-receipt.json").exists()
    monkeypatch.setattr(archive, "cleanup", cleanup)
    previous = len(remote.events)
    archive.archive_model(root, "fixture", remote)
    assert all(event == "download" for event, name in remote.events[previous:])
    assert not list(output.glob("task-*.npz"))
    assert not (output / "runtime").exists()


def test_active_worker_and_held_worker_lock_are_never_archived(completed, monkeypatch):
    root, output = completed
    remote = Remote()
    monkeypatch.setattr(archive, "verified_process", lambda state: True)
    assert archive.archive_model(root, "fixture", remote) is None
    assert_payload_present(output)
    assert not remote.events
    monkeypatch.setattr(archive, "verified_process", lambda state: False)
    with campaign.lock(output / "worker.lock", nonblocking=True), pytest.raises(BlockingIOError):
        archive.archive_model(root, "fixture", remote)
    assert_payload_present(output)
    assert not remote.events


def test_removed_partial_smoke_and_nonfinite_evidence_are_preserved(completed):
    root, output = completed
    remote = Remote()
    progress = campaign.read(output / "progress.json")
    for status in ("paused", "failed", "storage_paused", "smoke_complete"):
        campaign.atomic(output / "progress.json", dict(progress, status=status))
        assert archive.archive_model(root, "fixture", remote) is None
    campaign.atomic(output / "progress.json", progress)
    result = campaign.read(output / "results.json")
    result["exact_empirical_crps"] = float("nan")
    (output / "results.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="score summary mismatch"):
        archive.archive_model(root, "fixture", remote)
    assert_payload_present(output)
    assert not remote.events


def test_raw_corruption_and_bounded_size_fail_before_upload(completed):
    root, output = completed
    remote = Remote()
    with pytest.raises(ValueError, match="bounded archival size"):
        archive.archive_model(root, "fixture", remote, max_bytes=1)
    assert_payload_present(output)
    path = output / "task-0000.npz"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="identity/checksum"):
        archive.archive_model(root, "fixture", remote)
    assert_payload_present(output)
    assert not remote.events


@pytest.mark.parametrize("name", ["../escape.json", "/absolute.json", "model/../escape.json"])
def test_tar_verifier_refuses_unsafe_paths_even_with_matching_digest(tmp_path, name):
    path = tmp_path / "unsafe.tar.gz"
    data = b"{}"
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    inventory = {"archive_sha256": campaign.file_hash(path), "archive_bytes": path.stat().st_size,
                 "members": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}}}
    with pytest.raises(ValueError, match="unsafe"):
        archive.verify_archive(path, inventory)
    assert not (tmp_path / "escape.json").exists()


def test_tar_verifier_reads_every_member_and_rejects_member_hash_mismatch(tmp_path):
    path = tmp_path / "bad-member.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name in ("first.json", "last.json"):
            info = tarfile.TarInfo(name)
            info.size = 2
            tar.addfile(info, io.BytesIO(b"{}"))
    inventory = {"archive_sha256": campaign.file_hash(path), "archive_bytes": path.stat().st_size,
                 "members": {name: {"sha256": hashlib.sha256(b"{}").hexdigest(), "bytes": 2}
                             for name in ("first.json", "last.json")}}
    inventory["members"]["last.json"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="member checksum"):
        archive.verify_archive(path, inventory)


def test_github_existing_asset_is_downloaded_and_never_clobbered(tmp_path):
    """Exercise the real gh wrapper without network, including collision recovery."""
    class ExistingRelease(archive.GitHubRelease):
        def run(self, *args):
            self.calls = getattr(self, "calls", []) + [args]
            if args[0] == "view":
                return json.dumps({"assets": [{"name": "evidence.tar.gz", "url": "https://example.test/asset"}]})
            if args[0] == "download":
                destination = Path(args[args.index("--dir") + 1]) / args[args.index("--pattern") + 1]
                destination.write_bytes(b"same evidence")
            return ""

    remote = ExistingRelease("Hutch2064/Simfolio-Forecasting-Methodology", "test-only")
    path = tmp_path / "evidence.tar.gz"
    path.write_bytes(b"same evidence")
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    remote.put(path, recovery)
    assert not any(call[0] == "upload" for call in remote.calls)
    assert not list(recovery.iterdir())
    path.write_bytes(b"different evidence")
    with pytest.raises(ValueError, match="refusing overwrite"):
        remote.put(path, recovery)
    assert not any("--clobber" in call for call in remote.calls)
