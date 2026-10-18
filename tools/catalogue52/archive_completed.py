"""Bounded, lossless release archival of terminal, fully scored single52 models.

No network or cleanup happens on import. Partial/removed models remain local.
Every deletion requires an independently downloaded and exhaustively verified
archive, inventory and receipt. Runtime compiler caches are regenerable.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

import numpy as np
from campaign import (
    EXPECTED,
    PROTOCOL,
    atomic,
    digest,
    file_hash,
    lock,
    read,
    storage_ok,
    verified_process,
    verify_snapshot,
)

MAX_ARCHIVE_BYTES = 256 * 1024 ** 2


def checked_name(name):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
        raise ValueError("unsafe model name")
    return name


class GitHubRelease:
    """Existing release only; never overwrite an existing asset."""

    def __init__(self, repo, release):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("invalid GitHub repository")
        self.repo, self.release = repo, release
        result = self.run("view", release, "--json", "assets")
        self.assets = {asset["name"]: asset["url"] for asset in json.loads(result)["assets"]}

    def run(self, *args):
        return subprocess.run(["gh", "release", *args, "--repo", self.repo],
                              check=True, capture_output=True, text=True).stdout

    def download(self, name, directory):
        destination = Path(directory) / name
        if destination.exists():
            raise ValueError("download must use fresh staging")
        self.run("download", self.release, "--pattern", name, "--dir", str(directory))
        if not destination.is_file() or destination.is_symlink():
            raise ValueError("release download missing")
        return destination

    def put(self, path, recovery):
        path = Path(path)
        if path.name in self.assets:
            restored = self.download(path.name, recovery)
            if file_hash(restored) != file_hash(path):
                raise ValueError("existing remote asset differs; refusing overwrite")
            restored.unlink()
        else:
            self.run("upload", self.release, str(path))
            self.assets[path.name] = self.url(path.name)
        return self.url(path.name)

    def url(self, name):
        from urllib.parse import quote
        return f"https://github.com/{self.repo}/releases/download/{quote(self.release, safe='')}/{quote(name)}"


def validate_complete(output):
    """Rebuild the origin-first score from every byte-checked raw checkpoint."""
    spec, manifest = read(output / "registration.json"), read(output / "manifest.json")
    progress, result = read(output / "progress.json"), read(output / "results.json")
    if (spec["indices"] is not None or progress.get("status") != "complete"
            or progress.get("completed_tasks") != EXPECTED["tasks"]
            or progress.get("scheduled_tasks") != EXPECTED["tasks"]
            or progress.get("required_tasks") != EXPECTED["tasks"]
            or progress.get("full_score_eligible") is not True
            or result.get("all_denominator_gates_passed") is not True
            or any(result.get(key) != value for key, value in EXPECTED.items())
            or result.get("simulations") != 240 or result.get("block") != "single52"):
        raise ValueError("full-score denominator gates failed")
    if (manifest["protocol"] != PROTOCOL or manifest["registration"] != spec
            or digest(manifest["campaign"]) != spec["campaign_sha256"]):
        raise ValueError("manifest registration/campaign mismatch")
    binding = digest(manifest)
    if progress["manifest_sha256"] != binding or result["manifest_sha256"] != binding:
        raise ValueError("score manifest binding mismatch")
    identities = manifest["task_identities"]
    if len(identities) != EXPECTED["tasks"] or len({item["task_id"] for item in identities}) != len(identities):
        raise ValueError("task identity denominator mismatch")
    names = sorted({item["portfolio_id"] for item in identities})
    width = max(item["horizon_days"] for item in identities)
    if len(names) != EXPECTED["portfolios"] or len(names) * width != EXPECTED["cells"]:
        raise ValueError("portfolio/cell denominator mismatch")
    positions = {name: i for i, name in enumerate(names)}
    sums, counts = np.zeros((len(names), width)), np.zeros((len(names), width), dtype=np.int64)
    seconds = 0.
    for index, identity in enumerate(identities):
        path = output / f"task-{index:04d}.npz"
        task = read(path.with_suffix(".json"))
        horizon = identity["horizon_days"]
        if (task["index"] != index or task["manifest_sha256"] != binding
                or task["sha256"] != file_hash(path) or task["horizon"] != horizon
                or task["portfolio"] != identity["portfolio_id"] or task["seed"] != identity["seed"]
                or task["origin"] != identity["origin_date"]):
            raise ValueError("raw task identity/checksum mismatch")
        with np.load(path, allow_pickle=False) as saved:
            losses = saved["losses"]
            if (str(saved["manifest"]) != binding or losses.shape != (horizon,)
                    or not np.isfinite(losses).all() or (losses < 0).any()):
                raise ValueError("raw task loss shape/finiteness mismatch")
            row = positions[identity["portfolio_id"]]
            sums[row, :horizon] += losses
            counts[row, :horizon] += 1
        if not math.isfinite(task["seconds"]) or task["seconds"] < 0:
            raise ValueError("invalid worker timing")
        seconds += task["seconds"]
    if (counts == 0).any():
        raise ValueError("incomplete cell denominators")
    cells = sums / counts
    if file_hash(output / "cells.npz") != result["cells_sha256"]:
        raise ValueError("cells checksum mismatch")
    with np.load(output / "cells.npz", allow_pickle=False) as saved:
        if (not np.array_equal(saved["portfolio"], names)
                or not np.array_equal(saved["counts"], counts)
                or not np.array_equal(saved["losses"], cells)):
            raise ValueError("raw tasks/cells reduction mismatch")
    if (not math.isfinite(result["exact_empirical_crps"])
            or result["exact_empirical_crps"] != float(cells.mean())
            or result["portfolio_scores"] != dict(zip(names, cells.mean(axis=1).tolist()))
            or result["summed_worker_seconds"] != seconds):
        raise ValueError("score summary mismatch")
    verify_snapshot(spec["snapshot"], spec["source_sha256"])
    return spec


def payloads(output, spec):
    """Explicit evidence allowlist excludes credentials, inputs and runtime caches."""
    paths = {}
    for pattern in ("task-*.npz", "task-*.json", "failure-*.json"):
        paths.update({f"model/{p.name}": p for p in output.glob(pattern)})
    for name in ("cells.npz", "results.json", "manifest.json", "registration.json", "progress.json",
                 "process.json", "ready.json", "error.json", "interruption.json", "worker.log"):
        path = output / name
        if path.exists():
            paths[f"model/{name}"] = path
    snapshot = Path(spec["snapshot"])
    paths["sourcesnapshot.json"] = snapshot / "snapshot.json"
    for path in paths.values():
        if not path.is_file() or path.is_symlink():
            raise ValueError("evidence must be regular files without symlinks")
    return paths


def build_archive(output, spec, staging, max_bytes):
    paths = payloads(output, spec)
    members = {name: {"sha256": file_hash(path), "bytes": path.stat().st_size}
               for name, path in sorted(paths.items())}
    # Bound source and raw evidence before writing staging; never sweep a campaign.
    if sum(item["bytes"] for item in members.values()) > max_bytes:
        raise ValueError("model exceeds bounded archival size; preserve local evidence")
    config = read(output.parent.parent / "campaign.json")
    if not storage_ok(output.parent.parent, config,
                      additional_bytes=2 * sum(item["bytes"] for item in members.values()) + 2 * 1024 ** 2):
        raise ValueError("archive staging exceeds campaign budget/free-space reserve")
    identity = digest(members)
    stem = f"catalogue52-{checked_name(output.name)}-{identity[:20]}"
    archive = staging / f"{stem}.tar.gz"
    with (archive.open("wb") as raw,
          gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="", compresslevel=6) as zipped,
          tarfile.open(fileobj=zipped, mode="w|") as tar):
        for name, path in sorted(paths.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode = members[name]["bytes"], 0o644
            with path.open("rb") as stream:
                tar.addfile(info, stream)
    if archive.stat().st_size > max_bytes:
        raise ValueError("compressed archive exceeds bounded archival size")
    inventory = {"format": "catalogue52-lossless-archive-v1", "model": output.name,
                 "source_sha256": spec["source_sha256"], "archive": archive.name,
                 "archive_sha256": file_hash(archive), "archive_bytes": archive.stat().st_size,
                 "members": members}
    manifest_path = staging / f"{stem}.manifest.json"
    atomic(manifest_path, inventory)
    return archive, manifest_path, inventory


def verify_archive(archive, inventory):
    """Read every restored member; never trust tar extraction paths or links."""
    if file_hash(archive) != inventory["archive_sha256"] or archive.stat().st_size != inventory["archive_bytes"]:
        raise ValueError("downloaded archive checksum/size mismatch")
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute() or ".." in name.parts
                    or str(name) != member.name or member.name in seen
                    or member.name not in inventory["members"]):
                raise ValueError("unsafe, duplicate or unexpected archive member")
            expected = inventory["members"][member.name]
            if member.size != expected["bytes"]:
                raise ValueError("restored member size mismatch")
            with tar.extractfile(member) as stream:
                data = stream.read()
            if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
                raise ValueError("restored member checksum mismatch")
            if member.name.endswith(".json"):
                json.loads(data)
            elif member.name.endswith(".npz"):
                with np.load(io.BytesIO(data), allow_pickle=False) as values:
                    for key in values.files:
                        values[key]  # Force ZIP CRC and array readability for every member.
            seen.add(member.name)
    if seen != inventory["members"].keys():
        raise ValueError("archive missing inventory members")


def remote_proof(remote, receipt, staging):
    inventory_path = remote.download(receipt["manifest"]["name"], staging)
    if file_hash(inventory_path) != receipt["manifest"]["sha256"]:
        raise ValueError("downloaded inventory checksum mismatch")
    inventory = read(inventory_path)
    if (inventory["archive"] != receipt["archive"]["name"]
            or inventory["archive_sha256"] != receipt["archive"]["sha256"]):
        raise ValueError("archive receipt/inventory mismatch")
    archive = remote.download(receipt["archive"]["name"], staging)
    verify_archive(archive, inventory)
    return archive, inventory


def cleanup(output, inventory):
    """Remove only checksum-matched evidence whose remote restoration was proved."""
    runtime = output / "runtime"
    if runtime.is_symlink():
        raise ValueError("refusing runtime symlink cleanup")
    removals = []
    for name, item in inventory["members"].items():
        relative = PurePosixPath(name)
        if len(relative.parts) != 2 or relative.parts[0] != "model":
            continue
        leaf = relative.parts[1]
        if leaf == "cells.npz" or re.fullmatch(r"task-\d{4}\.(npz|json)", leaf):
            path = output / leaf
            if path.exists():
                if path.is_symlink() or file_hash(path) != item["sha256"]:
                    raise ValueError("local evidence changed; refusing cleanup")
                removals.append(path)
    # Check every deletion first; a mismatch must leave the whole payload intact.
    for path in removals:
        path.unlink()
    if runtime.exists():
        shutil.rmtree(runtime)


def archive_model(campaign, name, remote, *, max_bytes=MAX_ARCHIVE_BYTES):
    campaign = Path(campaign).resolve()
    output = campaign / "models" / checked_name(name)
    if output.is_symlink() or output.parent.is_symlink():
        raise ValueError("model storage must not be a symlink")
    with lock(output / "worker.lock", nonblocking=True):
        state = read(output / "process.json")
        if verified_process(state):
            return None
        progress = read(output / "progress.json")
        if progress.get("status") != "complete":
            return None  # Removed, paused, failed and smoke evidence are preserved.
        before = shutil.disk_usage(campaign).free
        receipt_path = output / "archive-receipt.json"
        # A prior proof may precede an interrupted cleanup. Re-prove it remotely.
        with tempfile.TemporaryDirectory(prefix=".archive-one-", dir=campaign) as temporary:
            staging = Path(temporary)
            downloaded = staging / "downloaded"
            downloaded.mkdir()
            if receipt_path.exists():
                local = read(receipt_path)
                receipt = local["remote_receipt"]
                if receipt["repo"] != remote.repo or receipt["release"] != remote.release:
                    raise ValueError("archive destination differs from saved receipt")
                saved_receipt = remote.download(receipt["receipt_name"], downloaded)
                if file_hash(saved_receipt) != local["remote_receipt_sha256"] or read(saved_receipt) != receipt:
                    raise ValueError("downloaded receipt mismatch")
                _, inventory = remote_proof(remote, receipt, downloaded)
            else:
                spec = validate_complete(output)
                archive, manifest, inventory = build_archive(output, spec, staging, max_bytes)
                remote.put(archive, downloaded)
                remote.put(manifest, downloaded)
                receipt_name = archive.name.removesuffix(".tar.gz") + ".receipt.json"
                receipt = {"format": "catalogue52-archive-receipt-v1", "status": "archived",
                           "model": name, "repo": remote.repo, "release": remote.release,
                           "receipt_name": receipt_name,
                           "archive": {"name": archive.name, "sha256": file_hash(archive), "url": remote.url(archive.name)},
                           "manifest": {"name": manifest.name, "sha256": file_hash(manifest), "url": remote.url(manifest.name)},
                           "receipt_url": remote.url(receipt_name),
                           "kept_local_paths": [str(p) for p in sorted(output.iterdir())
                               if p.name != "runtime" and not re.fullmatch(r"task-\d{4}\.(npz|json)|cells\.npz", p.name)],
                           "restore_command": shlex.join(["python", str(Path(__file__).resolve()),
                               "--campaign", str(campaign), "--repo", remote.repo,
                               "--release", remote.release, "--restore", name])}
                remote_proof(remote, receipt, downloaded)
                receipt_file = staging / receipt_name
                atomic(receipt_file, receipt)
                remote.put(receipt_file, downloaded)
                restored_receipt = remote.download(receipt_name, downloaded)
                if restored_receipt.read_bytes() != receipt_file.read_bytes():
                    raise ValueError("downloaded receipt mismatch")
                local = {"status": "archived", "remote_receipt": receipt,
                         "remote_receipt_sha256": file_hash(restored_receipt)}
                atomic(receipt_path, local)  # Durable proof precedes any deletion.
            if verified_process(read(output / "process.json")):
                raise ValueError("model became active; refusing cleanup")
            cleanup(output, inventory)
        local["storage"] = {"free_before_bytes": before, "free_after_bytes": shutil.disk_usage(campaign).free}
        atomic(receipt_path, local)
        return local


def restore_model(campaign, name, remote):
    campaign = Path(campaign).resolve()
    output = campaign / "models" / checked_name(name)
    with lock(output / "worker.lock", nonblocking=True):
        if verified_process(read(output / "process.json")):
            raise ValueError("cannot restore an active model")
        local = read(output / "archive-receipt.json")
        receipt = local["remote_receipt"]
        if (remote.repo, remote.release) != (receipt["repo"], receipt["release"]):
            raise ValueError("restore destination differs from saved receipt")
        with tempfile.TemporaryDirectory(prefix=".restore-one-", dir=campaign) as temporary:
            staging = Path(temporary)
            downloaded_receipt = remote.download(receipt["receipt_name"], staging)
            if file_hash(downloaded_receipt) != local["remote_receipt_sha256"] or read(downloaded_receipt) != receipt:
                raise ValueError("downloaded receipt mismatch")
            archive, inventory = remote_proof(remote, receipt, staging)
            needed = sum(item["bytes"] for name, item in inventory["members"].items()
                         if name.startswith("model/") and not (output / PurePosixPath(name).name).exists())
            if not storage_ok(campaign, read(campaign / "campaign.json"), additional_bytes=needed):
                raise ValueError("restoration exceeds campaign budget/free-space reserve")
            with tarfile.open(archive, "r:gz") as tar:
                for member in tar:
                    relative = PurePosixPath(member.name)
                    if len(relative.parts) != 2 or relative.parts[0] != "model":
                        continue
                    target = output / relative.parts[1]
                    if target.exists():
                        if target.is_symlink() or file_hash(target) != inventory["members"][member.name]["sha256"]:
                            raise ValueError("restore refuses to overwrite differing local evidence")
                        continue
                    with tar.extractfile(member) as source, target.with_suffix(target.suffix + ".restore-tmp").open("xb") as stream:
                        shutil.copyfileobj(source, stream)
                    target.with_suffix(target.suffix + ".restore-tmp").replace(target)
            validate_complete(output)
        local["status"] = "restored"
        atomic(output / "archive-restored.json", local)
        (output / "archive-receipt.json").unlink()
        return local


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--once", action="store_true", help="Archive at most one completed model")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--restore", metavar="MODEL")
    args = parser.parse_args(argv)
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be at least 1")
    os.nice(15)
    remote = GitHubRelease(args.repo, args.release)
    if args.restore:
        restore_model(args.campaign, args.restore, remote)
        return
    while True:
        for output in sorted((args.campaign / "models").iterdir()):
            progress = output / "progress.json"
            if (not output.is_dir() or not progress.exists()
                    or read(progress).get("status") != "complete"):
                continue
            if (output / "archive-receipt.json").exists() and not any(output.glob("task-*.npz")) and not (output / "runtime").exists():
                continue
            try:
                result = archive_model(args.campaign, output.name, remote)
            except BlockingIOError:
                continue
            except (OSError, ValueError, subprocess.CalledProcessError) as error:
                if args.once:
                    raise
                print(json.dumps({"model": output.name, "status": "preserved_local",
                                  "error": type(error).__name__, "message": str(error)}), flush=True)
                continue
            if result:
                print(json.dumps({"model": output.name, **result["storage"]}), flush=True)
                break
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
