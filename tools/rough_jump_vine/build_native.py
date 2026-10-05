"""Build/install the parity-tested C++ vine optimization from pinned sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

UPSTREAM = "508973584f0824ea079003d935a502b81a39caa0"
HEADERS = (
    (
        "boost_1_84_0.tar.bz2",
        "https://archives.boost.io/release/1.84.0/source/boost_1_84_0.tar.bz2",
        "cc4b893acf645c9d4b698e9a0f08ca8846aa5d6c68275c14c3e7949c24109454",
    ),
    (
        "eigen-3.4.0.tar.gz",
        "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz",
        "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72",
    ),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(arguments, **kwargs):
    subprocess.run(arguments, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "cmake==4.4.4",
            "ninja==1.13.2",
            "nanobind==3.1.0",
            "scikit-build-core==1.1.1",
        ]
    )
    source = workspace / "pyvinecopulib"
    if not source.exists():
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "v0.7.5",
                "https://github.com/vinecopulib/pyvinecopulib.git",
                str(source),
            ]
        )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    assert revision == UPSTREAM, "native upstream revision differs"
    run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], cwd=source)
    for filename, url, digest in HEADERS:
        archive = workspace / filename
        if not archive.exists():
            urllib.request.urlretrieve(url, archive)
        assert sha(archive) == digest, f"header archive differs: {filename}"
        with tarfile.open(archive) as handle:
            members = handle.getmembers()
            if filename.startswith("boost"):
                members = [
                    m
                    for m in members
                    if m.name.startswith("boost_1_84_0/boost/")
                    or m.name == "boost_1_84_0/LICENSE_1_0.txt"
                ]
            handle.extractall(workspace, members=members, filter="data")
    patch = Path(__file__).with_name("native.patch")
    library = source / "lib/vinecopulib"
    applied = (
        subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch)],
            cwd=library,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )
    if not applied:
        run(["git", "apply", "--check", str(patch)], cwd=library)
        run(["git", "apply", str(patch)], cwd=library)
    actual_patch = subprocess.check_output(
        ["git", "diff", "--no-color", "--no-ext-diff"], cwd=library
    )
    assert actual_patch == patch.read_bytes(), "native source contains other changes"
    environment = dict(os.environ)
    environment.update(
        Boost_INCLUDE_DIR=str(workspace / "boost_1_84_0"),
        EIGEN3_INCLUDE_DIR=str(workspace / "eigen-3.4.0"),
        CMAKE_BUILD_PARALLEL_LEVEL="6",
    )
    wheels = workspace / "wheels"
    wheels.mkdir(exist_ok=True)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            str(source),
            "--wheel-dir",
            str(wheels),
            f"-Cbuild-dir={workspace / 'build'}",
            "-Ccmake.define.CMAKE_POLICY_VERSION_MINIMUM=3.5",
        ],
        env=environment,
    )
    wheel = max(wheels.glob("*.whl"), key=lambda p: p.stat().st_mtime)
    run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)])
    receipt = {
        "upstream_revision": revision,
        "vinecopulib_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=library, text=True
        ).strip(),
        "patch_sha256": sha(patch),
        "wheel_sha256": sha(wheel),
        "headers": {filename: digest for filename, _, digest in HEADERS},
    }
    (workspace / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
