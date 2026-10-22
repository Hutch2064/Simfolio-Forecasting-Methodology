"""Fail-closed scanner for material intended for the public repository."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path

_TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".ini",
    ".cfg",
    ".sh",
}

# Construct sensitive literals in pieces so this scanner does not flag its own
# source merely because it names a forbidden token in a rule table.
_FORBIDDEN = (
    "from app" + ".engine",
    "import app" + ".engine",
    "simfolio" + "-engine",
    "simfolio" + "-web",
    "fly" + ".toml",
    "vercel" + ".json",
    "SUPABASE" + "_SERVICE_ROLE_KEY",
    "DATABASE" + "_URL",
    "AWS" + "_SECRET_ACCESS_KEY",
    "OPENAI" + "_API_KEY",
    "/Users/" + "aidan",
)

# Values are never returned in findings: diagnostics identify only the rule.
_SECRET_PATTERNS = {
    "github credential": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "signed URL": re.compile(r"[?&](?:X-Amz-Signature|X-Goog-Signature|Signature|sig)=[A-Za-z0-9%/+_-]{16,}", re.IGNORECASE),
    "credential in URL": re.compile(r"https?://[^\s/:]+:[^\s/@]+@", re.IGNORECASE),
    "developer home path": re.compile(r"(?:/Users/|/home/)[A-Za-z0-9_.-]+/"),
    "private provider or full-path payload": re.compile(
        r'"(?:private_provider_payload|raw_provider_data|provider_price_history|full_asset_paths|asset_path_cube)"\s*:'),
}

_PRIVATE_KEY_MARKERS = (
    "-----BEGIN " + "PRIVATE KEY-----",
    "-----BEGIN RSA " + "PRIVATE KEY-----",
    "-----BEGIN OPENSSH " + "PRIVATE KEY-----",
)

_EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
}

# These categories describe provenance, never credentials. An exact reviewed
# path+SHA permits only its listed categories; changed/new bytes fail closed.
_PROVENANCE = {
    "simfolio" + "-engine": "private_archive_reference",
    "simfolio" + "-web": "private_archive_reference",
    "from app" + ".engine": "literal_original_import",
    "import app" + ".engine": "literal_original_import",
    "/Users/" + "aidan": "developer_source_path",
    "developer home path": "developer_source_path",
}
_PAYLOAD_SUFFIXES = {
    ".csv", ".npz", ".npy", ".parquet", ".sqlite", ".sqlite3", ".db",
    ".pkl", ".pickle", ".feather", ".h5", ".hdf5", ".arrow", ".zip",
    ".tar", ".gz", ".xlsx", ".xls", ".bin", ".dat",
}
_BINARY_PREFIXES = {
    "github credential": (b"gh", b"github_pat_"),
    "AWS access key": (b"akia", b"asia"),
    "signed URL": tuple(separator + key + b"=" for separator in (b"?", b"&")
                        for key in (b"x-amz-signature", b"x-goog-signature", b"signature", b"sig")),
    "credential in URL": (b"http",),
    "developer home path": (b"/users/", b"/home/"),
    "private provider or full-path payload": tuple(b'"' + key for key in (
        b"private_provider_payload", b"raw_provider_data", b"provider_price_history",
        b"full_asset_paths", b"asset_path_cube")),
}


def _binary_findings(data: bytes) -> list[str]:
    lower = data.lower()
    findings = [marker for marker in (*_FORBIDDEN, *_PRIVATE_KEY_MARKERS)
                if marker.lower().encode() in lower]
    for name, pattern in _SECRET_PATTERNS.items():
        binary = re.compile(pattern.pattern.encode(), pattern.flags & ~re.UNICODE)
        found = False
        for prefix in _BINARY_PREFIXES[name]:
            start = lower.find(prefix)
            while start >= 0:
                if binary.match(data, start):
                    findings.append(name)
                    found = True
                    break
                start = lower.find(prefix, start + 1)
            if found:
                break
    return findings


def _content_findings(data: bytes, suffix: str) -> list[str]:
    """Scan archive contents too; approval never suppresses a credential."""
    if suffix in {".zip", ".npz", ".xlsx"}:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return [finding for name in archive.namelist() if not name.endswith("/")
                    for finding in (scan_text(name)
                                    + _content_findings(archive.read(name), Path(name).suffix))]
    if suffix == ".gz":
        return _content_findings(gzip.decompress(data), "")
    if suffix not in _TEXT_SUFFIXES and suffix:
        # Avoid interpreting large numerical arrays as Unicode; credentials in
        # binary payloads are still checked, including approved exact fixtures.
        return _binary_findings(data)
    return scan_text(data.decode("utf-8", errors="replace"))


def scan_text(text: str) -> list[str]:
    findings: list[str] = []
    for marker in (*_FORBIDDEN, *_PRIVATE_KEY_MARKERS):
        if marker.lower() in text.lower():
            findings.append(marker)
    findings.extend(name for name, pattern in _SECRET_PATTERNS.items() if pattern.search(text))
    return findings


def iter_public_text_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        if any(part in _EXCLUDED_PARTS for part in path.parts):
            continue
        yield path


def scan_tree(root: str | Path, *, approvals: dict | None = None) -> dict[str, list[str]]:
    base = Path(root)
    if approvals is None:
        approvals = json.loads((Path(__file__).parent / "resources/publication_safety_review.json")
                               .read_text())["files"]
    failures: dict[str, list[str]] = {}
    for path in base.rglob("*"):
        if not path.is_file() or any(part in _EXCLUDED_PARTS for part in path.parts):
            continue
        name = str(path.relative_to(base))
        data = path.read_bytes()
        approval = approvals.get(name, {})
        exact = approval.get("sha256") == hashlib.sha256(data).hexdigest()
        categories = (set(approval.get("provenance_categories", [])) & set(_PROVENANCE.values())
                      if exact else set())
        try:
            findings = scan_text(name) + _content_findings(data, path.suffix.lower())
        except (zipfile.BadZipFile, OSError, EOFError):
            findings = ["invalid payload archive"]
        findings = [finding for finding in findings
                    if _PROVENANCE.get(finding) not in categories]
        if path.suffix.lower() in _PAYLOAD_SUFFIXES and not (
                exact and approval.get("payload_review") in {
                    "reviewed_public_numeric_data", "published_source_or_evidence_archive"}):
            findings.append("unreviewed data or archive payload")
        if findings:
            failures[name] = sorted(set(findings))
    return failures


def assert_publication_safe(root: str | Path) -> None:
    failures = scan_tree(root)
    if failures:
        formatted = "; ".join(
            f"{path}: {', '.join(items)}" for path, items in sorted(failures.items())
        )
        raise RuntimeError(f"publication safety scan failed: {formatted}")
