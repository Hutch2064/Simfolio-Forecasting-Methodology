"""Fail-closed scanner for material intended for the public repository."""

from __future__ import annotations

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
    "dist",
    "build",
}


def scan_text(text: str) -> list[str]:
    findings: list[str] = []
    for marker in (*_FORBIDDEN, *_PRIVATE_KEY_MARKERS):
        if marker.lower() in text.lower():
            findings.append(marker)
    return findings


def iter_public_text_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        if any(part in _EXCLUDED_PARTS for part in path.parts):
            continue
        yield path


def scan_tree(root: str | Path) -> dict[str, list[str]]:
    base = Path(root)
    failures: dict[str, list[str]] = {}
    for path in iter_public_text_files(base):
        findings = scan_text(path.read_text(encoding="utf-8", errors="replace"))
        if findings:
            failures[str(path.relative_to(base))] = findings
    return failures


def assert_publication_safe(root: str | Path) -> None:
    failures = scan_tree(root)
    if failures:
        formatted = "; ".join(
            f"{path}: {', '.join(items)}" for path, items in sorted(failures.items())
        )
        raise RuntimeError(f"publication safety scan failed: {formatted}")
