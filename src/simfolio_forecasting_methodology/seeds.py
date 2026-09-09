"""Deterministic random-seed derivation retained by the research harnesses."""

from __future__ import annotations

import hashlib
from typing import Any


def deterministic_seed(*parts: Any) -> int:
    """Derive the exact 32-bit seed used by retained Simfolio research code.

    Each part is converted with ``str``, UTF-8 encoded, and null-delimited into
    an 8-byte BLAKE2b digest. The little-endian digest integer is reduced modulo
    ``2**32 - 1``.
    """

    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "little") % (2**32 - 1)
