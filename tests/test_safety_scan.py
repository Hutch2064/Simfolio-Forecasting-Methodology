from pathlib import Path

import pytest

from simfolio_forecasting_methodology.safety_scan import assert_publication_safe, scan_text


def test_scanner_rejects_production_import():
    assert scan_text("from app.engine import Something")


def test_scanner_rejects_private_key_material():
    assert scan_text("-----BEGIN PRIVATE KEY-----")


def test_scanner_allows_normal_methodology_text():
    assert scan_text("exact empirical CRPS on terminal log returns") == []


def test_repository_tree_is_publication_safe():
    root = Path(__file__).resolve().parents[1]
    assert_publication_safe(root)
