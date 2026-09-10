from pathlib import Path

from simfolio_forecasting_methodology.safety_scan import assert_publication_safe, scan_text


def test_scanner_rejects_production_import():
    forbidden = "from " + "app" + ".engine import Something"
    assert scan_text(forbidden)


def test_scanner_rejects_private_key_material():
    forbidden = "-----BEGIN " + "PRIVATE KEY-----"
    assert scan_text(forbidden)


def test_scanner_allows_normal_methodology_text():
    assert scan_text("exact empirical CRPS on terminal log returns") == []


def test_repository_tree_is_publication_safe():
    root = Path(__file__).resolve().parents[1]
    assert_publication_safe(root)


def test_scanner_rejects_credentials_and_signed_urls_without_echoing_values():
    samples = [
        "gh" + "p_" + "a" * 36,
        "AK" + "IA" + "A" * 16,
        "https://example.org/data?X-Amz-" + "Signature=" + "a" * 64,
        "https://user:" + "secret@example.org/path",
        "/home/" + "researcher/private/data.csv",
    ]
    for sample in samples:
        findings = scan_text(sample)
        assert findings
        assert sample not in " ".join(findings)
