import hashlib
import io
import zipfile
from pathlib import Path

from simfolio_forecasting_methodology.safety_scan import (
    assert_publication_safe,
    scan_text,
    scan_tree,
)


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


def reviewed(data, *, categories=(), payload=None):
    return {"sha256": hashlib.sha256(data).hexdigest(),
            "provenance_categories": list(categories), "payload_review": payload}


def test_only_exact_reviewed_provenance_categories_are_permitted(tmp_path):
    data = ("https://github.com/example/" + "simfolio" + "-engine/releases/tag/research\n"
            + "/Users/" + "researcher/source.py\n").encode()
    path = tmp_path / "receipt.json"
    path.write_bytes(data)
    approvals = {path.name: reviewed(data, categories=["private_archive_reference",
                                                     "developer_source_path"])}
    assert scan_tree(tmp_path, approvals=approvals) == {}
    assert scan_text(data.decode())  # Unknown text has no provenance exception.
    assert scan_tree(tmp_path, approvals={})
    path.write_bytes(data + b"changed")
    assert scan_tree(tmp_path, approvals=approvals)


def test_exact_provenance_approval_cannot_suppress_any_secret(tmp_path):
    samples = ["gh" + "p_" + "a" * 36, "-----BEGIN " + "PRIVATE KEY-----",
               "https://example.org/?X-Amz-" + "Signature=" + "a" * 64,
               "https://user:" + "secret@example.org/path"]
    for sample in samples:
        data = sample.encode()
        (tmp_path / "receipt.json").write_bytes(data)
        approval = reviewed(data, categories=["private_archive_reference", "developer_source_path",
                                              "literal_original_import", None])
        assert scan_tree(tmp_path, approvals={"receipt.json": approval})


def test_payloads_require_separate_exact_public_review(tmp_path):
    path = tmp_path / "provider.csv"
    path.write_bytes(b"date,value\n2020-01-01,10\n")
    approval = reviewed(path.read_bytes(), categories=["developer_source_path"])
    assert "unreviewed data or archive payload" in scan_tree(
        tmp_path, approvals={path.name: approval})[path.name]
    approval["payload_review"] = "reviewed_public_numeric_data"
    assert scan_tree(tmp_path, approvals={path.name: approval}) == {}
    path.write_bytes(path.read_bytes() + b"2020-01-02,11\n")
    assert scan_tree(tmp_path, approvals={path.name: approval})
    assert scan_text('{"' + 'private_' + 'provider_payload": [[10, 11]]}')


def test_exact_public_archive_review_never_suppresses_embedded_credential(tmp_path):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("metadata.txt", "gh" + "p_" + "a" * 36)
    data = output.getvalue()
    (tmp_path / "fixture.zip").write_bytes(data)
    approval = reviewed(data, payload="published_source_or_evidence_archive")
    assert "github credential" in scan_tree(tmp_path, approvals={"fixture.zip": approval})["fixture.zip"]


def test_binary_scanner_preserves_strict_secret_rules(tmp_path):
    samples = ["gh" + "p_" + "a" * 36, "github_" + "pat_" + "b" * 45,
               "AK" + "IA" + "A" * 16, "https://example.org/?sig=" + "a" * 20,
               "https://user:" + "secret@example.org/path",
               "-----BEGIN " + "PRIVATE KEY-----"]
    for sample in samples:
        data = b"\0" + sample.encode() + b"\0"
        (tmp_path / "fixture.npy").write_bytes(data)
        approval = reviewed(data, payload="reviewed_public_numeric_data")
        findings = scan_tree(tmp_path, approvals={"fixture.npy": approval})["fixture.npy"]
        assert set(scan_text(sample)).issubset(findings)


def test_secret_checks_include_unknown_file_extensions(tmp_path):
    (tmp_path / "credential.pem").write_text("-----BEGIN " + "PRIVATE KEY-----")
    assert scan_tree(tmp_path, approvals={})["credential.pem"]
