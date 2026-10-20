# French551 public-only search — 2026-10-20

The previous official sweep was cancelled by the user. No nonpublic scores from that campaign are included here. The only scored model in this track is the completed public adaptive-block reference: 551 series, 28,101/28,101 origins, primary scoring-v2 value 0.7272852027781417. Its exact preserved summary is `public-baseline-v2.json`.

Two additional public comparators are implemented and independently validated, but have zero scored origins: uniform bounded circular blocks with replacement, and uniform feasible blocks with no observation reuse within a history cycle. The user-selected exhaustion rule refills the unused-history pool only after each complete cycle; available blocks shorten as the current cycle is depleted. See `COMPARATORS.md`, `model_catalogue.json`, and the [comparator](comparator-validation-receipt.json) and [scoring-r2](scoring-r2-validation-receipt.json) validation receipts.

The scorer and reporters are frozen source-only artifacts. The full-panel comparator control is stopped; no comparator forecast or reporting daemon had been launched at this publication snapshot. Representative development is authorized; the full-panel run remains gated on the stated research stopping criterion. This publication contains source, tests, the one completed public baseline, and compact status metadata. It does not contain generated plan arrays, private input payloads, credentials, or any cancelled-campaign nonpublic scores.

Validated source is under `../../../scripts/` and tests under `../../../tests/`. Run focused checks from the repository root with:

```bash
PYTHONPATH=src:scripts pytest -q tests/test_bounded_public.py tests/test_bounded_public_without_replacement.py
PYTHONPATH=src:scripts pytest -q tests/test_french_score_sidecar.py tests/test_french_score_sidecar_r2.py
```
