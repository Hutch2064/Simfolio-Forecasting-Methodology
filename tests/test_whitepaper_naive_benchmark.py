"""Bind the corrected whitepaper comparator to complete expanded-panel evidence."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/results/expanded132/whitepaper-naive'


def test_full_benchmark_cells_and_equal_block_aggregate():
    aggregate = json.loads((OUT / 'aggregate.json').read_text())
    index = json.loads((OUT / 'files.json').read_text())
    for name, ref in index.items():
        raw = (OUT / name).read_bytes()
        assert len(raw) == ref['bytes']
        assert hashlib.sha256(raw).hexdigest() == ref['sha256']
    for block, shape, tasks in [('original80', (80, 8766), 4080), ('single52', (52, 8766), 2652)]:
        result = json.loads((OUT / f'{block}.json').read_text())
        manifest = json.loads(gzip.decompress((OUT / f'{block}-manifest.json.gz').read_bytes()))
        assert result['all_denominator_gates_passed']
        assert result['tasks'] == tasks
        assert len(manifest['task_identities']) == tasks
        assert manifest['requested_model_id'] == aggregate['model_id']
        assert manifest['simulations'] == 240
        source = 'tools/whitepaper_naive/models.py'
        assert manifest['source_hashes'][source] == hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
        with np.load(OUT / f'{block}-cells.npz', allow_pickle=False) as z:
            assert z['losses'].shape == shape
            assert np.isfinite(z['losses']).all() and (z['losses'] >= 0).all()
            assert (z['counts'] > 0).all()
            assert float(z['losses'].mean()) == result['exact_empirical_crps'] == aggregate['scores'][block]
    expected = .5 * aggregate['scores']['original80'] + .5 * aggregate['scores']['single52']
    assert expected == aggregate['equal_block_score'] == 0.6145512010153478
    assert aggregate['best_candidate_improvement_vs_naive_percent'] == 100 * (1 - .40781825383484904 / expected)


def test_retained_whitepaper_reproduction_and_correct_model_identity():
    audit = json.loads((OUT / 'whitepaper-reproduction-audit.json').read_text())
    binding = json.loads((OUT / 'source-binding.json').read_text())
    assert audit['passed'] and audit['old_and_new_original80_scores_byte_equal']
    assert audit['original80_score'] == .3819285166552024
    assert audit['all_8766_horizon_means_max_abs_difference'] < 1e-14
    assert audit['candidate_id'] == 'asset_level_naive_politis_white_stationary_bootstrap_rebalanced'
    assert binding['smoke']['selector_matches_retained_whitepaper_engine_cases'] == 21
    assert binding['smoke']['repeat_and_policy_cases'] == 4
