"""Causal constituent interval rows for the unchanged fixed panel targets."""
from pathlib import Path
from fractions import Fraction
import json
import numpy as np


class AssetInputs:
    def __init__(self, root):
        self.root = Path(root)
        self.panel = json.loads((self.root/'panel.json').read_text())
        self.by_id = {r['series_id']: r for r in self.panel['series']}

    def source(self, row):
        return (np.load(self.root/row['dates_file'], mmap_mode='r', allow_pickle=False),
                np.load(self.root/row['returns_file'], mmap_mode='r', allow_pickle=False))

    def training(self, row, stop):
        dates, target = self.source(row)
        if stop < 2 or stop > len(dates):
            raise ValueError('Invalid forecast training prefix')
        if row['panel_branch'] == 'assets':
            matrix = np.asarray(target[:stop]).reshape(-1, 1).copy()
            matrix.flags.writeable = False
            return matrix, np.ones(1), (row['series_id'],)
        if row['construction'] != 'constant_weights_common_observed_endpoint':
            raise ValueError('Unknown fixed-panel portfolio policy')
        rational = row['rational_weights']
        ids = tuple(sid for sid, _, _ in rational)
        weights = np.asarray([float(Fraction(n, d)) for _, n, d in rational])
        anchor = np.datetime64(row['provenance']['valuation_anchor'], 'D')
        endpoints = np.concatenate(([anchor], dates[:stop]))
        columns = []
        for sid in ids:
            ds, values = self.source(self.by_id[sid])
            end = int(np.searchsorted(ds, dates[stop-1], side='right'))
            ds, values = ds[:end], values[:end]  # never pass a future asset observation to a fit
            positions = np.searchsorted(ds, endpoints)
            if np.any(positions >= len(ds)) or not np.array_equal(ds[positions], endpoints):
                raise ValueError('Constituent observed endpoints do not match portfolio history')
            columns.append(np.diff(np.cumsum(values, dtype=np.float64)[positions]))
        matrix = np.stack(columns, axis=1)
        rebuilt = np.log1p(np.sum(np.expm1(matrix)*weights, axis=1))
        if not np.allclose(rebuilt, target[:stop], rtol=0, atol=2e-14):
            raise ValueError('Constituent accounting differs from fixed target stream')
        matrix.flags.writeable = False
        return matrix, weights, ids
