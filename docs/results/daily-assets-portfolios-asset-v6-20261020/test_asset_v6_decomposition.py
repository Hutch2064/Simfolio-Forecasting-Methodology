import unittest
import numpy as np
from audit_asset_v6_full_pair import independent_reduce
from diagnose_asset_v6_pair import origin_weights


class AdditiveDiagnosticTests(unittest.TestCase):
    def roster(self):
        result, metadata, records = [], {}, []
        rng = np.random.default_rng(71)
        for i, (branch, cls, family, count) in enumerate([
            ('assets', 'equity', 'value', 3), ('assets', 'equity', 'value', 1),
            ('assets', 'bonds', 'duration', 2), ('portfolios', 'mix', 'balanced', 3),
            ('portfolios', 'mix', 'balanced', 1), ('portfolios', 'mix', 'concentrated', 2)]):
            sid = str(i)
            metadata[sid] = dict(panel_branch=branch, asset_class=cls, family=family, series_id=sid)
            for j in range(count):
                date = f'2000-02-{j+1:02}'
                result.append(((i, j), metadata[sid], date))
                records.append((sid, date, float(rng.uniform(0, 3))))
        m = metadata['0']
        result.append(((0, 9), m, '1990-02-01'))
        records.append(('0', '1990-02-01', .7))
        return result, metadata, records

    def test_flattened_weights_equal_nested_canonical_average(self):
        roster, metadata, records = self.roster()
        weights = origin_weights(roster)
        value = sum(weights[key]*record[2] for (key, _, _), record in zip(roster, records))
        self.assertAlmostEqual(value, independent_reduce(records, metadata)['main_score'], places=15)

    def test_available_portfolio_branch_has_half_quarter_weight(self):
        roster, _, _ = self.roster()
        weights = origin_weights(roster)
        newer = sum(weights[key] for key, m, date in roster if date.startswith('2000') and m['panel_branch'] == 'portfolios')
        older = sum(weights[key] for key, m, date in roster if date.startswith('1990'))
        self.assertAlmostEqual(newer, .25, places=15)
        self.assertAlmostEqual(older, .5, places=15)


if __name__ == '__main__':
    unittest.main()
