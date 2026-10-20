"""Reference-selection contract independent of the live forecast processes."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent/'asset_level_v6'))
import reporter


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = reporter.PUBLIC_IDS
        self.meta = {'a': {'panel_branch': 'assets', 'asset_class': 'equities', 'family': 'broad'},
                     'p': {'panel_branch': 'portfolios', 'asset_class': 'mixed_portfolios', 'family': 'balanced'}}
        self.expected = {'a', 'p'}

    def records(self, a, p):
        return {sid: {'series_id': sid, 'origin_date': '2001-01-02', 'score': v}
                for sid, v in [('a', a), ('p', p)]}

    def test_higher_main_score_is_reference_and_sign_is_correct(self):
        data = {self.a: self.records(1., 1.), self.b: self.records(2., 2.),
                'candidate': self.records(.5, .5)}
        out = reporter.reduce_records(data, self.meta, self.expected, {k: True for k in data})
        self.assertEqual(out['reference_model_id'], self.b)
        self.assertEqual(out['paired_comparisons']['candidate']['percent_better'], 75.)
        self.assertFalse(out['reference_selection_preliminary'])

    def test_reference_not_selected_from_incomplete_smoke(self):
        data = {self.a: self.records(1., 1.), self.b: self.records(2., 2.)}
        out = reporter.reduce_records(data, self.meta, self.expected, {self.a: False, self.b: False})
        self.assertIsNone(out['reference_model_id'])
        self.assertEqual(out['paired_comparisons'], {})
        self.assertTrue(out['reference_selection_preliminary'])

    def test_candidate_partial_coverage_has_matched_denominator(self):
        data = {self.a: self.records(1., 1.), self.b: self.records(2., 4.),
                'candidate': {'a': self.records(1., 1.)['a']}}
        out = reporter.reduce_records(data, self.meta, self.expected,
                                      {self.a: True, self.b: True, 'candidate': False})
        pair = out['paired_comparisons']['candidate']
        self.assertEqual(pair['matched_successful_tasks'], 1)
        self.assertEqual(pair['reference_main_score'], 2.)
        self.assertEqual(pair['percent_better'], 50.)
        self.assertTrue(pair['preliminary'])
        self.assertNotIn('candidate', [r['model_id'] for r in out['complete_coverage_leaderboard']])

    def test_portfolio_only_coverage_is_pending_not_score(self):
        data = {self.a: {'p': self.records(1., 1.)['p']},
                self.b: {'p': self.records(2., 2.)['p']}}
        out = reporter.reduce_records(data, self.meta, self.expected, {self.a: False, self.b: False})
        self.assertIsNone(out['reference_model_id'])
        self.assertFalse(out['complete_coverage_leaderboard'])


if __name__ == '__main__':
    unittest.main()
