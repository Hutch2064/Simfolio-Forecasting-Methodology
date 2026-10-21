import unittest
import numpy as np
import score_aligned_v7 as score


class AlignedScoreTests(unittest.TestCase):
    def test_exact_equal_day_formula_uses_native_not_calendar_lead(self):
        history = np.array([-.02, .01, .03, -.01])
        losses = np.array([.1, .2, .3])
        dates = np.array(['2026-01-02', '2026-01-05', '2026-01-06'], dtype='datetime64[D]')
        expected = np.mean(losses / (np.std(history, ddof=1) * np.sqrt([1., 2., 3.])))
        self.assertEqual(score.origin_score(losses, history, '2026-01-01', dates), expected)

    def test_common_positive_unit_change_cancels(self):
        history = np.array([-.02, .01, .03, -.01])
        losses = np.array([.1, .2])
        dates = np.array(['2026-01-02', '2026-01-05'], dtype='datetime64[D]')
        base = score.origin_score(losses, history, '2026-01-01', dates)
        self.assertAlmostEqual(base, score.origin_score(losses * 100, history * 100, '2026-01-01', dates))

    def test_invalid_training_scale_and_dates_fail_closed(self):
        for history, dates in [([0., 0.], ['2026-01-02']),
                               ([0., np.nan], ['2026-01-02']),
                               ([0., 1.], ['2025-12-31'])]:
            with self.assertRaises(ValueError):
                score.origin_score([.1], history, '2026-01-01', np.array(dates, dtype='datetime64[D]'))

    def test_hierarchy_caps_portfolios_and_rejects_unmatched_quarters(self):
        metadata = {
            'a': {'panel_branch': 'assets', 'asset_class': 'equity', 'family': 'equity'},
            'p': {'panel_branch': 'portfolios', 'asset_class': 'mixed', 'family': 'mixed'},
        }
        rows = [{'series_id': 'a', 'origin_date': '2026-01-01', 'score': 2.},
                {'series_id': 'p', 'origin_date': '2026-01-01', 'score': 4.}]
        result = score.aggregate_records(rows, metadata)
        self.assertEqual(result['score'], 3.)
        self.assertEqual(result['maximum_portfolio_weight'], .5)
        with self.assertRaises(ValueError):
            score.aggregate_records([rows[1]], metadata)


if __name__ == '__main__':
    unittest.main()
