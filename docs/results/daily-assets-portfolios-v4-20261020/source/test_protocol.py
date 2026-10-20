from fractions import Fraction
import json
import numpy as np
import pytest
import common
import construction as pc
import score_main as score


def test_observed_common_intervals_keep_weekend_returns_and_compound_before_weighting():
    equity_dates = np.array(['2020-01-03', '2020-01-06', '2020-01-07'], dtype='datetime64[D]')
    crypto_dates = np.arange(np.datetime64('2020-01-03'), np.datetime64('2020-01-08'))
    assets = {'E': (equity_dates, np.array([.01, .02, -.01])),
              'D': (crypto_dates, np.array([.1, .1, .1, .1, -.02]))}
    dates, actual, anchor = pc.portfolio_returns(assets, {'E': Fraction(1, 2), 'D': Fraction(1, 2)})
    assert anchor == '2020-01-03'
    np.testing.assert_array_equal(dates, equity_dates[1:])
    expected = np.log1p(.5*np.expm1([.02, -.01])+.5*np.expm1([.3, -.02]))
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    assert abs(actual[0]-(.02+.3)/2) > 1e-4


def test_weights_order_are_deterministic_long_only_and_unlevered():
    one = pc.canonical_weights({'A': Fraction(3, 5), 'B': Fraction(2, 5)})
    assert one == pc.canonical_weights({'B': Fraction(2, 5), 'A': Fraction(3, 5)})
    with pytest.raises(ValueError):
        pc.canonical_weights({'A': Fraction(6, 5), 'B': Fraction(-1, 5)})
    with pytest.raises(ValueError):
        pc.canonical_weights({'A': Fraction(1)})


def test_portfolio_catalogue_covers_all_class_subsets_concentrations_and_broad_sectors():
    rows = json.loads((common.CAMPAIGN/'inputs/panel.json').read_text())['series']
    definitions, exclusions = pc.catalogue(rows)
    assert len(definitions) > 100  # Coverage rule, not a production count cap.
    assert len({r['series_id'] for r in definitions}) == len(definitions)
    assert len({tuple(tuple(w) for w in r['rational_weights']) for r in definitions}) == len(definitions)
    assert sum(r['purpose'] == 'material_directional_concentration_stress' for r in definitions) == 30
    assert sum(r['purpose'] == 'complete_broad_sector_concentration_coverage' for r in definitions) == 5
    assert {r['family'] for r in definitions if r['family'].startswith('balanced_')} == {
        'balanced_3_classes', 'balanced_4_classes', 'balanced_5_classes', 'balanced_6_classes'}
    assert len({r['family'] for r in definitions if r['family'].startswith('cross_class_pair__')}) == 15
    assert exclusions and all('identical_exact_asset_weights' in r['reason'] for r in exclusions)


def test_single_score_is_exact_former_sqrt_h_formula_and_no_estimation_discount():
    history = np.array([.01, -.03, .02, .005])
    losses = np.array([.1, .3, .6])
    future = np.array(['2020-01-06', '2020-01-07', '2020-01-08'], dtype='datetime64[D]')
    w = score.legacy.calendar_log_weights('2020-01-03', future)
    expected = np.dot(losses/(np.std(history, ddof=1)*np.sqrt([1, 2, 3])), w)
    assert score.origin_score(losses, history, '2020-01-03', future) == expected


def test_two_branch_cap_and_clone_invariance_within_family():
    meta = {'A': {'panel_branch': 'assets', 'asset_class': 'equities', 'family': 'broad'},
            'P': {'panel_branch': 'portfolios', 'asset_class': 'mixed_portfolios', 'family': 'balanced'}}
    rows = [{'series_id': 'A', 'origin_date': '2000-01-02', 'score': 2.},
            {'series_id': 'A', 'origin_date': '2000-04-02', 'score': 4.},
            {'series_id': 'P', 'origin_date': '2000-04-02', 'score': 10.}]
    result = score.aggregate_records(rows, meta)
    assert result['quarter_scores'] == {'2000-Q1': 2., '2000-Q2': 7.}
    assert result['score'] == 4.5
    assert max(result['portfolio_weight_by_quarter'].values()) == .5
    for i in range(30):
        sid = 'P'+str(i)
        meta[sid] = dict(meta['P'])
        rows.append({'series_id': sid, 'origin_date': '2000-04-02', 'score': 10.})
    assert score.aggregate_records(rows, meta)['score'] == result['score']


def test_portfolio_only_quarters_fail_closed_instead_of_gaining_full_weight():
    meta = {'P': {'panel_branch': 'portfolios', 'asset_class': 'mixed_portfolios', 'family': 'balanced'}}
    with pytest.raises(ValueError, match='lack matched'):
        score.aggregate_records([{'series_id': 'P', 'origin_date': '2000-01-02', 'score': 2.}], meta)


def test_original_asset_branch_keeps_exact_legacy_hierarchy_and_diagnostics_are_ignored():
    meta = {'A': {'panel_branch': 'assets', 'asset_class': 'equities', 'family': 'broad'},
            'B': {'panel_branch': 'assets', 'asset_class': 'fixed_income', 'family': 'treasury'}}
    records = [{'series_id': 'A', 'origin_date': '2000-01-02', 'score': 1., 'coverage_90': .1},
               {'series_id': 'B', 'origin_date': '2000-01-02', 'score': 5., 'coverage_90': .9}]
    actual = score.aggregate_records(records, meta)
    assert actual['score'] == score.legacy.aggregate_records(records, meta)['score'] == 3.
    assert actual['diagnostics_in_score'] is False
    assert actual['maximum_portfolio_weight'] == 0
