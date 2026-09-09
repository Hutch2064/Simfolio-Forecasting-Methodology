from simfolio_forecasting_methodology.catalog import FRONTIER_SOURCE_ID, load_canonical_175


def test_canonical_catalogue_has_exact_membership_and_frontier_rank_one():
    rows = load_canonical_175()
    assert len(rows) == 175
    assert rows[0].canonical_rank == 1
    assert rows[0].source_rank == 12
    assert rows[0].model_id == FRONTIER_SOURCE_ID
    assert rows[-1].canonical_rank == 175
    assert rows[-1].source_rank == 186


def test_canonical_catalogue_scores_are_sorted():
    rows = load_canonical_175()
    scores = [row.exact_empirical_crps for row in rows]
    assert scores == sorted(scores)
