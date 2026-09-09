from simfolio_forecasting_methodology.catalog import load_canonical_175
from simfolio_forecasting_methodology.naming import public_name


def test_all_canonical_public_names_are_unique_and_ranked():
    rows = load_canonical_175()
    names = [public_name(row) for row in rows]
    assert len(names) == len(set(names)) == 175
    assert names[0].startswith("M001 — Asset FastMAP")
    assert names[-1].startswith("M175 — ")
