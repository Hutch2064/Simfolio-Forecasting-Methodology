from simfolio_forecasting_methodology.catalog import load_canonical_175
from simfolio_forecasting_methodology.specifications import parse_compositional_spec


def test_all_pipe_delimited_canonical_ids_parse_to_known_components():
    rows = load_canonical_175()
    pipe_rows = [row for row in rows if "|" in row.model_id]
    assert len(pipe_rows) == 85
    for row in pipe_rows:
        spec = parse_compositional_spec(row.model_id)
        assert spec.source_id == row.model_id
