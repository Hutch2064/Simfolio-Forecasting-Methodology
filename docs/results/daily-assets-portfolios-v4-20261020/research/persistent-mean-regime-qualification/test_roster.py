from pathlib import Path
import json,sys
import pytest
OWN=Path(__file__).resolve().parent
V4=OWN.parent/'portfolio_expansion_v4'
sys.path.insert(0,str(V4))
import score_main


def test_distinct_rosters_cover_every_family_with_supported_score_quarters():
    plan=json.loads((OWN/'plan.json').read_text())
    panel=json.loads((V4/'inputs/panel.json').read_text())
    metadata={r['series_id']:r for r in panel['series']}
    expected={(r['panel_branch'],r['asset_class'],r['family']) for r in panel['series']}
    keys=[]
    for phase,cases in plan['phases'].items():
        assert {(metadata[c['series_id']]['panel_branch'],metadata[c['series_id']]['asset_class'],metadata[c['series_id']]['family']) for c in cases}==expected
        selected={(c['series_id'],c['origin']['origin_label']) for c in cases}
        assert len(selected)==len(cases)
        keys.append(selected)
        records=[{'series_id':c['series_id'],'origin_date':c['origin']['origin_date'],'score':1.} for c in cases]
        result=score_main.aggregate_records(records,metadata)
        assert result['score']==1.
        assert result['maximum_portfolio_weight']==.5
    assert not keys[0]&keys[1]
