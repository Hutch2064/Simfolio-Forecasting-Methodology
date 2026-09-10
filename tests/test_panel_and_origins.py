import hashlib
import json
from importlib import resources

import numpy as np

from simfolio_forecasting_methodology.data import canonical_calendar
from simfolio_forecasting_methodology.experiment import build_experiment_schedule
from simfolio_forecasting_methodology.origins import (
    eligible_quarter_end_positions,
    expected_dense_cell_count,
    expected_origin_tasks,
    origin_schedule,
    rolling_horizon_cap,
    temporal_holdouts,
)
from simfolio_forecasting_methodology.panel import (
    appendix_panel_provenance,
    generate_equal_class_history_panel,
    load_scored52_portfolio_panel,
    panel_semantic_fingerprint,
    rebalance_counts,
)


def test_canonical_panel_shape_and_rebalance_counts():
    panel = load_scored52_portfolio_panel()
    assert len(panel) == 80
    assert len({ticker for p in panel for ticker in p.tickers}) == 52
    assert rebalance_counts(panel) == {
        "annually": 19,
        "monthly": 26,
        "none": 18,
        "quarterly": 17,
    }
    assert panel_semantic_fingerprint(panel) == (
        "c017963c9eb772e9bac09bb7a84ae975443cbb99cd127d666a9b0ac8103933d3"
    )


def test_canonical_default_is_frozen_scored_panel():
    panel = generate_equal_class_history_panel()
    assert panel[0].name == "equal_class_history_001"
    assert panel[0].rebalance == "annually"
    assert panel[0].tickers == (
        "IWMSIM",
        "XLPSIM",
        "IEISIM",
        "ZROZSIM",
        "GLDSIM",
        "REITSIM",
    )
    assert panel[1].tickers == (
        "SPXUSIM",
        "TNASIM",
        "SHYSIM",
        "TLTSIM",
        "GSGSIM",
        "SLVSIM",
    )


def test_appendix_panel_is_provenance_only():
    assert appendix_panel_provenance() == {
        "status": "provenance_only_not_executable",
        "row_count": 80,
        "ticker_count": 38,
        "rebalance_counts": {"annually": 18, "monthly": 20, "none": 24, "quarterly": 18},
        "normalized_lf_sha256": "6691eb1daf38cc2aae6708fe1d155aab7975f29ebc6ada197f46bd8d9a8ecf70",
        "original_crlf_sha256": "847946335fea3473e987af23f528483d66ab265e85596f5d226216091fb476b4",
    }


def test_dense_origin_contract():
    assert expected_origin_tasks() == 4080
    holdouts = temporal_holdouts(11687)
    assert len(holdouts) == 3
    assert [h.train_fraction for h in holdouts] == [0.25, 0.50, 0.75]
    assert [h.position for h in holdouts] == [2920, 5842, 8764]
    assert [h.max_horizon for h in holdouts] == [8766, 5844, 2922]
    assert expected_dense_cell_count(11687) == 701280


def test_source_equivalent_rolling_selection_and_masks():
    dates = canonical_calendar()
    descriptors = origin_schedule(dates)
    assert len(descriptors) == 51
    assert descriptors[0].position >= 504
    assert descriptors[0].evaluation_split == "rolling_origin"
    assert descriptors[0].max_horizon == rolling_horizon_cap(descriptors[0].position, len(dates))
    mask = descriptors[-1].horizon_mask(len(dates))
    assert mask.dtype == np.bool_
    assert mask.sum() == descriptors[-1].max_horizon


def test_origin_schedule_matches_archived_source_reference_digests():
    resource = resources.files("simfolio_forecasting_methodology").joinpath(
        "resources", "protocols", "origin_reference_digests.json"
    )
    reference = json.loads(resource.read_text(encoding="utf-8"))
    dates = canonical_calendar()
    descriptors = origin_schedule(dates)

    def digest(payload):
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(body).hexdigest()

    eligible = eligible_quarter_end_positions(dates)
    assert len(eligible) == reference["selection"]["eligible_count"]
    assert digest([int(position) for position in eligible]) == reference["selection"]["eligible_position_sha256"]
    assert digest([dates[int(position)].date().isoformat() for position in eligible]) == reference["selection"]["eligible_date_sha256"]
    selected = [item.position for item in descriptors if item.evaluation_split == "rolling_origin"]
    selected_dates = [dates[int(position)].date().isoformat() for position in selected]
    assert digest([int(position) for position in selected]) == reference["selection"]["selected_position_sha256"]
    assert digest(selected_dates) == reference["selection"]["selected_date_sha256"]

    rows = [
        {
            "label": item.origin_label,
            "split": item.evaluation_split,
            "position": int(item.position),
            "origin_date": item.origin_date,
            "max_horizon": int(item.max_horizon),
        }
        for item in descriptors
    ]
    masks = [
        [int(position) for position in np.flatnonzero(item.horizon_mask(len(dates)))]
        for item in descriptors
    ]
    assert len(descriptors) == reference["descriptors"]["count"]
    assert digest(rows) == reference["descriptors"]["descriptor_sha256"]
    assert digest(masks) == reference["descriptors"]["horizon_mask_sha256"]


def test_value_free_canonical_schedule_has_fixed_counts():
    schedule = build_experiment_schedule()
    assert schedule.task_count == 4080
    assert schedule.cell_capacity == 701280
    assert len(schedule.common_dates) == 11687
    assert sum(task.descriptor.evaluation_split == "rolling_origin" for task in schedule.tasks) == 3840
    assert sum(task.descriptor.evaluation_split != "rolling_origin" for task in schedule.tasks) == 240
