from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import score_requested_horizon_crps_v11 as v11
import score_series_buckets_v9 as v9


def make_pair(bucket: str, index: int, branch: str, score: float) -> tuple[dict, dict, dict]:
    origin_end = {
        "short": ("2020-01-01", "2021-01-01"),
        "medium": ("2000-01-01", "2005-01-01"),
        "long": ("1970-01-01", "1991-01-01"),
    }
    origin, end = origin_end[bucket]
    sid = f"{bucket}_{branch}_{index}"
    request_id = f"[{bucket},{branch},{index}]"
    row = {
        "request_id": request_id,
        "series_id": sid,
        "origin_date": origin,
        "requested_end_date": end,
        "requested_native_h": 4,
        "case_index": index,
        "origin_index": index,
        "raw_mean_crps": score,
    }
    scale = {"series_id": sid, "origin_date": origin, "requested_end_date": end,
             "case_index": index, "origin_index": index, "training_sd": 0.5, "native_h": 4}
    metadata = {
        "series_id": sid,
        "panel_branch": branch,
        "asset_class": "fixture_class",
        "family": f"fixture_family_{branch}",
    }
    return row, scale, metadata


class RequestedHorizonV11Tests(unittest.TestCase):
    def test_request_formula_and_exact_horizon_factor(self):
        self.assertEqual(v11.standardized_request_crps(2.0, 0.5, 4), 2.0)
        self.assertAlmostEqual(v11.standardized_request_crps(2.0, 0.5, 16), 1.0)

    def test_positive_unit_rescaling_leaves_request_score_unchanged(self):
        base = v11.standardized_request_crps(0.3, 0.05, 10)
        changed_units = v11.standardized_request_crps(12.0, 2.0, 10)
        self.assertAlmostEqual(base, changed_units, places=15)

    def test_constant_request_divisor_differs_from_v7_per_lead_divisors(self):
        daily_losses = [1.0, 4.0]
        v11_value = v11.standardized_request_crps(sum(daily_losses) / 2, 2.0, 2)
        v7_value = sum(loss / (2.0 * math.sqrt(h)) for h, loss in enumerate(daily_losses, 1)) / 2
        self.assertAlmostEqual(v11_value, 2.5 / (2.0 * math.sqrt(2.0)), places=15)
        self.assertNotAlmostEqual(v11_value, v7_value, places=8)

    def test_fixed_v9_branch_and_equal_bucket_hierarchy_with_unequal_counts(self):
        rows = []
        scales = {}
        metadata = {}
        scores = {
            "short": {"assets": [2.0], "portfolios": [4.0]},
            "medium": {"assets": [10.0, 10.0], "portfolios": [20.0, 20.0]},
            "long": {"assets": [100.0, 100.0, 100.0], "portfolios": [200.0, 200.0, 200.0]},
        }
        for bucket, branches in scores.items():
            for branch, values in branches.items():
                for i, score in enumerate(values):
                    row, scale, meta = make_pair(bucket, i, branch, score)
                    rows.append(row)
                    scales[row["request_id"]] = scale
                    metadata[meta["series_id"]] = meta
        planned = [{k: r[k] for k in ("request_id", "series_id", "origin_date", "requested_end_date")} for r in rows]
        weights = v9.build_weights(planned, metadata)
        self.assertEqual(weights["planned_request_count"], 12)
        out = v11.aggregate_records(rows, metadata, weights=weights, scales_by_request=scales)
        # Standardized request values divide by .5*sqrt(4)=1. Each bucket has
        # 50/50 asset/portfolio support; the bucket Qs are 3, 15 and 150.
        bucket_scores = {b: out["buckets"][b]["partial_score"] for b in ("short", "medium", "long")}
        self.assertEqual(bucket_scores["short"], 3.0)
        self.assertEqual(bucket_scores["medium"], 15.0)
        self.assertAlmostEqual(bucket_scores["long"], 150.0, places=12)
        self.assertAlmostEqual(out["score"], 56.0, places=12)
        self.assertEqual(out["weight_plan_sha256"], weights["weight_plan_sha256"])
        self.assertTrue(out["complete"])
        same = v11.compare(out, out)
        self.assertEqual(same["percent_better_than_baseline"], 0.0)
        self.assertEqual(same["matched_request_count"], 12)

    def test_incomplete_support_has_no_final_q_or_comparison(self):
        rows = []
        scales = {}
        metadata = {}
        for bucket in ("short", "medium", "long"):
            row, scale, meta = make_pair(bucket, 0, "assets", 1.0)
            rows.append(row)
            scales[row["request_id"]] = scale
            metadata[meta["series_id"]] = meta
        planned = [{k: r[k] for k in ("request_id", "series_id", "origin_date", "requested_end_date")} for r in rows]
        weights = v9.build_weights(planned, metadata)
        partial = v11.aggregate_records(rows[:-1], metadata, weights=weights, scales_by_request=scales)
        self.assertFalse(partial["complete"])
        self.assertIsNone(partial["score"])
        with self.assertRaises(ValueError):
            v11.compare(partial, partial)

    def test_exact_zero_baseline_has_no_relative_ratio(self):
        rows = []
        scales = {}
        metadata = {}
        for bucket in ("short", "medium", "long"):
            row, scale, meta = make_pair(bucket, 0, "assets", 0.0)
            rows.append(row)
            scales[row["request_id"]] = scale
            metadata[meta["series_id"]] = meta
        planned = [{k: r[k] for k in ("request_id", "series_id", "origin_date", "requested_end_date")} for r in rows]
        weights = v9.build_weights(planned, metadata)
        zero = v11.aggregate_records(rows, metadata, weights=weights, scales_by_request=scales)
        model_rows = [dict(r, raw_mean_crps=1.0) for r in rows]
        model = v11.aggregate_records(model_rows, metadata, weights=weights, scales_by_request=scales)
        comparison = v11.compare(model, zero)
        self.assertEqual(comparison["baseline_Q"], 0.0)
        self.assertIsNone(comparison["percent_better_than_baseline"])
        self.assertEqual(comparison["relative_gain_defined"], False)

    def test_horizon_and_series_identity_must_match_scale_index(self):
        rows = []
        scales = {}
        metadata = {}
        for bucket in ("short", "medium", "long"):
            row, scale, meta = make_pair(bucket, 0, "assets", 1.0)
            rows.append(row)
            scales[row["request_id"]] = scale
            metadata[meta["series_id"]] = meta
        row = rows[0]
        planned = [{k: r[k] for k in ("request_id", "series_id", "origin_date", "requested_end_date")} for r in rows]
        weights = v9.build_weights(planned, metadata)
        with self.assertRaises(ValueError):
            v11.aggregate_records([dict(row, requested_native_h=5)], metadata, weights=weights, scales_by_request=scales)
        with self.assertRaises(ValueError):
            v11.aggregate_records([dict(row, series_id="wrong")], metadata, weights=weights, scales_by_request=scales)
        with self.assertRaises(ValueError):
            v11.aggregate_records([dict(row, requested_end_date="2022-01-01")], metadata, weights=weights, scales_by_request=scales)
        with self.assertRaises((TypeError, ValueError)):
            v11.aggregate_records([dict(row, case_index=0.9)], metadata, weights=weights, scales_by_request=scales)

    def test_comparison_binds_consumed_scale_values(self):
        rows = []
        scales = {}
        metadata = {}
        for bucket in ("short", "medium", "long"):
            row, scale, meta = make_pair(bucket, 0, "assets", 1.0)
            rows.append(row)
            scales[row["request_id"]] = scale
            metadata[meta["series_id"]] = meta
        planned = [{k: r[k] for k in ("request_id", "series_id", "origin_date", "requested_end_date")} for r in rows]
        weights = v9.build_weights(planned, metadata)
        left = v11.aggregate_records(rows, metadata, weights=weights, scales_by_request=scales)
        changed = {key: dict(value) for key, value in scales.items()}
        changed[rows[0]["request_id"]]["training_sd"] = 0.25
        right = v11.aggregate_records(rows, metadata, weights=weights, scales_by_request=changed)
        with self.assertRaises(ValueError):
            v11.compare(left, right)

    def test_invalid_domain_is_rejected_without_epsilon(self):
        for sd in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(sd=sd), self.assertRaises(ValueError):
                v11.standardized_request_crps(1.0, sd, 1)
        for h in (0, -1, True, 1.0):
            with self.subTest(h=h), self.assertRaises((TypeError, ValueError)):
                v11.standardized_request_crps(1.0, 1.0, h)
        for raw in (-1.0, float("nan"), float("inf")):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                v11.standardized_request_crps(raw, 1.0, 1)


if __name__ == "__main__":
    unittest.main()
