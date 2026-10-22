"""Synthetic fixtures only: no fits, forecasts, production calls, or real scores."""
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import preserve_study as helper


class StudyFixtures(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.methodology = self.root / "methodology"
        self.master = self.root / "master"
        self.methodology.mkdir()
        self.master.mkdir()
        self.runner = self.write("runner.py", "# synthetic runner\n")
        self.auditor = self.write("independent_auditor.py", "# synthetic independent auditor\n")
        self.reducer = self.write("summarize.py", "# synthetic nested checkpoint reducer\n")
        self.growth = self.write("growth_summary.py", "# synthetic nested growth reducer\n")
        self.publication_sources = [self.auditor, self.reducer, self.growth]
        self.source = self.write("methodology.md", "Synthetic fixture methodology, not real evidence.\n")
        self.control = self.write("controls.bin", "synthetic saved control cells")
        self.panels = {panel: self.panel(panel) for panel in helper.PANELS}
        self.study = {
            "schema": "unified-frontier-study-v1", "study_id": "UF-FIXTURE-1",
            "purpose": "Synthetic catalogue fixture", "methodology": "Training-only unified fixture law",
            "uniform_asset_law": True, "data_manifest_sha256": helper.DATA_SHA256,
            "revisions": {"fixture_repository": "a" * 40},
            "sources": [self.source, *self.publication_sources],
            "variants": [{"label": "candidate", "research_id": "UF-FIXTURE-V1",
                          "methodology": "Synthetic law", "decision": "rejected",
                          "rationale": "Fixture preserves an unsuccessful variant."}],
            "panels": self.panels,
            "growth_diagnostics": {"status": "not_evaluated",
                                   "scope": "diagnostic_no_historical_return_floor"},
            "runtime_evidence": {"status": "not_measured", "scope": "No timings in fixture.",
                                 "production_browser_ctv": False},
            "validation": {"canonical_use": "exploratory_reused_panels",
                           "seed_robustness": {"status": "not_tested"},
                           "independent_post_selection": {"status": "not_tested"},
                           "production": "not_validated"}}
        self.close_archive()
        self.packet = self.root / "study.json"
        self.save_packet()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
        return {"path": name, "sha256": helper.file_sha256(path)}

    def panel(self, panel):
        entities, origins, cells, contributions = helper.PANELS[panel]
        identities = [{"task_id": f"{panel}-{index}", "seed": index, "origin_date": "2000-01-01",
                       "training_digest": "t", "realized_digest": "r", "future_dates_digest": "f"}
                      for index in range(origins)]
        labels = ["M261_saved_control", "fast_saved_control", "candidate"]
        scores = dict(zip(labels, [1.0, 1.1, 1.2]))
        manifest = {"simulations": 240, "task_identities": {panel: identities},
                    "labels": {panel: labels}, "source_hashes": {
                        ref["path"]: ref["sha256"] for ref in (self.runner, self.control)},
                    "snapshot_sources": {self.source["path"]: self.source["sha256"]}}
        prefix = panel + "/"
        refs = {
            "manifest": self.write(prefix + "manifest.json", manifest),
            "reference_manifest": self.write(prefix + "reference.json", {
                "data_manifest_sha256": helper.DATA_SHA256, "task_identities": identities}),
            "completion": self.write(prefix + "completion.json", {
                "complete_origins": {panel: origins}, "required_origins": {panel: origins},
                "failures": {panel: {}}}),
            "results": self.write(prefix + "results.json", {
                "complete_origins": origins, "required_origins": origins,
                "complete_portfolios": entities, "scores": scores}),
            "cells": self.write(prefix + "cells.bin", "SYNTHETIC fixture: no actual CRPS cells")}
        denominator = {"origins": origins, "portfolios": entities, "simulations": 240,
                       "cells_per_model": cells, "origin_horizon_contributions": contributions}
        refs["analysis"] = self.write(prefix + "analysis.json", {panel: {
            **denominator, "models": {key: {"CRPS": value} for key, value in scores.items()}}})
        audit = {"schema": "unified-frontier-publication-audit-v1", "panel": panel,
                 "independent_from_runner": True, "data_manifest_sha256": helper.DATA_SHA256,
                 "complete_origins": origins, "failures": 0,
                 "all_task_identities_match_reference": True,
                 "all_cells_and_counts_rebuilt_byte_exactly": True,
                 "all_saved_controls_byte_equal": True, "all_bound_sources_and_controls_match": True,
                 "task_identities_sha256": helper.digest_json(identities),
                 "counts_sha256": "d" * 64, "denominator_sha256": helper.digest_json({
                     **denominator, "horizons": 8766, "counts_sha256": "d" * 64}),
                 "artifact_sha256": {role: ref["sha256"] for role, ref in refs.items()},
                 "auditor_source": self.auditor,
                 "auditor_dependencies": [self.reducer, self.growth],
                 "required_publication_source_refs": self.publication_sources}
        refs["audit"] = self.write(prefix + "audit.json", audit)
        return {"status": "complete_scored_and_audited", "scores": scores, "evidence": refs,
                "controls": {label: self.control for label in labels[:2]},
                "runner_sources": [self.runner]}

    def close_archive(self):
        verification = self.write("remote_restore_receipt.json", {"synthetic": True})
        all_refs = [*self.study["sources"], self.runner, self.control]
        for panel in self.study["panels"].values():
            all_refs += list(panel.get("evidence", {}).values())
        self.study["archive_receipt"] = self.write("archive-closure.json", {
            "schema": "unified-frontier-archive-closure-v1", "status": "verified",
            "study_id": self.study["study_id"], "independent_download_and_member_verification": True,
            "source_input_seed_environment_closure": True,
            "remote_receipt_url": "https://example.invalid/synthetic-receipt",
            "restore_command": "synthetic fixture only", "verification_sources": [verification],
            "closed_artifacts": {ref["path"]: ref["sha256"] for ref in all_refs}})

    def save_packet(self):
        self.packet.write_text(json.dumps(self.study), encoding="utf-8")

    def edit_artifact(self, role, function, panel="52"):
        refs = self.study["panels"][panel]["evidence"]
        value = helper.read_json(self.root / refs[role]["path"])
        function(value)
        refs[role] = self.write(refs[role]["path"], value)

    def validate(self):
        return helper.validate_study(self.root, self.study)

    def aggregation_fixture(self, raw=None, panel="52", proof=True):
        row = self.study["panels"][panel]
        scores = row["scores"]
        raw = raw or {**scores, "candidate": math.nextafter(scores["candidate"], math.inf)}
        row["raw_results_scores"] = raw
        self.edit_artifact("results", lambda value: value.update({"scores": raw}), panel)
        n = helper.PANELS[panel][2]
        epsilon = sys.float_info.epsilon
        gamma = n * epsilon / (1 - n * epsilon)
        receipt = {
            "schema": "unified-frontier-scalar-aggregation-v1",
            "method": "float64_gamma_n_mean_absolute_cells", "dtype": "float64",
            "canonical_order": "C", "n": n, "epsilon": epsilon, "gamma_n": gamma,
            "canonical_scores": scores, "raw_results_scores": raw,
            "per_label": {label: {"mean_absolute_cells": score,
                "absolute_difference": abs(raw[label] - score), "error_bound": gamma * score,
                "ulp_difference": helper.ulp_distance(raw[label], score),
                "within_bound": abs(raw[label] - score) <= gamma * score}
                for label, score in scores.items()}}
        def update_audit(value):
            value["artifact_sha256"]["results"] = row["evidence"]["results"]["sha256"]
            if proof:
                value["score_aggregation"] = receipt
        self.edit_artifact("audit", update_audit, panel)
        self.close_archive()

    def test_complete_fixture_and_report(self):
        result = self.validate()
        self.assertEqual(result, self.study)
        self.assertIn("Full 52", helper.report({"studies": [result]}))
        self.assertIn("Independent post-selection: not_tested", helper.report({"studies": [result]}))

    def test_one_ulp_scalar_roundoff_requires_and_accepts_closed_proof(self):
        self.aggregation_fixture()
        result = self.validate()
        self.assertNotEqual(result["panels"]["52"]["raw_results_scores"], result["panels"]["52"]["scores"])
        self.assertIn("1 ULP", helper.report({"studies": [result]}))

    def test_roundoff_proof_and_raw_scores_preserve_byte_idempotence(self):
        self.aggregation_fixture()
        self.save_packet()
        helper.prepare(self.root, self.packet, self.methodology, self.master)
        before = {p.name: p.read_bytes() for p in self.methodology.iterdir()}
        helper.prepare(self.root, self.packet, self.methodology, self.master)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.methodology.iterdir()})
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.master.iterdir()})

    def test_public_scalar_proof_must_equal_independent_closed_audit(self):
        self.aggregation_fixture()
        row = self.study["panels"]["52"]
        row["score_aggregation"] = copy.deepcopy(helper.read_json(self.root / row["evidence"]["audit"]["path"])["score_aggregation"])
        row["score_aggregation"]["n"] += 1
        with self.assertRaisesRegex(ValueError, "public scalar aggregation proof differs"):
            self.validate()

    def test_one_ulp_scalar_roundoff_without_proof_rejected(self):
        self.aggregation_fixture(proof=False)
        with self.assertRaisesRegex(ValueError, "disagree without aggregation proof"):
            self.validate()

    def test_material_score_change_rejected_with_self_reported_proof(self):
        self.aggregation_fixture(raw={**self.study["panels"]["52"]["scores"], "candidate": 1.21})
        with self.assertRaisesRegex(ValueError, "derived roundoff bound"):
            self.validate()

    def test_roundoff_receipt_cannot_inflate_error_bound(self):
        self.aggregation_fixture()
        self.edit_artifact("audit", lambda value: value["score_aggregation"]["per_label"]["candidate"].update({"error_bound": 1.0}))
        self.close_archive()
        with self.assertRaisesRegex(ValueError, "misstates the derived roundoff bound"):
            self.validate()

    def test_roundoff_receipt_cannot_inflate_mean_absolute_cells(self):
        self.aggregation_fixture()
        def inflate(value):
            receipt = value["score_aggregation"]
            receipt["per_label"]["candidate"].update({"mean_absolute_cells": 1e100,
                                                      "error_bound": receipt["gamma_n"] * 1e100})
        self.edit_artifact("audit", inflate)
        self.close_archive()
        with self.assertRaisesRegex(ValueError, "mean absolute cells must equal canonical score"):
            self.validate()

    def test_roundoff_receipt_must_report_correct_ulp_count(self):
        self.aggregation_fixture()
        self.edit_artifact("audit", lambda value: value["score_aggregation"]["per_label"]["candidate"].update({"ulp_difference": 2}))
        self.close_archive()
        with self.assertRaisesRegex(ValueError, "misstates the derived roundoff bound"):
            self.validate()

    def test_roundoff_receipt_cannot_change_canonical_cell_count(self):
        self.aggregation_fixture()
        self.edit_artifact("audit", lambda value: value["score_aggregation"].update({"n": 999999999}))
        self.close_archive()
        with self.assertRaisesRegex(ValueError, "derived binary64 gamma_n bound"):
            self.validate()

    def test_material_cell_change_not_excused_by_scalar_proof(self):
        self.aggregation_fixture()
        row = self.study["panels"]["52"]
        row["evidence"]["cells"] = self.write("52/cells.bin", "materially changed synthetic cells")
        self.close_archive()
        with self.assertRaisesRegex(ValueError, "close all scored artifacts"):
            self.validate()

    def test_canonical_publication_score_still_must_match_analysis_exactly(self):
        self.aggregation_fixture()
        self.study["panels"]["52"]["scores"]["candidate"] = math.nextafter(1.2, math.inf)
        with self.assertRaisesRegex(ValueError, "score receipts disagree"):
            self.validate()

    def test_pre_and_post_update_fixtures_preserve_all_existing_fields(self):
        old = {"schema": helper.SCHEMA, "studies": [{"study_id": "UF-OLD", "payload": [1, 2]}],
               "unrelated_metadata": {"must_survive": True}}
        snapshot = copy.deepcopy(old)
        updated = helper.add_study(old, self.validate())
        self.assertEqual(old, snapshot)
        self.assertEqual(updated["studies"][:-1], old["studies"])
        self.assertEqual(updated["unrelated_metadata"], old["unrelated_metadata"])
        self.assertEqual(helper.add_study(updated, self.study), updated)

    def test_both_outputs_equal_and_second_write_byte_identical(self):
        helper.prepare(self.root, self.packet, self.methodology, self.master)
        before = {(p.parent.name, p.name): p.read_bytes()
                  for folder in (self.methodology, self.master) for p in folder.iterdir()}
        helper.prepare(self.root, self.packet, self.methodology, self.master)
        after = {(p.parent.name, p.name): p.read_bytes()
                 for folder in (self.methodology, self.master) for p in folder.iterdir()}
        self.assertEqual(before, after)
        self.assertEqual((self.methodology / "unified_frontier_studies.json").read_bytes(),
                         (self.master / "unified_frontier_studies.json").read_bytes())

    def test_prepare_adds_to_existing_valid_sidecars_and_preserves_originals(self):
        prior = copy.deepcopy(self.study)
        prior["study_id"] = "UF-PRIOR-FIXTURE"
        before = {"schema": helper.SCHEMA, "studies": [prior], "prior_metadata": {"keep": True}}
        for folder in (self.methodology, self.master):
            (folder / "unified_frontier_studies.json").write_text(json.dumps(before))
            (folder / "refinement52.json").write_text("untouched historical catalogue fixture")
        result = helper.prepare(self.root, self.packet, self.methodology, self.master)
        self.assertEqual(result["studies"][0], prior)
        self.assertEqual(result["prior_metadata"], before["prior_metadata"])
        self.assertEqual(helper.prepare(self.root, self.packet, self.methodology, self.master), result)
        for folder in (self.methodology, self.master):
            self.assertEqual((folder / "refinement52.json").read_text(), "untouched historical catalogue fixture")

    def test_conflicting_existing_id_is_rejected(self):
        payload = helper.add_study({"schema": helper.SCHEMA, "studies": []}, self.study)
        altered = copy.deepcopy(self.study)
        altered["purpose"] = "changed"
        with self.assertRaisesRegex(ValueError, "immutable study ID conflicts"):
            helper.add_study(payload, altered)

    def test_different_destinations_rejected_before_writes(self):
        (self.master / "unified_frontier_studies.json").write_text(json.dumps({
            "schema": helper.SCHEMA, "studies": [], "different": True}))
        with self.assertRaisesRegex(ValueError, "differ before update"):
            helper.prepare(self.root, self.packet, self.methodology, self.master)
        self.assertFalse((self.methodology / "unified_frontier_studies.json").exists())

    def test_one_panel_unscored_stays_explicitly_unscored(self):
        self.study["panels"]["80"] = {"status": "not_evaluated", "reason": "Fixture pending"}
        self.close_archive()
        self.assertEqual(self.validate()["panels"]["80"]["status"], "not_evaluated")
        self.study["panels"]["80"]["scores"] = {"candidate": 0.1}
        with self.assertRaisesRegex(ValueError, "unscored panel cannot"):
            self.validate()

    def test_partial_completion_cannot_be_full(self):
        self.edit_artifact("completion", lambda value: value["complete_origins"].update({"52": 2651}))
        with self.assertRaisesRegex(ValueError, "completion is partial"):
            self.validate()

    def test_denominator_mismatch(self):
        self.edit_artifact("analysis", lambda value: value["52"].update({"cells_per_model": 455831}))
        with self.assertRaisesRegex(ValueError, "denominators are not canonical"):
            self.validate()

    def test_reference_identity_change(self):
        self.edit_artifact("reference_manifest", lambda value: value["task_identities"][0].update({"seed": 55}))
        with self.assertRaisesRegex(ValueError, "reference task/data identities differ"):
            self.validate()

    def test_audit_stale_artifact_binding(self):
        self.edit_artifact("audit", lambda value: value["artifact_sha256"].update({"cells": "0" * 64}))
        with self.assertRaisesRegex(ValueError, "close all scored artifacts"):
            self.validate()

    def test_missing_independent_rebuild(self):
        self.edit_artifact("audit", lambda value: value.update({"all_cells_and_counts_rebuilt_byte_exactly": False}))
        with self.assertRaisesRegex(ValueError, "cell-rebuild audit is incomplete"):
            self.validate()

    def test_nested_auditor_dependency_missing_from_study_sources(self):
        self.study["sources"].remove(self.reducer)
        with self.assertRaisesRegex(ValueError, "study.sources omits required publication source"):
            self.validate()

    def test_nested_auditor_dependency_stale_hash(self):
        (self.root / self.reducer["path"]).write_text("changed nested reducer")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch: summarize.py"):
            self.validate()

    def test_nested_auditor_dependency_missing_from_archive_closure(self):
        closure = helper.read_json(self.root / self.study["archive_receipt"]["path"])
        del closure["closed_artifacts"][self.growth["path"]]
        self.study["archive_receipt"] = self.write("archive-closure.json", closure)
        with self.assertRaisesRegex(ValueError, "archive closure omits"):
            self.validate()

    def test_declared_dependency_must_appear_in_required_publication_refs(self):
        self.edit_artifact("audit", lambda value: value["required_publication_source_refs"].remove(self.reducer))
        with self.assertRaisesRegex(ValueError, "required publication sources omit auditor dependency"):
            self.validate()

    def test_required_publication_refs_must_include_auditor_source(self):
        self.edit_artifact("audit", lambda value: value["required_publication_source_refs"].remove(self.auditor))
        with self.assertRaisesRegex(ValueError, "required publication sources omit auditor_source"):
            self.validate()

    def test_source_bytes_changed(self):
        (self.root / self.runner["path"]).write_text("changed source")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            self.validate()

    def test_unclosed_archive(self):
        closure = helper.read_json(self.root / self.study["archive_receipt"]["path"])
        del closure["closed_artifacts"][self.source["path"]]
        self.study["archive_receipt"] = self.write("archive-closure.json", closure)
        with self.assertRaisesRegex(ValueError, "archive closure omits"):
            self.validate()

    def test_runtime_cannot_imply_browser_ctv(self):
        self.study["runtime_evidence"]["production_browser_ctv"] = True
        with self.assertRaisesRegex(ValueError, "separate from browser CTV"):
            self.validate()

    def test_seed_robustness_does_not_grant_independent_validation(self):
        self.study["validation"]["seed_robustness"] = {"status": "tested", "evidence": [self.source]}
        self.assertEqual(self.validate()["validation"]["independent_post_selection"]["status"], "not_tested")
        self.study["validation"]["independent_post_selection"] = {"status": "tested"}
        with self.assertRaisesRegex(ValueError, "validation evidence is missing"):
            self.validate()

    def test_symlink_and_escape_rejected(self):
        alias = self.root / "alias"
        alias.symlink_to(self.root / self.source["path"])
        for raw in ("alias", "../outside", str(self.root / self.source["path"])):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                helper.artifact(self.root, {"path": raw, "sha256": self.source["sha256"]})


if __name__ == "__main__":
    unittest.main()
