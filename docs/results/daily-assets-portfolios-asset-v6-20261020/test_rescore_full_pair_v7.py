"""Small, synthetic checks for the new v7 task-score index writer."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import rescore_full_pair_v7 as runner


class ScoreIndexTests(unittest.TestCase):
    def setUp(self):
        self.previous_index_path = runner.SCORE_INDEX_OUTPUT
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

    def tearDown(self):
        runner.SCORE_INDEX_OUTPUT = self.previous_index_path

    def test_roundtrip_binds_task_identity_and_baselines(self):
        output = Path(self.tempdir.name) / "tiny-v7-scores.sqlite3"
        runner.SCORE_INDEX_OUTPUT = output
        rows = {}
        for model_no, model_id in enumerate(runner.MODEL_IDS):
            rows[model_id] = [(
                model_id,
                '["series-A",{"origin_date":"2020-01-02","max_horizon":1}]',
                "panel-fingerprint",
                runner.EXPECTED_PROTOCOL,
                f"execution-{model_no}",
                "000000/000",
                0,
                0,
                "series-A",
                "2020-01-02",
                1,
                2,
                json.dumps({"model_id": model_id, "seed": 11}, sort_keys=True),
                "a" * 64,
                "b" * 64,
                0.25 + model_no,
                0.5 + model_no,
            )]
        metadata = {
            "protocol_id": runner.EXPECTED_PROTOCOL,
            "panel_fingerprint": "panel-fingerprint",
            "baseline_model_ids": list(runner.MODEL_IDS),
            "reference_model_id_under_v7": runner.MODEL_IDS[0],
            "coverage": {"tasks_per_model": 1},
        }

        digest, counts = runner._write_v7_score_index(rows, metadata, expected_per_model=1)

        self.assertTrue(output.is_file())
        self.assertEqual(len(digest), 64)
        self.assertEqual(counts, {model_id: 1 for model_id in runner.MODEL_IDS})
        uri = f"file:{output}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            scored = connection.execute(
                "SELECT model_id,case_index,series_id,origin_date,native_horizon,"
                "training_observations,task_identity_json,source_checkpoint_sha256,"
                "loss_sha256,score_v7,legacy_score_v6 FROM task_scores_v7 ORDER BY model_id"
            ).fetchall()
            stored_metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        self.assertEqual(len(scored), 2)
        self.assertEqual([row[1:6] for row in scored], [
            ("000000/000", "series-A", "2020-01-02", 1, 2),
            ("000000/000", "series-A", "2020-01-02", 1, 2),
        ])
        self.assertTrue(all(row[7] == "a" * 64 and row[8] == "b" * 64 for row in scored))
        self.assertEqual(json.loads(stored_metadata["baseline_model_ids"]), list(runner.MODEL_IDS))
        self.assertEqual(json.loads(stored_metadata["protocol_id"]), runner.EXPECTED_PROTOCOL)

    def test_writer_refuses_to_overwrite_existing_file(self):
        output = Path(self.tempdir.name) / "existing.sqlite3"
        output.write_bytes(b"preserve")
        runner.SCORE_INDEX_OUTPUT = output
        with self.assertRaises(FileExistsError):
            runner._write_v7_score_index({}, {}, expected_per_model=0)
        self.assertEqual(output.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
