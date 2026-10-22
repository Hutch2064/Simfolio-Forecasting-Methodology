import unittest,tempfile,sqlite3,json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import series_buckets_v9_reporter as report

class ReporterTests(unittest.TestCase):
    def test_publication_requires_exact_admission(self):
        with tempfile.TemporaryDirectory()as d,patch.object(report,'OUT',Path(d)/'sidecar'),patch.object(report,'O',Path(d)):
            report.OUT.mkdir()
            out=dict(protocol_sha256='p',scorer_sha256='s',scientific_plan_fingerprint='f',weight_plan_sha256='w')
            self.assertFalse(report.publish_admitted(out))
            self.assertFalse((Path(d)/'authoritative_current_status_latest.json').exists())
            marker=dict(schema='headline-score-amendment-v9',required=True,expected_requests_per_model=320017,score_protocol_id=report.scorer.SCORE_PROTOCOL_ID,protocol_sha256='p',scorer_sha256='s',reporter_sha256=report.verified.sha(report.__file__),plan_fingerprint='f',weight_plan_sha256='w')
            path=report.OUT/'main_score_admission.json';path.write_text(json.dumps(dict(marker,reporter_sha256='bad')))
            with self.assertRaisesRegex(ValueError,'admission'):report.publish_admitted(out)
            path.write_text(json.dumps(marker));self.assertTrue(report.publish_admitted(out))
            self.assertEqual(json.loads((Path(d)/'final240_current_status_latest.json').read_text()),out)

    def test_watch_releases_lock_before_sleep(self):
        import fcntl
        with tempfile.TemporaryDirectory()as d,patch.object(report,'OUT',Path(d)),patch.object(report.os,'getpriority',return_value=0),patch.object(sys,'argv',['reporter','--watch','60']),patch.object(report,'Reporter')as constructor:
            constructor.return_value.snapshot.side_effect=[{'complete':False},{'complete':True}]
            def sleep(seconds):
                with (Path(d)/'reporter.lock').open('a')as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with patch.object(report.time,'sleep',side_effect=sleep)as sleeping:report.main()
            sleeping.assert_called_once_with(60.)
            self.assertEqual(constructor.return_value.snapshot.call_count,2)

    def test_bad_ledger_hash_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'ledger';p.write_text('bad')
            with self.assertRaisesRegex(ValueError,'hash/model/count'):
                report.validate_ledger(p,dict(ledger_sha256=report.LEDGER_SHA[report.verified.WR],model_id=report.verified.WR,request_count=24574),report.verified.WR)

    def test_seed_scores_and_count_binding(self):
        r=report.Reporter.__new__(report.Reporter);r.ledgers={};r.seed={(True,7,1,240):np.array([1.,2.,3.])}
        r.cache=sqlite3.connect(':memory:');r.cache.execute('CREATE TABLE raw_pairs(path TEXT PRIMARY KEY,signature TEXT,checkpoint_sha256 TEXT,items TEXT)')
        with tempfile.TemporaryDirectory()as d:
            cp=Path(d)/'x.npz';rp=Path(d)/'x.json';np.savez(cp,unused=np.array([0]));rp.write_text('{}')
            payload={'index':7,'paths':240,'kind':'short','checkpoint_sha256':'hash','items':[{'H':1}]}
            self.assertEqual(r.raw_items(cp,rp,payload)[0]['values'],[1.,2.,3.])
            self.assertEqual(r.raw_items(cp,rp,payload)[0]['values'],[1.,2.,3.])
            rp.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'changed'):r.raw_items(cp,rp,payload)
    def test_raw_loss_validation_no_silent_drop(self):
        r=report.Reporter.__new__(report.Reporter);r.ledgers={};r.seed={}
        r.cache=sqlite3.connect(':memory:');r.cache.execute('CREATE TABLE raw_pairs(path TEXT PRIMARY KEY,signature TEXT,checkpoint_sha256 TEXT,items TEXT)')
        with tempfile.TemporaryDirectory()as d:
            cp=Path(d)/'x.npz';rp=Path(d)/'x.json';rp.write_text('{}')
            np.savez(cp,**{f'{m}_1':np.array([-1.])for m in report.MODELS})
            with self.assertRaisesRegex(ValueError,'Invalid raw loss'):
                r.raw_items(cp,rp,{'index':7,'paths':240,'kind':'short','checkpoint_sha256':'x','items':[{'H':1}]})
if __name__=='__main__':unittest.main()
