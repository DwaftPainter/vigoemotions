"""Reporting must reject changed evidence without running a neural model."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_final_figures import verify_inputs


class FinalFigureEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.members = {
            f'tables/{split}_{table}.csv': b'fixture\n'
            for split in ('val', 'test')
            for table in ('comparison', 'per_label', 'frequency_tiers', 'cardinality', 'neutral')
        }
        for name in ('threshold_profile', 'global_sweep', 'generalization_gap'):
            self.members[f'tables/{name}.csv'] = b'fixture\n'
        for name in ('val', 'val_reproduced', 'test', 'test_probabilities'):
            self.members[f'predictions/{name}.npz'] = b'synthetic probabilities'
        self.put_json('thresholds.json', {'global_threshold': .4})
        runtime = {'device': 'cuda', 'packages': {'python': 'synthetic'}}
        self.put_json('validation_reproduction.json', {'status': 'verified', 'runtime': runtime})
        self.put_json('pre_test_lock.json', {
            'thresholds_sha256': self.checksum('thresholds.json'),
            'reproduction_sha256': self.checksum('validation_reproduction.json'),
            'validation_predictions_sha256': self.checksum('predictions/val.npz'),
        })
        self.attempt = {
            'status': 'completed', 'runtime': runtime,
            'lock_sha256': self.checksum('pre_test_lock.json'),
            'raw_probabilities_sha256': self.checksum('predictions/test_probabilities.npz'),
            'predictions_sha256': self.checksum('predictions/test.npz'),
        }
        self.put_json('test_attempt.json', self.attempt)
        self.flush()

    def put_json(self, name, value):
        self.members[name] = json.dumps(value).encode()

    def checksum(self, name):
        return hashlib.sha256(self.members[name]).hexdigest()

    def flush(self):
        for name, content in self.members.items():
            path = self.output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (self.output / 'execution_verification.json').write_text(json.dumps({
            'protocol_id': 'vigoemotions_final_rq1_28_v2',
            'imported_members_sha256': {name: self.checksum(name) for name in self.members},
        }))

    def test_verified_inputs_are_read_without_modification(self):
        before = {p: p.read_bytes() for p in self.output.rglob('*') if p.is_file()}
        verify_inputs(self.output)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_changed_metric_table_is_rejected(self):
        (self.output / 'tables/test_comparison.csv').write_text('altered metric\n')
        with self.assertRaisesRegex(ValueError, 'hash'):
            verify_inputs(self.output)

    def test_failed_test_is_rejected_even_with_updated_import_hash(self):
        self.attempt['status'] = 'failed'
        self.put_json('test_attempt.json', self.attempt)
        self.flush()
        with self.assertRaisesRegex(ValueError, 'completed'):
            verify_inputs(self.output)

    def test_changed_raw_scores_are_rejected_even_with_updated_import_hash(self):
        self.members['predictions/test_probabilities.npz'] = b'different probabilities'
        self.flush()
        with self.assertRaisesRegex(ValueError, 'binding'):
            verify_inputs(self.output)


if __name__ == '__main__':
    unittest.main()
