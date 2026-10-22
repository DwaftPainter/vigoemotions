import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT, LABELS, scores, write_json
from tune_phobert import RATES, prepare, load_plan, summarize


class PhoBERTTuningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)/'study'
        self.base = json.loads((ROOT/'configs/midterm.json').read_text())
        prepare(self.output, self.base)
        for count, name in zip((27, 20, 10), RATES):
            folder = self.output/name
            (folder/'predictions').mkdir(parents=True)
            (folder/'tables').mkdir()
            y = np.eye(27, dtype=int)
            p = np.full((27, 27), .1)
            p[np.arange(count), np.arange(count)] = .8
            np.savez_compressed(folder/'predictions/phobert_val.npz', sample_id=np.array(LABELS),
                                true_label_vector=y, probability_vector=p, labels=np.array(LABELS))
            measured = scores(y, p)
            config = json.loads((self.output/'configs'/f'{name}.json').read_text())
            write_json(folder/'phobert_run.json', dict(config=config, best_epoch=1,
                environment={'data_sha256': {'train': 'train', 'val': 'val', 'test': 'test'}},
                device='synthetic GPU fixture', train_seconds=1,
                history=[dict(epoch=i, loss=1/i, **measured) for i in range(1, 5)]))
            pd.DataFrame([dict(split='val', threshold=.5, **measured)]).to_csv(
                folder/'tables/phobert_metrics.csv', index=False)
            pd.DataFrame({'label': LABELS, 'f1': f1_score(y, p >= .5, average=None)}).to_csv(
                folder/'tables/phobert_per_label.csv', index=False)
            pd.DataFrame([dict(style='all', **measured)]).to_csv(
                folder/'tables/phobert_styles.csv', index=False)
            pd.DataFrame([dict(sample_id='synthetic', kind='FN', label='joy')]).to_csv(
                folder/'tables/phobert_error_cases_val.csv', index=False)

    def test_selection_is_recomputed_from_probabilities_and_audit_is_preserved(self):
        audit = self.output/'neural_error_review.csv'
        audit.write_text('reviewer,observation\nstudent,keep my manual review\n')
        comparison, winner = summarize(self.output)
        self.assertEqual(winner.run, 'lr_1e-5')
        self.assertEqual(winner.macro_f1, 1)
        self.assertEqual(comparison.set_index('run').loc['lr_2e-5', 'delta_macro_f1_vs_control'], 0)
        self.assertIn('keep my manual review', audit.read_text())
        self.assertEqual(json.loads((self.output/'study.json').read_text())['status'], 'completed')
        self.assertTrue((self.output/'learning_curves.png').is_file())
        with self.assertRaises(FileExistsError):
            prepare(self.output, self.base)

    def test_changed_non_lr_setting_is_rejected(self):
        path = self.output/'configs/lr_1e-5.json'
        config = json.loads(path.read_text()); config['phobert']['batch_size'] = 8
        write_json(path, config)
        with self.assertRaisesRegex(ValueError, 'only learning_rate'):
            load_plan(self.output)

    def test_mismatched_validation_ids_are_rejected(self):
        path = self.output/'lr_2e-5/predictions/phobert_val.npz'
        with np.load(path) as saved:
            values = {k: saved[k].copy() for k in saved.files}
        values['sample_id'][0] = 'wrong-id'
        np.savez_compressed(path, **values)
        with self.assertRaises(AssertionError):
            summarize(self.output)

    def test_inconsistent_metric_or_data_hash_is_rejected(self):
        path = self.output/'lr_2e-5/phobert_run.json'
        run = json.loads(path.read_text()); run['environment']['data_sha256']['train'] = 'changed'
        write_json(path, run)
        with self.assertRaisesRegex(ValueError, 'same dataset'):
            summarize(self.output)
        run['environment']['data_sha256']['train'] = 'train'
        run['history'][0]['macro_f1'] = .99
        write_json(path, run)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            summarize(self.output)


if __name__ == '__main__':
    unittest.main()
