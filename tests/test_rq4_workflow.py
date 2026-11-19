from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from final_inference import save_probabilities
from final_study import create_json, digest, read_json
from final_threshold_core import decode, read_predictions
from neutral_28_labels import LABELS28
from rq4_robustness import run_split, verify_clean

ROOT = Path(__file__).resolve().parents[1]


class SmallPairedStudy:
    """Synthetic text and scores, with the same paired evaluation artifact contract."""
    root = ROOT
    config = {'labels': LABELS28}

    def __init__(self, base):
        self.output = base/'rq1'
        self.data_dir = base/'data'
        self.frame_reads = []
        self.fitted = {'global_threshold': .4, 'per_label_thresholds': [.3] * 28}
        self.assets = {'fixture': 'synthetic'}
        self.probs = np.full((2, 28), .1, dtype=np.float32)
        self.probs[0, 0] = .8
        self.probs[1, 27] = .9
        for split in ('val','test'):
            path = self.output/f'predictions/{split}.npz'
            frame = self.frame(split)
            save_probabilities(path, frame, self.probs)
            values = read_predictions(path, self.ids(split))
            values.update(pred_fixed=decode(self.probs, .5), pred_global=decode(self.probs, .4),
                          pred_per_label=decode(self.probs, .3))
            np.savez_compressed(path, **values)
        self.frame_reads.clear()
        create_json(self.output/'thresholds.json', self.fitted)
        create_json(self.output/'validation_reproduction.json', {'assets': self.assets})
        create_json(self.output/'pre_test_lock.json', {
            'thresholds_sha256': digest(self.output/'thresholds.json'),
            'validation_predictions_sha256': digest(self.output/'predictions/val.npz'),
        })
        create_json(self.output/'test_attempt.json', {
            'status': 'completed',
            'lock_sha256': digest(self.output/'pre_test_lock.json'),
            'predictions_sha256': digest(self.output/'predictions/test.npz'),
        })

    def verify_lock(self):
        return read_json(self.output/'pre_test_lock.json')

    def ids(self, split):
        return [f'{split}-2', f'{split}-1']

    def frame(self, split):
        self.frame_reads.append(split)
        return pd.DataFrame({'id': self.ids(split), 'normalized': ['Đẹp 😀', 'just ordinary'],
                             'labels': ['0', '27'], 'label_ids': [[0], [27]]})

    def tiers(self):
        return {label: ('head', 'mid', 'tail')[c//9] for c, label in enumerate(LABELS28[:27])}


class PairedWorkflowTests(unittest.TestCase):
    def test_test_requires_validated_inputs_and_reuses_paired_baseline(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            study = SmallPairedStudy(base)
            output, private = base/'output', base/'private'
            runtime = {'device': 'cpu', 'packages': {'fixture': 'synthetic'}}
            degraded = study.probs.copy()
            degraded[0, 0] = .2

            def infer(_study, frame, _private, _device):
                self.assertEqual(frame.id.tolist(), study.ids(frame.id.iloc[0].rsplit('-', 1)[0]) * 3)
                self.assertEqual(frame.normalized.iloc[2], 'Dep 😀')
                return np.concatenate([study.probs, degraded, study.probs]), runtime

            protected = {path: digest(path) for path in study.output.rglob('*') if path.is_file()}
            with (patch('rq4_robustness.verify_assets', return_value=study.assets),
                  patch('rq4_robustness.predict_checkpoint', side_effect=infer),
                  patch('rq4_robustness.runtime_versions', return_value=runtime['packages'])):
                with self.assertRaisesRegex(ValueError, 'Complete RQ4 validation'):
                    run_split(study, output, 'test', private, 'cpu')
                self.assertEqual(study.frame_reads, [])
                run_split(study, output, 'val', private, 'cpu')
                coverage = output/'val/tables/noise_coverage.csv'
                original = coverage.read_bytes()
                coverage.write_bytes(original+b'\n')
                with self.assertRaisesRegex(ValueError, 'stale or modified'):
                    run_split(study, output, 'test', private, 'cpu')
                self.assertEqual(study.frame_reads, ['val'])
                coverage.write_bytes(original)
                run_split(study, output, 'test', private, 'cpu')
                with self.assertRaises(FileExistsError):
                    run_split(study, output, 'test', private, 'cpu')
            self.assertEqual(study.frame_reads, ['val','test'])
            self.assertEqual({p: digest(p) for p in protected}, protected)
            for split in ('val','test'):
                rows = pd.read_csv(output/split/'tables/comparison.csv')
                fixed = rows.query("strategy == 'fixed' and scope == 'all28'").set_index('condition')
                self.assertAlmostEqual(fixed.loc['no_diacritics', 'delta_macro_f1_pp_vs_original'], -100/28)
                self.assertEqual(fixed.loc['qwerty_typo', 'delta_macro_f1_pp_vs_original'], 0)
                values = read_predictions(output/split/'predictions/no_diacritics.npz', study.ids(split))
                np.testing.assert_array_equal(values['probability_vector'], degraded)
                self.assertTrue((output/split/'figures/macro_f1_delta.pdf').is_file())

    def test_original_score_check_rejects_fixed_decision_drift_even_within_tolerance(self):
        expected = np.full((1, 28), .50001)
        altered = np.full((1, 28), .49999)
        reference = dict(probability_vector=expected, pred_fixed=decode(expected, .5),
                         pred_global=decode(expected, .4), pred_per_label=decode(expected, .3))
        fitted = {'global_threshold': .4, 'per_label_thresholds': [.3]*28}
        with self.assertRaisesRegex(ValueError, 'fixed decisions'):
            verify_clean(altered, reference, fitted, 1e-4)


if __name__ == '__main__':
    unittest.main()
