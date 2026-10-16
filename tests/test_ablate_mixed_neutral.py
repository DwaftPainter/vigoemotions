import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import common
import ablate_mixed_neutral as ablation


def fixture():
    labels = [[c] for c in range(27)] * 2 + [[27], [27, 2], [1, 27, 1]]
    return pd.DataFrame({'id': [str(i) for i in range(len(labels))],
                         'label_ids': labels,
                         'normalized': [f'shared token{i % 27}' for i in range(len(labels))]})


class MixedNeutralTests(unittest.TestCase):
    def setUp(self):
        self.train = fixture()
        self.validation = self.train.iloc[[0, 27, 54, 55]].copy()
        self.validation['id'] = ['val0', 'val1', 'val2', 'val3']
        self.config = json.loads((common.ROOT/'configs/midterm.json').read_text())

    def test_detection_including_duplicates(self):
        for labels, expected in [([27], False), ([27, 27], False), ([2], False),
                                 ([2, 27], True), ([27, 2, 2], True)]:
            self.assertEqual(ablation.is_mixed_neutral(labels), expected)
        self.assertTrue(ablation.is_mixed_neutral('[0, 27]'))

    def test_exact_exclusion_and_neutral_only_preserved(self):
        original = self.train.copy(deep=True)
        conditions = ablation.training_conditions(self.train, expected_excluded=2)
        pd.testing.assert_frame_equal(conditions['A'], original)
        self.assertEqual(conditions['B'].id.tolist(), original.id.iloc[:-2].tolist())
        self.assertEqual(common.targets(conditions['B'])[-1].sum(), 0)
        self.assertEqual(common.targets(conditions['A'])[-2, 2], 1)
        pd.testing.assert_frame_equal(self.train, original)
        with self.assertRaises(ValueError):
            ablation.training_conditions(self.train)  # Required real-data count is strict.

    def test_independent_training_and_fixed_seed_reproducibility(self):
        p1, meta = ablation.fit_condition(self.train, self.validation, self.config)
        p2, _ = ablation.fit_condition(self.train, self.validation, self.config)
        np.testing.assert_allclose(p1, p2, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(p1 >= .5, p2 >= .5)
        changed_val = self.validation.copy()
        changed_val.normalized = 'validationexclusive'
        _, changed_meta = ablation.fit_condition(self.train, changed_val, self.config)
        self.assertEqual(meta['tfidf_features'], changed_meta['tfidf_features'])

    def test_validation_identity_metrics_and_no_test_input(self):
        original = self.validation.copy(deep=True)
        # Any attempt to reload data (including test) during fitting/evaluation fails.
        with tempfile.TemporaryDirectory() as folder, patch.object(
                ablation, 'load_data', side_effect=AssertionError('No test data allowed')):
            out = Path(folder)
            comparison, tiers = ablation.run_comparison(
                self.train, self.validation, self.config, out, expected_excluded=2)
            ablation.verify_predictions(out)
            pd.testing.assert_frame_equal(self.validation, original)
            self.assertEqual(sorted(np.concatenate(list(tiers.values())).tolist()), list(range(27)))
            for name in ('A', 'B'):
                with np.load(out/f'{name}_val.npz') as archive:
                    np.testing.assert_array_equal(archive['sample_id'], original.id)
                    np.testing.assert_array_equal(archive['true_label_vector'], common.targets(original))
                    expected = common.scores(common.targets(original), archive['probability_vector'])
                    for metric, value in expected.items():
                        self.assertEqual(comparison.set_index('metric').loc[metric, name], value)
                    self.assertEqual(set(archive.files), {'sample_id', 'true_label_vector',
                                                        'probability_vector', 'labels'})
            np.testing.assert_allclose(comparison.B_minus_A, comparison.B-comparison.A)
            np.testing.assert_allclose(comparison.absolute_difference, abs(comparison.B-comparison.A))
            self.assertEqual(pd.read_csv(out/'per_label_f1.csv').shape[0], 27)
            self.assertEqual(pd.read_csv(out/'validation_subgroup_comparison.csv').samples.sum(), 4)

    def test_original_csvs_unchanged_and_test_targets_irrelevant(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'data').mkdir()
            for split, rows in {
                'train': [('t0', 'overlap', '[2]'), ('t1', 'training', '[27, 2]'),
                          ('t2', 'neutral', '[27]')],
                'val': [('v0', 'validation', '[2, 27]')],
                'test': [('s0', 'overlap', '[0]')],
            }.items():
                pd.DataFrame(rows, columns=['id', 'text', 'labels']).to_csv(root/'data'/f'{split}.csv', index=False)
            paths = list((root/'data').glob('*.csv'))
            before = {p: p.read_bytes() for p in paths}
            with patch.object(common, 'ROOT', root):
                frames, excluded = common.load_data()
                ablation.training_conditions(frames['train'], expected_excluded=1)
                self.assertEqual(excluded.id.tolist(), ['t0'])
                self.assertEqual(before, {p: p.read_bytes() for p in paths})
                # Alter only synthetic test targets: cleaned training/validation are invariant.
                test_path = root/'data/test.csv'
                test_frame = pd.read_csv(test_path)
                test_frame['labels'] = '[26, 27]'
                test_frame.to_csv(test_path, index=False)
                changed, _ = common.load_data()
                for split in ('train', 'val'):
                    pd.testing.assert_frame_equal(frames[split], changed[split])

    def test_output_outside_study_is_refused(self):
        with patch.object(sys, 'argv', ['ablate_mixed_neutral.py', '--output', str(common.ROOT/'reports')]):
            with self.assertRaises(SystemExit):
                ablation.main()

    def test_existing_output_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            sentinel = out/'comparison.csv'
            sentinel.write_text('keep existing results')
            with patch.object(ablation, 'OUTPUT', out), patch.object(
                    sys, 'argv', ['ablate_mixed_neutral.py', '--output', str(out)]):
                with self.assertRaises(SystemExit):
                    ablation.main()
            self.assertEqual(sentinel.read_text(), 'keep existing results')


if __name__ == '__main__':
    unittest.main()
