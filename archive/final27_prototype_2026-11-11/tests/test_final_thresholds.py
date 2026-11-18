"""Controlled decoder experiments and pre-test gates, using synthetic data only."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import LABELS, targets
from final_threshold_core import decode, fit_thresholds, read_predictions, metric_row
from final_study import Study, IMPLEMENTATION, digest, read_json
from final_inference import restore, reproduce_validation, infer_test


class DecoderTests(unittest.TestCase):
    def test_boundary_and_scalar_vector_decoding_agree(self):
        p = np.array([[.25, .5], [.75, .49]], dtype=np.float32)
        np.testing.assert_array_equal(decode(p, .5), [[0, 1], [1, 0]])
        for threshold in (.35, .5, .75):
            np.testing.assert_array_equal(decode(p, threshold), decode(p, [threshold]*2))
        with self.assertRaises(ValueError):
            decode(p, [.5])

    def test_global_and_per_label_fit_match_exhaustive_objectives(self):
        y = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
        p = np.array([[.4, .1], [.6, .8], [.2, .65], [.3, .7]])
        grid = [.25, .5, .75]
        result = fit_thresholds(y, p, grid)
        objective = lambda threshold: f1_score(y, decode(p, threshold), average='macro', zero_division=0)
        expected = max(grid, key=lambda threshold: (objective(threshold), threshold))
        self.assertEqual(result['global_threshold'], expected)
        for c in range(2):
            best = max(grid, key=lambda t: (f1_score(y[:, c], decode(p, t)[:, c], zero_division=0), t))
            self.assertEqual(result['per_label_thresholds'][c], best)
        self.assertGreaterEqual(objective(result['global_threshold']), objective(.5))
        self.assertGreaterEqual(f1_score(y, decode(p, result['per_label_thresholds']), average='macro'), objective(expected))

    def test_ties_choose_highest_and_zero_support_falls_back(self):
        y = np.array([[1, 0], [0, 0]])
        p = np.array([[.9, .9], [.1, .1]])
        result = fit_thresholds(y, p, [.25, .5, .75])
        self.assertEqual(result['global_threshold'], .75)
        self.assertEqual(result['per_label_thresholds'], [.75, .5])
        self.assertEqual(result['fallback_label_indices'], [1])

    def test_invalid_probabilities_targets_and_grids_fail(self):
        for y, p, grid in [([[2]], [[.5]], [.5]), ([[1]], [[np.nan]], [.5]),
                           ([[1]], [[1.1]], [.5]), ([[1]], [[.5]], []),
                           ([[1]], [[.5]], [.5, .5]), ([[1]], [[.5]], [.6])]:
            with self.subTest(y=y, p=p, grid=grid), self.assertRaises(ValueError):
                fit_thresholds(y, p, grid)

    def test_empty_group_is_missing_and_lrap_does_not_change(self):
        y = np.array([[1, 0], [0, 1]])
        p = np.array([[.4, .3], [.2, .8]])
        self.assertIsNone(metric_row(y[:0], p[:0], .5)['macro_f1'])
        self.assertEqual(metric_row(y, p, .5)['lrap'], metric_row(y, p, [.3, .7])['lrap'])

    def test_prediction_contract_rejects_wrong_ids_labels_text_and_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'validation.npz'
            y = np.zeros((2, 27), dtype=int); y[0, 0] = 1
            values = dict(sample_id=np.array(['a', 'b']), labels=np.array(LABELS),
                          true_label_vector=y, probability_vector=y*.8+.1,
                          label_cardinality=y.sum(axis=1))
            np.savez_compressed(path, **values)
            read_predictions(path, ['a', 'b'])
            for key, replacement in [('sample_id', np.array(['b', 'a'])),
                                     ('labels', np.array(LABELS[::-1])),
                                     ('label_cardinality', np.array([0, 0])),
                                     ('text', np.array(['private', 'private'])),
                                     ('true_label_vector', np.full((2, 27), 2))]:
                with self.subTest(key=key):
                    np.savez_compressed(path, **{**values, key: replacement})
                    with self.assertRaises(ValueError):
                        read_predictions(path, ['a', 'b'])


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        repository = Path(__file__).resolve().parents[1]
        cfg = read_json(repository/'configs/final_thresholds.json')
        for name in (*IMPLEMENTATION, 'common.py', 'train_neural.py', 'import_equal20_results.py', 'q1_difficulty_common.py'):
            dest = self.root/'scripts'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repository/'scripts'/name, dest)
        self.write('docs/final_protocol.md', 'Synthetic protocol fixture; not a research result.\n')
        profile = read_json(repository/cfg['model']['source_profile'])
        self.write(cfg['model']['source_profile'], profile)
        labels = {'train': [[c] for c in range(27)], 'val': [[0], [1, 2], [27], [0, 1, 2, 3]], 'test': [[2, 3], [27]]}
        data_hashes = {}
        for split, rows in labels.items():
            ids = [f'{split}-{i}' for i in range(len(rows))]
            self.write(f'splits/{split}.csv', 'id\n'+'\n'.join(ids)+'\n')
            spec = cfg['data']['splits'][split]
            spec.update(n=len(ids), ordered_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest())
            frame = pd.DataFrame({'id': ids, 'text': [f'Synthetic {s}' for s in ids], 'labels': [str(y) for y in rows]})
            content = frame.to_csv(index=False).encode()
            data_hashes[split] = hashlib.sha256(content).hexdigest()
            if split == 'test': self.test_csv = content
            else: self.write(f'data/{split}.csv', content)
            if split == 'val':
                self.validation_frame = frame.assign(label_ids=rows)
        y = targets(self.validation_frame)
        self.p = y*.8+.1
        self.write(cfg['data']['validation_predictions'], b'')
        np.savez_compressed(self.root/cfg['data']['validation_predictions'], sample_id=self.validation_frame.id.to_numpy(dtype=str),
                            labels=np.array(LABELS), true_label_vector=y, probability_vector=self.p,
                            label_cardinality=y.sum(axis=1))
        cfg['data']['validation_predictions_sha256'] = digest(self.root/cfg['data']['validation_predictions'])
        measured = metric_row(y, self.p, .5)['macro_f1']
        cfg['data']['validation_reference_macro_f1_at_0_5'] = measured
        run = dict(config=profile, best_epoch=14, history=[dict(epoch=i, macro_f1=measured if i == 14 else measured/2) for i in range(1, 21)])
        run_path = 'reports/kaggle_equal20_2026-11-05/evidence/27/phobert/run.json'
        self.write(run_path, run)
        self.archive = self.root/'private.zip'
        with ZipFile(self.archive, 'w') as z:
            z.writestr(cfg['model']['checkpoint_member'], b'synthetic weights, not a real model')
            for name in ('VnCoreNLP-1.2.jar', 'models/wordsegmenter/wordsegmenter.rdr', 'models/wordsegmenter/vi-vocab'):
                z.writestr('resources/vncorenlp/'+name, b'synthetic segmentation fixture')
        cfg['model'].update(archive_sha256=digest(self.archive), checkpoint_sha256=hashlib.sha256(b'synthetic weights, not a real model').hexdigest())
        self.write('reports/kaggle_equal20_2026-11-05/evidence/27/manifest.json', dict(
            profile=profile, label_spaces=[27], validation_only=True,
            source_sha256={name: digest(self.root/'scripts'/name) for name in ('common.py', 'train_neural.py')},
            environment=dict(data_sha256=data_hashes),
            split_ids_sha256={s: v['ordered_ids_sha256'] for s, v in cfg['data']['splits'].items()}))
        self.write(cfg['model']['source_provenance'], dict(studies={'27': dict(
            zip_sha256=cfg['model']['archive_sha256'], exported_members=[
                dict(member='labels27/phobert/phobert_run.json', sha256=digest(self.root/run_path)),
                dict(member='labels27/phobert/predictions/phobert_val.npz', sha256=cfg['data']['validation_predictions_sha256'])])}))
        self.write('configs/final_thresholds.json', cfg)
        tiers = pd.DataFrame([dict(rank=c+1, label_id=c, label=label, train_count=1,
                                   tier='head' if c < 9 else 'mid' if c < 18 else 'tail') for c, label in enumerate(LABELS)])
        self.write('docs/final_label_tiers.csv', tiers.to_csv(index=False))
        self.study = Study(root=self.root)
        self.private = self.root/'data/final'
        self.runtime = dict(device='synthetic fixture', packages={'synthetic': 'test-only'})
        versions = patch('final_inference.runtime_versions', return_value=self.runtime['packages'])
        versions.start(); self.addCleanup(versions.stop)
        restore(self.study, self.archive, self.private)

    def write(self, relative, value):
        path = self.root/relative; path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, dict): path.write_text(json.dumps(value))
        elif isinstance(value, bytes): path.write_bytes(value)
        else: path.write_text(value)

    def reproduce(self):
        with patch('final_inference.predict_checkpoint', return_value=(self.p.copy(), self.runtime)):
            return reproduce_validation(self.study, self.private)

    def lock(self):
        self.reproduce(); self.study.fit(); self.study.lock()

    def test_reference_and_tiers_never_read_test_csv(self):
        self.assertFalse((self.root/'data/test.csv').exists())
        self.assertEqual(self.study.reference()['probability_vector'].shape, (4, 27))
        self.assertEqual(len(self.study.tiers()), 27)

    def test_fit_requires_reproduction_and_lock_requires_fit(self):
        with self.assertRaises(FileNotFoundError): self.study.fit()
        self.reproduce()
        with self.assertRaises(FileNotFoundError): self.study.lock()
        self.study.fit(); self.study.lock()
        self.study.verify_lock()
        with self.assertRaises(FileExistsError): self.study.fit()
        with self.assertRaises(FileExistsError): self.study.lock()

    def test_failed_reproduction_produces_no_verified_artifacts(self):
        with patch('final_inference.predict_checkpoint', return_value=(self.p*.5, {})):
            with self.assertRaises(ValueError): reproduce_validation(self.study, self.private)
        self.assertFalse((self.study.output/'validation_reproduction.json').exists())
        self.assertFalse((self.study.output/'thresholds.json').exists())

    def test_lock_detects_changed_protocol_thresholds_and_reproduced_scores(self):
        self.lock()
        for path in [self.root/'docs/final_protocol.md', self.study.output/'thresholds.json',
                     self.study.output/'predictions/val_reproduced.npz']:
            original = path.read_bytes()
            path.write_bytes(original+b' ')
            with self.subTest(path=path), self.assertRaises(ValueError): self.study.verify_lock()
            path.write_bytes(original)

    def test_unlocked_test_is_rejected_before_inference_or_test_data_loading(self):
        with patch('final_inference.predict_checkpoint', side_effect=AssertionError('No inference before lock')):
            with self.assertRaises(FileNotFoundError): infer_test(self.study, self.private)
        self.assertFalse((self.study.output/'test_attempt.json').exists())

    def test_test_scores_once_then_saved_outputs_reproduce_decoders(self):
        self.lock(); self.write('data/test.csv', self.test_csv)
        p = np.full((2, 27), .1); p[0, [2, 3]] = .9
        with patch('final_inference.predict_checkpoint', return_value=(p, self.runtime)) as inference:
            rows = infer_test(self.study, self.private)
            with self.assertRaises(FileExistsError): infer_test(self.study, self.private)
            self.assertEqual(inference.call_count, 1)
        self.assertEqual(len(rows), 3)
        saved = read_predictions(self.study.output/'predictions/test.npz', self.study.ids('test'))
        fitted = self.study.fitted()
        for strategy, cut in [('fixed', .5), ('global', fitted['global_threshold']), ('per_label', fitted['per_label_thresholds'])]:
            np.testing.assert_array_equal(saved['pred_'+strategy], decode(p, cut))
        self.assertEqual(read_json(self.study.output/'test_attempt.json')['status'], 'completed')

    def test_changed_weights_block_test_before_targets_are_read(self):
        self.lock()
        (self.private/'best.pt').write_bytes(b'wrong weights')
        with self.assertRaises(ValueError): infer_test(self.study, self.private)
        self.assertFalse((self.study.output/'test_attempt.json').exists())

    def test_failed_technical_inference_requires_explicit_unchanged_resume(self):
        self.lock(); self.write('data/test.csv', self.test_csv)
        with patch('final_inference.predict_checkpoint', side_effect=RuntimeError('synthetic device failure')):
            with self.assertRaises(RuntimeError): infer_test(self.study, self.private)
        with self.assertRaises(FileExistsError): infer_test(self.study, self.private)
        p = np.full((2, 27), .1)
        with patch('final_inference.predict_checkpoint', return_value=(p, self.runtime)):
            infer_test(self.study, self.private, resume_failed=True)
        self.assertEqual(read_json(self.study.output/'test_attempt.json')['attempts'], 2)

    def test_saved_test_probabilities_resume_without_second_model_inference(self):
        self.lock(); self.write('data/test.csv', self.test_csv)
        with patch('final_inference.predict_checkpoint', return_value=(np.full((2, 27), .1), self.runtime)) as inference:
            with patch('final_inference.write_evaluation', side_effect=RuntimeError('synthetic reporting failure')):
                with self.assertRaises(RuntimeError): infer_test(self.study, self.private)
            self.assertEqual(inference.call_count, 1)
        with patch('final_inference.predict_checkpoint', side_effect=AssertionError('Scores already persisted')):
            infer_test(self.study, self.private, resume_failed=True)

    def test_report_regenerates_tables_without_touching_probabilities(self):
        self.lock(); self.write('data/test.csv', self.test_csv)
        with patch('final_inference.predict_checkpoint', return_value=(np.full((2, 27), .1), self.runtime)):
            infer_test(self.study, self.private)
        before = {p.name: digest(p) for p in (self.study.output/'predictions').glob('*.npz')}
        (self.study.output/'tables/test_comparison.csv').unlink()
        (self.study.output/'tables/generalization_gap.csv').unlink()
        with patch('final_inference.predict_checkpoint', side_effect=AssertionError('No inference during report')):
            self.study.report()
        self.assertEqual(before, {p.name: digest(p) for p in (self.study.output/'predictions').glob('*.npz')})
        self.assertTrue((self.study.output/'tables/generalization_gap.csv').exists())

    def test_validation_fit_ignores_test_csv_contents(self):
        self.write('data/test.csv', b'tampered held-out labels must not be read by fitting')
        self.reproduce(); self.study.fit()
        self.assertTrue((self.study.output/'thresholds.json').exists())

    def test_modified_test_source_rejected_before_model_inference(self):
        self.lock(); self.write('data/test.csv', self.test_csv.replace(b'[27]', b'[26]'))
        with patch('final_inference.predict_checkpoint', side_effect=AssertionError('No inference on modified test source')):
            with self.assertRaises(ValueError): infer_test(self.study, self.private)

    def test_protocol_settings_and_public_weight_locations_cannot_be_overridden(self):
        config = read_json(self.study.config_path)
        config['test_gate']['validation_reproduction']['absolute_tolerance'] = .1
        self.write('configs/final_thresholds.json', config)
        with self.assertRaises(ValueError): Study(root=self.root)
        with self.assertRaises(ValueError): restore(self.study, self.archive, self.study.output/'weights')

    def test_output_cannot_overwrite_evidence_or_private_data(self):
        for folder in ('reports/kaggle_equal20_2026-11-05', 'data', 'docs'):
            with self.subTest(folder=folder), self.assertRaises(ValueError):
                Study(root=self.root, output=self.root/folder)

    def test_lock_rejects_foreign_validation_outputs(self):
        self.reproduce(); self.study.fit()
        path = self.study.output/'predictions/val.npz'
        values = read_predictions(path, self.study.ids('val'))
        values['probability_vector'] = np.zeros((4, 27))
        values.update({f'pred_{s}': np.zeros((4, 27), dtype=bool) for s in ('fixed', 'global', 'per_label')})
        np.savez_compressed(path, **values)
        with self.assertRaises(ValueError): self.study.lock()

    def test_runtime_change_blocks_test_before_loading_targets(self):
        self.lock()
        with patch('final_inference.runtime_versions', return_value={'synthetic': 'changed'}):
            with self.assertRaises(ValueError): infer_test(self.study, self.private)
        self.assertFalse((self.study.output/'test_attempt.json').exists())


class PackageTests(unittest.TestCase):
    def test_code_bundle_keeps_runtime_inputs_without_shipping_private_assets(self):
        import ast
        import nbformat
        from package_final import package
        with tempfile.TemporaryDirectory() as tmp:
            path = package(Path(tmp)/'code.zip')
            with ZipFile(path) as z:
                names = z.namelist()
                self.assertFalse(any(n.startswith('data/') or n.endswith(('.pt', '.jar', '.joblib')) for n in names))
                self.assertIn('reports/kaggle_equal20_2026-11-05/evidence/27/phobert/val_predictions.npz', names)
                nb = json.loads(z.read('notebooks/10_kaggle_final_validation.ipynb'))
                nbformat.validate(nb)
                for cell in nb['cells']:
                    if cell['cell_type'] != 'code': continue
                    self.assertEqual(cell['outputs'], [])
                    source = ''.join(cell['source'])
                    self.assertNotIn("run('infer-test'", source)
                    ast.parse('\n'.join(line for line in source.splitlines() if not line.startswith('%')))


if __name__ == '__main__':
    unittest.main()
