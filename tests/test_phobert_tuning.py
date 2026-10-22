import json
import contextlib
from copy import deepcopy
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT, LABELS, scores, write_json
from tune_phobert import RATES, prepare, load_plan, summarize
import tune_phobert as tuning


class PhoBERTTuningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)/'study'
        self.base = json.loads((ROOT/'configs/midterm.json').read_text())
        self.runtime = dict(python='fixture-python', platform='fixture-platform', processor='fixture-cpu',
            packages={'torch': 'fixture-torch', 'transformers': 'fixture-transformers'},
            data_sha256={'train': 'train', 'val': 'val', 'test': 'test'})
        self.checkpoints = Path(self.temp.name)/'data/checkpoints/study'
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
                environment=deepcopy(self.runtime),
                device='synthetic GPU fixture', parameter_count=123, train_seconds=1,
                history=[dict(epoch=i, loss=1/i, **measured) for i in range(1, 5)]))
            pd.DataFrame([dict(model='phobert', split='val', threshold=.5,
                               parameter_count=123, train_seconds=1, **measured)]).to_csv(
                folder/'tables/phobert_metrics.csv', index=False)
            pd.DataFrame({'label': LABELS, 'f1': f1_score(y, p >= .5, average=None)}).to_csv(
                folder/'tables/phobert_per_label.csv', index=False)
            pd.DataFrame([dict(style='all', **measured)]).to_csv(
                folder/'tables/phobert_styles.csv', index=False)
            pd.DataFrame([dict(sample_id='synthetic', kind='FN', label='joy')]).to_csv(
                folder/'tables/phobert_error_cases_val.csv', index=False)
            checkpoint = self.checkpoints/name/'best.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(f'checkpoint fixture {name}'.encode())
        self.saved_runs = {name: self.snapshot(self.output/name) for name in RATES}
        self.training_calls = []

    @staticmethod
    def snapshot(folder):
        return {str(path.relative_to(folder)): path.read_bytes()
                for path in folder.rglob('*') if path.is_file()}

    def make_missing(self, name):
        shutil.rmtree(self.output/name)
        shutil.rmtree(self.checkpoints/name)

    def train_missing(self, arguments, **kwargs):
        folder = Path(arguments[arguments.index('--output-dir')+1])
        checkpoint = Path(arguments[arguments.index('--checkpoint')+1])
        self.training_calls.append(folder.name)
        folder.mkdir(exist_ok=False)
        for relative, content in self.saved_runs[folder.name].items():
            path = folder/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        checkpoint.parent.mkdir(parents=True, exist_ok=False)
        with checkpoint.open('xb') as stream:
            stream.write(b'new checkpoint fixture')

    def run_cli(self, resume=True, runtime=None, device='synthetic GPU fixture', train=None):
        arguments = ['tune_phobert.py', 'run', '--output', str(self.output)]
        if resume:
            arguments.append('--resume')
        with patch.object(sys, 'argv', arguments), \
             patch.object(tuning, 'gpu_device', return_value=device), \
             patch.object(tuning, 'ROOT', Path(self.temp.name)), \
             patch.object(tuning, 'environment', return_value=runtime or self.runtime, create=True), \
             patch.object(tuning.subprocess, 'run', side_effect=train or self.train_missing) as launch, \
             patch.object(tuning, 'summarize', wraps=summarize) as summary:
            self.last_launch, self.last_summary = launch, summary
            tuning.main()
        return launch, summary

    def assert_resume_refused(self, explanation, **kwargs):
        before = self.snapshot(Path(self.temp.name))
        error = io.StringIO()
        with contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
            self.run_cli(**kwargs)
        self.last_launch.assert_not_called()
        self.last_summary.assert_not_called()
        self.assertIn(explanation, error.getvalue())
        self.assertEqual(before, self.snapshot(Path(self.temp.name)))

    def test_resume_skips_valid_first_run_and_trains_only_missing_rates(self):
        first = self.snapshot(self.output/'lr_1e-5')
        checkpoint = (self.checkpoints/'lr_1e-5/best.pt').read_bytes()
        for name in ('lr_2e-5', 'lr_3e-5'):
            self.make_missing(name)
        launch, summary = self.run_cli()
        self.assertEqual(self.training_calls, ['lr_2e-5', 'lr_3e-5'])
        self.assertEqual(launch.call_count, 2)
        summary.assert_called_once()
        self.assertEqual(first, self.snapshot(self.output/'lr_1e-5'))
        self.assertEqual(checkpoint, (self.checkpoints/'lr_1e-5/best.pt').read_bytes())
        self.assertEqual(json.loads((self.output/'study.json').read_text())['status'], 'completed')

    def test_resume_all_complete_runs_does_not_train_or_touch_run_files(self):
        before = {name: self.snapshot(self.output/name) for name in RATES}
        checkpoints = self.snapshot(self.checkpoints)
        launch, summary = self.run_cli()
        launch.assert_not_called()
        summary.assert_called_once()
        self.assertEqual(before, {name: self.snapshot(self.output/name) for name in RATES})
        self.assertEqual(checkpoints, self.snapshot(self.checkpoints))

    def test_resume_can_run_all_missing_rates(self):
        for name in RATES:
            self.make_missing(name)
        _, summary = self.run_cli()
        self.assertEqual(self.training_calls, list(RATES))
        summary.assert_called_once()

    def test_resume_preflights_later_partial_directory_before_training(self):
        self.make_missing('lr_2e-5')
        self.make_missing('lr_3e-5')
        (self.output/'lr_3e-5').mkdir()
        self.assert_resume_refused('lr_3e-5: incomplete')

    def test_resume_refuses_orphaned_checkpoint_without_output(self):
        shutil.rmtree(self.output/'lr_2e-5')
        self.assert_resume_refused('lr_2e-5: incomplete')

    def test_resume_refuses_corrupt_core_artifacts(self):
        folder = self.output/'lr_1e-5'
        run_path = folder/'phobert_run.json'
        original = run_path.read_bytes()
        for kind in ('config', 'short_history', 'duplicate_epoch', 'nonfinite_history'):
            with self.subTest(kind=kind):
                run = json.loads(original)
                if kind == 'config':
                    run['config']['phobert']['batch_size'] = 8
                elif kind == 'short_history':
                    run['history'].pop()
                elif kind == 'duplicate_epoch':
                    run['history'][1]['epoch'] = 1
                else:
                    run['history'][2]['loss'] = float('nan')
                write_json(run_path, run)
                self.assert_resume_refused('lr_1e-5: incomplete')
                run_path.write_bytes(original)
        for relative in ('phobert_run.json', 'tables/phobert_metrics.csv', 'predictions/phobert_val.npz'):
            with self.subTest(missing=relative):
                path = folder/relative
                content = path.read_bytes(); path.unlink()
                self.assert_resume_refused('lr_1e-5: incomplete')
                path.write_bytes(content)
        metrics_path = folder/'tables/phobert_metrics.csv'
        content = metrics_path.read_bytes()
        metrics = pd.read_csv(metrics_path); metrics['macro_f1'] = .1
        metrics.to_csv(metrics_path, index=False)
        self.assert_resume_refused('lr_1e-5: incomplete')
        metrics_path.write_bytes(content)
        archive = folder/'predictions/phobert_val.npz'
        with np.load(archive) as saved:
            values = {key: saved[key].copy() for key in saved.files}
        values['probability_vector'][0, 0] = 2
        np.savez_compressed(archive, **values)
        self.assert_resume_refused('lr_1e-5: incomplete')

    def test_resume_reports_current_runtime_mismatches_without_training(self):
        for section, key, value in [('packages', 'torch', 'different-version'),
                                    ('data_sha256', 'train', 'different-data'),
                                    ('data_sha256', 'val', 'different-validation')]:
            with self.subTest(section=section, key=key):
                runtime = deepcopy(self.runtime); runtime[section][key] = value
                self.assert_resume_refused(f'environment.{section}.{key}', runtime=runtime)
        self.assert_resume_refused('device', device='different GPU')

    def test_resume_rejects_incompatible_completed_runs(self):
        path = self.output/'lr_2e-5/phobert_run.json'
        run = json.loads(path.read_text()); run['environment']['packages']['torch'] = 'different-version'
        write_json(path, run)
        self.assert_resume_refused('environment.packages.torch')

    def test_run_without_resume_still_refuses_existing_runs_and_checkpoints(self):
        self.assert_resume_refused('Run output already exists', resume=False)
        for name in RATES:
            shutil.rmtree(self.output/name)
        self.assert_resume_refused('Private checkpoint directory exists', resume=False)

    def test_run_without_resume_can_still_train_a_fresh_study(self):
        for name in RATES:
            self.make_missing(name)
        self.checkpoints.rmdir()
        _, summary = self.run_cli(resume=False)
        self.assertEqual(self.training_calls, list(RATES))
        summary.assert_called_once()

    def test_resume_requires_an_existing_study(self):
        shutil.rmtree(self.output)
        self.assert_resume_refused('requires an existing prepared study')

    def test_resume_rejects_unreadable_probability_archive(self):
        archive = self.output/'lr_1e-5/predictions/phobert_val.npz'
        archive.write_bytes(archive.read_bytes()[:50])
        self.assert_resume_refused('lr_1e-5: incomplete')

    def test_resume_stops_on_training_failure_without_summarizing(self):
        self.make_missing('lr_2e-5'); self.make_missing('lr_3e-5')
        first = self.snapshot(self.output/'lr_1e-5')
        def fail(arguments, **kwargs):
            raise tuning.subprocess.CalledProcessError(1, arguments)
        with self.assertRaises(tuning.subprocess.CalledProcessError):
            self.run_cli(train=fail)
        self.last_summary.assert_not_called()
        self.assertEqual(first, self.snapshot(self.output/'lr_1e-5'))
        self.assertFalse((self.output/'lr_3e-5').exists())

    def test_resume_rechecks_new_run_before_selecting_a_winner(self):
        self.make_missing('lr_2e-5'); self.make_missing('lr_3e-5')
        def incompatible(arguments, **kwargs):
            self.train_missing(arguments, **kwargs)
            path = self.output/'lr_2e-5/phobert_run.json'
            run = json.loads(path.read_text()); run['environment']['packages']['torch'] = 'changed-during-run'
            write_json(path, run)
        error = io.StringIO()
        with contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
            self.run_cli(train=incompatible)
        self.assertIn('environment.packages.torch', error.getvalue())
        self.assertEqual(self.training_calls, ['lr_2e-5'])
        self.assertFalse((self.output/'comparison.csv').exists())

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
