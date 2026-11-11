"""Source verification and immutable validation-to-test artifacts for RQ1."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, LABELS, targets
from import_equal20_results import file_variant, validate_history
from q1_difficulty_common import align_frame
from final_threshold_core import BASE_FIELDS, decode, fit_thresholds, read_predictions, metric_row, write_evaluation

IMPLEMENTATION = ('final_threshold_core.py', 'final_study.py', 'final_inference.py', 'final_thresholds.py')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def create_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


class Study:
    def __init__(self, root=ROOT, config=None, output=None, data_dir=None):
        self.root = Path(root).resolve()
        self.config_path = Path(config or self.root/'configs/final_thresholds.json').resolve()
        self.config = read_json(self.config_path)
        self.data_dir = Path(data_dir or self.root/'data').resolve()
        self.output = Path(output or self.root/'reports/final').resolve()
        allowed = self.root/'reports/final'
        if self.output == self.root or (self.root in self.output.parents and not self.output.is_relative_to(allowed)):
            raise ValueError('Final output must stay in reports/final or an external reproduction directory')
        if self.output == self.data_dir or self.output in self.data_dir.parents or self.data_dir in self.output.parents:
            raise ValueError('Public output and private data directories must be separate')
        cfg, dec = self.config, self.config['decoding']
        if (cfg['protocol_id'] != 'vigoemotions_final_rq1_v1' or cfg['labels'] != LABELS
                or cfg['model']['label_count'] != 27 or cfg['model']['selected_epoch'] != 14
                or cfg['model']['retraining'] or cfg['seed'] != 42):
            raise ValueError('Unsupported model/label/seed contract; use the recorded RQ1 v1 protocol')
        expected = dict(positive_comparison='>=', fixed_threshold=.5, tuning_split='val',
                        candidate_thresholds=[i/100 for i in range(5, 81, 5)],
                        global_objective='macro_f1_over_all_27_labels', per_label_objective='binary_f1_for_each_label',
                        f1_zero_division=0, equal_objective_tie_break='highest_threshold; exact computed F1 equality, no rounded comparison',
                        no_validation_positive_fallback=.5, selected_global_threshold=None, selected_per_label_thresholds=None)
        if dec != expected:
            raise ValueError('Decoder settings differ from the implemented v1 protocol')
        gate = cfg['test_gate']
        if (gate['validation_reproduction'] != dict(absolute_tolerance=1e-4, relative_tolerance=0)
                or gate['test_inference_runs'] != 1 or gate['test_based_tuning']):
            raise ValueError('The validation reproduction and one-shot test gates cannot be weakened')
        if (cfg['analysis']['frequency_tiers']['rank_ranges_inclusive'] != {'head': [1, 9], 'mid': [10, 18], 'tail': [19, 27]}
                or cfg['analysis']['cardinality']['groups'] != ['0', '1', '2', '3', '>=4']
                or cfg['analysis']['zero_division'] != 0):
            raise ValueError('Unsupported tier/cardinality/zero-division settings')

    def ids(self, split):
        spec = self.config['data']['splits'][split]
        frame = pd.read_csv(self.root/spec['path'], dtype=str)
        if list(frame.columns) != ['id'] or frame.id.isna().any() or frame.id.duplicated().any():
            raise ValueError(f'Invalid fixed {split} IDs')
        ids = frame.id.tolist()
        checksum = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
        if len(ids) != spec['n'] or checksum != spec['ordered_ids_sha256']:
            raise ValueError(f'Fixed {split} ID/order hash mismatch')
        return ids

    def manifest(self):
        folder = self.root/'reports/kaggle_equal20_2026-11-05'
        manifest = read_json(folder/'evidence/27/manifest.json')
        profile = read_json(self.root/self.config['model']['source_profile'])
        run = read_json(folder/'evidence/27/phobert/run.json')
        model = self.config['model']
        if (manifest['profile'] != profile or run['config'] != profile or manifest['label_spaces'] != [27]
                or not manifest['validation_only'] or profile['phobert']['revision'] != model['revision']
                or profile['phobert']['checkpoint'] != model['pretrained_checkpoint']
                or run['best_epoch'] != model['selected_epoch'] or len(run['history']) != model['training_epochs_completed']):
            raise ValueError('Original run/profile differs from the selected frozen model')
        prep = self.config['preprocessing']
        if (prep['max_length'] != profile['phobert']['max_length'] or prep['inference_batch_size'] != profile['phobert']['batch_size']
                or prep['inference_shuffle'] or not prep['truncation'] or prep['padding'] != 'max_length' or prep['model_mode'] != 'eval'):
            raise ValueError('Unsupported preprocessing/inference configuration')
        for name in ('common.py', 'train_neural.py'):
            file_variant((self.root/'scripts'/name).read_bytes(), manifest['source_sha256'][name])
        for split in ('train', 'val', 'test'):
            self.ids(split)  # ID files only; never reads test CSV/targets.
            if self.config['data']['splits'][split]['ordered_ids_sha256'] != manifest['split_ids_sha256'][split]:
                raise ValueError('Split configuration differs from the original manifest')
        provenance = read_json(self.root/model['source_provenance'])['studies']['27']
        if provenance['zip_sha256'] != model['archive_sha256']:
            raise ValueError('Archive provenance mismatch')
        for suffix, path in [('phobert_run.json', folder/'evidence/27/phobert/run.json'),
                             ('phobert_val.npz', self.root/self.config['data']['validation_predictions'])]:
            checks = [r['sha256'] for r in provenance['exported_members'] if r['member'].endswith('/'+suffix)]
            if checks != [digest(path)]:
                raise ValueError(f'Original {suffix} checksum mismatch')
        return manifest

    def frame(self, split):
        manifest = self.manifest()
        path = self.data_dir/f'{split}.csv'
        file_variant(path.read_bytes(), manifest['environment']['data_sha256'][split])
        source = pd.read_csv(path, dtype=str)
        if split != 'train' and len(source) != len(self.ids(split)):
            raise ValueError(f'Unexpected extra {split} source IDs')
        return align_frame(source, self.ids(split))

    def reference(self):
        self.manifest()
        path = self.root/self.config['data']['validation_predictions']
        if digest(path) != self.config['data']['validation_predictions_sha256']:
            raise ValueError('Source validation probability hash mismatch')
        values = read_predictions(path, self.ids('val'))
        if not np.array_equal(values['true_label_vector'], targets(self.frame('val'))):
            raise ValueError('Validation targets differ from source data')
        measured = metric_row(values['true_label_vector'], values['probability_vector'], .5)['macro_f1']
        if not np.isclose(measured, self.config['data']['validation_reference_macro_f1_at_0_5'], rtol=0, atol=1e-12):
            raise ValueError('Source fixed-threshold Macro-F1 mismatch')
        run = read_json(self.root/'reports/kaggle_equal20_2026-11-05/evidence/27/phobert/run.json')
        validate_history(run, 27, measured)
        return values

    def tiers(self):
        frame = pd.read_csv(self.root/self.config['analysis']['frequency_tiers']['membership_path'])
        counts = targets(self.frame('train')).sum(axis=0)
        order = sorted(range(27), key=lambda c: (-counts[c], c))
        expected = pd.DataFrame([dict(rank=r, label_id=c, label=LABELS[c], train_count=int(counts[c]),
                                      tier='head' if r <= 9 else 'mid' if r <= 18 else 'tail')
                                 for r, c in enumerate(order, 1)])
        if not frame.equals(expected):
            raise ValueError('Tier membership must match retained train-only frequencies')
        return dict(zip(frame.label, frame.tier))

    def sources(self):
        self.reference(); self.tiers()
        paths = ['docs/final_protocol.md', self.config['analysis']['frequency_tiers']['membership_path'],
                 self.config['model']['source_profile'], self.config['model']['source_provenance'],
                 self.config['data']['validation_predictions'],
                 'reports/kaggle_equal20_2026-11-05/evidence/27/manifest.json',
                 'reports/kaggle_equal20_2026-11-05/evidence/27/phobert/run.json',
                 'scripts/common.py', 'scripts/train_neural.py', 'scripts/import_equal20_results.py',
                 'scripts/q1_difficulty_common.py', *['scripts/'+p for p in IMPLEMENTATION]]
        paths += [spec['path'] for spec in self.config['data']['splits'].values()]
        result = {p: digest(self.root/p) for p in paths}
        result['final_config'] = digest(self.config_path)
        result.update({f'private_source/{s}.csv': digest(self.data_dir/f'{s}.csv') for s in ('train', 'val')})
        return result

    def reproduction(self):
        report = read_json(self.output/'validation_reproduction.json')
        if report['status'] != 'verified' or report['sources_sha256'] != self.sources():
            raise ValueError('Validation reproduction is missing, failed or stale')
        path = self.output/'predictions/val_reproduced.npz'
        if digest(path) != report['prediction_sha256'] or report['checkpoint_sha256'] != self.config['model']['checkpoint_sha256']:
            raise ValueError('Reproduction probabilities/checkpoint changed')
        actual, reference = read_predictions(path, self.ids('val')), self.reference()
        if not np.array_equal(actual['true_label_vector'], reference['true_label_vector']):
            raise ValueError('Reproduced targets differ from validation')
        a, b = actual['probability_vector'], reference['probability_vector']
        if not np.allclose(a, b, rtol=0, atol=1e-4) or not np.array_equal(decode(a, .5), decode(b, .5)):
            raise ValueError('Full validation probabilities or fixed decisions do not reproduce')
        return report

    def expected_thresholds(self):
        values = self.reference()
        fitted = fit_thresholds(values['true_label_vector'], values['probability_vector'], self.config['decoding']['candidate_thresholds'])
        return dict(protocol_id=self.config['protocol_id'], labels=LABELS, **fitted,
                    sources_sha256=self.sources(), reproduction_sha256=digest(self.output/'validation_reproduction.json'))

    def fit(self):
        self.reproduction()
        if any((self.output/p).exists() for p in ('thresholds.json', 'pre_test_lock.json', 'test_attempt.json')):
            raise FileExistsError('Fitted/locked studies are immutable; use the existing outputs')
        fitted = self.expected_thresholds()
        create_json(self.output/'thresholds.json', fitted)
        tiers = self.tiers()
        counts = pd.read_csv(self.root/self.config['analysis']['frequency_tiers']['membership_path']).set_index('label').train_count
        rows = write_evaluation(self.output, 'val', self.reference(), fitted, tiers)
        pd.DataFrame([dict(label=label, threshold=fitted['per_label_thresholds'][c],
                           validation_support=fitted['validation_support'][c], train_count=int(counts[label]), tier=tiers[label],
                           fallback=c in fitted['fallback_label_indices']) for c, label in enumerate(LABELS)]).to_csv(
            self.output/'tables/threshold_profile.csv', index=False)
        pd.DataFrame(fitted['global_sweep']).sort_values('threshold').to_csv(self.output/'tables/global_sweep.csv', index=False)
        return rows

    def fitted(self):
        self.reproduction()
        fitted = read_json(self.output/'thresholds.json')
        if fitted != self.expected_thresholds():
            raise ValueError('Thresholds/source bindings differ from validation-only optimization')
        return fitted

    def lock(self):
        self.regenerate_split('val', self.fitted(), self.tiers())
        bindings = dict(sources_sha256=self.sources(), thresholds_sha256=digest(self.output/'thresholds.json'),
                        reproduction_sha256=digest(self.output/'validation_reproduction.json'),
                        validation_predictions_sha256=digest(self.output/'predictions/val.npz'))
        create_json(self.output/'pre_test_lock.json', dict(protocol_id=self.config['protocol_id'], **bindings))
        return bindings

    def verify_lock(self):
        self.fitted()
        lock = read_json(self.output/'pre_test_lock.json')
        expected = dict(protocol_id=self.config['protocol_id'], sources_sha256=self.sources(),
                        thresholds_sha256=digest(self.output/'thresholds.json'),
                        reproduction_sha256=digest(self.output/'validation_reproduction.json'),
                        validation_predictions_sha256=digest(self.output/'predictions/val.npz'))
        if lock != expected:
            raise ValueError('Pre-test lock changed or its inputs are stale')
        return lock

    def report(self):
        """Regenerate derived tables without rewriting or re-inferring probabilities."""
        fitted, tiers = self.fitted(), self.tiers()
        self.regenerate_split('val', fitted, tiers)
        attempt_path = self.output/'test_attempt.json'
        if not attempt_path.exists():
            return dict(status='validation_tables_regenerated', test_evaluated=False)
        self.verify_lock()
        attempt = read_json(attempt_path)
        if attempt['status'] != 'completed' or digest(self.output/'predictions/test.npz') != attempt['predictions_sha256']:
            raise ValueError('Test inference must complete before regenerating its tables')
        self.regenerate_split('test', fitted, tiers)
        self.write_gap()
        return dict(status='validation_and_test_tables_regenerated', inference_performed=False)

    def write_gap(self):
        fitted = self.fitted()
        arrays = {s: read_predictions(self.output/f'predictions/{s}.npz', self.ids(s)) for s in ('val', 'test')}
        rows = []
        for strategy, cut in [('fixed', .5), ('global', fitted['global_threshold']), ('per_label', fitted['per_label_thresholds'])]:
            metrics = {s: metric_row(a['true_label_vector'], a['probability_vector'], cut) for s, a in arrays.items()}
            rows.extend(dict(strategy=strategy, metric=m, val=metrics['val'][m], test=metrics['test'][m],
                             gap=metrics['val'][m]-metrics['test'][m]) for m in ('macro_f1', 'micro_f1'))
        pd.DataFrame(rows).to_csv(self.output/'tables/generalization_gap.csv', index=False)

    def regenerate_split(self, split, fitted, tiers):
        values = read_predictions(self.output/f'predictions/{split}.npz', self.ids(split))
        for strategy, cut in [('fixed', .5), ('global', fitted['global_threshold']), ('per_label', fitted['per_label_thresholds'])]:
            if not np.array_equal(values.get('pred_'+strategy), decode(values['probability_vector'], cut)):
                raise ValueError('Saved decoder outputs differ from fitted thresholds')
        if split == 'val':
            reference = self.reference()
            if any(not np.array_equal(values[key], reference[key]) for key in BASE_FIELDS):
                raise ValueError('Saved validation output differs from the source NPZ')
        write_evaluation(self.output, split, values, fitted, tiers, save_predictions=False)
