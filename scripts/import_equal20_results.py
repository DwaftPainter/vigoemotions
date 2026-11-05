"""Verify the two Kaggle archives and publish text-free midterm evidence.

Reads checkpoints only to check archive membership/size; never unpickles weights,
re-runs inference, trains, or scores test. Original ZIPs/notebooks stay untouched.
"""
import ast
from io import BytesIO, StringIO
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from common import ROOT, LABELS, load_data, targets, scores, write_json
from evaluate import save_evaluation
import neutral_28_labels as neutral

OUT = ROOT/'reports/kaggle_equal20_2026-11-05'
MODELS = ('tfidf_lr', 'gru', 'phobert')
METRICS = ('macro_f1', 'micro_f1', 'hamming_loss', 'lrap')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_variant(data, expected):
    """Prove exact LF/CRLF byte variants, without normalizing content or JSON."""
    lf = data.replace(b'\r\n', b'\n')
    for label, candidate in (('exact', data), ('LF', lf), ('CRLF', lf.replace(b'\n', b'\r\n'))):
        if digest(candidate) == expected:
            return label
    raise ValueError('Checksum is not an exact LF/CRLF variant of the local file')


def validate_predictions(saved, frame, labels):
    if set(saved) - {'sample_id', 'labels', 'true_label_vector', 'probability_vector', 'label_cardinality'}:
        raise ValueError('Unexpected NPZ fields; do not export potential source text')
    names = LABELS if labels == 27 else neutral.LABELS28
    expected = targets(frame) if labels == 27 else neutral.targets28(frame)
    np.testing.assert_array_equal(saved['sample_id'], frame.id.to_numpy(dtype=str))
    np.testing.assert_array_equal(saved['labels'], names)
    y, p = saved['true_label_vector'], saved['probability_vector']
    np.testing.assert_array_equal(y, expected)
    if 'label_cardinality' in saved:
        np.testing.assert_array_equal(saved['label_cardinality'], expected.sum(axis=1))
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid validation probability matrix')
    return y, p


def validate_history(run, labels, measured):
    history = run['history']
    if [r['epoch'] for r in history] != list(range(1, 21)):
        raise ValueError('Neural training did not complete all 20 epochs')
    key = 'macro_f1' if labels == 27 else 'selection_macro_f1'
    best = max(history, key=lambda row: row[key])
    if run['best_epoch'] != best['epoch'] or not np.isclose(best[key], measured, atol=1e-10, rtol=0):
        raise ValueError('Saved probabilities do not match the best checkpoint criterion')


def notebook_evidence(labels):
    stem = f'{8 if labels == 27 else 9:02d}_kaggle_equal_20ep_{labels}_labels'
    path = ROOT/'notebooks'/f'{stem}_executed.ipynb'
    notebook = json.loads(path.read_text())
    template = json.loads((ROOT/'notebooks'/f'{stem}.ipynb').read_text())
    code = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    template_code = [c['source'] for c in template['cells'] if c['cell_type'] == 'code']
    if [c['source'] for c in code] != template_code:
        raise ValueError(f'{path.name}: executed code differs from the declared notebook')
    if any(c.get('execution_count') is None for c in code):
        raise ValueError('Executed notebook has unexecuted code cells')
    if any(o['output_type'] == 'error' for c in code for o in c.get('outputs', [])):
        raise ValueError('Executed notebook has an error output')
    # Cross-check the printed init manifest with the archive, not a later draft.
    printed = []
    for c in code:
        for o in c.get('outputs', []):
            value = o.get('text', '')
            text = ''.join(value) if isinstance(value, list) else value
            for line in text.splitlines():
                if line.startswith('Manifest: '):
                    printed.append(ast.literal_eval(line[len('Manifest: '):]))
    if len(printed) != 1:
        raise ValueError('Expected one printed study manifest in the executed notebook')
    return dict(path=str(path.relative_to(ROOT)), sha256=digest(path.read_bytes()),
                code_cells=len(code), output_blocks=sum(len(c.get('outputs', [])) for c in code)), printed[0]


def verify_archive(labels, frames, destination):
    path = ROOT/f'vigoemotions_kaggle_equal20_{labels}labels_private_results.zip'
    record, printed = notebook_evidence(labels)
    profile = json.loads((ROOT/'configs/kaggle_equal_20ep.json').read_text())
    rows, per_label, checks = [], [], []
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate archive members')
        if any('test_predictions' in n or n.endswith('test_lock.json') or n.endswith('_test.npz') for n in names):
            raise ValueError('Unexpected test-model evaluation in midterm study')
        read_json = lambda name: json.loads(archive.read(name))
        manifest = read_json('manifest.json')
        if manifest != printed:
            raise ValueError('Notebook and ZIP manifests differ')
        status = read_json('status.json')
        if (manifest['profile'] != profile or manifest['label_spaces'] != [labels]
                or not manifest['validation_only'] or read_json('config27.json') != profile
                or status['status'] != 'completed' or status['completed_runs'] != 3
                or status['expected_runs'] != 3):
            raise ValueError('Archive profile/scope/completion does not match the declared study')
        variants = {'profile': file_variant((ROOT/'configs/kaggle_equal_20ep.json').read_bytes(), manifest['profile_sha256'])}
        for name, checksum in manifest['source_sha256'].items():
            variants[f'scripts/{name}'] = file_variant((ROOT/'scripts'/name).read_bytes(), checksum)
        for split, frame in frames.items():
            variants[f'data/{split}.csv'] = file_variant((ROOT/'data'/f'{split}.csv').read_bytes(), manifest['environment']['data_sha256'][split])
            if digest('\n'.join(frame.id).encode()) != manifest['split_ids_sha256'][split]:
                raise ValueError('Fixed split ID/order mismatch')
        comparison = pd.read_csv(StringIO(archive.read('comparison.csv').decode()))
        if len(comparison) != 3 or set(comparison.model) != set(MODELS):
            raise ValueError('Expected one comparison row per model')
        for model in MODELS:
            folder = f'labels27/{model}' if labels == 27 else f'labels28/models/{model}'
            metadata = f'{folder}/{model}_run.json' if labels == 27 else f'{folder}/run.json'
            prediction = f'{folder}/predictions/{model}_val.npz' if labels == 27 else f'{folder}/val_predictions.npz'
            run = read_json(metadata)
            if run['config'] != (profile if labels == 27 else neutral.study_config(profile)):
                raise ValueError('Model run config differs from frozen profile')
            if labels == 28 and run['labels'] != neutral.LABELS28:
                raise ValueError('Model label order mismatch')
            with np.load(BytesIO(archive.read(prediction)), allow_pickle=False) as saved:
                y, p = validate_predictions(saved, frames['val'], labels)
                measured, common = scores(y, p), scores(y[:, :27], p[:, :27])
            if model != 'tfidf_lr':
                validate_history(run, labels, common['macro_f1'])
            checkpoint = (f'{folder}/model.joblib' if model == 'tfidf_lr' else
                          f'{folder}/pretrained/model.safetensors' if labels == 28 and model == 'phobert' else f'{folder}/best.pt')
            if archive.getinfo(checkpoint).file_size == 0:
                raise ValueError('Empty checkpoint')
            recorded = comparison.loc[comparison.model == model].iloc[0]
            values = {**measured, **{f'emotion27_{k}': v for k, v in common.items()}}
            for key, value in values.items():
                if not np.isclose(recorded[key], value, atol=1e-10, rtol=0):
                    raise ValueError(f'{labels}/{model}: archived {key} differs from NPZ')
            row = dict(model=model, label_count=labels, split='val', threshold=.5,
                       epochs_trained=20 if model != 'tfidf_lr' else None,
                       best_epoch=run.get('best_epoch'), **values,
                       parameter_count=run['parameter_count'], train_seconds=run['train_seconds'], device=run['device'])
            for key in ('label_count', 'threshold', 'parameter_count', 'train_seconds'):
                if not np.isclose(recorded[key], row[key], atol=1e-10, rtol=0):
                    raise ValueError(f'Archived metadata mismatch: {key}')
            if model != 'tfidf_lr' and (recorded.best_epoch != row['best_epoch'] or recorded.epochs_trained != 20):
                raise ValueError('Archived epoch metadata mismatch')
            rows.append(row)
            precision, recall, f1, support = precision_recall_fscore_support(y, p >= .5, zero_division=0)
            for i, name in enumerate(LABELS if labels == 27 else neutral.LABELS28):
                per_label.append(dict(model=model, label_count=labels, label=name, support=int(support[i]),
                                      precision=precision[i], recall=recall[i], f1=f1[i]))
            # Export an allowlist of ID-only evidence; never export weights/vocab/raw comments.
            target = destination/str(labels)/model
            target.mkdir(parents=True, exist_ok=True)
            for member, filename in ((metadata, 'run.json'), (prediction, 'val_predictions.npz')):
                (target/filename).write_bytes(archive.read(member))
                checks.append(dict(member=member, sha256=digest(archive.read(member))))
            save_evaluation(model, frames, {'val': p[:, :27]},
                            {k: row[k] for k in ('parameter_count', 'train_seconds')},
                            output_dir=destination.parent/'analysis'/f'{labels}_emotion27'/model)
        target = destination/str(labels)
        for member in ('manifest.json', 'config27.json', 'status.json', 'comparison.csv'):
            (target/member).write_bytes(archive.read(member))
        if labels == 28:
            for member in ('labels28/config.json', 'labels28/study.json', 'labels28/annotation_counts.csv'):
                (target/Path(member).name).write_bytes(archive.read(member))
    provenance = dict(zip=path.name, zip_sha256=digest(path.read_bytes()), notebook=record,
                      verified_file_variants=variants, exported_members=checks,
                      checkpoint_members_verified=True, raw_text_exported=False,
                      test_model_scoring=False, environment=manifest['environment'])
    return rows, per_label, provenance


def main():
    frames, _ = load_data()
    evidence = OUT/'evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    rows, per_label, provenance = [], [], {}
    for labels in (27, 28):
        r, p, source = verify_archive(labels, frames, evidence)
        rows.extend(r); per_label.extend(p); provenance[str(labels)] = source
    combined = pd.DataFrame(rows)
    combined.to_csv(OUT/'model_comparison.csv', index=False)
    pd.DataFrame(per_label).to_csv(OUT/'per_label_metrics.csv', index=False)
    projected = combined[['model', 'label_count', *[f'emotion27_{k}' for k in METRICS]]].copy()
    projected.to_csv(OUT/'emotion27_comparison.csv', index=False)
    write_json(OUT/'provenance.json', dict(studies=provenance,
        verification='Recomputed all six validation metrics from archived NPZ; verified histories, checkpoints, IDs, labels, config and source/data checksums',
        byte_differences='28-label source/config/CSV hashes match exact CRLF variants; parsed profile and contents match 27-label inputs',
        historical_results_used=False, trained_locally=False))
    print(combined.to_string(index=False))


if __name__ == '__main__':
    main()
