"""Canonical Kaggle study: 27/28 labels, 20 epochs for both neural models."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd

from common import ROOT, LABELS, scores, targets, write_json
import neutral_28_labels as neutral

PROFILE = ROOT/'configs/kaggle_equal_20ep.json'
DEFAULT_OUTPUT = ROOT/'reports/neutral_28_labels/kaggle_equal_20ep'
MODELS = ('tfidf_lr', 'gru', 'phobert')


def load_profile(path):
    config = neutral.read_json(path)
    if config['label_count'] != 27 or config['threshold'] != .5:
        raise ValueError('Expected the canonical 27-label base profile at threshold 0.5')
    if any(config[name]['epochs'] != 20 for name in ('gru', 'phobert')):
        raise ValueError('Both neural models must run exactly 20 epochs; use a different study for another budget')
    if config.get('checkpoint_selection_metric') != 'emotion27_macro_f1':
        raise ValueError('Both label spaces must select checkpoints using the same 27 emotions')
    return config


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_identity(data_dir):
    identity = neutral.runtime(data_dir)
    identity['gpu'] = None
    if importlib.util.find_spec('torch') is not None:
        import torch
        if torch.cuda.is_available():
            identity['gpu'] = torch.cuda.get_device_name(0)
    return identity


def study_path(path):
    path = Path(path).resolve()
    if path != DEFAULT_OUTPUT.resolve() and DEFAULT_OUTPUT.resolve() not in path.parents:
        raise ValueError(f'Choose an isolated output inside {DEFAULT_OUTPUT}')
    return path


def label_spaces(out):
    path = Path(out)/'manifest.json'
    spaces = neutral.read_json(path).get('label_spaces', [27, 28]) if path.exists() else [27, 28]
    if spaces not in ([27], [28], [27, 28]):
        raise ValueError('Invalid label scope in study manifest')
    return spaces


def init(out, data_dir, profile=PROFILE, labels=None):
    out = study_path(out)
    if Path(data_dir).resolve() != (ROOT/'data').resolve():
        raise ValueError('Place the paired source CSVs in this Kaggle project data/ directory')
    config = load_profile(profile)
    frames, _, _ = neutral.load_frames(data_dir)
    manifest = dict(profile=config, profile_sha256=digest(profile),
        environment=runtime_identity(data_dir),
        source_sha256={name: digest(ROOT/'scripts'/name) for name in
            ('kaggle_equal_epochs.py', 'common.py', 'train_tfidf.py', 'train_neural.py', 'neutral_28_labels.py', 'evaluate.py')},
        split_ids_sha256={s: hashlib.sha256('\n'.join(f.id).encode()).hexdigest() for s, f in frames.items()},
        validation_only=True, checkpoint_selection_metric='emotion27_macro_f1',
        label_spaces=[labels] if labels is not None else [27, 28])
    if out.exists():
        if not (out/'manifest.json').is_file() or neutral.read_json(out/'manifest.json') != manifest:
            raise FileExistsError('Existing study differs; preserve it and choose a new named output')
    else:
        out.mkdir(parents=True)
        write_json(out/'manifest.json', manifest)
        write_json(out/'config27.json', config)
    if 28 in manifest['label_spaces']:
        neutral.init_study(out/'labels28', data_dir, out/'config27.json', out/'labels27')
    report(out, data_dir)
    return out


def checked(out, data_dir, lock_runtime=False):
    out = study_path(out)
    if Path(data_dir).resolve() != (ROOT/'data').resolve():
        raise ValueError('Use the canonical Kaggle project data/ directory for all runs')
    manifest = neutral.read_json(out/'manifest.json')
    if load_profile(out/'config27.json') != manifest['profile']:
        raise ValueError('Frozen profile changed')
    if neutral.runtime(data_dir)['data_sha256'] != manifest['environment']['data_sha256']:
        raise ValueError('Source CSV hashes changed')
    if lock_runtime and runtime_identity(data_dir) != manifest['environment']:
        raise ValueError('Runtime/packages/data changed; preserve this study and start a separate one')
    for name, expected in manifest['source_sha256'].items():
        if digest(ROOT/'scripts'/name) != expected:
            raise ValueError('Training source changed: '+name)
    frames, _, _ = neutral.load_frames(data_dir)
    for split, frame in frames.items():
        if hashlib.sha256('\n'.join(frame.id).encode()).hexdigest() != manifest['split_ids_sha256'][split]:
            raise ValueError('Split identity changed')
    return out, manifest['profile'], frames


def _locations(out, labels, model):
    if labels == 27:
        folder = out/'labels27'/model
        return folder, folder/f'{model}_run.json', folder/'predictions'/f'{model}_val.npz'
    folder = out/'labels28/models'/model
    return folder, folder/'run.json', folder/'val_predictions.npz'


def result_rows(out, frames):
    rows = []
    for labels in label_spaces(out):
        for model in MODELS:
            folder, metadata, prediction = _locations(out, labels, model)
            if not metadata.exists():
                continue
            run = neutral.read_json(metadata)
            expected_config = neutral.read_json(out/'config27.json')
            if labels == 28:
                expected_config = neutral.study_config(expected_config)
            if run['config'] != expected_config:
                raise ValueError('A recorded run does not match the frozen Kaggle profile')
            with np.load(prediction, allow_pickle=False) as saved:
                y, p = saved['true_label_vector'], saved['probability_vector']
                expected = targets(frames['val']) if labels == 27 else neutral.targets28(frames['val'])
                np.testing.assert_array_equal(saved['sample_id'], frames['val'].id.to_numpy(dtype=str))
                np.testing.assert_array_equal(saved['labels'], LABELS if labels == 27 else neutral.LABELS28)
                np.testing.assert_array_equal(y, expected)
                if p.shape != expected.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                    raise ValueError('Invalid saved probabilities')
                common = scores(y[:, :27], p[:, :27])
                measured = scores(y, p)
            if run['config']['gru']['epochs'] != 20 or run['config']['phobert']['epochs'] != 20:
                raise ValueError('A recorded run has an incompatible epoch budget')
            epochs = None
            if model != 'tfidf_lr':
                history = run['history']
                if [h['epoch'] for h in history] != list(range(1, 21)):
                    raise ValueError('Incomplete neural history; never label a partial run complete')
                key = 'macro_f1' if labels == 27 else 'selection_macro_f1'
                best = max(history, key=lambda h: h[key])
                if best['epoch'] != run['best_epoch'] or not np.isclose(best[key], common['macro_f1'], atol=1e-10, rtol=0):
                    raise ValueError('Checkpoint selection does not match the saved validation probabilities')
                checkpoint = folder/'best.pt' if labels == 27 else (folder/'best.pt' if model == 'gru' else folder/'pretrained/model.safetensors')
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                epochs = 20
            elif not (folder/'model.joblib').is_file():
                raise FileNotFoundError(folder/'model.joblib')
            rows.append(dict(model=model, label_count=labels, split='val', threshold=.5,
                epochs_trained=epochs, best_epoch=run.get('best_epoch'),
                **measured, **{f'emotion27_{k}': v for k, v in common.items()},
                parameter_count=run['parameter_count'], train_seconds=run['train_seconds'],
                device=run['device'], source=str(metadata.relative_to(out))))
    return pd.DataFrame(rows)


def report(out, data_dir):
    out, _, frames = checked(out, data_dir)
    rows = result_rows(out, frames)
    spaces = label_spaces(out)
    expected = len(spaces) * len(MODELS)
    status = dict(status='completed' if len(rows) == expected else 'prepared_or_partial',
        label_spaces=spaces, completed_runs=len(rows), expected_runs=expected, neural_epochs=20,
        tfidf_budget='Same OvR Logistic Regression solver/max_iter in both label spaces; epochs not applicable',
        checkpoint_selection='Highest validation emotion27 Macro-F1 at 0.5; ties keep earliest epoch',
        source_of_truth='This Kaggle manifest, config snapshots, run metadata and verified probabilities; no legacy scores imported',
        limitation='Equal epochs do not imply equal compute; GRU/PhoBERT use their documented batch sizes and architectures')
    write_json(out/'status.json', status)
    if not rows.empty:
        rows.to_csv(out/'comparison.csv', index=False)
        print(rows.to_string(index=False))
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return rows


def _launch(arguments, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as stream:
        with subprocess.Popen(arguments, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace') as process:
            for line in process.stdout:
                print(line, end='', flush=True); stream.write(line); stream.flush()
            result = process.wait()
    if result:
        raise subprocess.CalledProcessError(result, arguments)


def train(out, data_dir, labels, model):
    out, config, frames = checked(out, data_dir, lock_runtime=True)
    if labels not in label_spaces(out):
        raise ValueError('This study is scoped to a different label space; use its matching notebook')
    existing = result_rows(out, frames)
    if not existing.empty and ((existing.label_count == labels) & (existing.model == model)).any():
        print('SKIP verified completed run:', labels, model)
        return
    folder, metadata, _ = _locations(out, labels, model)
    if folder.exists():
        raise FileExistsError(f'Incomplete run preserved: {folder}; use a new study to retry')
    if model != 'tfidf_lr':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Enable a Kaggle GPU before neural training')
    if model == 'phobert':
        neutral.ensure_segmenter(out/'resources/vncorenlp')
    if labels == 27:
        if model == 'tfidf_lr':
            from train_tfidf import main as fit
            fit(config=config, output_dir=folder)
        else:
            _launch([sys.executable, '-u', str(ROOT/'scripts/train_neural.py'), '--model', model,
                '--config', str(out/'config27.json'), '--output-dir', str(folder),
                '--checkpoint', str(folder/'best.pt'), '--vncorenlp-dir', str(out/'resources/vncorenlp')],
                out/'logs'/f'27_{model}.log')
            if model == 'gru':
                shutil.copy2(ROOT/'data/derived/gru_vocab.json', folder/'vocab.json')
        # Publish only this run's ID-only evaluation files into the paired 27-label reference directory.
        reference = out/'labels27'
        for category in ('tables', 'predictions'):
            (reference/category).mkdir(exist_ok=True)
            for path in (folder/category).glob('*'):
                shutil.copy2(path, reference/category/path.name)
        shutil.copy2(metadata, reference/metadata.name)
    else:
        _launch([sys.executable, '-u', str(ROOT/'scripts/neutral_28_labels.py'), 'train',
                 '--model', model, '--output', str(out/'labels28'), '--data-dir', str(data_dir),
                 '--vncorenlp-dir', str(out/'resources/vncorenlp')], out/'logs'/f'28_{model}.log')
    report(out, data_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'train', 'report', 'archive'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--labels', type=int, choices=[27, 28])
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args()
    if args.action == 'init':
        init(args.output, args.data_dir, labels=args.labels)
    elif args.action == 'train':
        if args.labels is None or args.model is None:
            parser.error('train requires --labels and --model')
        train(args.output, args.data_dir, args.labels, args.model)
    else:
        if args.labels is not None and label_spaces(args.output) != [args.labels]:
            raise ValueError('The output study does not match this notebook label scope')
        rows = report(args.output, args.data_dir)
        if args.action == 'archive':
            spaces = label_spaces(args.output)
            if len(rows) != len(spaces) * len(MODELS):
                raise ValueError('Complete and verify all runs for this label scope before archiving')
            suffix = f'_{spaces[0]}labels' if len(spaces) == 1 else ''
            target = ROOT/f'vigoemotions_kaggle_equal20{suffix}_private_results.zip'
            with ZipFile(target, 'w', ZIP_DEFLATED) as archive:
                for path in args.output.rglob('*'):
                    if path.is_file(): archive.write(path, path.relative_to(args.output))
            print('Private results:', target)


if __name__ == '__main__':
    main()
