"""Controlled PhoBERT learning-rate study; selection uses validation at 0.5 only."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import BadZipFile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/vigo-matplotlib')

import numpy as np
import pandas as pd

from common import ROOT, LABELS, scores, write_json, environment

RATES = {'lr_1e-5': 1e-5, 'lr_2e-5': 2e-5, 'lr_3e-5': 3e-5}
ARTIFACT_ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, AssertionError, EOFError, BadZipFile)


def prepare(output, base):
    output.mkdir(parents=True, exist_ok=False)
    for name, rate in RATES.items():
        config = deepcopy(base)
        config['phobert']['learning_rate'] = rate
        write_json(output/'configs'/f'{name}.json', config)
    write_json(output/'study.json', {
        'status': 'prepared', 'rates': RATES, 'base_config': base,
        'selection': 'Maximum validation Macro-F1 at 0.5; ties choose lower learning rate',
        'control': 'lr_2e-5', 'test_evaluated': False,
    })


def load_plan(output):
    plan = json.loads((output/'study.json').read_text())
    if plan['rates'] != RATES or plan['base_config']['threshold'] != 0.5:
        raise ValueError('Study must use the fixed learning-rate grid and threshold 0.5')
    for name, rate in RATES.items():
        expected = deepcopy(plan['base_config'])
        expected['phobert']['learning_rate'] = rate
        if json.loads((output/'configs'/f'{name}.json').read_text()) != expected:
            raise ValueError(f'{name}: only learning_rate may differ from the base config')
    return plan


def validate_run(output, name):
    """Read and verify completion artifacts without changing outputs or checkpoints."""
    folder = output/name
    run = json.loads((folder/'phobert_run.json').read_text())
    expected = json.loads((output/'configs'/f'{name}.json').read_text())
    metadata = run.get('environment', {})
    if (not isinstance(metadata, dict) or not isinstance(metadata.get('packages'), dict)
            or not metadata['packages'] or not isinstance(metadata.get('data_sha256'), dict)
            or any(key not in metadata['data_sha256'] for key in ('train', 'val', 'test'))
            or any(not isinstance(metadata.get(key), str) for key in ('python', 'platform', 'processor'))
            or not isinstance(run.get('device'), str) or not run['device']):
        raise ValueError(f'{name}: missing environment, dataset identity or hardware metadata')
    history = run['history']
    if run['config'] != expected or len(history) != expected['phobert']['epochs']:
        raise ValueError(f'{name}: configuration or epoch count mismatch')
    if ([h['epoch'] for h in history] != list(range(1, expected['phobert']['epochs']+1))
            or any(type(h['epoch']) is not int for h in history)):
        raise ValueError(f'{name}: epoch history must contain every epoch exactly once, in order')
    for h in history:
        for key in ('loss', 'macro_f1', 'micro_f1', 'hamming_loss', 'lrap'):
            value = h[key]
            if (type(value) not in (int, float) or not np.isfinite(value) or value < 0
                    or (key != 'loss' and value > 1)):
                raise ValueError(f'{name}: invalid epoch {h["epoch"]} {key}')
    best = max(history, key=lambda h: (h['macro_f1'], -h['epoch']))
    if type(run['best_epoch']) is not int or run['best_epoch'] != best['epoch']:
        raise ValueError(f'{name}: invalid checkpoint selection')
    with np.load(folder/'predictions/phobert_val.npz', allow_pickle=False) as saved:
        ids, y, p, labels = (saved[k].copy() for k in
            ('sample_id', 'true_label_vector', 'probability_vector', 'labels'))
        if 'label_cardinality' in saved:
            np.testing.assert_array_equal(saved['label_cardinality'], y.sum(axis=1))
    if (ids.ndim != 1 or len(ids) == 0 or p.shape != y.shape or y.shape != (len(ids), 27)
            or len(set(ids.tolist())) != len(ids) or not np.isin(y, [0, 1]).all()
            or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or labels.tolist() != LABELS):
        raise ValueError(f'{name}: invalid validation archive')
    measured = scores(y, p)
    recorded = pd.read_csv(folder/'tables/phobert_metrics.csv')
    if (len(recorded) != 1 or recorded.iloc[0]['model'] != 'phobert'
            or recorded.iloc[0]['split'] != 'val' or recorded.iloc[0]['threshold'] != .5):
        raise ValueError(f'{name}: expected validation metrics at 0.5 only')
    for metric, value in measured.items():
        if not (np.isclose(value, recorded.iloc[0][metric], atol=1e-10, rtol=0)
                and np.isclose(value, best[metric], atol=1e-10, rtol=0)):
            raise ValueError(f'{name}: {metric} does not match saved probabilities/checkpoint')
    for key in ('parameter_count', 'train_seconds'):
        if (not np.isfinite(run[key]) or run[key] < 0
                or not np.isclose(run[key], recorded.iloc[0][key], atol=1e-10, rtol=0)):
            raise ValueError(f'{name}: invalid or inconsistent {key}')
    row = dict(run=name, learning_rate=RATES[name], best_epoch=run['best_epoch'],
               **measured, train_seconds=run['train_seconds'])
    return run, (ids, y, labels), row


def require_compatible(saved, current, context):
    differences = []

    def compare(left, right, path):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                compare(left.get(key, '<missing>'), right.get(key, '<missing>'), f'{path}.{key}' if path else key)
        elif left != right:
            differences.append(f'{path}: saved={left!r}, current={right!r}')

    compare(saved, current, '')
    if differences:
        raise ValueError(f'{context}: use the same dataset, package versions and hardware; '
                         f'incompatible runtime: {"; ".join(differences)}. Existing artifacts were preserved.')


def run_environment(run):
    return {'environment': run['environment'], 'device': run['device']}


def gpu_device():
    import torch
    if not torch.cuda.is_available():
        raise ValueError('Run this study in a Colab GPU runtime')
    return torch.cuda.get_device_name(0)


def validate_runs(output, names):
    runs, rows, identities = {}, [], []
    for name in names:
        run, identity, row = validate_run(output, name)
        if identities:
            for actual, reference in zip(identity, identities[0]):
                np.testing.assert_array_equal(actual, reference,
                    err_msg=f'{name}: validation IDs/targets/labels differ from {next(iter(runs))}')
            require_compatible(run_environment(next(iter(runs.values()))), run_environment(run), name)
        identities.append(identity)
        runs[name] = run
        rows.append(row)
    return runs, rows


def resume_pending(output, checkpoint_dir, runtime):
    """Preflight every run before launching any missing run; partial recovery is not supported."""
    if checkpoint_dir.exists() and not checkpoint_dir.is_dir():
        raise ValueError(f'Incomplete checkpoint layout: {checkpoint_dir} is not a directory; nothing was overwritten')
    pending, completed = [], []
    for name in RATES:
        folder, checkpoint = output/name, checkpoint_dir/name
        if not folder.exists() and not folder.is_symlink():
            if checkpoint.exists() or checkpoint.is_symlink():
                raise ValueError(f'{name}: incomplete run: checkpoint output exists without results. '
                                 'Restore the complete artifacts or use a separate study; nothing was overwritten.')
            pending.append(name)
            continue
        try:
            validate_run(output, name)
        except ARTIFACT_ERRORS as error:
            raise ValueError(f'{name}: incomplete or invalid run: {error}. '
                             'Restore the complete artifacts or use a separate study. '
                             'Partial-run recovery is not supported; nothing was overwritten.') from error
        completed.append(name)
    runs, _ = validate_runs(output, completed)
    for name, run in runs.items():
        require_compatible(run_environment(run), runtime, f'{name} versus current Colab runtime')
        print(f'SKIP {name}: completed artifacts validated; preserving outputs and checkpoints.', flush=True)
    return pending


def summarize(output):
    plan = load_plan(output)
    runs, rows = validate_runs(output, RATES)
    comparison = pd.DataFrame(rows)
    control = comparison.set_index('run').loc['lr_2e-5']
    comparison['delta_macro_f1_vs_control'] = comparison.macro_f1 - control.macro_f1
    winner = comparison.sort_values(['macro_f1', 'learning_rate'], ascending=[False, True]).iloc[0]
    comparison.to_csv(output/'comparison.csv', index=False)
    for suffix in ('per_label', 'styles'):
        key = 'label' if suffix == 'per_label' else 'style'
        metric = 'f1' if suffix == 'per_label' else 'macro_f1'
        reference = pd.read_csv(output/'lr_2e-5'/f'tables/phobert_{suffix}.csv').set_index(key)
        combined = []
        for name in RATES:
            frame = pd.read_csv(output/name/f'tables/phobert_{suffix}.csv')
            frame['run'] = name
            frame[f'delta_{metric}_vs_control'] = frame[metric] - frame[key].map(reference[metric])
            combined.append(frame)
        pd.concat(combined, ignore_index=True).to_csv(output/f'{suffix}_comparison.csv', index=False)
    # Empty review columns distinguish automatically selected errors from a human audit.
    audit = pd.read_csv(output/winner.run/'tables/phobert_error_cases_val.csv')
    audit['reviewer'] = ''; audit['observation'] = ''; audit['hypothesis'] = ''
    audit_path = output/'neural_error_review.csv'
    if not audit_path.exists():
        audit.to_csv(audit_path, index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, run in runs.items():
        history = pd.DataFrame(run['history'])
        axes[0].plot(history.epoch, history.loss, 'o-', label=name)
        axes[1].plot(history.epoch, history.macro_f1, 'o-', label=name)
    for axis, title in zip(axes, ('Train BCE loss', 'Validation Macro-F1 at 0.5')):
        axis.set(title=title, xlabel='Epoch'); axis.legend()
    fig.tight_layout(); fig.savefig(output/'learning_curves.png', dpi=150); plt.close(fig)
    # Hash the probability archives so the selection can be traced to concrete outputs.
    hashes = {name: hashlib.sha256((output/name/'predictions/phobert_val.npz').read_bytes()).hexdigest()
              for name in RATES}
    plan.update(status='completed', selected_run=winner.run, prediction_sha256=hashes)
    write_json(output/'study.json', plan)
    lines = ['# PhoBERT learning-rate study', '',
             'Completed: three controlled runs, validation at threshold 0.5; test not evaluated.', '',
             comparison.to_string(index=False), '',
             f'Selected run: {winner.run}; Macro-F1 {winner.macro_f1:.4f}; '
             f'delta vs same-runtime 2e-5 control {winner.delta_macro_f1_vs_control:+.4f}.', '',
             'The original baseline remains in reports/tables/model_comparison.csv. '
             'One seed and four epochs per run do not establish statistical significance or convergence. '
             'Read learning curves and per-label/style deltas before explaining causes. '
             'neural_error_review.csv is an unreviewed candidate list; fill review columns manually.']
    (output/'experiment_summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return comparison, winner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'summarize'])
    parser.add_argument('--output', type=Path, default=ROOT/'reports/phobert_tuning')
    parser.add_argument('--vncorenlp-dir', type=Path, default=ROOT/'data/vncorenlp')
    parser.add_argument('--resume', action='store_true', help='Validate existing runs and train only missing rates (run only)')
    args = parser.parse_args()
    if args.resume and args.action != 'run':
        parser.error('--resume is only supported with the run command')
    output = args.output.resolve()
    if args.action == 'prepare':
        prepare(output, json.loads((ROOT/'configs/midterm.json').read_text()))
    elif args.action == 'summarize':
        comparison, winner = summarize(output)
        print(comparison.to_string(index=False)); print('Selected:', winner.run)
    else:
        try:
            device = gpu_device()
        except ValueError as error:
            parser.error(str(error))
        if not output.exists():
            if args.resume:
                parser.error('--resume requires an existing prepared study; no artifacts were changed')
            prepare(output, json.loads((ROOT/'configs/midterm.json').read_text()))
        if args.resume:
            try:
                load_plan(output)
            except ARTIFACT_ERRORS as error:
                parser.error(f'Invalid study plan: {error}; no artifacts were changed')
        else:
            load_plan(output)
        checkpoint_dir = ROOT/'data/checkpoints'/output.name
        pending = list(RATES)
        if args.resume:
            try:
                runtime = {'environment': environment(), 'device': device}
                pending = resume_pending(output, checkpoint_dir, runtime)
            except ARTIFACT_ERRORS as error:
                parser.error(str(error))
        else:
            if any((output/name).exists() for name in RATES):
                parser.error('Run output already exists. Summarize completed runs or use a new --output path.')
            if checkpoint_dir.exists():
                parser.error('Private checkpoint directory exists; use a new --output name')
        for name in pending:
            print(f'RUN {name}', flush=True)
            subprocess.run([sys.executable, str(ROOT/'scripts/train_neural.py'), '--model', 'phobert',
                '--config', str(output/'configs'/f'{name}.json'), '--output-dir', str(output/name),
                '--checkpoint', str(checkpoint_dir/name/'best.pt'),
                '--vncorenlp-dir', str(args.vncorenlp_dir.resolve())], check=True, cwd=ROOT)
            if args.resume:
                try:
                    run, _, _ = validate_run(output, name)
                    require_compatible(runtime, run_environment(run), f'{name} after training')
                except ARTIFACT_ERRORS as error:
                    parser.error(str(error))
        if args.resume:
            try:
                validate_runs(output, RATES)
            except ARTIFACT_ERRORS as error:
                parser.error(str(error))
        comparison, winner = summarize(output)
        print(comparison.to_string(index=False)); print('Selected:', winner.run)


if __name__ == '__main__':
    main()
