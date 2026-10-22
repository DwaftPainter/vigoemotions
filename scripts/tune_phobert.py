"""Controlled PhoBERT learning-rate study; selection uses validation at 0.5 only."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/vigo-matplotlib')

import numpy as np
import pandas as pd

from common import ROOT, LABELS, scores, write_json

RATES = {'lr_1e-5': 1e-5, 'lr_2e-5': 2e-5, 'lr_3e-5': 3e-5}


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


def summarize(output):
    plan = load_plan(output)
    runs, rows, identities = {}, [], []
    for name, rate in RATES.items():
        folder = output/name
        run = json.loads((folder/'phobert_run.json').read_text())
        expected = json.loads((output/'configs'/f'{name}.json').read_text())
        if run['config'] != expected or len(run['history']) != expected['phobert']['epochs']:
            raise ValueError(f'{name}: configuration or epoch count mismatch')
        best = max(run['history'], key=lambda h: (h['macro_f1'], -h['epoch']))
        if run['best_epoch'] != best['epoch']:
            raise ValueError(f'{name}: invalid checkpoint selection')
        with np.load(folder/'predictions/phobert_val.npz', allow_pickle=False) as saved:
            ids, y, p, labels = (saved[k].copy() for k in
                ('sample_id', 'true_label_vector', 'probability_vector', 'labels'))
        if (p.shape != y.shape or y.shape != (len(ids), 27) or len(set(ids)) != len(ids)
                or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all()
                or (p < 0).any() or (p > 1).any() or labels.tolist() != LABELS):
            raise ValueError(f'{name}: invalid validation archive')
        if identities:
            for actual, reference in zip((ids, y, labels), identities[0]):
                np.testing.assert_array_equal(actual, reference)
            previous = next(iter(runs.values()))
            if (run['environment'] != previous['environment'] or run['device'] != previous['device']):
                raise ValueError('Use the same dataset, package versions and hardware for all three runs')
        identities.append((ids, y, labels))
        measured = scores(y, p)
        recorded = pd.read_csv(folder/'tables/phobert_metrics.csv')
        if len(recorded) != 1 or recorded.iloc[0]['split'] != 'val' or recorded.iloc[0]['threshold'] != .5:
            raise ValueError(f'{name}: expected validation metrics at 0.5 only')
        for metric, value in measured.items():
            if not (np.isclose(value, recorded.iloc[0][metric], atol=1e-10, rtol=0)
                    and np.isclose(value, best[metric], atol=1e-10, rtol=0)):
                raise ValueError(f'{name}: {metric} does not match saved probabilities/checkpoint')
        runs[name] = run
        rows.append(dict(run=name, learning_rate=rate, best_epoch=run['best_epoch'],
                         **measured, train_seconds=run['train_seconds']))
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
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == 'prepare':
        prepare(output, json.loads((ROOT/'configs/midterm.json').read_text()))
    elif args.action == 'summarize':
        comparison, winner = summarize(output)
        print(comparison.to_string(index=False)); print('Selected:', winner.run)
    else:
        import torch
        if not torch.cuda.is_available():
            parser.error('Run this study in a Colab GPU runtime')
        if not output.exists():
            prepare(output, json.loads((ROOT/'configs/midterm.json').read_text()))
        load_plan(output)
        if any((output/name).exists() for name in RATES):
            parser.error('Run output already exists. Summarize completed runs or use a new --output path.')
        checkpoint_dir = ROOT/'data/checkpoints'/output.name
        if checkpoint_dir.exists():
            parser.error('Private checkpoint directory exists; use a new --output name')
        for name in RATES:
            subprocess.run([sys.executable, str(ROOT/'scripts/train_neural.py'), '--model', 'phobert',
                '--config', str(output/'configs'/f'{name}.json'), '--output-dir', str(output/name),
                '--checkpoint', str(checkpoint_dir/name/'best.pt'),
                '--vncorenlp-dir', str(args.vncorenlp_dir.resolve())], check=True, cwd=ROOT)
        comparison, winner = summarize(output)
        print(comparison.to_string(index=False)); print('Selected:', winner.run)


if __name__ == '__main__':
    main()
