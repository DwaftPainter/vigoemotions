"""Render post-test figures from imported, hash-verified artifacts; no inference."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neutral_28_labels import LABELS28

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ('fixed', 'global', 'per_label')
COLORS = ('#607d8b', '#167c80', '#c27628')
TABLES = ('comparison', 'per_label', 'frequency_tiers', 'cardinality', 'neutral')
MEMBERS = {
    'validation_reproduction.json', 'thresholds.json', 'pre_test_lock.json', 'test_attempt.json',
    *[f'predictions/{name}.npz' for name in ('val', 'val_reproduced', 'test', 'test_probabilities')],
    *[f'tables/{split}_{name}.csv' for split in ('val', 'test') for name in TABLES],
    *[f'tables/{name}.csv' for name in ('threshold_profile', 'global_sweep', 'generalization_gap')],
}


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_inputs(output):
    output = Path(output)
    manifest = read_json(output / 'execution_verification.json')
    if manifest['protocol_id'] != 'vigoemotions_final_rq1_28_v2' or set(manifest['imported_members_sha256']) != MEMBERS:
        raise ValueError('Expected the complete imported 28-label Final evidence')
    for relative, expected in manifest['imported_members_sha256'].items():
        if checksum(output / relative) != expected:
            raise ValueError(f'Imported artifact hash mismatch: {relative}')
    attempt = read_json(output / 'test_attempt.json')
    reproduction = read_json(output / 'validation_reproduction.json')
    if attempt['status'] != 'completed' or reproduction['status'] != 'verified':
        raise ValueError('Validation must be verified and test completed before reporting')
    if attempt['runtime'] != reproduction['runtime']:
        raise ValueError('Validation/test runtime differs')
    lock = read_json(output / 'pre_test_lock.json')
    bindings = [
        (attempt['lock_sha256'], 'pre_test_lock.json'),
        (attempt['raw_probabilities_sha256'], 'predictions/test_probabilities.npz'),
        (attempt['predictions_sha256'], 'predictions/test.npz'),
        (lock['thresholds_sha256'], 'thresholds.json'),
        (lock['reproduction_sha256'], 'validation_reproduction.json'),
        (lock['validation_predictions_sha256'], 'predictions/val.npz'),
    ]
    if any(expected != checksum(output / relative) for expected, relative in bindings):
        raise ValueError('Completed test/lock artifact binding mismatch')


def build(output=ROOT / 'reports/final28'):
    output = Path(output)
    verify_inputs(output)
    figures, analysis = output / 'figures', output / 'analysis'
    figures.mkdir(exist_ok=True)
    analysis.mkdir(exist_ok=True)
    generated = []

    def save(fig, name, bottom=0):
        fig.tight_layout(rect=(0, bottom, 1, 1))
        for extension in ('png', 'pdf'):
            path = figures / f'{name}.{extension}'
            fig.savefig(path, dpi=180)
            generated.append(path)
        plt.close(fig)

    def table(name):
        return pd.read_csv(output / 'tables' / f'{name}.csv')

    profile = table('threshold_profile')
    if profile.label.tolist() != LABELS28:
        raise ValueError('Threshold profile is not in canonical 28-label order')
    global_threshold = read_json(output / 'thresholds.json')['global_threshold']
    names = ('Fixed 0.5', f'Global {global_threshold:g}', 'Per-label')
    detail = profile.copy()
    for split in ('val', 'test'):
        per_label = table(f'{split}_per_label')
        values = per_label.pivot(index='label', columns='strategy', values='f1').loc[LABELS28]
        support = per_label[per_label.strategy == 'fixed'].set_index('label').loc[LABELS28, 'support']
        detail[f'{split}_support'] = support.to_numpy()
        for strategy in STRATEGIES:
            detail[f'{split}_{strategy}_f1'] = values[strategy].to_numpy()
        for strategy in STRATEGIES[1:]:
            detail[f'{split}_{strategy}_delta_pp'] = 100 * (values[strategy] - values.fixed).to_numpy()
    detail_path = analysis / 'label_deltas.csv'
    detail.to_csv(detail_path, index=False)
    generated.append(detail_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(3)
    for ax, metric, title in zip(axes, ('macro_f1', 'micro_f1'), ('Macro-F1', 'Micro-F1')):
        for shift, split, color in ((-.19, 'val', '#9babb5'), (.19, 'test', '#167c80')):
            frame = table(f'{split}_comparison').query("scope == 'all28'").set_index('strategy').loc[list(STRATEGIES)]
            bars = ax.bar(x + shift, 100 * frame[metric], .36, label=split, color=color)
            ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=3)
        ax.set(xticks=x, xticklabels=names, ylabel=f'{title} (%)', ylim=(0, 68), title=f'{title}; all 28 labels')
        ax.legend()
    save(fig, 'validation_test')

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(profile.label, profile.threshold, color=[COLORS[2] if s == 'neutral' else COLORS[1] for s in profile.tier])
    ax.axvline(.5, color=COLORS[0], linestyle='--', label='Fixed 0.5')
    ax.axvline(global_threshold, color=COLORS[2], linestyle=':', label=f'Global {global_threshold:g}')
    ax.invert_yaxis()
    ax.set(xlim=(0, .9), xlabel='Validation-selected per-label threshold', title='Frozen threshold profile; 28 labels')
    fig.legend(loc='lower center', ncol=2, frameon=False)
    save(fig, 'threshold_profile', bottom=.06)

    fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharey=True, sharex=True)
    y = np.arange(28)
    for ax, split in zip(axes, ('val', 'test')):
        for shift, strategy, color in ((-.18, 'global', COLORS[1]), (.18, 'per_label', COLORS[2])):
            ax.barh(y + shift, detail[f'{split}_{strategy}_delta_pp'], .34, label=strategy, color=color)
        ax.axvline(0, color='#444444', linewidth=.8)
        ax.set(yticks=y, yticklabels=LABELS28, xlabel='F1 difference vs fixed (percentage points)', title=f'{split}; F1 changes by label')
        ax.tick_params(axis='y', labelsize=8)
        ax.legend(loc='lower right')
    axes[0].invert_yaxis()
    save(fig, 'label_deltas')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name, field, order, title in (
        (axes[0], 'test_frequency_tiers', 'tier', ['head', 'mid', 'tail'], 'Test emotion27; train-only tiers'),
        (axes[1], 'test_cardinality', 'cardinality', ['1', '2', '3', '>=4'], 'Test all28; true label cardinality'),
    ):
        frame = table(name)
        if 'scope' in frame:
            frame = frame[frame.scope == ('all28' if field == 'cardinality' else 'emotion27')]
        frame[field] = frame[field].astype(str)
        x = np.arange(len(order))
        for shift, strategy, color in ((-.18, 'global', COLORS[1]), (.18, 'per_label', COLORS[2])):
            rows = frame[frame.strategy == strategy].set_index(field).loc[order]
            ax.bar(x + shift, 100 * rows.delta_macro_f1_vs_fixed, .34, label=strategy, color=color)
        ticks = order
        if field == 'cardinality':
            counts = frame[frame.strategy == 'fixed'].set_index(field).loc[order, 'n']
            ticks = [f'{group}\n(n={n})' for group, n in zip(order, counts)]
        ax.axhline(0, color='#444444', linewidth=.8)
        ax.set(xticks=x, xticklabels=ticks, ylabel='Macro-F1 difference vs fixed (pp)', title=title)
        ax.legend()
    save(fig, 'tier_cardinality_deltas')

    (analysis / 'figure_provenance.json').write_text(json.dumps({
        'builder': 'scripts/build_final_figures.py', 'builder_sha256': checksum(__file__),
        'execution_verification_sha256': checksum(output / 'execution_verification.json'),
        'generated_sha256': {str(p.relative_to(output)): checksum(p) for p in generated},
        'inference_performed': False,
    }, indent=2) + '\n')
    return generated


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, default=ROOT / 'reports/final28')
    for path in build(parser.parse_args().study_dir):
        print(path)
