"""Controlled validation-only mixed-neutral ablation; never export comments."""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from threadpoolctl import threadpool_limits

from common import (ROOT, LABELS, environment, load_data, parse_labels, scores,
                    targets, write_json)

METRICS = ('macro_f1', 'micro_f1', 'hamming_loss', 'lrap')
OUTPUT = ROOT / 'reports/ablation_mixed_neutral'


def is_mixed_neutral(labels):
    labels = set(parse_labels(labels))
    return 27 in labels and bool(labels - {27})


def mixed_mask(frame):
    return frame.label_ids.map(is_mixed_neutral).to_numpy(dtype=bool)


def training_conditions(train, expected_excluded=251):
    mask = mixed_mask(train)
    if int(mask.sum()) != expected_excluded:
        raise ValueError(f'Expected {expected_excluded} mixed-neutral training rows, got {mask.sum()}')
    return {'A': train.copy(deep=True), 'B': train.loc[~mask].copy(deep=True)}


def frequency_tiers(train, config):
    frequency = targets(train).sum(axis=0)
    ranking = np.lexsort((np.arange(27), -frequency))
    return {tier: ranking[start - 1:end] for tier in ('head', 'mid', 'tail')
            for start, end in [config['frequency_tiers'][f'{tier}_ranks']]}


def fit_condition(train, validation, config):
    params = dict(config['tfidf'])
    params['ngram_range'] = tuple(params['ngram_range'])
    vectorizer = TfidfVectorizer(**params)
    started = time.perf_counter()
    x = vectorizer.fit_transform(train.normalized)
    model = OneVsRestClassifier(LogisticRegression(**config['logistic_regression']), n_jobs=1)
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, targets(train))
    elapsed = time.perf_counter() - started
    p = model.predict_proba(vectorizer.transform(validation.normalized))
    if p.shape != (len(validation), 27) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid validation probabilities')
    return p, dict(train_seconds=elapsed, training_samples=len(train),
                   tfidf_features=len(vectorizer.vocabulary_))


def compare(a, b, keys):
    row = {}
    for key in keys:
        row.update({f'A_{key}': a[key], f'B_{key}': b[key],
                    f'B_minus_A_{key}': b[key] - a[key],
                    f'absolute_difference_{key}': abs(b[key] - a[key])})
    return row


def save_predictions(path, validation, y, p):
    np.savez_compressed(path, sample_id=validation.id.to_numpy(dtype=str),
                        true_label_vector=y, probability_vector=p, labels=np.array(LABELS))


def verify_predictions(out):
    with np.load(out/'A_val.npz') as a, np.load(out/'B_val.npz') as b:
        for field in ('sample_id', 'true_label_vector', 'labels'):
            np.testing.assert_array_equal(a[field], b[field])


def run_comparison(train, validation, config, out, expected_excluded=251):
    """Accept only training and validation frames; test is not an input."""
    conditions = training_conditions(train, expected_excluded)
    tiers = frequency_tiers(train, config)
    y = targets(validation)
    val_ids = validation.id.to_numpy(dtype=str).copy()
    val_mixed = mixed_mask(validation)
    results, probabilities = {}, {}
    for name, subset in conditions.items():
        p, metadata = fit_condition(subset, validation, config)
        np.testing.assert_array_equal(validation.id.to_numpy(dtype=str), val_ids)
        np.testing.assert_array_equal(targets(validation), y)
        save_predictions(out/f'{name}_val.npz', validation, y, p)
        results[name] = dict(**scores(y, p, .5), **metadata)
        probabilities[name] = p
        pd.DataFrame({'sample_id': subset.id.to_numpy(dtype=str)}).to_csv(
            out/f'{name}_training_ids.csv', index=False)
    verify_predictions(out)
    keys = (*METRICS, 'train_seconds', 'training_samples', 'tfidf_features')
    comparison = pd.DataFrame([dict(metric=k, A=results['A'][k], B=results['B'][k],
        B_minus_A=results['B'][k]-results['A'][k],
        absolute_difference=abs(results['B'][k]-results['A'][k])) for k in keys])
    comparison.to_csv(out/'comparison.csv', index=False)
    per_label = {name: precision_recall_fscore_support(y, p >= .5, zero_division=0)
                 for name, p in probabilities.items()}
    frequency = {name: targets(subset).sum(axis=0) for name, subset in conditions.items()}
    label_rows = []
    for c, label in enumerate(LABELS):
        values = {name: dict(zip(('precision', 'recall', 'f1'), [v[c] for v in stats[:3]]))
                  for name, stats in per_label.items()}
        label_rows.append(dict(label_id=c, label=label,
            tier=next(t for t, ids in tiers.items() if c in ids),
            A_train_count=int(frequency['A'][c]), B_train_count=int(frequency['B'][c]),
            validation_support=int(y[:, c].sum()),
            **compare(values['A'], values['B'], ('precision', 'recall', 'f1'))))
    pd.DataFrame(label_rows).to_csv(out/'per_label_f1.csv', index=False)
    tier_rows = []
    for tier, ids in tiers.items():
        values = {name: scores(y[:, ids], p[:, ids]) for name, p in probabilities.items()}
        tier_rows.append(dict(tier=tier, label_count=len(ids),
                              **compare(values['A'], values['B'], METRICS)))
    pd.DataFrame(tier_rows).to_csv(out/'frequency_tier_comparison.csv', index=False)
    subgroup_rows = []
    for group, mask in [('mixed_neutral', val_mixed), ('non_mixed_neutral', ~val_mixed)]:
        if not mask.any():
            raise ValueError(f'Empty validation subgroup: {group}')
        values = {name: scores(y[mask], p[mask]) for name, p in probabilities.items()}
        subgroup_rows.append(dict(subgroup=group, samples=int(mask.sum()),
                                  **compare(values['A'], values['B'], METRICS)))
    pd.DataFrame(subgroup_rows).to_csv(out/'validation_subgroup_comparison.csv', index=False)
    return comparison, tiers


def write_summary(out, comparison):
    # Import the existing plotting setup without calling its threshold-sweep evaluator.
    from evaluate import plt
    indexed = comparison.set_index('metric')
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    for ax, metric in zip(axes, METRICS):
        ax.bar(['A', 'B'], indexed.loc[metric, ['A', 'B']], color=['#167d9a', '#ce5863'])
        ax.set(title=metric, ylim=(0, 1))
    fig.suptitle('Validation only; fixed threshold 0.5 (Hamming Loss: lower is better)')
    fig.tight_layout()
    fig.savefig(out/'comparison.png', dpi=160)
    plt.close(fig)
    table = '\n'.join(f'| {r.metric} | {r.A:.8f} | {r.B:.8f} | {r.B_minus_A:+.8f} |'
                      for r in comparison.itertuples())
    breakdown_rows = []
    for filename, group_column in [('frequency_tier_comparison.csv', 'tier'),
                                    ('validation_subgroup_comparison.csv', 'subgroup')]:
        for row in pd.read_csv(out/filename).to_dict('records'):
            breakdown_rows.append(
                f"| {row[group_column]} | {row['A_macro_f1']:.8f} | "
                f"{row['B_macro_f1']:.8f} | {row['B_minus_A_macro_f1']:+.8f} |")
    breakdown = '\n'.join(breakdown_rows)
    delta = indexed.loc['macro_f1', 'B_minus_A']
    direction = 'improved' if delta > 0 else 'worsened' if delta < 0 else 'did not change'
    (out/'experiment_summary.md').write_text(f'''# Mixed-neutral label ablation (Midterm)

## Research question and annotation issue
Does retaining emotion labels of mixed-neutral training samples improve predictive
performance compared with excluding those samples entirely? Neutral (27) sometimes
co-occurs with emotion labels, creating conflicting annotations under the project's
27-emotion convention. Neutral-only examples have all-zero targets.

## Design
A retains all 16,530 cleaned training rows and removes only the Neutral label.
B excludes exactly 251 mixed-neutral training rows, leaving 16,279 rows.
Both use the same 2,066 validation IDs and target vectors, including all 19 mixed-neutral
validation rows. Their saved prediction archives are checked for exact ID/target equality.
All normalization, TF-IDF settings, OvR Logistic Regression settings, seed 42, and
threshold 0.5 are held constant using configs/midterm.json. Vocabulary and IDF are
fit independently on each training subset. The changed training subset can therefore
also change the learned feature space. No threshold search is performed.
The existing loader reads test data for the established overlap audit only; no test
predictions, metrics, tuning, or selection are performed. Original CSVs are read-only.
The Final research question on threshold optimization remains unchanged.

## Validation results
| Metric | A: emotion priority | B: exclude mixed | B minus A |
|---|---:|---:|---:|
{table}

Removing mixed-neutral training samples **{direction} validation Macro-F1**
by {abs(delta):.8f}. Higher F1/LRAP and lower Hamming Loss are better. Differences
are descriptive absolute-scale differences, not relative percentages; CSVs include
both signed B-minus-A values and their absolute magnitudes.

The changes should be assessed metric by metric: an F1 gain can coexist with a
ranking-quality (LRAP) loss. This run does not justify a universal performance claim.

| Label tier or validation subgroup | A Macro-F1 | B Macro-F1 | B minus A |
|---|---:|---:|---:|
{breakdown}

See per_label_f1.csv for precision, recall, F1, and support; frequency_tier_comparison.csv
for head/mid/tail metrics; and validation_subgroup_comparison.csv for subgroup metrics.
Tiers are frozen from A's cleaned training frequencies: descending count, ties by
label ID, nine labels per tier. Tier metrics operate on those nine columns only;
tier LRAP ranks within the tier, with the existing all-zero target convention.
Macro-F1 always includes every column in its evaluated group, with zero_division=0.
Training time includes TF-IDF fitting and classifier fitting, excluding data loading,
validation transformation, and prediction. Timing is a single hardware-dependent run.

## Implications and limitations
These results inform whether retaining partially conflicting training annotations is
useful for this baseline. Better predictive performance does not establish which
original annotation is correct. This is one fixed split, seed, model, and threshold;
no significance test or independent test evaluation was performed. Removing rows also
reduces training size and changes label prevalence and learned vocabulary. The optional
random-exclusion control was not run, so annotation effects cannot be separated from
data-volume effects. The 19 mixed-neutral validation cases are supplementary and
are too few for statistically conclusive claims. Validation targets retain the emotion
priority convention, so the comparison is conditional on that evaluation convention.
Cross-environment reproduction may introduce numerical and runtime differences.

## Reproduction
Run `.venv/bin/python scripts/ablate_mixed_neutral.py` from the repository root.
Existing output directories are refused; use `--output reports/ablation_mixed_neutral/rerun_01`
for a new run. Archives contain ordered IDs, targets, continuous probabilities, and
label names. Training-ID manifests, input hashes, configuration, and package versions
are recorded without raw comments, vocabulary, model checkpoints, or text snippets.
''', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    out = args.output.resolve()
    if out != OUTPUT.resolve() and OUTPUT.resolve() not in out.parents:
        parser.error('Output must be inside reports/ablation_mixed_neutral/')
    if out.exists():
        parser.error('Output already exists; choose a fresh subdirectory to preserve results')
    config = json.loads((ROOT/'configs/midterm.json').read_text())
    if (config['seed'], config['logistic_regression']['random_state'], config['threshold']) != (42, 42, .5):
        raise ValueError('This study requires seed 42 and threshold 0.5')
    before = environment()
    frames, excluded = load_data()
    train, validation = frames['train'], frames['val']
    if (len(train), len(validation), len(excluded), int(mixed_mask(validation).sum())) != (16530, 2066, 1, 19):
        raise ValueError('Dataset counts do not match the pre-specified study')
    training_conditions(train)  # Fail before creating artifacts if the 251-row contract differs.
    del frames  # Only train/validation are passed to the experiment.
    out.mkdir(parents=True)
    comparison, tiers = run_comparison(train, validation, config, out)
    after_hashes = {s: hashlib.sha256((ROOT/'data'/f'{s}.csv').read_bytes()).hexdigest()
                    for s in ('train', 'val', 'test')}
    if after_hashes != before['data_sha256']:
        raise RuntimeError('Input CSV changed during experiment')
    write_json(out/'experiment_config.json', dict(
        status='completed', config=config, environment=before, threshold=.5,
        conditions={'A': 'emotion_priority', 'B': 'exclude_mixed_neutral'},
        excluded_mixed_neutral=251, removed_overlap_ids=excluded.id.tolist(),
        validation_samples=2066, validation_mixed_neutral=19,
        tiers={k: v.tolist() for k, v in tiers.items()},
        tiers_reference='A cleaned training frequencies, frozen across experiments',
        test_usage='existing leakage audit only; no evaluation, tuning, or selection',
        input_hashes_unchanged=True, optional_control='not run',
        time_definition='TF-IDF fit + OvR fit; excludes data loading and validation'))
    write_summary(out, comparison)
    print(comparison.to_string(index=False))


if __name__ == '__main__':
    main()
