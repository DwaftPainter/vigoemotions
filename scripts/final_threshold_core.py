"""Pure decoder fitting and text-free evaluation; no model or test-data loading."""
from pathlib import Path

import numpy as np
import pandas as pd

from common import LABELS as EMOTIONS, scores
from neutral_28_labels import LABELS28 as LABELS

BASE_FIELDS = {'sample_id', 'labels', 'true_label_vector', 'probability_vector', 'label_cardinality', 'emotion_cardinality'}
DECODERS = ('fixed', 'global', 'per_label')


def validate_arrays(y, probabilities):
    y, p = np.asarray(y), np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 2 or y.shape != p.shape or not y.shape[1] or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected aligned binary n-by-label targets and scores')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite in [0, 1]')
    return y, p


def decode(probabilities, thresholds):
    # Use one comparison precision for scalar/vector thresholds and float32/64 NPZs.
    p, t = np.asarray(probabilities, dtype=np.float64), np.asarray(thresholds, dtype=np.float64)
    if p.ndim != 2 or t.ndim > 1 or (t.ndim == 1 and t.shape != (p.shape[1],)):
        raise ValueError('Threshold must be a scalar or one value per label')
    if not np.isfinite(t).all() or ((t < 0) | (t > 1)).any():
        raise ValueError('Thresholds must be finite in [0, 1]')
    validate_arrays(np.zeros(p.shape, dtype=bool), p)
    return p >= t


def label_f1(y, pred):
    numerator = 2 * (y.astype(bool) & pred).sum(axis=0)
    denominator = y.sum(axis=0) + pred.sum(axis=0)
    return np.divide(numerator, denominator, out=np.zeros(y.shape[1]), where=denominator != 0)


def fit_thresholds(y, probabilities, candidates, fallback=.5):
    y, p = validate_arrays(y, probabilities)
    grid = np.asarray(candidates, dtype=np.float64)
    if not len(y) or grid.ndim != 1 or not len(grid) or len(np.unique(grid)) != len(grid):
        raise ValueError('A nonempty validation set and unique candidate grid are required')
    if not np.isfinite(grid).all() or ((grid < 0) | (grid > 1)).any() or fallback not in grid:
        raise ValueError('Grid must contain the fallback and finite candidates in [0, 1]')
    grid = np.sort(grid)[::-1]  # argmax keeps the highest threshold on exact F1 ties.
    f1 = np.stack([label_f1(y, decode(p, threshold)) for threshold in grid])
    macro = f1.mean(axis=1)
    selected = grid[f1.argmax(axis=0)]
    missing = np.flatnonzero(y.sum(axis=0) == 0)
    selected[missing] = fallback
    return dict(global_threshold=float(grid[macro.argmax()]), per_label_thresholds=selected.tolist(),
                fallback_label_indices=missing.tolist(), validation_support=y.sum(axis=0).astype(int).tolist(),
                global_sweep=[dict(threshold=float(t), macro_f1=float(m)) for t, m in zip(grid, macro)])


def read_predictions(path, expected_ids):
    with np.load(path, allow_pickle=False) as saved:
        required = BASE_FIELDS - {'label_cardinality', 'emotion_cardinality'}
        if not required <= set(saved.files) or set(saved.files) - BASE_FIELDS - {'pred_'+s for s in DECODERS}:
            raise ValueError('Unexpected prediction fields or missing required arrays')
        values = {key: saved[key].copy() for key in saved.files}
    ids = values['sample_id']
    if ids.ndim != 1 or ids.dtype.kind not in 'US' or len(set(ids.tolist())) != len(ids):
        raise ValueError('Prediction IDs must be unique strings')
    if not np.array_equal(ids, np.asarray(expected_ids, dtype=str)) or not np.array_equal(values['labels'], LABELS):
        raise ValueError('Prediction IDs/order or canonical label order mismatch')
    y, p = validate_arrays(values['true_label_vector'], values['probability_vector'])
    if y.shape != (len(ids), len(LABELS)):
        raise ValueError('Prediction shape differs from the fixed split')
    card = y.sum(axis=1)
    if 'label_cardinality' in values and not np.array_equal(values['label_cardinality'], card):
        raise ValueError('Ground-truth cardinality mismatch')
    values['label_cardinality'] = card
    emotion_card = y[:, :27].sum(axis=1)
    if 'emotion_cardinality' in values and not np.array_equal(values['emotion_cardinality'], emotion_card):
        raise ValueError('27-emotion cardinality mismatch')
    values['emotion_cardinality'] = emotion_card
    for strategy in DECODERS:
        if 'pred_'+strategy in values:
            validate_arrays(values['pred_'+strategy], p)
    return values


def metric_row(y, probabilities, thresholds):
    y, p = validate_arrays(y, probabilities)
    values = scores(y, p, np.asarray(thresholds, dtype=np.float64)) if len(y) else dict.fromkeys(
        ('macro_f1', 'micro_f1', 'hamming_loss', 'lrap'))
    if len(y) and y.shape[1] == 1:
        # sklearn treats n×1 as a binary task; its macro/micro then include negatives.
        # A label-scope metric must instead score that label's positive detections.
        f1 = float(label_f1(y, decode(p, thresholds))[0])
        values.update(macro_f1=f1, micro_f1=f1)
    return dict(n=len(y), positive_support=int(y.sum()), **values)


def write_evaluation(output, split, values, fitted, tiers, save_predictions=True):
    """Derive all tables and decoder outputs from one probability matrix."""
    output = Path(output)
    tables, predictions = output/'tables', output/'predictions'
    tables.mkdir(parents=True, exist_ok=True); predictions.mkdir(parents=True, exist_ok=True)
    y, p = validate_arrays(values['true_label_vector'], values['probability_vector'])
    cuts = {'fixed': .5, 'global': fitted['global_threshold'], 'per_label': fitted['per_label_thresholds']}
    overall, per_label, tier_rows, card_rows, neutral_rows = [], [], [], [], []
    payload = {key: values[key] for key in BASE_FIELDS}
    for strategy, cut in cuts.items():
        pred = decode(p, cut)
        payload['pred_'+strategy] = pred
        for scope, width in [('all28', 28), ('emotion27', 27)]:
            selected_cut = np.asarray(cut)[:width] if np.ndim(cut) else cut
            overall.append(dict(split=split, strategy=strategy, scope=scope, label_count=width,
                                **metric_row(y[:, :width], p[:, :width], selected_cut)))
        tp, fp, fn = (y.astype(bool) & pred).sum(axis=0), ((y == 0) & pred).sum(axis=0), ((y == 1) & ~pred).sum(axis=0)
        f1 = label_f1(y, pred)
        for c, label in enumerate(LABELS):
            per_label.append(dict(split=split, strategy=strategy, label=label, support=int(y[:, c].sum()),
                                  tp=int(tp[c]), fp=int(fp[c]), fn=int(fn[c]), f1=float(f1[c])))
        neutral_rows.append(dict(split=split, strategy=strategy, label='neutral', support=int(y[:, 27].sum()),
                                 threshold=float(np.asarray(cut)[27] if np.ndim(cut) else cut),
                                 tp=int(tp[27]), fp=int(fp[27]), fn=int(fn[27]), f1=float(f1[27]),
                                 precision=float(tp[27]/(tp[27]+fp[27])) if tp[27]+fp[27] else 0.,
                                 recall=float(tp[27]/(tp[27]+fn[27])) if tp[27]+fn[27] else 0.))
        for tier in ('head', 'mid', 'tail'):
            columns = np.array([c for c, label in enumerate(EMOTIONS) if tiers[label] == tier])
            selected_cut = np.asarray(cut)[columns] if np.ndim(cut) else cut
            tier_rows.append(dict(split=split, strategy=strategy, scope='emotion27', tier=tier, label_count=len(columns),
                                  **metric_row(y[:, columns], p[:, columns], selected_cut)))
        for scope, width, field in [('all28', 28, 'label_cardinality'), ('emotion27', 27, 'emotion_cardinality')]:
            cardinality = values[field]
            selected_cut = np.asarray(cut)[:width] if np.ndim(cut) else cut
            for group in ('0', '1', '2', '3', '>=4'):
                mask = cardinality >= 4 if group == '>=4' else cardinality == int(group)
                card_rows.append(dict(split=split, strategy=strategy, scope=scope, cardinality=group,
                                      **metric_row(y[mask, :width], p[mask, :width], selected_cut)))
    for name, rows in [('comparison', overall), ('per_label', per_label),
                       ('frequency_tiers', tier_rows), ('cardinality', card_rows), ('neutral', neutral_rows)]:
        frame = pd.DataFrame(rows)
        keys = {'comparison': ['split', 'scope'], 'per_label': ['split', 'label'], 'neutral': ['split'],
                'frequency_tiers': ['split', 'tier'], 'cardinality': ['split', 'scope', 'cardinality']}[name]
        metrics = ['f1'] if name in ('per_label', 'neutral') else ['macro_f1', 'micro_f1', 'hamming_loss', 'lrap']
        baseline = frame[frame.strategy == 'fixed'][keys+metrics].rename(columns={m: 'fixed_'+m for m in metrics})
        frame = frame.merge(baseline, on=keys, validate='many_to_one')
        for metric in metrics:
            frame['delta_'+metric+'_vs_fixed'] = frame[metric]-frame['fixed_'+metric]
        frame.drop(columns=['fixed_'+m for m in metrics]).to_csv(tables/f'{split}_{name}.csv', index=False)
    if save_predictions:
        np.savez_compressed(predictions/f'{split}.npz', **payload)
    return overall
