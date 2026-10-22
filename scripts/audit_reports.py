"""Check stored reports against data and saved probabilities; never train or edit reports."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import analyze_data
from ablate_mixed_neutral import training_conditions, frequency_tiers
from common import ROOT, LABELS, load_data, targets, scores
from evaluate import save_evaluation
from tune_phobert import load_plan


def same_table(actual, expected, keys=None):
    if keys:
        actual = actual.sort_values(keys).reset_index(drop=True)
        expected = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False,
                                  check_exact=False, rtol=0, atol=1e-10)


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


def main():
    reports = ROOT/'reports'
    frames, _ = load_data()
    y = targets(frames['val'])
    hashes = {split: hashlib.sha256((ROOT/'data'/f'{split}.csv').read_bytes()).hexdigest()
              for split in frames}
    for path in [reports/'eda_environment.json', *reports.glob('*_run.json')]:
        metadata = json.loads(path.read_text())
        if metadata.get('environment', metadata)['data_sha256'] != hashes:
            raise ValueError(f'{path.name}: source data checksum mismatch')

    # Recreate aggregates/evaluation only in a temporary directory. No model is fitted.
    with TemporaryDirectory() as directory:
        temporary = Path(directory)
        with patch.object(analyze_data, 'ROOT', temporary), contextlib.redirect_stdout(io.StringIO()):
            analyze_data.main()
        for path in (temporary/'reports/tables').glob('*.csv'):
            # Order within equal co-occurrence counts differed between old pandas versions.
            keys = ['label_a', 'label_b'] if path.name == 'train_top_pairs.csv' else None
            same_table(pd.read_csv(reports/'tables'/path.name), pd.read_csv(path), keys)
        for path in (temporary/'splits').glob('*.csv'):
            same_table(pd.read_csv(ROOT/'splits'/path.name, dtype=str), pd.read_csv(path, dtype=str))
        print('PASS: EDA tables, fixed split IDs and source checksums')

        for name in ('tfidf_lr', 'gru', 'phobert'):
            run = json.loads((reports/f'{name}_run.json').read_text())
            with np.load(reports/'predictions'/f'{name}_val.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['sample_id'], frames['val'].id.to_numpy(dtype=str))
                np.testing.assert_array_equal(saved['true_label_vector'], y)
                np.testing.assert_array_equal(saved['labels'], LABELS)
                np.testing.assert_array_equal(saved['label_cardinality'], y.sum(axis=1))
                probabilities = saved['probability_vector'].copy()
            out = temporary/name
            with contextlib.redirect_stdout(io.StringIO()):
                save_evaluation(name, frames, {'val': probabilities},
                    {key: run[key] for key in ('parameter_count', 'train_seconds')}, output_dir=out)
            for path in (out/'tables').glob('*.csv'):
                same_table(pd.read_csv(reports/'tables'/path.name), pd.read_csv(path))
            expected = json.loads((out/f'{name}_analysis.json').read_text())
            actual = json.loads((reports/f'{name}_analysis.json').read_text())
            for key, value in expected.items():
                if isinstance(value, float):
                    close(actual[key], value)
            if 'history' in run:
                best = max(run['history'], key=lambda row: (row['macro_f1'], -row['epoch']))
                if best['epoch'] != run['best_epoch']:
                    raise ValueError(f'{name}: incorrect best_epoch')
                for metric, value in scores(y, probabilities).items():
                    close(best[metric], value)
            print(f'PASS: {name} prediction identity, 7 evaluation tables, analysis and checkpoint selection')

    comparison = pd.read_csv(reports/'tables/model_comparison.csv').set_index('model')
    for name, title in [('tfidf_lr', 'TF-IDF + OvR LR'), ('gru', 'BiGRU'), ('phobert', 'PhoBERT')]:
        row = pd.read_csv(reports/'tables'/f'{name}_metrics.csv').iloc[0]
        for key in ('macro_f1', 'micro_f1', 'hamming_loss', 'lrap', 'parameter_count', 'train_seconds'):
            close(comparison.loc[title, key], row[key])
    styles = {name: pd.read_csv(reports/'tables'/f'{name}_styles.csv').set_index('style')
              for name in ('gru', 'phobert')}
    for row in pd.read_csv(reports/'tables/neural_style_comparison.csv').itertuples():
        for name in styles:
            close(row.n, styles[name].loc[row.style, 'n'])
            close(getattr(row, f'{name}_macro_f1'), styles[name].loc[row.style, 'macro_f1'])
        close(row.delta_macro_f1, row.phobert_macro_f1-row.gru_macro_f1)
    print('PASS: model and neural-style comparison tables')

    out = reports/'ablation_mixed_neutral'
    metadata = json.loads((out/'experiment_config.json').read_text())
    if metadata['status'] != 'completed' or metadata['environment']['data_sha256'] != hashes:
        raise ValueError('Ablation metadata/data mismatch')
    conditions = training_conditions(frames['train'])
    tiers = frequency_tiers(frames['train'], metadata['config'])
    comparison = pd.read_csv(out/'comparison.csv').set_index('metric')
    probabilities = {}
    for name, train in conditions.items():
        same_table(pd.read_csv(out/f'{name}_training_ids.csv', dtype=str),
                   pd.DataFrame({'sample_id': train.id.to_numpy(dtype=str)}))
        with np.load(out/f'{name}_val.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['sample_id'], frames['val'].id.to_numpy(dtype=str))
            np.testing.assert_array_equal(saved['true_label_vector'], y)
            np.testing.assert_array_equal(saved['labels'], LABELS)
            probabilities[name] = saved['probability_vector'].copy()
        for metric, value in scores(y, probabilities[name]).items():
            close(comparison.loc[metric, name], value)
    close(comparison.B_minus_A, comparison.B-comparison.A)
    close(comparison.absolute_difference, abs(comparison.B-comparison.A))
    per_label = pd.read_csv(out/'per_label_f1.csv')
    for name in conditions:
        stats = precision_recall_fscore_support(y, probabilities[name] >= .5, zero_division=0)
        for metric, values in zip(('precision', 'recall', 'f1'), stats[:3]):
            close(per_label[f'{name}_{metric}'], values)
        close(per_label[f'{name}_train_count'], targets(conditions[name]).sum(axis=0))
    mixed = frames['val'].label_ids.map(lambda labels: 27 in labels and len(set(labels)) > 1).to_numpy()
    groups = {'mixed_neutral': mixed, 'non_mixed_neutral': ~mixed}
    for filename, masks, key in [('frequency_tier_comparison.csv', tiers, 'tier'),
                                 ('validation_subgroup_comparison.csv', groups, 'subgroup')]:
        table = pd.read_csv(out/filename).set_index(key)
        for group, mask in masks.items():
            for name, p in probabilities.items():
                actual, predicted = (y[:, mask], p[:, mask]) if key == 'tier' else (y[mask], p[mask])
                for metric, value in scores(actual, predicted).items():
                    close(table.loc[group, f'{name}_{metric}'], value)
    print('PASS: ablation IDs, probabilities, metrics and group breakdowns')

    plan = load_plan(reports/'phobert_tuning')
    print('Tuning status:', plan['status'])
    if plan['status'] == 'prepared':
        if list((reports/'phobert_tuning').glob('lr_*/phobert_run.json')):
            raise ValueError('Prepared tuning contains run artifacts; inspect an interrupted study')
    elif plan['status'] == 'completed':
        # summarize validates scores, hashes and config; it also regenerates derived outputs.
        # Keep this audit read-only by using a temporary copy of the study.
        from shutil import copytree
        from tune_phobert import summarize
        with TemporaryDirectory() as directory:
            study = copytree(reports/'phobert_tuning', Path(directory)/'study')
            summarize(study)
        print('PASS: completed tuning verified on a temporary copy')
    else:
        raise ValueError('Unknown tuning status')
    print('Audit finished. No training or report files were changed.')


if __name__ == '__main__':
    main()
