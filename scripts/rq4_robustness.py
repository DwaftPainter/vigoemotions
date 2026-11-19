"""Evaluate new input perturbations against the frozen RQ1 PhoBERT-28."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from common import ROOT
from final_inference import predict_checkpoint, runtime_versions, save_probabilities, verify_assets
from final_study import Study, create_json, digest, read_json
from final_threshold_core import decode, read_predictions, write_evaluation
from rq4_noise import CONDITIONS, prepare_conditions

CONFIG = Path('configs/rq4_noise_robustness.json')
RQ1_OUTPUT = Path('reports/final28')
RQ4_OUTPUT = Path('reports/rq4_noise')
TABLES = {
    'comparison': (['scope'], ['macro_f1', 'micro_f1', 'hamming_loss', 'lrap']),
    'frequency_tiers': (['tier'], ['macro_f1', 'micro_f1']),
    'per_label': (['label'], ['f1']),
    'neutral': (['label'], ['f1', 'precision', 'recall']),
}


def verify_rq1(study):
    """Use completed RQ1 evidence as a read-only control and source of cutoffs."""
    lock = study.verify_lock()
    files = ('pre_test_lock.json', 'thresholds.json', 'validation_reproduction.json',
             'test_attempt.json', 'predictions/val.npz', 'predictions/test.npz')
    hashes = {name: digest(study.output/name) for name in files}
    attempt = read_json(study.output/'test_attempt.json')
    if (attempt['status'] != 'completed' or attempt['lock_sha256'] != hashes['pre_test_lock.json']
            or attempt['predictions_sha256'] != hashes['predictions/test.npz']
            or lock['thresholds_sha256'] != hashes['thresholds.json']
            or lock['validation_predictions_sha256'] != hashes['predictions/val.npz']):
        raise ValueError('RQ4 requires completed RQ1 results and their original locked thresholds')
    return hashes


def protocol(study):
    config = read_json(study.root/CONFIG)
    if (config['protocol_id'] != 'vigoemotions_rq4_noise_v1'
            or config['conditions'] != list(CONDITIONS)
            or type(config['seed']) is not int or not 0 < config['qwerty_letter_probability'] <= 1
            or config['clean_probability_atol'] != 1e-4):
        raise ValueError('Invalid RQ4 protocol configuration')
    bindings = ['scripts/rq4_noise.py', 'scripts/rq4_robustness.py', str(CONFIG), 'docs/rq4_noise_robustness.md']
    return config, {name: digest(study.root/name) for name in bindings}


def output_directory(study, output, private):
    output = Path(output or study.root/RQ4_OUTPUT).resolve()
    if (output.is_relative_to(study.root) and not output.is_relative_to(study.root/RQ4_OUTPUT)):
        raise ValueError('RQ4 output must stay under reports/rq4_noise or an external directory')
    for protected in (study.output, study.data_dir, Path(private).resolve()):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('RQ4 output must be separate from RQ1, private model assets and private data')
    return output


def compare_to_original(frame, groups, metrics):
    """Compare conditions within a split, a decoder and a declared label scope."""
    keys = ['split', 'strategy', *groups]
    control = frame.loc[frame.condition == 'original', keys+metrics].rename(
        columns={metric: f'original_{metric}' for metric in metrics})
    result = frame.merge(control, on=keys, validate='many_to_one')
    if len(result) != len(frame):
        raise ValueError('Each condition requires a matching original baseline row')
    for metric in metrics:
        result[f'delta_{metric}_pp_vs_original'] = 100 * (result[metric] - result.pop(f'original_{metric}'))
    return result


def write_tables(directory, split, fitted, tiers):
    for condition in CONDITIONS:
        path = directory/'predictions'/f'{condition}.npz'
        # IDs and target correspondence were checked against the fixed RQ1 split at save time.
        with np.load(path, allow_pickle=False) as saved:
            values = dict(saved.items())
        write_evaluation(directory, condition, values, fitted, tiers)
    tables = directory/'tables'
    comparisons = {}
    for name, (groups, metrics) in TABLES.items():
        frames = []
        for condition in CONDITIONS:
            frame = pd.read_csv(tables/f'{condition}_{name}.csv').rename(columns={'split': 'condition'})
            frame.insert(0, 'split', split)
            frames.append(frame)
        combined = compare_to_original(pd.concat(frames, ignore_index=True), groups, metrics)
        combined.to_csv(tables/f'{name}.csv', index=False)
        comparisons[name] = combined
    return comparisons['comparison']


def write_plot(directory, rows, split):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    noisy = rows.query("scope == 'all28' and condition != 'original'")
    kinds = ('no_diacritics', 'qwerty_typo')
    strategy_names = {'fixed': 'Fixed 0.5', 'global': 'Global', 'per_label': 'Per-label'}
    x = np.arange(len(kinds))
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for index, (strategy, name) in enumerate(strategy_names.items()):
        values = noisy.loc[noisy.strategy == strategy].set_index('condition').loc[list(kinds)]
        ax.bar(x + (index - 1) * .24, values.delta_macro_f1_pp_vs_original, width=.22, label=name,
               color=['.85', '.55', '.25'][index], edgecolor='black')
    ax.axhline(0, color='black', linewidth=.8)
    ax.set_xticks(x, ['Diacritics removed', 'QWERTY substitution'])
    ax.set_ylabel('Macro-F1 change vs. original (pp)')
    ax.set_title(f'PhoBERT-28 noise robustness ({split})')
    ax.legend(frameon=False)
    fig.tight_layout()
    folder = directory/'figures'
    folder.mkdir()
    fig.savefig(folder/'macro_f1_delta.png', dpi=200)
    fig.savefig(folder/'macro_f1_delta.pdf')
    plt.close(fig)


def verify_clean(clean, reference, fitted, tolerance):
    expected = reference['probability_vector']
    difference = float(np.max(np.abs(np.asarray(clean, dtype=np.float64) - expected)))
    if not np.isfinite(difference) or difference > tolerance:
        raise ValueError(f'Original-text RQ4 scores do not reproduce RQ1 (max absolute error {difference:.6g})')
    cuts = {'fixed': .5, 'global': fitted['global_threshold'], 'per_label': fitted['per_label_thresholds']}
    agreements = {name: bool(np.array_equal(decode(clean, cut), reference['pred_'+name])) for name, cut in cuts.items()}
    if not agreements['fixed']:
        raise ValueError('Original-text fixed decisions do not reproduce RQ1')
    return dict(max_absolute_error=difference, absolute_tolerance=tolerance, decisions_identical_to_rq1=agreements)


def validation_record(output, config_hashes, rq1_hashes):
    directory = output/'val'
    if not (directory/'manifest.json').is_file():
        raise ValueError('Complete RQ4 validation before evaluating the held-out split')
    record = read_json(directory/'manifest.json')
    if (record['status'] != 'completed' or record['split'] != 'val' or record['protocol_sha256'] != config_hashes
            or record['rq1_sha256'] != rq1_hashes or record['output_sha256'] != output_hashes(directory)):
        raise ValueError('RQ4 validation evidence is stale or modified')
    return record


def output_hashes(directory):
    return {p.relative_to(directory).as_posix(): digest(p)
            for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'manifest.json'}


def run_split(study, output, split, private, device=None):
    config, config_hashes = protocol(study)
    output = output_directory(study, output, private)
    destination = output/split
    if destination.exists():
        raise FileExistsError(f'RQ4 {split} output already exists: {destination}')
    rq1_hashes = verify_rq1(study)
    assets = verify_assets(study, private)
    reproduction = read_json(study.output/'validation_reproduction.json')
    if assets != reproduction['assets']:
        raise ValueError('Use the original RQ1 frozen inference assets')
    validation_sha = None
    if split == 'test':
        validation = validation_record(output, config_hashes, rq1_hashes)
        validation_sha = digest(output/'val/manifest.json')
        runtime = validation['runtime']
        device = device or runtime['device']
        if runtime['device'] != device or runtime['packages'] != runtime_versions():
            raise ValueError('Use the same runtime and device as RQ4 validation')

    reference = read_predictions(study.output/f'predictions/{split}.npz', study.ids(split))
    fitted = read_json(study.output/'thresholds.json')
    frame = study.frame(split)
    frames, coverage = prepare_conditions(frame, seed=config['seed'], rate=config['qwerty_letter_probability'])
    inputs = pd.concat([frames[condition] for condition in CONDITIONS], ignore_index=True)
    scores, actual_runtime = predict_checkpoint(study, inputs, private, device)
    if scores.shape != (len(inputs), len(study.config['labels'])):
        raise ValueError('Inference scores do not match the paired split and conditions')
    if split == 'test' and actual_runtime != runtime:
        raise ValueError('Inference runtime differs from RQ4 validation')
    clean_check = verify_clean(scores[:len(frame)], reference, fitted, config['clean_probability_atol'])
    if verify_rq1(study) != rq1_hashes or protocol(study)[1] != config_hashes or verify_assets(study, private) != assets:
        raise ValueError('RQ4 or RQ1 inputs changed during inference')

    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'.{split}-', dir=output) as temporary:
        directory = Path(temporary)/split
        directory.mkdir()
        prediction_root = directory/'predictions'
        for condition, probabilities in zip(CONDITIONS, np.split(scores, len(CONDITIONS)), strict=True):
            path = prediction_root/f'{condition}.npz'
            save_probabilities(path, frames[condition], probabilities)
            saved = read_predictions(path, reference['sample_id'])
            if not np.array_equal(saved['true_label_vector'], reference['true_label_vector']):
                raise ValueError('Noisy targets differ from the original RQ1 split')
        rows = write_tables(directory, split, fitted, study.tiers())
        write_plot(directory, rows, split)
        pd.DataFrame([{'split': split, 'condition': name, **values} for name, values in coverage.items()]).to_csv(
            directory/'tables/noise_coverage.csv', index=False)
        record = dict(status='completed', split=split, conditions=list(CONDITIONS), n=len(frame), runtime=actual_runtime,
                      clean_vs_rq1=clean_check, protocol_sha256=config_hashes, rq1_sha256=rq1_hashes,
                      rq4_validation_manifest_sha256=validation_sha, output_sha256=output_hashes(directory))
        create_json(directory/'manifest.json', record)
        if destination.exists():
            raise FileExistsError(f'RQ4 {split} output already exists: {destination}')
        directory.rename(destination)
    return dict(status='completed', split=split, samples=len(frame), clean_vs_rq1=clean_check,
                coverage=coverage, output=str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('verify', 'validate', 'test'))
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--private-dir', type=Path, default=ROOT/'data/final28')
    parser.add_argument('--rq1-dir', type=Path, default=ROOT/RQ1_OUTPUT)
    parser.add_argument('--output-dir', type=Path, default=ROOT/RQ4_OUTPUT)
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    args = parser.parse_args()
    study = Study(output=args.rq1_dir, data_dir=args.data_dir)
    if args.stage == 'verify':
        _, bindings = protocol(study)
        result = dict(status='verified', rq1_sha256=verify_rq1(study), protocol_sha256=bindings)
        assets = verify_assets(study, args.private_dir)
        if assets != read_json(study.output/'validation_reproduction.json')['assets']:
            raise ValueError('Use the original RQ1 frozen inference assets')
    else:
        split = 'val' if args.stage == 'validate' else 'test'
        result = run_split(study, args.output_dir, split, args.private_dir, args.device)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
