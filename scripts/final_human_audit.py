"""Prepare private human-review worksheets from frozen test outputs; no inference."""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_final_figures import verify_inputs
from final_study import Study, digest, read_json
from final_threshold_core import read_predictions
from neutral_28_labels import LABELS28, targets28

ROOT = Path(__file__).resolve().parents[1]
REVIEW_FIELDS = ['reviewer', 'status', 'read_private_text', 'reviewed_on', 'error_category',
                 'reference_assessment', 'observation', 'hypothesis']
CATEGORIES = {'threshold_tradeoff', 'negation', 'implicit_emotion', 'sarcasm', 'slang_emoji',
              'missing_context', 'label_overlap', 'annotation_query', 'other', 'unclear'}
ASSESSMENTS = {'consistent', 'questionable', 'insufficient_context'}
CASE_FIELDS = ['review_id', 'selection_group', 'sample_id', 'error_type', 'label_id', 'target_label',
               'pair_labels', 'true_labels', 'label_cardinality', 'emotion_cardinality', 'mixed_neutral',
               'neutral_only', 'target_probability', 'target_per_label_threshold', 'fixed_threshold',
               'global_threshold', 'probability_vector', 'per_label_threshold_vector',
               *[field for strategy in ('fixed', 'global', 'per_label')
                 for field in (f'pred_{strategy}_labels', f'{strategy}_outcome')]]


def rank(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def select_pairs(ids, y, pred, kind, count):
    if kind not in ('FN', 'FP'):
        raise ValueError('Expected FN or FP')
    mask = (y == 1) & ~pred if kind == 'FN' else (y == 0) & pred
    ordered = sorted([(int(i), int(c)) for i, c in np.argwhere(mask)],
                     key=lambda t: (rank(f'42|{kind}|{ids[t[0]]}|{t[1]}'), str(ids[t[0]]), t[1]))
    diverse, remaining, seen = [], [], set()
    for i, c in ordered:
        if str(ids[i]) in seen:
            remaining.append((i, c))
        else:
            diverse.append((i, c))
            seen.add(str(ids[i]))
    return (diverse + remaining)[:count]


def build_cases(values, fitted, train_y):
    ids, y = values['sample_id'], values['true_label_vector'].astype(bool)
    pred = values['pred_per_label'].astype(bool)
    p, cuts = values['probability_vector'], fitted['per_label_thresholds']
    rows, selected_indices = [], set()
    design = {'seed': 42, 'primary_decoder': 'per_label', 'cooccurrence_pairs': [], 'core': {}}

    def add(group, i, c=None, kind='context', pair=None):
        selected_indices.add(i)
        labels = lambda mask: ';'.join(np.asarray(LABELS28)[mask].tolist())
        row = dict(review_id='audit_' + rank(f'42|{group}|{ids[i]}|{c}')[:20],
                   selection_group=group, sample_id=str(ids[i]), error_type=kind,
                   label_id='' if c is None else c, target_label='' if c is None else LABELS28[c],
                   pair_labels='' if pair is None else ';'.join(LABELS28[a] for a in pair),
                   true_labels=labels(y[i]), label_cardinality=int(y[i].sum()),
                   emotion_cardinality=int(y[i, :27].sum()),
                   mixed_neutral='yes' if y[i, 27] and y[i, :27].any() else 'no',
                   neutral_only='yes' if y[i, 27] and not y[i, :27].any() else 'no',
                   target_probability='' if c is None else repr(float(p[i, c])),
                   target_per_label_threshold='' if c is None else cuts[c],
                   fixed_threshold=.5, global_threshold=fitted['global_threshold'],
                   probability_vector=json.dumps(p[i].astype(float).tolist()),
                   per_label_threshold_vector=json.dumps(cuts))
        for strategy in ('fixed', 'global', 'per_label'):
            decoded = values['pred_' + strategy][i].astype(bool)
            row[f'pred_{strategy}_labels'] = labels(decoded)
            row[f'{strategy}_outcome'] = '' if c is None else (
                ('TP' if y[i, c] else 'FP') if decoded[c] else ('FN' if y[i, c] else 'TN'))
        rows.append({key: str(value) for key, value in row.items()})

    for kind in ('FN', 'FP'):
        selected = select_pairs(ids, y, pred, kind, 20)
        mask = y & ~pred if kind == 'FN' else ~y & pred
        design['core'][kind] = dict(available_pairs=int(mask.sum()), selected_pairs=len(selected),
                                    selected_unique_comments=len({i for i, c in selected}),
                                    population_shortfall=max(0, 20 - int(mask.sum())))
        for i, c in selected:
            add(f'core_{kind}', i, c, kind)
        # Neutral is contextual coverage in addition to the unchanged core sample.
        if not any(c == 27 for i, c in selected):
            neutral = sorted(np.flatnonzero(mask[:, 27]),
                             key=lambda i: (rank(f'42|{kind}|{ids[i]}|27'), str(ids[i])))
            for i in neutral[:1]:
                add(f'neutral_{kind}', int(i), 27, kind)

    high = sorted(np.flatnonzero(y.sum(axis=1) >= 4),
                  key=lambda i: (rank(f'42|high_cardinality|{ids[i]}'), str(ids[i])))
    for i in high[:5]:
        add('high_cardinality', int(i))
    design['high_cardinality'] = dict(available_comments=len(high), selected_comments=min(5, len(high)))

    emotion_train = train_y[:, :27].astype(np.int64)
    counts = emotion_train.T @ emotion_train
    pairs = sorted([(a, b) for a in range(27) for b in range(a + 1, 27)],
                   key=lambda pair: (-counts[pair], *pair))[:3]
    seen_pair_comments = set()
    for a, b in pairs:
        eligible = np.flatnonzero((y[:, a] & y[:, b]) | (pred[:, a] & pred[:, b]))
        available = [int(i) for i in eligible if str(ids[i]) not in seen_pair_comments]
        chosen = sorted(available, key=lambda i: (rank(f'42|pair|{a}|{b}|{ids[i]}'), str(ids[i])))[:5]
        design['cooccurrence_pairs'].append(dict(label_id_a=a, label_id_b=b, label_a=LABELS28[a],
            label_b=LABELS28[b], train_count=int(counts[a, b]), eligible_test_comments=len(eligible),
            available_after_cross_pair_dedup=len(available), selected_comments=len(chosen)))
        for i in chosen:
            add(f'pair_{a}_{b}', i, pair=(a, b))
            seen_pair_comments.add(str(ids[i]))

    mixed = y[:, 27] & y[:, :27].any(axis=1)
    if not any(mixed[i] for i in selected_indices):
        extra = sorted(np.flatnonzero(mixed), key=lambda i: (rank(f'42|mixed_neutral|{ids[i]}'), str(ids[i])))
        for i in extra[:1]:
            add('mixed_neutral', int(i))
    cases = pd.DataFrame(rows, columns=CASE_FIELDS)
    if cases.review_id.duplicated().any():
        raise ValueError('Audit review IDs must be unique')
    design.update(selected_rows=len(cases), unique_comments=int(cases.sample_id.nunique()),
                  mixed_neutral_rows=int(cases.mixed_neutral.eq('yes').sum()),
                  neutral_target_rows=int(cases.target_label.eq('neutral').sum()),
                  supplemental_selection='Neutral FN/FP: first same-kind hash-ranked case if absent from core; mixed: first 42|mixed_neutral|ID case only if absent overall')
    return cases, design


def spreadsheet_text(text):
    # Source comments must be literal text when opened in a spreadsheet.
    return "'" + text if text.lstrip().startswith(('=', '+', '-', '@')) else text


def review_packet(cases, texts):
    packet = cases.copy()
    packet['text'] = packet.sample_id.map(lambda sample: spreadsheet_text(texts[sample]))
    for field in REVIEW_FIELDS:
        packet[field] = 'pending' if field == 'status' else ''
    return packet


def align_packet(cases, packet):
    if list(packet.columns) != [*cases.columns, 'text', *REVIEW_FIELDS]:
        raise ValueError('Keep the original review worksheet columns')
    if packet.review_id.duplicated().any() or set(packet.review_id) != set(cases.review_id):
        raise ValueError('Review IDs must match the complete original packet')
    packet = packet.set_index('review_id').loc[cases.review_id].reset_index()
    if not packet[list(cases.columns)].astype(str).equals(cases.astype(str).reset_index(drop=True)):
        raise ValueError('Readonly sample/target/prediction fields changed')
    return packet


def validate_reviews(cases, packet, texts):
    packet = align_packet(cases, packet)
    expected_text = packet.sample_id.map(lambda sample: spreadsheet_text(texts[sample]))
    if not packet.text.equals(expected_text):
        raise ValueError('Private source text was changed')
    for field in REVIEW_FIELDS:
        packet[field] = packet[field].astype(str).str.strip()
    if not packet.status.isin(['pending', 'complete']).all():
        raise ValueError('Use status pending or complete')
    complete = packet[packet.status == 'complete']
    for _, row in complete.iterrows():
        if not row.reviewer or row.reviewer.casefold() in {'ai', 'chatgpt', 'codex', 'assistant', 'unknown', 'tbd'}:
            raise ValueError('A completed review requires the human reviewer identifier')
        if row.read_private_text != 'yes' or not row.observation or not row.hypothesis:
            raise ValueError('Completed reviews require private-text confirmation, observation and hypothesis')
        try:
            date.fromisoformat(row.reviewed_on)
        except ValueError as error:
            raise ValueError('Use a valid reviewed_on date YYYY-MM-DD') from error
        if row.error_category not in CATEGORIES or row.reference_assessment not in ASSESSMENTS:
            raise ValueError('Invalid error category or reference assessment')
    result = dict(selected_rows=len(cases), unique_comments=int(cases.sample_id.nunique()),
                  completed_rows=len(complete), pending_rows=len(packet) - len(complete),
                  reviewer_count=int(complete.reviewer.nunique()), audit_complete=len(complete) == len(cases),
                  completion_basis='Human-declared fields; the validator does not independently verify human identity or reading',
                  completed_categories={str(k): int(v) for k, v in complete.error_category.value_counts().items()})
    for kind in ('FN', 'FP'):
        key = kind.lower()
        selected = int(cases.selection_group.eq(f'core_{kind}').sum())
        result[f'selected_core_{key}'] = selected
        result[f'completed_core_{key}'] = int(complete.selection_group.eq(f'core_{kind}').sum())
        result[f'{key}_population_shortfall'] = max(0, 20 - selected)
    result['status'] = 'complete' if result['audit_complete'] else ('in_progress' if len(complete) else 'prepared_pending_human_reviews')
    return result


def preserve_packet(path, expected):
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
        cases = expected.drop(columns=['text', *REVIEW_FIELDS])
        old = align_packet(cases, old)
        if old.text.tolist() != expected.text.tolist():
            raise ValueError('Existing private packet text differs; refusing to replace human work')
        return
    expected.to_csv(path, index=False, encoding='utf-8-sig')


def private_directory(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / 'data'):
        raise ValueError('Private review text must stay under the git-ignored data directory')
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_once(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'Existing audit evidence differs: {path.name}; refusing to overwrite')
    else:
        path.write_bytes(content)


def execute(command, private):
    private = private_directory(private)
    study = Study()
    verify_inputs(study.output)
    study.verify_lock()
    values = read_predictions(study.output / 'predictions/test.npz', study.ids('test'))
    test = study.frame('test')
    if not np.array_equal(values['true_label_vector'], targets28(test)):
        raise ValueError('Test targets differ from authorized source')
    texts = dict(zip(test.id, test.text))
    output = study.output / 'human_audit'
    output.mkdir(exist_ok=True)
    cases, design = build_cases(values, read_json(study.output / 'thresholds.json'), targets28(study.frame('train')))
    design.update(protocol_id=study.config['protocol_id'], preparation_script_sha256=digest(__file__),
                  config_sha256=digest(study.config_path), test_predictions_sha256=digest(study.output / 'predictions/test.npz'),
                  pre_test_lock_sha256=digest(study.output / 'pre_test_lock.json'),
                  private_worksheet=str((private / 'review_private.csv').relative_to(ROOT)), raw_text_in_public_outputs=False)
    worksheet = private / 'review_private.csv'
    case_bytes = cases.to_csv(index=False).encode()
    design_bytes = (json.dumps(design, ensure_ascii=False, indent=2) + '\n').encode()
    if command == 'prepare':
        write_once(output / 'cases.csv', case_bytes)
        write_once(output / 'sampling_design.json', design_bytes)
        preserve_packet(worksheet, review_packet(cases, texts))
    elif (output / 'cases.csv').read_bytes() != case_bytes or (output / 'sampling_design.json').read_bytes() != design_bytes:
        raise ValueError('Audit sampling or source bindings changed')
    packet = pd.read_csv(worksheet, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    summary = validate_reviews(cases, packet, texts)
    summary.update(worksheet_sha256=digest(worksheet), cases_sha256=digest(output / 'cases.csv'),
                   sampling_design_sha256=digest(output / 'sampling_design.json'), inference_performed=False)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'check'])
    parser.add_argument('--private-dir', type=Path, default=ROOT / 'data/final28/human_audit')
    args = parser.parse_args()
    print(json.dumps(execute(args.command, args.private_dir), ensure_ascii=False, indent=2))
