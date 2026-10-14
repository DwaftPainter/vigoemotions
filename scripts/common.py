"""Shared 27-emotion data contract; never export raw comments."""
import ast
import hashlib
import importlib.metadata
import json
import platform
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, label_ranking_average_precision_score

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['amusement', 'excitement', 'joy', 'love', 'desire', 'optimism',
          'caring', 'pride', 'admiration', 'gratitude', 'relief', 'approval',
          'realization', 'surprise', 'curiosity', 'confusion', 'fear',
          'nervousness', 'remorse', 'embarrassment', 'disappointment',
          'sadness', 'grief', 'disgust', 'anger', 'annoyance', 'disapproval']


def parse_labels(value):
    labels = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(labels, list) or not labels:
        raise ValueError('Expected a nonempty list of label IDs')
    if any(type(x) is not int or not 0 <= x <= 27 for x in labels):
        raise ValueError('Label IDs must be integers in [0, 27]')
    return labels


def multihot(labels):
    # Project convention: neutral alone -> zero; mixed neutral -> retain emotions.
    result = np.zeros(27, dtype=np.int64)
    for label in set(parse_labels(labels)) - {27}:
        result[label] = 1
    return result


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()


def load_data():
    frames = {}
    for split in ('train', 'val', 'test'):
        df = pd.read_csv(ROOT / 'data' / f'{split}.csv', dtype=str)
        if list(df.columns) != ['id', 'text', 'labels'] or df.isna().any().any():
            raise ValueError(f'Invalid schema or missing values: {split}')
        df['label_ids'] = df.labels.map(parse_labels)
        df['normalized'] = df.text.map(normalize)
        if df.normalized.eq('').any():
            raise ValueError('Empty text after normalization')
        frames[split] = df
    all_ids = pd.concat([df.id for df in frames.values()])
    if all_ids.duplicated().any():
        raise ValueError('Duplicate IDs across splits')
    # Only compare text for leakage auditing; no test targets affect training.
    heldout = set(frames['val'].normalized) | set(frames['test'].normalized)
    excluded = frames['train'][frames['train'].normalized.isin(heldout)]
    frames['train'] = frames['train'][~frames['train'].normalized.isin(heldout)].copy()
    if set(frames['val'].normalized) & set(frames['test'].normalized):
        raise ValueError('Validation/test text overlap must be resolved before training')
    return frames, excluded


def targets(df):
    return np.stack(df.label_ids.map(multihot))


def scores(y, probabilities, threshold=0.5):
    pred = probabilities >= threshold
    return dict(macro_f1=float(f1_score(y, pred, average='macro', zero_division=0)),
                micro_f1=float(f1_score(y, pred, average='micro', zero_division=0)),
                hamming_loss=float(hamming_loss(y, pred)),
                lrap=float(label_ranking_average_precision_score(y, probabilities)))


def style_masks(df):
    # Heuristics, not a validated linguistic classifier; groups can overlap.
    # Python Unicode boundaries are stable across pandas object/Arrow string backends.
    teencode = re.compile(r'\b(?:ko|khum|hong|hok|k|kh|dc|đc|j|z|mn|mik|bn|ntn|vc|vl)\b', re.IGNORECASE)
    emoji = df.text.map(lambda s: sum(0x1F300 <= ord(c) <= 0x1FAFF or
                                      0x2600 <= ord(c) <= 0x27BF for c in s))
    return {'all': np.ones(len(df), dtype=bool),
            'short_le_5': df.normalized.str.split().str.len().le(5).to_numpy(),
            'emoji_heavy_ge_2': emoji.ge(2).to_numpy(),
            'teencode': df.normalized.map(lambda text: bool(teencode.search(text))).to_numpy()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def environment():
    names = ['numpy', 'pandas', 'scikit-learn', 'scipy', 'matplotlib',
             'torch', 'transformers', 'py-vncorenlp']
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'processor': platform.processor(), 'packages': versions,
            'data_sha256': {s: hashlib.sha256((ROOT/'data'/f'{s}.csv').read_bytes()).hexdigest()
                            for s in ('train', 'val', 'test')}}
