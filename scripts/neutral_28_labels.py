"""Isolated 28-label study. Never writes to the original 27-label artifacts."""
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from threadpoolctl import threadpool_limits

from common import ROOT, LABELS, normalize, parse_labels, scores, write_json

LABELS28 = [*LABELS, 'neutral']
MODELS = ('tfidf_lr', 'gru', 'phobert')
NAMES = {'tfidf_lr': 'TF-IDF + Logistic Regression', 'gru': 'BiGRU', 'phobert': 'PhoBERT'}
DEFAULT_OUTPUT = ROOT / 'reports/neutral_28_labels'
METRICS = ('macro_f1', 'micro_f1', 'hamming_loss', 'lrap')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def multihot28(labels):
    """Preserve original annotations, including mixed Neutral/emotion positives."""
    y = np.zeros(28, dtype=np.int64)
    y[list(set(parse_labels(labels)))] = 1
    return y


def targets28(frame):
    y = np.stack(frame.label_ids.map(multihot28))
    assert y.shape == (len(frame), 28)
    return y


def load_frames(data_dir):
    """Same checks/leakage rule as common.load_data, with an explicit data path.

    common.load_data has no path argument. Reuse its parser/normalizer and mirror
    its split-preserving audit without monkey-patching common.ROOT.
    """
    data_dir = Path(data_dir)
    frames, counts = {}, []
    for split in ('train', 'val', 'test'):
        path = data_dir / f'{split}.csv'
        if not path.is_file():
            raise FileNotFoundError(f'Missing required data: {path}')
        df = pd.read_csv(path, dtype=str)
        if list(df.columns) != ['id', 'text', 'labels'] or df.isna().any().any():
            raise ValueError(f'Invalid schema/missing values: {split}')
        df['label_ids'] = df.labels.map(parse_labels)
        df['normalized'] = df.text.map(normalize)
        if df.normalized.eq('').any():
            raise ValueError('Empty normalized text')
        frames[split] = df
        sets = df.label_ids.map(set)
        counts.append(dict(split=split, raw_samples=len(df),
            neutral_only=int(sets.map(lambda x: x == {27}).sum()),
            mixed_neutral=int(sets.map(lambda x: 27 in x and len(x) > 1).sum()),
            neutral_positive=int(sets.map(lambda x: 27 in x).sum())))
    if pd.concat([f.id for f in frames.values()]).duplicated().any():
        raise ValueError('Duplicate IDs across splits')
    heldout = set(frames['val'].normalized) | set(frames['test'].normalized)
    overlap = frames['train'].normalized.isin(heldout)
    excluded = frames['train'].loc[overlap].copy()
    frames['train'] = frames['train'].loc[~overlap].copy()
    if set(frames['val'].normalized) & set(frames['test'].normalized):
        raise ValueError('Validation/test text overlap')
    return frames, excluded, pd.DataFrame(counts)


def runtime(data_dir):
    packages = {}
    for name in ('numpy', 'pandas', 'scikit-learn', 'scipy', 'torch', 'transformers', 'py-vncorenlp'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(python=platform.python_version(), platform=platform.platform(), packages=packages,
        data_sha256={s: digest(Path(data_dir)/f'{s}.csv') for s in ('train', 'val', 'test')})


def output_path(path):
    path = Path(path).resolve()
    # All new artifacts stay in the new namespace, including optional named studies.
    if path != DEFAULT_OUTPUT.resolve() and DEFAULT_OUTPUT.resolve() not in path.parents:
        raise ValueError(f'Output must be inside {DEFAULT_OUTPUT}')
    return path


def study_config(base):
    config = deepcopy(base)
    config.update(label_count=28, neutral_id=27, threshold=0.5,
        neutral_policy='preserve_original_annotations',
        checkpoint_selection='Highest validation 28-label Macro-F1 at 0.5; ties keep earliest epoch')
    if base.get('label_count') != 27 or base.get('neutral_id') != 27 or base.get('threshold') != .5:
        raise ValueError('Expected a 27-label baseline config with Neutral ID 27 and threshold 0.5')
    return config


def init_study(out, data_dir, base_config, baseline_reports):
    out = output_path(out)
    frames, excluded, counts = load_frames(data_dir)
    for split, frame in frames.items():
        y = targets28(frame)
        neutral = frame.label_ids.map(lambda labels: 27 in labels).to_numpy(dtype=int)
        if not np.array_equal(y[:, 27], neutral):
            raise ValueError(f'{split}: Neutral annotation/target mismatch')
    config = study_config(read_json(base_config))
    identity = runtime(data_dir)
    identity['split_ids_sha256'] = {s: hashlib.sha256('\n'.join(f.id).encode()).hexdigest()
                                   for s, f in frames.items()}
    manifest = dict(config=config, baseline_config=read_json(base_config),
        baseline_reports=str(Path(baseline_reports).resolve()), environment=identity,
        labels=LABELS28, annotation_counts=counts.to_dict('records'))
    if out.exists():
        if not (out/'study.json').is_file() or read_json(out/'study.json') != manifest:
            raise FileExistsError('Different/incomplete study exists; choose a NEW named output directory')
        print('Existing identical study preserved:', out)
        return out
    # Compare fixed IDs when available; never rewrite the old split files.
    for split, frame in frames.items():
        old = ROOT/'splits'/f'{split}.csv'
        if old.is_file() and pd.read_csv(old, dtype=str).id.tolist() != frame.id.tolist():
            raise ValueError(f'{split}: IDs/order differ from the repository fixed split')
    out.mkdir(parents=True)
    write_json(out/'study.json', manifest)
    write_json(out/'config.json', config)
    counts.to_csv(out/'annotation_counts.csv', index=False)
    (out/'splits').mkdir()
    for split, frame in frames.items():
        frame[['id']].to_csv(out/'splits'/f'{split}.csv', index=False)
    excluded[['id']].to_csv(out/'splits/excluded_train.csv', index=False)
    pd.DataFrame(columns=['model', 'input_index', 'text', 'predicted_emotions', 'neutral_predicted',
                          'predicted_probabilities', 'top5']).to_csv(out/'prediction_examples.csv', index=False)
    print('Seed:', config['seed'], '| labels:', len(LABELS28), '| threshold: 0.5')
    print(counts.to_string(index=False))
    print('Excluded train overlaps:', len(excluded))
    return out


def checked_study(out, data_dir):
    out = output_path(out)
    study = read_json(out/'study.json')
    if study['labels'] != LABELS28 or read_json(out/'config.json') != study['config']:
        raise ValueError('Study config/label mapping changed; use a new study')
    frames, _, _ = load_frames(data_dir)
    if runtime(data_dir)['data_sha256'] != study['environment']['data_sha256']:
        raise ValueError('Source data checksum changed')
    for split, frame in frames.items():
        if frame.id.tolist() != pd.read_csv(out/'splits'/f'{split}.csv', dtype=str).id.tolist():
            raise ValueError('Split identity changed')
    return out, study, frames


def validate_probabilities(y, p):
    if (y.ndim != 2 or y.shape[1] != 28 or p.shape != y.shape or
            not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or
            (p < 0).any() or (p > 1).any()):
        raise ValueError('Expected binary targets and finite probabilities with 28 columns')


def save_evaluation(out, model, split, frame, probabilities):
    y, p = targets28(frame), np.asarray(probabilities)
    validate_probabilities(y, p)
    pred = p >= .5
    folder = out/'models'/model
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/f'{split}_predictions.npz', sample_id=frame.id.to_numpy(dtype=str),
        true_label_vector=y, probability_vector=p, labels=np.array(LABELS28))
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, zero_division=0)
    rows = [dict(model=model, split=split, label_id=i, label=label, precision=precision[i],
        recall=recall[i], f1=f1[i], support=int(support[i]), predicted_positive=int(pred[:, i].sum()))
        for i, label in enumerate(LABELS28)]
    pd.DataFrame(rows).to_csv(folder/f'{split}_per_class.csv', index=False)
    row = dict(model=model, split=split, label_count=28, threshold=.5, n=len(frame), **scores(y, p),
        **{f'emotion27_{k}': v for k, v in scores(y[:, :27], p[:, :27]).items()})
    pd.DataFrame([row]).to_csv(folder/f'{split}_metrics.csv', index=False)
    return row


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    # TF-IDF does not require torch. Neural dependencies are imported lazily.


def gru_inputs(texts, vocab, max_length):
    import torch
    sequences = [[vocab.get(w, 1) for w in normalize(t).lower().split()][:max_length] for t in texts]
    ids = torch.zeros((len(sequences), max_length), dtype=torch.long)
    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return ids, ids.ne(0).long()


def make_gru(config, vocab_size):
    import torch
    from train_neural import BiGRU
    model = BiGRU(vocab_size, config['embedding_dim'], config['hidden_dim'], config['dropout'])
    model.head = torch.nn.Linear(2*config['hidden_dim'], 28)
    assert model.head.out_features == 28
    return model


def train_tfidf(train, val, config, folder):
    seed_all(config['seed'])
    params = deepcopy(config['tfidf'])
    params['ngram_range'] = tuple(params['ngram_range'])
    vectorizer = TfidfVectorizer(**params)
    started = time.perf_counter()
    x = vectorizer.fit_transform(train.normalized)
    model = OneVsRestClassifier(LogisticRegression(**config['logistic_regression']), n_jobs=1)
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, targets28(train))
    elapsed = time.perf_counter()-started
    p = model.predict_proba(vectorizer.transform(val.normalized))
    assert len(model.estimators_) == 28 and p.shape[1] == 28
    joblib.dump(dict(vectorizer=vectorizer, model=model, labels=LABELS28), folder/'model.joblib')
    return p, dict(train_seconds=elapsed, device='cpu', features=len(vectorizer.vocabulary_),
        parameter_count=sum(getattr(e, 'coef_', np.array([])).size +
                            getattr(e, 'intercept_', np.array([])).size for e in model.estimators_),
        time_definition='TF-IDF fit + OvR fit; excludes data loading/evaluation')


def train_neural28(name, train, val, config, folder, segmenter_dir, allow_cpu=False):
    from collections import Counter
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm.auto import tqdm
    from train_neural import logits, predict, segmented_texts

    seed_all(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    elif not allow_cpu:
        raise RuntimeError('Enable a GPU runtime, or explicitly allow slow CPU training')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = config[name]
    frames = dict(train=train, val=val)  # Test never enters preprocessing or epoch selection.
    prep_started = time.perf_counter()
    if name == 'gru':
        words = Counter(w for t in train.normalized for w in t.lower().split())
        vocab = {w: i+2 for i, (w, _) in enumerate(words.most_common(cfg['max_vocab']-2))}
        write_json(folder/'vocab.json', vocab)
        model = make_gru(cfg, len(vocab)+2)
        features = {s: gru_inputs(f.normalized.tolist(), vocab, cfg['max_length']) for s, f in frames.items()}
    else:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        text = segmented_texts(frames, Path(segmenter_dir))
        tokenizer = AutoTokenizer.from_pretrained(cfg['checkpoint'], revision=cfg['revision'], use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(cfg['checkpoint'], revision=cfg['revision'],
            num_labels=28, problem_type='multi_label_classification',
            id2label=dict(enumerate(LABELS28)), label2id={v: i for i, v in enumerate(LABELS28)})
        assert model.config.num_labels == 28 and model.classifier.out_proj.out_features == 28
        tokenizer.save_pretrained(folder/'tokenizer')
        features = {}
        for split, texts in text.items():
            encoded = tokenizer(texts, padding='max_length', truncation=True,
                                max_length=cfg['max_length'], return_tensors='pt')
            features[split] = (encoded['input_ids'], encoded['attention_mask'])
    assert all(p.requires_grad for p in model.parameters()), 'Expected full backbone + head fine-tuning'
    loaders = {}
    for split, (ids, mask) in features.items():
        y = torch.tensor(targets28(frames[split]), dtype=torch.float32)
        loaders[split] = DataLoader(TensorDataset(ids, mask, y), batch_size=cfg['batch_size'],
            shuffle=split == 'train', num_workers=0,
            generator=torch.Generator().manual_seed(config['seed']))
    prep_seconds = time.perf_counter()-prep_started
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                 weight_decay=cfg.get('weight_decay', 0.0))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best, best_epoch, history = -1.0, None, []
    started = time.perf_counter()
    print('Device:', device, '| seed:', config['seed'], '| outputs: 28', flush=True)
    for epoch in range(1, cfg['epochs']+1):
        model.train()
        total = 0.0
        for ids, mask, y in tqdm(loaders['train'], desc=f'{name} epoch {epoch}/{cfg["epochs"]}'):
            optimizer.zero_grad(set_to_none=True)
            output = logits(model, name, ids.to(device), mask.to(device))
            assert output.shape == y.shape and output.shape[1] == 28
            loss = loss_fn(output, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()*len(y)
        p = predict(model, name, loaders['val'], device)
        row = dict(epoch=epoch, loss=total/len(train), **scores(targets28(val), p))
        history.append(row)
        pd.DataFrame(history).to_csv(folder/'history.csv', index=False)
        print(row, flush=True)
        if row['macro_f1'] > best:
            best, best_epoch = row['macro_f1'], epoch
            torch.save(model.state_dict(), folder/'best.pt')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    model.load_state_dict(torch.load(folder/'best.pt', map_location=device, weights_only=True))
    if name == 'phobert':
        model.save_pretrained(folder/'pretrained')  # Local reload without downloading the backbone again.
        (folder/'best.pt').unlink()  # Only redundant NEW-study weights; never original artifacts.
    p = predict(model, name, loaders['val'], device)
    return p, dict(history=history, best_epoch=best_epoch, train_seconds=seconds, prep_seconds=prep_seconds,
        device=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
        parameter_count=sum(p.numel() for p in model.parameters()),
        trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        time_definition='Training loop + per-epoch validation/checkpoint writes; excludes preprocessing',
        deterministic_note='Fixed seeds/cuDNN; exact equality across runtimes/GPUs is not guaranteed')


def train_model(out, data_dir, name, segmenter_dir=None, allow_cpu=False):
    out, study, frames = checked_study(out, data_dir)
    if (out/'test_lock.json').exists():
        raise RuntimeError('Test protocol locked; training is disabled for this study')
    folder = out/'models'/name
    if folder.exists():
        if (folder/'run.json').is_file():
            run = read_json(folder/'run.json')
            if run['config'] != study['config']:
                raise ValueError('Completed model config mismatch')
            load_predictor(out, name, segmenter_dir)  # Confirm checkpoint can actually reload.
            print('SKIP completed model; preserved:', name)
            return
        raise FileExistsError(f'Incomplete run preserved: {folder}. Use a NEW study; no mid-epoch resume.')
    folder.mkdir(parents=True)
    config = study['config']
    if name == 'tfidf_lr':
        p, meta = train_tfidf(frames['train'], frames['val'], config, folder)
    else:
        if name == 'phobert':
            ensure_segmenter(segmenter_dir or out/'resources/vncorenlp')
        p, meta = train_neural28(name, frames['train'], frames['val'], config, folder,
            segmenter_dir or out/'resources/vncorenlp', allow_cpu)
    save_evaluation(out, name, 'val', frames['val'], p)
    write_json(folder/'run.json', dict(model=name, config=config, labels=LABELS28,
        environment=runtime(data_dir), selection_split='val', threshold=.5, **meta))
    summarize(out)


def ensure_segmenter(path):
    import py_vncorenlp
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    if not (path/'VnCoreNLP-1.2.jar').is_file():
        py_vncorenlp.download_model(save_dir=str(path))
    return path


def load_predictor(out, name, segmenter_dir=None):
    """Load this study's saved artifacts; no test targets or training involved."""
    out = output_path(out)
    folder = out/'models'/name
    run = read_json(folder/'run.json')
    if run['labels'] != LABELS28 or run['config']['label_count'] != 28:
        raise ValueError('Checkpoint label mapping mismatch')
    cfg = run['config'].get(name, {})
    if name == 'tfidf_lr':
        bundle = joblib.load(folder/'model.joblib')  # Load only your own trusted checkpoint.
        if bundle['labels'] != LABELS28 or len(bundle['model'].estimators_) != 28:
            raise ValueError('TF-IDF checkpoint must have 28 outputs')
        def predictor(texts):
            return bundle['model'].predict_proba(bundle['vectorizer'].transform([normalize(t) for t in texts]))
        return predictor
    import torch
    from train_neural import logits
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'gru':
        vocab = read_json(folder/'vocab.json')
        model = make_gru(cfg, len(vocab)+2)
        model.load_state_dict(torch.load(folder/'best.pt', map_location=device, weights_only=True))
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model = AutoModelForSequenceClassification.from_pretrained(folder/'pretrained', local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(folder/'tokenizer', use_fast=False, local_files_only=True)
        if model.config.num_labels != 28 or [model.config.id2label[i] for i in range(28)] != LABELS28:
            raise ValueError('PhoBERT checkpoint must have 28 ordered labels')
        segmenter = None  # JVM starts lazily, only once for this predictor.
    model.to(device).eval()
    def predictor(texts):
        nonlocal segmenter
        texts = [normalize(t) for t in texts]
        if not texts or any(not t for t in texts):
            raise ValueError('Enter at least one nonempty comment')
        rows = []
        for start in range(0, len(texts), cfg['batch_size']):
            batch = texts[start:start+cfg['batch_size']]
            if name == 'gru':
                ids, mask = gru_inputs(batch, vocab, cfg['max_length'])
            else:
                if segmenter is None:
                    import py_vncorenlp
                    segmenter = py_vncorenlp.VnCoreNLP(annotators=['wseg'],
                        save_dir=str(ensure_segmenter(segmenter_dir or out/'resources/vncorenlp')))
                segmented = [' '.join(segmenter.word_segment(t)) for t in batch]
                encoded = tokenizer(segmented, padding='max_length', truncation=True,
                                    max_length=cfg['max_length'], return_tensors='pt')
                ids, mask = encoded['input_ids'], encoded['attention_mask']
            with torch.no_grad():
                rows.append(torch.sigmoid(logits(model, name, ids.to(device), mask.to(device))).cpu().numpy())
        p = np.concatenate(rows)
        validate_probabilities(np.zeros((len(texts), 28), dtype=int), p)
        return p
    # Define a cell for the closure even for GRU, which never uses it.
    if name == 'gru':
        segmenter = None
    return predictor


def prediction_tables(texts, probabilities):
    validate_probabilities(np.zeros((len(texts), 28), dtype=int), probabilities)
    summary, detail = [], []
    for i, (text, p) in enumerate(zip(texts, probabilities)):
        predicted = np.flatnonzero(p >= .5)
        top = np.argsort(-p, kind='stable')[:5]
        summary.append(dict(input_index=i, text=text,
            predicted_emotions=';'.join(LABELS28[j] for j in predicted if j != 27),
            neutral_predicted=bool(p[27] >= .5),
            predicted_probabilities=json.dumps({LABELS28[j]: float(p[j]) for j in predicted}),
            top5=json.dumps({LABELS28[j]: float(p[j]) for j in top})))
        for j in sorted(set(predicted) | set(top), key=lambda j: -p[j]):
            detail.append(dict(input_index=i, text=text, label_id=int(j), label=LABELS28[j],
                probability=float(p[j]), predicted=bool(p[j] >= .5), top5=bool(j in top)))
    return pd.DataFrame(summary), pd.DataFrame(detail)


def infer_examples(out, texts, segmenter_dir=None):
    """Only caller-supplied examples are exported with text; never dataset comments."""
    rows, details = [], []
    for name in MODELS:
        if not (Path(out)/'models'/name/'run.json').exists():
            continue
        predictor = load_predictor(out, name, segmenter_dir)
        summary, detail = prediction_tables(texts, predictor(texts))
        rows.append(summary.assign(model=name))
        details.append(detail.assign(model=name))
        del predictor
    if not rows:
        raise ValueError('No completed models available for inference')
    summary, detail = pd.concat(rows, ignore_index=True), pd.concat(details, ignore_index=True)
    summary.to_csv(Path(out)/'prediction_examples.csv', index=False)
    detail.to_csv(Path(out)/'prediction_example_probabilities.csv', index=False)
    return summary, detail


def find_validation_cases(out, data_dir, display_text=False):
    """Measured categories, not promised predictions. Export IDs only; no raw data."""
    out, _, frames = checked_study(out, data_dir)
    rows = []
    for name in MODELS:
        path = out/'models'/name/'val_predictions.npz'
        if not (out/'models'/name/'run.json').exists():
            continue
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['sample_id'], frames['val'].id.to_numpy(dtype=str))
            np.testing.assert_array_equal(saved['true_label_vector'], targets28(frames['val']))
            np.testing.assert_array_equal(saved['labels'], LABELS28)
            validate_probabilities(saved['true_label_vector'], saved['probability_vector'])
            pred = saved['probability_vector'] >= .5
            emotions = pred[:, :27].sum(axis=1)
            masks = {'neutral_only': pred[:, 27] & (emotions == 0),
                     'single_emotion': ~pred[:, 27] & (emotions == 1),
                     'multi_emotion': ~pred[:, 27] & (emotions >= 2),
                     'neutral_and_emotion': pred[:, 27] & (emotions >= 1)}
            for kind, mask in masks.items():
                indices = np.flatnonzero(mask)
                row = dict(model=name, category=kind, available_count=len(indices),
                           sample_id=str(saved['sample_id'][indices[0]]) if len(indices) else '')
                rows.append(row)
                print(name, kind, ':', row)
                if len(indices) and display_text:
                    # Private runtime display only. Never saved to the report CSV.
                    index = indices[0]
                    text = frames['val'].set_index('id').loc[row['sample_id'], 'text']
                    print('Private input:', text)
                    # Saved validation probabilities already correspond to this exact ID/target.
                    # Reusing them avoids restarting the VnCoreNLP JVM after inference.
                    probability = saved['probability_vector'][index:index+1]
                    print(prediction_tables([text], probability)[1].to_string(index=False))
    result = pd.DataFrame(rows)
    result.to_csv(out/'validation_prediction_cases.csv', index=False)
    return result


def baseline_row(out, study, name):
    """Import real 27-label scores; verify comparable IDs/targets when NPZ exists."""
    base = Path(study['baseline_reports'])
    row = dict(model=NAMES[name], model_key=name, label_count=27, split='val', threshold=.5,
               status='missing', source=str(base), conditions='No recorded 27-label validation metrics')
    table = base/'tables'/f'{name}_metrics.csv'
    if not table.exists():
        return row
    df = pd.read_csv(table)
    df = df[(df['split'] == 'val') & (df['threshold'] == .5)]
    if len(df) != 1:
        row['conditions'] = 'No unique baseline validation row at threshold 0.5'
        return row
    row.update(status='recorded', source=str(table), **{k: float(df.iloc[0][k]) for k in METRICS})
    row.update({f'emotion27_{k}': row[k] for k in METRICS})
    conditions = ['27-label metrics exclude Neutral; 28-label Macro-F1 has a different label set']
    run_path = base/f'{name}_run.json'
    if run_path.exists():
        old = read_json(run_path)
        old_cfg, new_cfg = old['config'], study['config']
        keys = ('tfidf', 'logistic_regression', 'seed') if name == 'tfidf_lr' else (name, 'seed')
        differences = [key for key in keys if old_cfg.get(key) != new_cfg.get(key)]
        conditions.append('Model settings match' if not differences else 'DIFFERENT settings: '+', '.join(differences))
        conditions.append('Baseline checkpoint selected by 27-label Macro-F1; new checkpoint by 28-label Macro-F1')
        new_run_path = out/'models'/name/'run.json'
        if new_run_path.exists() and old.get('device') != read_json(new_run_path).get('device'):
            conditions.append('Different or unverified hardware/device')
        for key in ('python', 'platform', 'packages', 'data_sha256'):
            if old.get('environment', {}).get(key) != study['environment'].get(key):
                conditions.append(f'Runtime/source difference: {key}')
    else:
        conditions.append('Baseline run metadata missing; hyperparameters/runtime unverified')
    new = out/'models'/name/'val_predictions.npz'
    archive = base/'predictions'/f'{name}_val.npz'
    if new.exists() and archive.exists():
        with np.load(new, allow_pickle=False) as n, np.load(archive, allow_pickle=False) as b:
            match = (b['labels'].tolist() == LABELS and
                np.array_equal(n['sample_id'], b['sample_id']) and
                np.array_equal(n['true_label_vector'][:, :27], b['true_label_vector']))
            if match:
                measured = scores(b['true_label_vector'], b['probability_vector'])
                match = all(np.isclose(measured[k], row[k], atol=1e-9, rtol=0) for k in METRICS)
            conditions.append('IDs, emotion targets and recorded metrics verified' if match else
                              'NOT PAIRED: baseline archive IDs/targets/metrics mismatch')
            row['paired_validation_verified'] = bool(match)
    else:
        conditions.append('Paired archive check unavailable')
        row['paired_validation_verified'] = False
    row['conditions'] = '; '.join(conditions)
    return row


def summarize(out):
    out = output_path(out)
    study = read_json(out/'study.json')
    metrics, classes, comparison = [], [], []
    for name in MODELS:
        comparison.append(baseline_row(out, study, name))
        row = dict(model=NAMES[name], model_key=name, label_count=28, split='val', threshold=.5,
                   status='not_run', source=str(out/'models'/name), conditions='Primary: preserve original annotations')
        folder = out/'models'/name
        if (folder/'run.json').exists():
            for split in ('val', 'test'):
                table = folder/f'{split}_metrics.csv'
                if not table.exists():
                    continue
                measured = pd.read_csv(table)
                metrics.append(measured)
                classes.append(pd.read_csv(folder/f'{split}_per_class.csv'))
                if split == 'val':
                    row.update(status='completed', **{k: measured.iloc[0][k] for k in
                               (*METRICS, *(f'emotion27_{k}' for k in METRICS))})
        comparison.append(row)
    metric_columns = ['model', 'split', 'label_count', 'threshold', 'n', *METRICS,
                      *(f'emotion27_{k}' for k in METRICS)]
    class_columns = ['model', 'split', 'label_id', 'label', 'precision', 'recall', 'f1', 'support', 'predicted_positive']
    summary = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame(columns=metric_columns)
    per_class = pd.concat(classes, ignore_index=True) if classes else pd.DataFrame(columns=class_columns)
    neutral = per_class[per_class.label_id == 27].copy()
    summary.to_csv(out/'metrics_summary.csv', index=False)
    per_class.to_csv(out/'per_class_metrics.csv', index=False)
    neutral.to_csv(out/'neutral_analysis.csv', index=False)
    compare = pd.DataFrame(comparison)
    compare.to_csv(out/'comparison_27_vs_28.csv', index=False)
    # Per-emotion paired changes, only when baseline archives actually match.
    changes = []
    for name in MODELS:
        old = next(r for r in comparison if r['model_key'] == name and r['label_count'] == 27)
        if not old.get('paired_validation_verified'):
            continue
        with np.load(Path(study['baseline_reports'])/'predictions'/f'{name}_val.npz', allow_pickle=False) as b:
            old_f1 = precision_recall_fscore_support(b['true_label_vector'], b['probability_vector'] >= .5,
                                                   zero_division=0)[2]
        new = per_class[(per_class.model == name) & (per_class.split == 'val')].set_index('label_id')
        for i, label in enumerate(LABELS):
            changes.append(dict(model=name, label_id=i, label=label, f1_27=old_f1[i],
                                f1_28=float(new.loc[i, 'f1']), delta_f1=float(new.loc[i, 'f1'])-old_f1[i]))
    pd.DataFrame(changes, columns=['model', 'label_id', 'label', 'f1_27', 'f1_28', 'delta_f1']).to_csv(
        out/'emotion27_per_class_comparison.csv', index=False)
    write_report(out, study, summary, neutral, compare)
    print(summary.to_string(index=False) if len(summary) else 'No 28-label models trained yet.')
    return summary, neutral, compare


def markdown_table(df):
    if df.empty:
        return '_Chưa có kết quả đo._'
    def value(x):
        return f'{x:.6f}' if isinstance(x, (float, np.floating)) else str(x).replace('|', '/')
    return '\n'.join(['| '+' | '.join(df.columns)+' |', '| '+' | '.join(['---']*len(df.columns))+' |',
                      *['| '+' | '.join(value(v) for v in row)+' |' for row in df.itertuples(index=False, name=None)]])


def write_report(out, study, summary, neutral, comparison):
    conclusions = []
    for name in MODELS:
        old = comparison[(comparison.model_key == name) & (comparison.label_count == 27)].iloc[0]
        new = summary[(summary.model == name) & (summary.split == 'val')]
        if new.empty:
            conclusions.append(f'- {NAMES[name]}: chưa chạy 28 nhãn; không có kết luận về cải thiện.')
        elif old.status == 'recorded' and old.get('paired_validation_verified', False):
            delta = float(new.iloc[0].emotion27_macro_f1)-float(old.macro_f1)
            conclusions.append(f'- {NAMES[name]}: delta Macro-F1 trên cùng 27 cảm xúc = {delta:+.6f}; '
                               'đọc các khác biệt điều kiện trong bảng so sánh trước khi quy nguyên nhân cho Neutral.')
        else:
            conclusions.append(f'- {NAMES[name]}: có kết quả 28 nhãn nhưng chưa xác minh được đối chiếu paired với baseline.')
    text = f'''# Thí nghiệm Neutral 28 nhãn

## 1. Mục đích và giao thức
Thêm Neutral (ID 27) vào 27 cảm xúc IDs 0–26. Không thay thế thí nghiệm 27 nhãn.
Seed {study['config']['seed']}; ngưỡng cố định 0.5. Giữ split và quy tắc loại train trùng heldout của project.
Văn bản NFC/chuẩn hóa khoảng trắng như `common.normalize`; TF-IDF và neural giữ hyperparameters từ config baseline.
BiGRU {study['config']['gru']['epochs']} epoch; PhoBERT {study['config']['phobert']['epochs']} epoch.
Chọn epoch bằng validation Macro-F1 trên 28 nhãn; hòa chọn epoch sớm nhất.
Đổi tập nhãn chọn checkpoint là hệ quả của 28 đầu ra và được nêu rõ khi so sánh baseline 27 nhãn.

## 2. Mã hóa Neutral và bất nhất annotation
Vector multi-hot có đúng 28 phần tử. [27] → chỉ phần tử 27 bằng 1;
[2,27] → cả joy và neutral bằng 1. ID lặp được gộp, annotation gốc không bị viết lại.
Theo guideline do người dùng cung cấp, Neutral không đi cùng cảm xúc, nhưng dữ liệu được báo có
725 mẫu Neutral-only và 306 mẫu mixed. Thí nghiệm chính giữ mọi nhãn gốc, không giả định nhãn nào sai.
Số liệu thực đọc từ CSV (trước loại train overlap):

{markdown_table(pd.DataFrame(study['annotation_counts']))}

## 3. Kết quả ba mô hình (số đo thật, ô trống nghĩa là chưa chạy)
{markdown_table(summary)}

## 4. So sánh 27 và 28 nhãn
Macro/Micro-F1, Hamming và LRAP trên 27 nhãn và 28 nhãn có tập đầu ra khác nhau, không coi delta đó
là bằng chứng trực tiếp tốt hơn. Các cột emotion27_* chiếu dự đoán 28 nhãn về cùng 27 cảm xúc.
Baseline được đọc từ artifact thật; không dựng lại kết quả thiếu. Baseline chưa có test không được so với test 28 nhãn.

{markdown_table(comparison)}

## 5. Neutral precision, recall, F1 và số lượng
Support là số mẫu có Neutral thật, bao gồm mixed; predicted_positive là số mẫu được dự đoán Neutral.
Pipeline 27 nhãn không có đầu ra Neutral, nên Neutral metric baseline là không xác định; không tự gán bằng 0.

{markdown_table(neutral)}

## 6. Ảnh hưởng đến 27 cảm xúc
{chr(10).join(conclusions)}

Xem emotion27_per_class_comparison.csv (chỉ tạo delta khi xác minh ID/target/metric baseline)
và per_class_metrics.csv cho F1 đủ 28 nhãn. Không chỉ dựa việc có thể dự đoán Neutral để kết luận tốt hơn.

## 7. Inference và các loại dự đoán
Inference nạp model đã lưu, không hardcode kết quả. prediction_examples.csv chỉ chứa comment ví dụ do người dùng
nhập; validation_prediction_cases.csv chỉ chứa ID và số ca tìm được. Nếu model không dự đoán loại Neutral-only,
single-emotion, multi-emotion hoặc mixed nào thì báo không tìm thấy, không dựng ví dụ.

## 8. Giới hạn
Một seed/split; annotation mixed có thể nhiễu hoặc phản ánh mơ hồ; LRAP 27 nhãn của project cho all-zero target
điểm 1, còn Neutral-only 28 nhãn có target dương nên LRAP toàn bộ không trực tiếp tương đương.
TF-IDF OvR fit độc lập nên thêm nhãn Neutral dự kiến không đổi 27 classifier cảm xúc khi cùng môi trường/config;
neural chia sẻ backbone nên có thể cải thiện hoặc xung đột gradient, cần số đo để đánh giá.
Không có validation loss trong pipeline gốc nên không kết luận overfitting chỉ từ train loss.
Khác GPU/package/dữ liệu/epoch được ghi trong conditions; checksum CSV khác có thể chỉ do định dạng,
vì vậy kiểm tra thêm ID/target trước khi so sánh. Không bảo đảm bitwise-identical giữa runtime.
Test chỉ đánh giá qua bước khóa riêng, không dùng fit/chọn epoch/ngưỡng. Sau test không retrain trong study này.
Checkpoint lưu trọng số để inference, không có optimizer state để resume giữa epoch.
Toàn bộ output/checkpoint/vocabulary cần giữ private. Không xuất comment nguồn từ dataset.
'''
    (out/'experiment_report.md').write_text(text, encoding='utf-8')


def evaluate_test(out, data_dir, segmenter_dir=None):
    out, study, frames = checked_study(out, data_dir)
    if not all((out/'models'/name/'run.json').exists() for name in MODELS):
        raise ValueError('Complete all three models and freeze choices before evaluating test')
    # Lock before reading any test targets for scoring; prevents subsequent retraining.
    fingerprints = {p.relative_to(out).as_posix(): digest(p) for name in MODELS
        for p in (out/'models'/name).rglob('*') if p.is_file() and
        (p.suffix in ('.pt', '.joblib', '.safetensors') or p.name in ('run.json', 'config.json', 'vocab.json'))}
    lock = dict(threshold=.5, selected_by='validation only', model_artifacts=fingerprints,
                data_sha256=study['environment']['data_sha256'])
    if (out/'test_lock.json').exists() and read_json(out/'test_lock.json') != lock:
        raise ValueError('Locked model artifacts changed; test comparison refused')
    write_json(out/'test_lock.json', lock)
    for name in MODELS:
        if (out/'models'/name/'test_metrics.csv').exists():
            print('Preserving existing test result:', name)
            continue
        predictor = load_predictor(out, name, segmenter_dir)
        save_evaluation(out, name, 'test', frames['test'], predictor(frames['test'].normalized.tolist()))
        del predictor
    summarize(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'train', 'report', 'test', 'examples'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--base-config', type=Path, default=ROOT/'configs/neutral28_12ep_base.json')
    parser.add_argument('--baseline-reports', type=Path, default=ROOT/'reports')
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--vncorenlp-dir', type=Path)
    parser.add_argument('--allow-cpu', action='store_true')
    args = parser.parse_args()
    if args.action == 'init':
        init_study(args.output, args.data_dir, args.base_config, args.baseline_reports)
        summarize(args.output)
    elif args.action == 'train':
        if args.model is None:
            parser.error('train requires --model')
        train_model(args.output, args.data_dir, args.model, args.vncorenlp_dir, args.allow_cpu)
    elif args.action == 'report':
        summarize(args.output)
    elif args.action == 'test':
        evaluate_test(args.output, args.data_dir, args.vncorenlp_dir)
    else:
        find_validation_cases(args.output, args.data_dir)


if __name__ == '__main__':
    main()
