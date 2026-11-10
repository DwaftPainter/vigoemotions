"""Shared contracts for supplementary Q1 research; no training or test scoring."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, LABELS, normalize, parse_labels, targets, style_masks, write_json
from import_equal20_results import OUT as EVIDENCE, file_variant, validate_predictions

OUTPUT = ROOT / 'reports/q1_label_difficulty'
PRIVATE = ROOT / 'data/q1_label_difficulty'
FOCUS = ('gratitude', 'disapproval')
THRESHOLD = .5
SEED = 42


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def isolated_output(path):
    path = Path(path).resolve()
    if path == OUTPUT or OUTPUT in path.parents:
        path.mkdir(parents=True, exist_ok=True)
        return path
    if ROOT in path.parents or path == ROOT:
        raise ValueError('Research outputs must stay under reports/q1_label_difficulty; official artifacts are protected')
    path.mkdir(parents=True, exist_ok=True)  # permits a temporary independent reproduction
    return path


def private_directory(path=PRIVATE):
    path = Path(path).resolve()
    if ROOT/'data' not in path.parents:
        raise ValueError('Private review/text files must stay inside the git-ignored data directory')
    path.mkdir(parents=True, exist_ok=True)
    return path


def align_frame(source, fixed_ids):
    if list(source.columns) != ['id', 'text', 'labels'] or source.isna().any().any():
        raise ValueError('Expected complete id,text,labels source CSV; never print source rows')
    fixed_ids = list(fixed_ids)
    if source.id.duplicated().any() or len(set(fixed_ids)) != len(fixed_ids):
        raise ValueError('Duplicate source or fixed IDs')
    if not set(fixed_ids).issubset(set(source.id)):
        raise ValueError('Source is missing fixed split IDs')
    frame = source.set_index('id').loc[fixed_ids].reset_index()
    frame['label_ids'] = frame.labels.map(parse_labels)
    frame['normalized'] = frame.text.map(normalize)
    if frame.normalized.eq('').any():
        raise ValueError('Empty normalized source text')
    return frame


def validate_saved(saved, frame):
    return validate_predictions(saved, frame, 27)


def load_evidence(data_dir=ROOT/'data'):
    """Use existing fixed IDs: no test CSV or test predictions are read."""
    data_dir = Path(data_dir)
    manifest_path = EVIDENCE/'evidence/27/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    config = json.loads((ROOT/'configs/kaggle_equal_20ep.json').read_text())
    if manifest['profile'] != config or manifest['label_spaces'] != [27] or not manifest['validation_only']:
        raise ValueError('Expected the verified 27-output validation-only equal-20 study')
    for name in ('common.py','train_neural.py'):
        if digest_file(ROOT/'scripts'/name) != manifest['source_sha256'][name]:
            raise ValueError(f'Canonical preprocessing/inference source {name} differs from the Kaggle manifest')
    provenance = json.loads((EVIDENCE/'provenance.json').read_text())
    if provenance['historical_results_used'] or provenance['trained_locally']:
        raise ValueError('Historical/local studies are not permitted')
    frames, variants, files = {}, {}, [manifest_path, EVIDENCE/'provenance.json', ROOT/'configs/kaggle_equal_20ep.json',
                                     ROOT/'scripts/common.py', ROOT/'scripts/train_neural.py']
    for split in ('train', 'val'):
        csv_path, ids_path = data_dir/f'{split}.csv', ROOT/f'splits/{split}.csv'
        if not csv_path.is_file():
            raise FileNotFoundError(f'Authorized private {split}.csv required in --data-dir')
        variants[split] = file_variant(csv_path.read_bytes(), manifest['environment']['data_sha256'][split])
        ids = pd.read_csv(ids_path, dtype=str).id.tolist()
        if hashlib.sha256('\n'.join(ids).encode()).hexdigest() != manifest['split_ids_sha256'][split]:
            raise ValueError(f'Fixed {split} IDs differ from verified evidence')
        frames[split] = align_frame(pd.read_csv(csv_path, dtype=str), ids)
        files += [csv_path, ids_path]
    path = EVIDENCE/'evidence/27/phobert/val_predictions.npz'
    records = [r for r in provenance['studies']['27']['exported_members'] if r['member'].endswith('/phobert_val.npz')]
    if len(records) != 1 or digest_file(path) != records[0]['sha256']:
        raise ValueError('Saved PhoBERT-27 NPZ checksum differs from verified provenance')
    with np.load(path, allow_pickle=False) as saved:
        y, probabilities = validate_saved(saved, frames['val'])
        y, probabilities = y.copy(), probabilities.copy()
    run_path = EVIDENCE/'evidence/27/phobert/run.json'
    run_records = [r for r in provenance['studies']['27']['exported_members'] if r['member'].endswith('/phobert_run.json')]
    if len(run_records) != 1 or digest_file(run_path) != run_records[0]['sha256']:
        raise ValueError('Saved PhoBERT-27 run checksum differs from verified provenance')
    run = json.loads(run_path.read_text())
    if run['config'] != config or len(run['history']) != 20 or run['best_epoch'] != 14:
        raise ValueError('Unexpected frozen PhoBERT-27 run configuration/history')
    files += [path, run_path, EVIDENCE/'per_label_metrics.csv',
              EVIDENCE/'analysis/27_emotion27/phobert/tables/phobert_per_label.csv']
    sources = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): digest_file(p) for p in files}
    meta = dict(study='Kaggle equal20 / 27 outputs / PhoBERT', threshold=THRESHOLD, seed=SEED,
                label_order=LABELS, validation_only=True, sources_sha256=sources, csv_variants=variants,
                checkpoint_epoch=run['best_epoch'], model=config['phobert'],
                implementation_sha256={name:digest_file(ROOT/'scripts'/name) for name in
                    ('q1_difficulty_common.py','analyze_q1_difficulty.py','analyze_q1_annotations.py','evaluate_q1_interventions.py')},
                packages={name:importlib.metadata.version(name) for name in ('numpy','pandas','scipy','scikit-learn','matplotlib')})
    return frames, y, probabilities, meta


def ratio(a, b):
    return float(a/b) if b else None


def wilson(k, n):
    if not n:
        return None, None
    z = 1.959963984540054
    p = k/n; d = 1+z*z/n
    center = (p+z*z/(2*n))/d
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return float(max(0,center-half)), float(min(1,center+half))


def binary_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Expected aligned binary targets and finite probability scores')
    truth, pred = y.astype(bool), p >= THRESHOLD
    tp, fp = int((truth&pred).sum()), int((~truth&pred).sum())
    fn, tn = int((truth&~pred).sum()), int((~truth&~pred).sum())
    result = dict(n=len(y), positive_support=tp+fn, negative_support=fp+tn, predicted_positive=tp+fp,
                  tp=tp, fp=fp, fn=fn, tn=tn, precision=ratio(tp,tp+fp), recall=ratio(tp,tp+fn),
                  f1=ratio(2*tp,2*tp+fp+fn), prevalence=ratio(tp+fn,len(y)),
                  false_positive_rate=ratio(fp,fp+tn), false_negative_rate=ratio(fn,tp+fn))
    for name,k,n in [('precision',tp,tp+fp),('recall',tp,tp+fn),('prevalence',tp+fn,len(y)),('false_positive_rate',fp,fp+tn)]:
        result[name+'_ci_low'], result[name+'_ci_high'] = wilson(k,n)
    return result


def subgroup_metrics(y, p, mask):
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != np.asarray(y).shape:
        raise ValueError('Subgroup mask must align exactly with targets')
    return binary_metrics(np.asarray(y)[mask],np.asarray(p)[mask])


def f1_bootstrap(metrics, repeats=2000, seed=SEED):
    """Multinomial category resampling equals row bootstrap for a binary F1."""
    if metrics['positive_support'] < 5 or metrics['predicted_positive'] < 5 or metrics['n'] < 20:
        return dict(f1_ci_low=None,f1_ci_high=None,f1_bootstrap_valid=0)
    counts=np.array([metrics[k] for k in ('tp','fp','fn','tn')])
    draws=np.random.default_rng(seed).multinomial(metrics['n'],counts/metrics['n'],size=repeats)
    denominator=2*draws[:,0]+draws[:,1]+draws[:,2]
    values=2*draws[denominator>0,0]/denominator[denominator>0]
    if not len(values):
        return dict(f1_ci_low=None,f1_ci_high=None,f1_bootstrap_valid=0)
    low,high=np.quantile(values,[.025,.975])
    return dict(f1_ci_low=float(low),f1_ci_high=float(high),f1_bootstrap_valid=len(values))


def bootstrap_f1_gap(y, p, repeats=2000, seed=SEED):
    """Same sampled comments for both labels; preserves their dependence."""
    y,p=np.asarray(y,dtype=bool),np.asarray(p)
    if y.shape!=p.shape or y.ndim!=2 or y.shape[1]!=2 or len(y)==0:
        raise ValueError('Expected n×2 aligned paired label arrays')
    rng=np.random.default_rng(seed); values=[]; pred=p>=THRESHOLD
    for start in range(0,repeats,128):
        ids=rng.integers(0,len(y),size=(min(128,repeats-start),len(y)))
        t,b=y[ids],pred[ids];tp=(t&b).sum(axis=1)
        den=t.sum(axis=1)+b.sum(axis=1)
        valid=(den>0).all(axis=1)
        f1=2*tp[valid]/den[valid]
        values.extend((f1[:,0]-f1[:,1]).tolist())
    if not values:
        return dict(ci_low=None,ci_high=None,valid_replicates=0,unit='validation comment',repeats=repeats)
    low,high=np.quantile(values,[.025,.975])
    return dict(ci_low=float(low),ci_high=float(high),valid_replicates=len(values),unit='validation comment',repeats=repeats)


def outcome_masks(y,p):
    t=np.asarray(y).astype(bool);b=np.asarray(p)>=THRESHOLD
    return {'TP':t&b,'FP':~t&b,'FN':t&~b,'TN':~t&~b}


def stable_seed(name):
    return int(hashlib.sha256(f'{SEED}:{name}'.encode()).hexdigest()[:8],16)


def write_csv(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False)
