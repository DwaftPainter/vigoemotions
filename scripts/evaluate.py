"""Shared evaluation on validation probabilities; no threshold tuning on test."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vigo-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_recall_fscore_support

from common import ROOT, LABELS, targets, scores, style_masks, write_json


def save_evaluation(name, frames, probabilities, metadata):
    out = ROOT/'reports'
    for folder in ('tables', 'figures', 'predictions'):
        (out/folder).mkdir(parents=True, exist_ok=True)
    rows, label_rows, style_rows = [], [], []
    freq = targets(frames['train']).sum(axis=0)
    for split, p in probabilities.items():
        df = frames[split]
        y = targets(df)
        if p.shape != y.shape or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError('Invalid probability matrix')
        # Save scores BEFORE any threshold decoding, with IDs and fixed class ordering.
        np.savez_compressed(out/'predictions'/f'{name}_{split}.npz',
                            sample_id=df.id.to_numpy(dtype=str), true_label_vector=y,
                            probability_vector=p, labels=np.array(LABELS),
                            label_cardinality=y.sum(axis=1))
        rows.append(dict(model=name, split=split, threshold=.5, **scores(y,p), **metadata))
        precision, recall, f1, support = precision_recall_fscore_support(y, p >= .5, zero_division=0)
        for c, label in enumerate(LABELS):
            label_rows.append(dict(model=name, split=split, label=label, train_count=freq[c],
                support=support[c], precision=precision[c], recall=recall[c], f1=f1[c]))
        for style, mask in style_masks(df).items():
            if mask.any():
                style_rows.append(dict(model=name, split=split, style=style, n=int(mask.sum()),
                                       **scores(y[mask],p[mask])))
    pd.DataFrame(rows).to_csv(out/'tables'/f'{name}_metrics.csv', index=False)
    pd.DataFrame(label_rows).to_csv(out/'tables'/f'{name}_per_label.csv', index=False)
    pd.DataFrame(style_rows).to_csv(out/'tables'/f'{name}_styles.csv', index=False)
    y, p = targets(frames['val']), probabilities['val']
    sweep = pd.DataFrame([dict(threshold=float(t), **scores(y,p,t))
                          for t in np.round(np.arange(.05,.801,.05),2)])
    sweep.to_csv(out/'tables'/f'{name}_thresholds_val.csv', index=False)
    best = sweep.sort_values(['macro_f1','threshold'], ascending=[False,False]).iloc[0]
    f1 = f1_score(y,p >= .5, average=None, zero_division=0)
    rho = spearmanr(freq,f1)
    write_json(out/f'{name}_analysis.json', {'best_global_threshold_val':float(best.threshold),
        'best_global_macro_f1_val':float(best.macro_f1),
        'frequency_f1_spearman':float(rho.statistic), 'spearman_pvalue':float(rho.pvalue),
        'note':'Exploratory validation analysis; test not evaluated by default.'})
    # Compare only train-selected pairs, on validation: no test correlation matrix.
    train_y = targets(frames['train'])
    co = train_y.T @ train_y
    pairs = sorted([(int(co[a,b]),a,b) for a in range(27) for b in range(a+1,27)], reverse=True)[:10]
    pred = p >= .5
    pair_rows = []
    for n,a,b in pairs:
        actual, predicted = y[:,a].astype(bool)&y[:,b].astype(bool), pred[:,a]&pred[:,b]
        joint_tp = int((actual & predicted).sum())
        pair_rows.append(dict(label_a=LABELS[a],label_b=LABELS[b],train_count=n,
            val_true_count=int(actual.sum()),val_pred_count=int(predicted.sum()),
            joint_true_positive=joint_tp, joint_recall=joint_tp/max(1,int(actual.sum()))))
    pd.DataFrame(pair_rows).to_csv(out/'tables'/f'{name}_pair_preservation_val.csv', index=False)
    # Deterministic edge-case sample; no text or reconstructed snippets in exported artifacts.
    errors = []
    masks = style_masks(frames['val'])
    for kind, mask in [('FN',(y==1)&~pred),('FP',(y==0)&pred)]:
        coords = np.argwhere(mask)
        order = sorted(coords, key=lambda ij: float(p[tuple(ij)]), reverse=kind=='FP')[:20]
        for i,c in order:
            errors.append(dict(kind=kind, sample_id=frames['val'].iloc[i].id, label=LABELS[c],
                probability=float(p[i,c]), true_labels=';'.join(np.array(LABELS)[y[i].astype(bool)]),
                predicted_labels=';'.join(np.array(LABELS)[pred[i]]), cardinality=int(y[i].sum()),
                styles=';'.join(k for k,v in masks.items() if k!='all' and v[i])))
    pd.DataFrame(errors).to_csv(out/'tables'/f'{name}_error_cases_val.csv', index=False)
    cards = []
    for label, mask in [('0',y.sum(axis=1)==0),('1',y.sum(axis=1)==1),
                        ('2',y.sum(axis=1)==2),('3',y.sum(axis=1)==3),('>=4',y.sum(axis=1)>=4)]:
        if mask.any():
            cards.append(dict(cardinality=label,n=int(mask.sum()),**scores(y[mask],p[mask])))
    pd.DataFrame(cards).to_csv(out/'tables'/f'{name}_cardinality_val.csv', index=False)
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(sweep.threshold,sweep.macro_f1,'o-',label='Macro-F1')
    axes[0].plot(sweep.threshold,sweep.micro_f1,'o-',label='Micro-F1')
    axes[0].axvline(.5,color='gray',linestyle='--');axes[0].legend()
    axes[0].set(title='Validation threshold sensitivity',xlabel='Threshold',ylabel='F1')
    axes[1].scatter(freq,f1); axes[1].set_xscale('log')
    axes[1].set(title=f'Frequency vs F1: Spearman rho={rho.statistic:.3f}',
                xlabel='Training label count (log)',ylabel='Validation per-label F1')
    fig.tight_layout();fig.savefig(out/'figures'/f'{name}_diagnostics.png',dpi=150);plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))
