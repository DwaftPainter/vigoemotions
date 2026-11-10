"""Experiment A: aggregate PhoBERT-27 validation diagnostics, never source text."""
import argparse
import os
from pathlib import Path
import re

os.environ.setdefault('MPLCONFIGDIR','/tmp/vigo-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT, LABELS, targets, style_masks, write_json
from q1_difficulty_common import (OUTPUT, EVIDENCE, FOCUS, SEED, load_evidence, isolated_output,
    binary_metrics, subgroup_metrics, f1_bootstrap, bootstrap_f1_gap, outcome_masks, stable_seed, write_csv)

REFERENCE = {'gratitude':dict(train_count=765,positive_support=96,tp=77,fp=11,fn=19,precision=.8750,recall=.8021,f1=.8370),
             'disapproval':dict(train_count=969,positive_support=123,tp=48,fp=94,fn=75,precision=.3380,recall=.3902,f1=.3623)}
# Pre-specified public lexical probes, not tokens/vocabulary extracted from private data.
CUES = {'gratitude':['cảm ơn','cám ơn','biết ơn','đội ơn','đa tạ'],
        'disapproval':['phản đối','không đồng ý','không chấp nhận','không tán thành']}
RELATED = {'gratitude':['admiration','love','approval','caring'],
           'disapproval':['anger','annoyance','disgust','disappointment','approval']}


def groups(frame, y, label):
    tokens=frame.normalized.str.split().str.len().to_numpy();card=y.sum(axis=1)
    styles=style_masks(frame)
    for low,high,name in [(1,6,'1–5'),(6,12,'6–11'),(12,25,'12–24'),(25,50,'25–49'),(50,np.inf,'≥50')]:
        yield 'whitespace_tokens',name,(tokens>=low)&(tokens<high)
    for value in (0,1,2,3):yield 'emotion_cardinality',str(value),card==value
    yield 'emotion_cardinality','≥4',card>=4
    for name in ('teencode','emoji_heavy_ge_2','short_le_5'):
        yield name,'yes',styles[name]
        yield name,'no',~styles[name]
    text=frame.normalized.str.casefold()
    cue=text.map(lambda s:any(re.search(r'(?<!\w)'+re.escape(c)+r'(?!\w)',s) for c in CUES[label])).to_numpy()
    for family,mask in [('public_lexical_probe',cue),
                        ('negation_probe',text.str.contains(r'(?<!\w)(?:không|chưa|chẳng|chả|đừng)(?!\w)',regex=True).to_numpy()),
                        ('question_mark',text.str.contains('?',regex=False).to_numpy()),
                        ('exclamation_mark',text.str.contains('!',regex=False).to_numpy()),
                        ('url_or_html',text.str.contains(r'https?://|www\.|<[^>]+>',regex=True).to_numpy())]:
        yield family,'yes',mask
        yield family,'no',~mask


def diagnostics(frames,y,p,repeats=2000):
    train_counts=targets(frames['train']).sum(axis=0)
    published=pd.read_csv(EVIDENCE/'per_label_metrics.csv')
    metrics,subgroups,probabilities,overlaps,features,train_probes=[],[],[],[],[],[]
    tokens=frames['val'].normalized.str.split().str.len().to_numpy()
    characters=frames['val'].normalized.str.len().to_numpy();card=y.sum(axis=1)
    for label in FOCUS:
        c=LABELS.index(label);r=binary_metrics(y[:,c],p[:,c]);r['train_count']=int(train_counts[c])
        for key,expected in REFERENCE[label].items():
            actual=round(r[key],4) if key in ('precision','recall','f1') else r[key]
            if actual!=expected:raise ValueError(f'Verified reference mismatch for {label}/{key}')
        row=published[(published.label_count==27)&(published.model=='phobert')&(published.label==label)].iloc[0]
        for key in ('precision','recall','f1'):
            if not np.isclose(r[key],row[key],atol=1e-12,rtol=0):raise ValueError('Published metric mismatch')
        metrics.append(dict(label=label,**r,**f1_bootstrap(r,repeats,stable_seed(label))))
        for family,name,mask in groups(frames['val'],y,label):
            s=subgroup_metrics(y[:,c],p[:,c],mask)
            subgroups.append(dict(label=label,family=family,group=name,**s,
                                  **f1_bootstrap(s,repeats,stable_seed(label+family+name))))
        train_y=targets(frames['train'])
        for family,name,mask in groups(frames['train'],train_y,label):
            if family=='public_lexical_probe':
                train_probes.append(dict(label=label,probe=name,n=int(mask.sum()),positive_support=int(train_y[mask,c].sum()),
                                         all_train_positive=int(train_counts[c])))
        for outcome,mask in outcome_masks(y[:,c],p[:,c]).items():
            values=p[mask,c]
            quantiles=np.quantile(values,[0,.1,.25,.5,.75,.9,1]).tolist() if len(values) else [None]*7
            probabilities.append(dict(label=label,outcome=outcome,n=int(mask.sum()),mean=float(values.mean()) if len(values) else None,
                **dict(zip(('minimum','q10','q25','median','q75','q90','maximum'),quantiles)),
                near_threshold=int(((values>=.4)&(values<=.6)).sum()),very_low=int((values<.1).sum()),very_high=int((values>=.9).sum())))
            for name,v in [('tokens',tokens),('characters',characters),('emotion_cardinality',card)]:
                features.append(dict(label=label,outcome=outcome,feature=name,n=int(mask.sum()),
                    mean=float(v[mask].mean()) if mask.any() else None,median=float(np.median(v[mask])) if mask.any() else None))
            for other in LABELS:
                if other==label:continue
                j=LABELS.index(other);true=y[:,j].astype(bool);pred=p[:,j]>=.5
                overlaps.append(dict(label=label,outcome=outcome,other_label=other,n=int(mask.sum()),
                    other_true=int((mask&true).sum()),other_predicted=int((mask&pred).sum()),
                    other_predicted_without_gt=int((mask&pred&~true).sum()),
                    other_true_and_predicted=int((mask&true&pred).sum()),
                    prespecified_related=other in RELATED[label]))
    focus=[LABELS.index(label) for label in FOCUS]
    gap=bootstrap_f1_gap(y[:,focus],p[:,focus],repeats,SEED)
    gap.update(point_difference=metrics[0]['f1']-metrics[1]['f1'],direction='Gratitude minus Disapproval',
               note='Paired comment bootstrap; conditional on this selected checkpoint and validation sample')
    return dict(metrics=metrics,subgroups=subgroups,probabilities=probabilities,overlaps=overlaps,
                outcome_features=features,train_lexical_probes=train_probes),gap


def figures(out,y,p,tables):
    folder=out/'figures';folder.mkdir(exist_ok=True)
    colors={'TP':'#007F86','FP':'#B55B40','FN':'#A57925','TN':'#657A90'}
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharex=True)
    for label,ax in zip(FOCUS,axes):
        c=LABELS.index(label)
        for name,mask in outcome_masks(y[:,c],p[:,c]).items():
            if mask.any():ax.hist(p[mask,c],bins=np.linspace(0,1,21),density=True,histtype='step',linewidth=1.7,
                                  label=f'{name} (n={mask.sum()})',color=colors[name])
        ax.axvline(.5,color='black',linestyle='--',linewidth=1)
        ax.set(title=label.capitalize(),xlabel='Sigmoid score (not calibrated confidence)',ylabel='Within-outcome density')
        ax.legend(fontsize=8,loc='upper center')
    fig.tight_layout();fig.savefig(folder/'probability_distributions.png',dpi=160);plt.close(fig)
    sub=pd.DataFrame(tables['subgroups'])
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharey=True)
    for row,label in enumerate(FOCUS):
        for column,family in enumerate(('whitespace_tokens','emotion_cardinality')):
            selected=sub[(sub.label==label)&(sub.family==family)]
            ax=axes[row,column];values=selected.f1.to_numpy(dtype=float)
            ax.bar(np.arange(len(selected)),values,color='#007F86')
            for x,(_,v) in enumerate(selected.iterrows()):
                if pd.notna(v.f1_ci_low):ax.plot([x,x],[v.f1_ci_low,v.f1_ci_high],color='#142D46',linewidth=2)
                if pd.isna(v.f1):ax.text(x,.025,'N/A',ha='center',fontsize=9,color='#657A90')
            ax.set_xticks(np.arange(len(selected)),[f'{v.group}\nn={v.n}, +={v.positive_support}' for v in selected.itertuples()],fontsize=9)
            ax.set_xlim(-.6,len(selected)-.4)
            ax.set(title=f'{label.capitalize()}: {family.replace("_"," ")}',ylabel='Binary label F1',ylim=(0,1))
    fig.suptitle('Validation diagnostics; + is target-positive support; intervals are conditional bootstrap estimates')
    fig.tight_layout();fig.savefig(folder/'subgroup_f1.png',dpi=160);plt.close(fig)


def markdown_table(rows,columns):
    def display(value):
        if value is None or (isinstance(value,float) and not np.isfinite(value)):return 'N/A'
        return f'{value:.4f}' if isinstance(value,float) else str(value)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |']+
                     ['| '+' | '.join(display(row.get(c)) for c in columns)+' |' for row in rows])


def research_report(out,tables,gap,meta):
    m=tables['metrics'];probs=tables['probabilities'];sub=tables['subgroups']
    observations=[]
    for label in FOCUS:
        fp=next(r for r in probs if r['label']==label and r['outcome']=='FP')
        fn=next(r for r in probs if r['label']==label and r['outcome']=='FN')
        observations.append(f"- **{label}:** {fp['very_high']}/{fp['n']} false positives score ≥0.9; "
                            f"{fn['very_low']}/{fn['n']} false negatives score <0.1. "
                            f"Only {fp['near_threshold']}/{fp['n']} FP and {fn['near_threshold']}/{fn['n']} FN lie in [0.4, 0.6].")
    candidate_rows=[]
    for label in FOCUS:
        rows=[r for r in tables['overlaps'] if r['label']==label and r['outcome']=='FN' and r['other_predicted_without_gt']>0]
        candidate_rows+=sorted(rows,key=lambda r:(-r['other_predicted_without_gt'],r['other_label']))[:3]
    lines=['# Q1: Gratitude versus Disapproval — supplementary research', '',
        '## Scope and reproducibility','',
        'Experiment A is completed on the saved PhoBERT **27-output** validation predictions at **threshold 0.5**. '
        'The backbone is unchanged: 20 epochs trained, checkpoint epoch 14 selected in the original Kaggle run. '
        'No inference, retraining, threshold selection, test predictions or official-report updates are performed. '
        'Experiments B and C are **pending actual human reviews and verified interventions**.', '',
        'Sources and SHA-256 checksums are recorded in `provenance.json`; fixed train/validation ID order and NPZ label order/targets are verified using the repository contracts. '
        'Only train supplies training-frequency statistics. Raw CSVs are read locally but never copied to public outputs.', '',
        '## Reproduced reference metrics','',markdown_table(m,['label','train_count','positive_support','tp','fp','fn','tn','precision','recall','f1']), '',
        f"The F1 gap is **{100*gap['point_difference']:.2f} percentage points**, with a paired-comment bootstrap 95% interval "
        f"[{100*gap['ci_low']:.2f}, {100*gap['ci_high']:.2f}] pp. Disapproval has more training positives (969 versus 765), "
        'so a simple lower-frequency explanation is insufficient for this pair. This does not isolate linguistic difficulty, annotation quality or a causal mechanism.', '',
        'Precision uses TP/(TP+FP); recall uses TP/(TP+FN); prevalence uses positives/all comments. '
        'F1 is 2TP/(2TP+FP+FN). Undefined denominators are N/A, not zero. The reference values match both published tables and saved predictions.', '',
        '## Error scores and threshold proximity','',markdown_table(probs,['label','outcome','n','median','q10','q90','near_threshold','very_low','very_high']), '',
        'Near-threshold means 0.4 ≤ score ≤ 0.6; very low is <0.1; very high is ≥0.9. These are descriptive score bands, not tuned decision thresholds. '
        'A sigmoid score is not a calibrated probability of correctness. FP and FN are shown separately; their denominators differ.', '',
        *observations, '',
        'Thus errors are not confined to a narrow band near 0.5. This observation does not establish whether an error comes from wording, semantics, annotation or missing context.', '',
        '![Within-outcome probability densities](figures/probability_distributions.png)', '',
        '## H3: text characteristics (observational)','',
        'Bins and lexical probes are fixed in the script; this supplementary exploratory study is not preregistered. `tables/subgroups.csv` includes **every** group, sample size, target-positive support, '
        'predicted-positive count, TP/FP/FN/TN, prevalence, precision/recall/F1 and uncertainty intervals. Length and cardinality bins partition their own families; '
        'style, punctuation, negation and public lexical probes overlap and must not be treated as independent samples.', '',
        markdown_table([r for r in sub if r['family'] in ('teencode','emoji_heavy_ge_2','public_lexical_probe')],
                       ['label','family','group','n','positive_support','predicted_positive','precision','recall','f1']), '',
        '![Length and cardinality diagnostics](figures/subgroup_f1.png)', '',
        'Cardinality uses the 27-emotion ground-truth vector and excludes Neutral. Style heuristics reuse `common.style_masks`; whitespace-token length is not BPE length. '
        'Subgroup differences do not control for other characteristics and are not evidence of causality.', '',
        'Outcome-specific length/cardinality summaries provide a separate view of false positives and false negatives. '
        'These are characteristics conditional on each outcome, not error probabilities for a characteristic.', '',
        markdown_table([r for r in tables['outcome_features'] if r['outcome'] in ('TP','FP','FN') and r['feature']!='characters'],
                       ['label','outcome','feature','n','mean','median']), '',
        '## H1 and H2: hypotheses requiring human evidence','',
        'The public lexical probes are a small, pre-specified proxy, not an explicit/implicit classifier. Cue absence does not imply implicit emotion; '
        'negation, quotations and targets can invalidate a cue. `tables/train_lexical_probes.csv` reports their training coverage separately. '
        'The label-specific cue lists are unequal, incomplete and not validated for sensitivity, so their coverage is not an unbiased cross-label explicitness comparison. '
        'Only the blinded annotation study can assess the intended categories; no completed human annotations are claimed.', '',
        '`tables/overlaps.csv` reports, separately within TP/FP/FN/TN, how often every other label is in the ground truth, predicted, '
        'predicted without that other label in ground truth, or both true and predicted. A correct prediction of another co-occurring emotion is **not** automatically semantic confusion. '
        'Even an unsupported other-label prediction is only an error pattern; human review is needed to identify ambiguity, annotation issues or a shared cue.', '',
        'The following are the three most frequent **unsupported** other-label predictions within each target’s FN set, not a single-label confusion matrix. '
        'Rows overlap; n is the target-FN denominator, not the count of true instances of the other emotion.', '',
        markdown_table(candidate_rows,['label','other_label','n','other_true','other_predicted_without_gt','other_true_and_predicted']), '',
        '## Uncertainty and interpretation','',
        'Precision, recall, prevalence and false-positive rate receive 95% Wilson intervals with their own denominators. F1 uses 2,000 ordinary nonparametric '
        'bootstrap replicates (or the explicitly recorded CLI count), seed 42/deterministic derived seeds. Binary-category multinomial resampling is equivalent to '
        'row resampling for single-label F1. F1 intervals are omitted when n<20, positive support<5 or predicted positives<5; point estimates and denominators remain visible. '
        'The label-gap bootstrap samples the **same comments for both labels**. These intervals condition on one selected checkpoint and this validation sample; '
        'they do not include training-seed variation, annotation uncertainty, repeated validation selection or multiple subgroup comparisons. No significance test or test-set improvement is claimed.', '',
        '## Study status and implications for Final','',
        '- **Observed:** the verified label gap, different FP/FN counts, score distributions, subgroup patterns and label overlaps.',
        '- **H1 unresolved:** explicitness must be human-annotated; lexical-probe associations are insufficient.',
        '- **H2 unresolved:** overlap diagnostics do not establish semantic confusion.',
        '- **H3 descriptive evidence only:** measured error associations are available; confounding, small groups and overlapping masks prevent causal claims.',
        '- **H4 pending:** no verified paraphrase intervention has been evaluated.',
        '- **Final:** use the proposed blinded review to prioritize errors and then evaluate human-verified, paired wording changes with fixed weights. '
        'Do not revise the official midterm results or silently replace its threshold based on these diagnostics.', '',
        '## Reproduce','',
        '```bash\n.venv/bin/python scripts/analyze_q1_difficulty.py\n```', '',
        'See `experiments/q1_label_difficulty/README.md` for the annotation and optional intervention commands. '
        'Private authorized train/val CSVs and the verified evidence directory are required; no test CSV is needed.', '',
        '## Method sources','',
        '- [NIST: binomial confidence intervals and Wilson score intervals](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbici.htm).',
        '- [Cohen (1960): A coefficient of agreement for nominal scales](https://doi.org/10.1177/001316446002000104). '
        'Agreement is reserved for actual independent reviews, not generated labels.', '']
    (out/'q1_analysis.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    parser.add_argument('--bootstrap-repeats',type=int,default=2000)
    args=parser.parse_args()
    if args.bootstrap_repeats<200:parser.error('Use at least 200 bootstrap replicates (default 2000)')
    frames,y,p,meta=load_evidence(args.data_dir)
    out=isolated_output(args.output_dir);tables,gap=diagnostics(frames,y,p,args.bootstrap_repeats)
    for name,rows in tables.items():write_csv(out/'tables'/f'{name}.csv',rows)
    meta.update(bootstrap_repeats=args.bootstrap_repeats,public_lexical_probes=CUES,
                related_label_hypotheses=RELATED,f1_gap=gap,annotation_status='pending',intervention_status='pending')
    write_json(out/'provenance.json',meta)
    figures(out,y,p,tables);research_report(out,tables,gap,meta)
    print('Q1 diagnostics complete: verified Gratitude/Disapproval metrics; aggregate-only outputs:',out)


if __name__=='__main__':main()
