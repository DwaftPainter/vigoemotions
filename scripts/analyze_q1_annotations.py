"""Experiment B: blinded packets, independent-review validation and weighted analysis."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from common import ROOT, LABELS, write_json
from q1_difficulty_common import (OUTPUT, PRIVATE, FOCUS, SEED, load_evidence, isolated_output,
    private_directory, outcome_masks, stable_seed, digest_file, write_csv, ratio)

FIELDS=['review_id','target_label','annotator_id','target_presence','expression','semantic_overlap','overlap_labels','rationale_code','status']
ADJ_FIELDS=FIELDS+['resolution_method']
CHOICES={'target_presence':{'yes','no','uncertain'},'expression':{'explicit','implicit','ambiguous','not_present'},
         'semantic_overlap':{'none','possible','shared_cue','uncertain'},
         'rationale_code':{'direct_emotion_phrase','inference_from_context','negation_or_target','multiple_interpretations','insufficient_context','no_target_signal'}}
DECISIONS=['target_presence','expression','semantic_overlap','overlap_labels']


def blinded_packet(key):
    rows=key[['review_id','target_label']].copy()
    for field in FIELDS[2:]:rows[field]=''
    return rows[FIELDS]


def preserve_write(path,frame,packet=False):
    """Never overwrite actual reviews when preparing reproducible packets again."""
    path=Path(path)
    if path.exists():
        old=pd.read_csv(path,dtype=str,keep_default_na=False)
        if packet:
            if list(old.columns)!=list(frame.columns) or old[['review_id','target_label']].to_dict('records')!=frame[['review_id','target_label']].to_dict('records'):
                raise ValueError('Existing review packet differs; use a new private directory')
        elif path.read_bytes()!=frame.to_csv(index=False).encode():
            raise ValueError('Existing coordinator key differs; use a new private directory')
        return
    frame.to_csv(path,index=False)


def prepare(frames,y,p,private_dir,out,per_cell=12):
    private_dir=private_directory(private_dir);key=[];design=[]
    for label in FOCUS:
        c=LABELS.index(label)
        for outcome,mask in outcome_masks(y[:,c],p[:,c]).items():
            if outcome=='TN':continue  # this review intentionally concentrates on positive/error strata
            candidates=np.flatnonzero(mask)
            chosen=sorted(candidates,key=lambda i:hashlib.sha256(f'{SEED}:{label}:{outcome}:{frames["val"].iloc[i].id}'.encode()).hexdigest())[:per_cell]
            for i in chosen:
                sample_id=frames['val'].iloc[i].id
                review_id='r_'+hashlib.sha256(f'{SEED}:{label}:{sample_id}'.encode()).hexdigest()[:16]
                key.append(dict(review_id=review_id,sample_id=sample_id,target_label=label,outcome=outcome,
                    truth=int(y[i,c]),predicted=int(p[i,c]>=.5),population_n=len(candidates),sample_n=len(chosen),
                    inclusion_probability=len(chosen)/len(candidates)))
            design.append(dict(target_label=label,outcome=outcome,population_n=len(candidates),sample_n=len(chosen),
                               inclusion_probability=ratio(len(chosen),len(candidates))))
    key=pd.DataFrame(key).sort_values('review_id').reset_index(drop=True)
    packet=blinded_packet(key)
    preserve_write(private_dir/'annotation_key.csv',key)
    for name in ('annotator_1.csv','annotator_2.csv'):preserve_write(private_dir/name,packet,packet=True)
    adj=packet.copy();adj['resolution_method']=''
    preserve_write(private_dir/'adjudication.csv',adj,packet=True)
    # Coordinator-only ID map is separate from the reviewer text packet.
    text=key[['review_id','target_label']].copy()
    text['text']=key.sample_id.map(frames['val'].set_index('id').text)
    preserve_write(private_dir/'review_text_private.csv',text)
    write_csv(out/'sampling_design.csv',design)
    write_json(out/'study_status.json',dict(status='prepared_pending_human_reviews',seed=SEED,per_cell=per_cell,
        sample_n=len(key),unique_comments=int(key.sample_id.nunique()),sample_outcomes=['TP','FP','FN'],
        blinding='Only review_id/target_label and blank review fields in annotation packets; predictions/GT/key remain coordinator-only',
        no_completed_annotations_created=True,key_sha256=digest_file(private_dir/'annotation_key.csv'),
        review_text_sha256=digest_file(private_dir/'review_text_private.csv')))
    return key


def read_reviews(path,key,adjudication=False):
    frame=pd.read_csv(path,dtype=str,keep_default_na=False)
    if list(frame.columns)!=(ADJ_FIELDS if adjudication else FIELDS):
        raise ValueError('Unexpected review columns; do not include predictions, confidence, raw text or free-text rationales')
    if frame.review_id.duplicated().any() or set(frame.review_id)!=set(key.review_id):
        raise ValueError('Review IDs must match the complete blinded packet exactly')
    frame=frame.set_index('review_id').loc[key.review_id].reset_index()
    if frame.target_label.tolist()!=key.target_label.tolist():raise ValueError('Target labels differ from coordinator key')
    if not frame.status.isin(['','pending','complete']).all():raise ValueError('Unknown review status')
    frame['annotator_id']=frame.annotator_id.str.strip().str.casefold()
    for i,r in frame[frame.status=='complete'].iterrows():
        if not r.annotator_id.strip():raise ValueError('Completed reviews require a human reviewer identifier')
        for field,values in CHOICES.items():
            if r[field] not in values:raise ValueError(f'Invalid completed {field}')
        overlap=set(filter(None,r.overlap_labels.split(';')))
        if not overlap.issubset(set(LABELS)-{r.target_label}):raise ValueError('Unknown or self-overlapping label')
        frame.loc[i,'overlap_labels']=';'.join(label for label in LABELS if label in overlap)
        if (r.semantic_overlap=='none')!= (len(overlap)==0):
            if r.semantic_overlap!='uncertain':raise ValueError('Overlap category and label set disagree')
        if (r.target_presence=='no') != (r.expression=='not_present'):
            raise ValueError('Absence must use not_present; presence must not')
        if r.target_presence=='uncertain' and r.expression!='ambiguous':raise ValueError('Uncertain presence requires ambiguous expression')
        if adjudication and r.resolution_method not in ('third_review','consensus_discussion'):
            raise ValueError('Completed adjudication needs a declared human resolution method')
    return frame


def agreement_metrics(a,b):
    if len(a)!=len(b) or not len(a):return dict(n=0,agreement=None,kappa=None)
    a,b=np.asarray(a),np.asarray(b)
    kappa=float(cohen_kappa_score(a,b)) if len(set(a)|set(b))>1 else None
    return dict(n=len(a),agreement=float((a==b).mean()),kappa=kappa if kappa is None or np.isfinite(kappa) else None)


def expression_recall(rows,complete,repeats=2000):
    """GT-positive denominator only; correct outcome-stratified sampling weights."""
    result=[]
    if rows.empty:return result
    for label in FOCUS:
        positive=rows[(rows.target_label==label)&(rows.truth==1)]
        for expression in ('explicit','implicit','ambiguous','not_present'):
            group=positive[positive.expression==expression]
            weighted_n=float(group.weight.sum());weighted_tp=float((group.weight*group.predicted).sum())
            estimate=ratio(weighted_tp,weighted_n) if complete else None
            low=high=None
            if complete and len(group)>=5:
                rng=np.random.default_rng(stable_seed(label+expression));values=[]
                # Resample the actual design strata, not the post-hoc expression groups.
                strata=[g for _,g in positive.groupby('outcome',sort=True)]
                for _ in range(repeats):
                    sampled=pd.concat([g.iloc[rng.integers(0,len(g),len(g))] for g in strata])
                    subset=sampled[sampled.expression==expression]
                    if subset.weight.sum():values.append(float((subset.weight*subset.predicted).sum()/subset.weight.sum()))
                if values:low,high=map(float,np.quantile(values,[.025,.975]))
            result.append(dict(target_label=label,expression=expression,sample_positive_n=len(group),
                sample_tp=int(group.predicted.sum()),sample_fn=int((1-group.predicted).sum()),
                sample_recall=ratio(group.predicted.sum(),len(group)),
                estimated_population_positive=weighted_n if complete else None,weighted_recall=estimate,
                ci_low=low,ci_high=high,weighted_estimate_available=bool(complete)))
    return result


def validate_key(key,frames,y,p):
    if key.review_id.duplicated().any() or key[['sample_id','target_label']].duplicated().any():raise ValueError('Duplicate coordinator cases')
    index={sample:i for i,sample in enumerate(frames['val'].id)}
    for r in key.itertuples():
        if r.sample_id not in index or r.target_label not in FOCUS:raise ValueError('Unknown coordinator sample/target')
        i,c=index[r.sample_id],LABELS.index(r.target_label)
        outcome=next(name for name,mask in outcome_masks(y[:,c],p[:,c]).items() if mask[i])
        population=int(outcome_masks(y[:,c],p[:,c])[outcome].sum())
        n=int(((key.target_label==r.target_label)&(key.outcome==r.outcome)).sum())
        if outcome!=r.outcome or int(y[i,c])!=r.truth or int(p[i,c]>=.5)!=r.predicted or population!=r.population_n or n!=r.sample_n:
            raise ValueError('Coordinator outcomes/counts differ from saved evidence')
        if not np.isclose(r.inclusion_probability,n/population,rtol=0,atol=1e-12):raise ValueError('Invalid inclusion probability')


def analyze(key,a,b,adjudication,out,private_dir,repeats=2000):
    ids_a=set(a.loc[a.status=='complete','annotator_id']);ids_b=set(b.loc[b.status=='complete','annotator_id'])
    if len(ids_a)>1 or len(ids_b)>1 or ids_a&ids_b:raise ValueError('Use one distinct, independent annotator per review file')
    paired=(a.status=='complete')&(b.status=='complete')
    agreements=[dict(field=field,**agreement_metrics(a.loc[paired,field],b.loc[paired,field])) for field in ('expression','target_presence','semantic_overlap')]
    resolved=[];disagreements=[]
    for i,k in key.iterrows():
        if not paired[i]:continue
        if all(a.loc[i,f]==b.loc[i,f] for f in DECISIONS):
            review=a.loc[i];method='two_annotator_agreement'
        else:
            disagreements.append(k.review_id)
            if adjudication.loc[i,'status']!='complete':continue
            review=adjudication.loc[i];method=review.resolution_method
            if method=='third_review' and review.annotator_id in ids_a|ids_b:
                raise ValueError('A third-review adjudicator must differ from both independent annotators')
        resolved.append({**k.to_dict(),**{f:review[f] for f in DECISIONS},'resolution':method,'weight':1/k.inclusion_probability})
    frame=pd.DataFrame(resolved)
    complete=len(frame)==len(key) and len(key)>0
    previous=out/'annotation_status.json'
    if previous.exists() and json.loads(previous.read_text())['status']=='complete' and not complete:
        raise ValueError('Do not replace completed results with a pending study; use a new --output-dir')
    results=expression_recall(frame,complete,repeats)
    write_csv(out/'agreement.csv',agreements)
    if results:write_csv(out/'expression_recall.csv',results)
    elif (out/'expression_recall.csv').exists():(out/'expression_recall.csv').unlink()
    if not frame.empty:frame.to_csv(private_dir/'resolved_annotations_private.csv',index=False)
    overlap=[]
    if not frame.empty:
        for (label,outcome),g in frame.groupby(['target_label','outcome'],sort=True):
            for other in LABELS:
                if other==label:continue
                overlap.append(dict(target_label=label,outcome=outcome,other_label=other,n=len(g),
                    overlap_judgments=int(g.overlap_labels.map(lambda s:other in s.split(';')).sum())))
    if overlap:write_csv(out/'semantic_overlap_judgments.csv',overlap)
    elif (out/'semantic_overlap_judgments.csv').exists():(out/'semantic_overlap_judgments.csv').unlink()
    state=dict(status='complete' if complete else 'pending',planned_n=len(key),paired_completed=int(paired.sum()),
               resolved_n=len(frame),unresolved_n=len(key)-len(frame),disagreement_n=len(disagreements),
               weighted_population_estimates_available=complete,
               agreement_scope='Paired completed review items in an outcome-balanced sample, not population agreement',
               recall_scope='Original ground-truth positives only; FP cases never enter recall denominators',
               uncertainty='Outcome-stratified bootstrap with fixed inverse-inclusion weights; no finite-population correction; no causal inference')
    write_json(out/'annotation_status.json',state)
    (out/'annotation_analysis.md').write_text('# Q1 human annotation study\n\n'+
        ('**Pending human reviews/adjudication.** No completed study or causal result is claimed.\n' if not complete else '**Completed supplied human-review records analyzed.**\n')+
        f'\nPlanned items: {len(key)}; paired completed: {paired.sum()}; resolved: {len(frame)}. '
        'Cohen κ and raw agreement use paired completed items; homogeneous-category κ is undefined. '
        'Expression recall excludes every ground-truth negative. Balanced TP/FN sampling changes naive recall, so '
        'population estimates use inverse inclusion weights and are withheld until the full planned sample is resolved. '
        'Small expression groups have no bootstrap interval. Different labels may reference the same comment; do not pool them as independent observations. '
        'Semantic overlap judgments are human assessments, not proof of confusion. No annotations are synthesized.\n')
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','analyze'])
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--private-dir',type=Path,default=PRIVATE)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT/'annotations')
    parser.add_argument('--per-cell',type=int,default=12)
    args=parser.parse_args()
    if args.per_cell<1:parser.error('--per-cell must be positive')
    out=isolated_output(args.output_dir);private=private_directory(args.private_dir)
    frames,y,p,meta=load_evidence(args.data_dir)
    if args.mode=='prepare':
        key=prepare(frames,y,p,private,out,args.per_cell)
        print(f'Prepared {len(key)} blinded items for two independent annotators; all annotations pending.')
    else:
        key_path=private/'annotation_key.csv'
        if not key_path.exists():raise FileNotFoundError('Run prepare first; coordinator key and authorized human reviews are private dependencies')
        key=pd.read_csv(key_path,dtype={'sample_id':str})
        sampling=json.loads((out/'study_status.json').read_text())
        if sampling['key_sha256']!=digest_file(key_path):raise ValueError('Coordinator key changed after deterministic sampling')
        if sampling['review_text_sha256']!=digest_file(private/'review_text_private.csv'):
            raise ValueError('Blinded review text changed after packet creation')
        validate_key(key,frames,y,p)
        a=read_reviews(private/'annotator_1.csv',key);b=read_reviews(private/'annotator_2.csv',key)
        adj=read_reviews(private/'adjudication.csv',key,True)
        state=analyze(key,a,b,adj,out,private)
        print('Annotation analysis:',state['status'],'—',state['resolved_n'],'resolved of',state['planned_n'])
    inputs=['annotation_key.csv','annotator_1.csv','annotator_2.csv','adjudication.csv','review_text_private.csv']
    write_json(out/'provenance.json',dict(evidence=meta,private_input_sha256={name:digest_file(private/name) for name in inputs if (private/name).is_file()},
                                        raw_text_exported=False,completed_annotations_generated=False))


if __name__=='__main__':main()
