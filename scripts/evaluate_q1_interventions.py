"""Experiment C: human-verified, frozen-checkpoint paired wording interventions."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from common import ROOT, LABELS, normalize, write_json
from q1_difficulty_common import (OUTPUT, PRIVATE, EVIDENCE, FOCUS, SEED, load_evidence,
    isolated_output, private_directory, digest_file, stable_seed, write_csv, ratio)

TRANSFORMS=('lexical_cue_removal','explicit_to_implicit','neutral_control')
PAIR_FIELDS=['pair_id','original_sample_id','paraphrase_id','target_label','transformation_type','original_sha256','paraphrase_sha256','status',
             'verifier_1','reviewer_type_1','semantic_equivalence_1','label_preserved_1',
             'verifier_2','reviewer_type_2','semantic_equivalence_2','label_preserved_2']


def verified_pairs(frame,truth_by_id):
    if list(frame.columns)!=PAIR_FIELDS:raise ValueError('Use the ID-only intervention schema; raw text belongs in the private text file')
    if frame.pair_id.duplicated().any() or frame.paraphrase_id.duplicated().any():raise ValueError('Duplicate pair/paraphrase IDs')
    for r in frame.itertuples():
        if not r.pair_id or not r.paraphrase_id:raise ValueError('Each pair needs stable IDs')
        if r.original_sample_id not in truth_by_id or r.target_label not in FOCUS:
            raise ValueError('Unknown validation original or target emotion')
        if not truth_by_id[r.original_sample_id][LABELS.index(r.target_label)]:
            raise ValueError('Intervention originals must be ground-truth positive for the preserved target')
        if r.transformation_type not in TRANSFORMS or r.status not in ('proposed','verified','rejected'):
            raise ValueError('Unknown transformation/status')
        if r.status=='verified':
            if not r.verifier_1.strip() or not r.verifier_2.strip() or r.verifier_1.strip().casefold()==r.verifier_2.strip().casefold():
                raise ValueError('Verified pairs require two distinct human verifiers')
            if any(getattr(r,f'reviewer_type_{i}')!='human' or getattr(r,f'semantic_equivalence_{i}')!='yes'
                   or getattr(r,f'label_preserved_{i}')!='yes' for i in (1,2)):
                raise ValueError('Both humans must verify semantic equivalence and target-label preservation; AI generation is insufficient')
            if any(len(getattr(r,key))!=64 or any(c not in '0123456789abcdef' for c in getattr(r,key)) for key in ('original_sha256','paraphrase_sha256')):
                raise ValueError('Verified pairs must bind human approvals to original/paraphrase SHA-256 hashes; run bind first')
    return frame[frame.status=='verified'].sort_values('pair_id').reset_index(drop=True)


def text_hash(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def validate_text_bindings(pairs,payload,frame):
    if list(payload.columns)!=['paraphrase_id','text'] or payload.paraphrase_id.duplicated().any():
        raise ValueError('Private paraphrases require unique paraphrase_id,text rows')
    texts=payload.set_index('paraphrase_id').text;originals=frame.set_index('id').normalized
    for r in pairs.itertuples():
        if r.paraphrase_id not in texts or not normalize(texts[r.paraphrase_id]):raise ValueError('Verified paraphrase text is missing')
        if text_hash(originals[r.original_sample_id])!=r.original_sha256 or text_hash(texts[r.paraphrase_id])!=r.paraphrase_sha256:
            raise ValueError('Text changed after human verification; bind and obtain new human reviews')


def bind_texts(pairs,payload,frame):
    if list(payload.columns)!=['paraphrase_id','text'] or payload.paraphrase_id.duplicated().any():raise ValueError('Invalid private text schema')
    texts=payload.set_index('paraphrase_id').text;originals=frame.set_index('id').normalized
    result=pairs.copy()
    for i,r in pairs.iterrows():
        if r.paraphrase_id not in texts or not normalize(texts[r.paraphrase_id]):continue
        original_hash,changed_hash=text_hash(originals[r.original_sample_id]),text_hash(texts[r.paraphrase_id])
        if original_hash!=r.original_sha256 or changed_hash!=r.paraphrase_sha256:
            result.loc[i,['original_sha256','paraphrase_sha256','status']]=[original_hash,changed_hash,'proposed']
            for field in PAIR_FIELDS[8:]:result.loc[i,field]=''
    return result


def verify_original_scores(ids,values,saved,atol=1e-4):
    values=np.asarray(values)
    if values.shape!=(len(ids),27) or not np.isfinite(values).all() or ((values<0)|(values>1)).any():
        raise ValueError('Invalid original probability matrix')
    if any(sample not in saved for sample in ids):raise ValueError('Unknown original prediction IDs')
    expected=np.stack([saved[sample] for sample in ids])
    if not np.allclose(values,expected,rtol=0,atol=atol):
        raise ValueError('Original scores do not reproduce saved PhoBERT-27 predictions (absolute tolerance 1e-4); check checkpoint/preprocessing/environment')


def mean_interval(values,repeats=2000,seed=SEED):
    values=np.asarray(values,dtype=float)
    if len(values)<2:return None,None
    rng=np.random.default_rng(seed)
    estimates=values[rng.integers(0,len(values),size=(repeats,len(values)))].mean(axis=1)
    return tuple(map(float,np.quantile(estimates,[.025,.975])))


def paired_summary(original,changed,original_ids,repeats=2000):
    original,changed=np.asarray(original,dtype=float),np.asarray(changed,dtype=float)
    if original.ndim!=1 or original.shape!=changed.shape or len(original_ids)!=len(original) or not len(original):
        raise ValueError('Paired scores and original IDs must align and be nonempty')
    if not np.isfinite(original).all() or not np.isfinite(changed).all() or ((original<0)|(original>1)|(changed<0)|(changed>1)).any():
        raise ValueError('Invalid paired scores')
    before,after=original>=.5,changed>=.5
    df=pd.DataFrame(dict(original_id=original_ids,delta=changed-original,flip=before!=after))
    clusters=df.groupby('original_id',sort=True)[['delta','flip']].mean()
    shift_low,shift_high=mean_interval(clusters.delta,repeats)
    flip_low,flip_high=mean_interval(clusters.flip,repeats)
    return dict(n_pairs=len(df),n_originals=len(clusters),mean_probability_shift=float(clusters.delta.mean()),
                probability_shift_ci_low=shift_low,probability_shift_ci_high=shift_high,
                flip_count=int((before!=after).sum()),flip_rate=float(clusters.flip.mean()),
                flip_ci_low=flip_low,flip_ci_high=flip_high,
                pair_weighted_probability_shift=float((changed-original).mean()),
                pair_weighted_flip_rate=float((before!=after).mean()),
                loss_count=int((before&~after).sum()),loss_denominator_pairs=int(before.sum()),
                positive_to_negative_rate_pairs=ratio((before&~after).sum(),before.sum()),
                gain_count=int((~before&after).sum()),gain_denominator_pairs=int((~before).sum()),
                negative_to_positive_rate_pairs=ratio((~before&after).sum(),(~before).sum()),
                primary_unit='original comment; variants averaged before bootstrap')


def prepare(frames,y,private_dir,per_label=8):
    from analyze_q1_annotations import preserve_write
    rows=[]
    for label in FOCUS:
        positive=np.flatnonzero(y[:,LABELS.index(label)])
        selected=sorted(positive,key=lambda i:hashlib.sha256(f'{SEED}:{label}:{frames["val"].iloc[i].id}'.encode()).hexdigest())[:per_label]
        for i in selected:
            sample=frames['val'].iloc[i].id
            for transformation in TRANSFORMS:
                token=hashlib.sha256(f'{SEED}:{label}:{sample}:{transformation}'.encode()).hexdigest()[:16]
                row={field:'' for field in PAIR_FIELDS}
                row.update(pair_id='p_'+token,original_sample_id=sample,paraphrase_id='v_'+token,
                           target_label=label,transformation_type=transformation,status='proposed',
                           original_sha256=text_hash(frames['val'].iloc[i].normalized))
                rows.append(row)
    pairs=pd.DataFrame(rows,columns=PAIR_FIELDS).sort_values('pair_id').reset_index(drop=True)
    preserve_write(private_dir/'intervention_pairs.csv',pairs)
    texts=pairs[['paraphrase_id']].copy();texts['text']=''
    preserve_write(private_dir/'intervention_text_private.csv',texts)
    originals=pairs[['original_sample_id','target_label']].drop_duplicates().copy()
    originals['text']=originals.original_sample_id.map(frames['val'].set_index('id').text)
    preserve_write(private_dir/'intervention_originals_private.csv',originals)
    return pairs


def verify_checkpoint(path):
    provenance=json.loads((EVIDENCE/'provenance.json').read_text())['studies']['27']
    archive_path=ROOT/provenance['zip']
    if not archive_path.is_file() or not Path(path).is_file():
        raise FileNotFoundError('The private 27-label Kaggle ZIP and extracted PhoBERT best.pt are required; neither is published')
    if digest_file(archive_path)!=provenance['zip_sha256']:raise ValueError('Private Kaggle ZIP checksum mismatch')
    expected=hashlib.sha256()
    with ZipFile(archive_path) as archive,archive.open('labels27/phobert/best.pt') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):expected.update(block)
    actual=digest_file(path)
    if actual!=expected.hexdigest():raise ValueError('Checkpoint bytes differ from the verified 27-label Kaggle best.pt')
    return actual


def infer(pairs,payload,frames,y,p,checkpoint,vncorenlp_dir,device_name):
    if checkpoint is None or vncorenlp_dir is None or not vncorenlp_dir.is_dir():
        raise FileNotFoundError('Verified pairs also require --checkpoint best.pt and --vncorenlp-dir with authorized segmentation assets')
    checkpoint_sha=verify_checkpoint(checkpoint)
    if list(payload.columns)!=['paraphrase_id','text'] or payload.paraphrase_id.duplicated().any():
        raise ValueError('Private paraphrases require unique paraphrase_id,text rows')
    selected=payload.set_index('paraphrase_id').reindex(pairs.paraphrase_id)
    if selected.text.isna().any() or selected.text.map(normalize).eq('').any():
        raise ValueError('Every verified pair requires a completed private paraphrase text')
    try:
        import torch
        from torch.utils.data import DataLoader,TensorDataset
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        from train_neural import segmented_texts,predict
    except ImportError:
        raise RuntimeError('Inference requires the original PyTorch/Transformers/py-vncorenlp environment; see requirements-colab.txt') from None
    config=json.loads((ROOT/'configs/kaggle_equal_20ep.json').read_text());settings=config['phobert']
    device=torch.device(device_name or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA device is unavailable')
    torch.manual_seed(SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    originals=sorted(set(pairs.original_sample_id))
    val=frames['val'].set_index('id')
    raw=[val.loc[sample,'normalized'] for sample in originals]+selected.text.map(normalize).tolist()
    batch_frames={'train':pd.DataFrame({'normalized':pd.Series([],dtype=str)}),'val':pd.DataFrame({'normalized':raw})}
    # Reuse the original segmentation helper and prediction loop, without any optimizer/training call.
    texts=segmented_texts(batch_frames,vncorenlp_dir)['val']
    tokenizer=AutoTokenizer.from_pretrained(settings['checkpoint'],revision=settings['revision'],use_fast=False)
    model=AutoModelForSequenceClassification.from_pretrained(settings['checkpoint'],revision=settings['revision'],
        num_labels=27,problem_type='multi_label_classification',id2label=dict(enumerate(LABELS)),label2id={l:i for i,l in enumerate(LABELS)})
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.to(device)
    encoded=tokenizer(texts,padding='max_length',truncation=True,max_length=settings['max_length'],return_tensors='pt')
    dataset=TensorDataset(encoded['input_ids'],encoded['attention_mask'],torch.zeros((len(texts),27)))
    loader=DataLoader(dataset,batch_size=settings['batch_size'],shuffle=False,num_workers=0)
    values=predict(model,'phobert',loader,device)
    saved={sample:p[i] for i,sample in enumerate(frames['val'].id)}
    verify_original_scores(originals,values[:len(originals)],saved)
    by_id={sample:values[i] for i,sample in enumerate(originals)}
    original=np.stack([by_id[sample] for sample in pairs.original_sample_id])
    return original,values[len(originals):],dict(checkpoint_sha256=checkpoint_sha,original_reproduction_atol=1e-4,
        source='verified 27-label Kaggle checkpoint',settings=settings,device=str(device),
        torch_version=torch.__version__,segmentation='Reused train_neural.segmented_texts',prediction='Reused train_neural.predict')


def analyze_pairs(pairs,original,changed,out):
    if original.shape!=(len(pairs),27) or changed.shape!=original.shape or not np.isfinite(changed).all() or ((changed<0)|(changed>1)).any():
        raise ValueError('Expected aligned finite n×27 intervention probability matrices')
    records=[]
    for i,r in enumerate(pairs.itertuples()):
        c=LABELS.index(r.target_label)
        records.append(dict(pair_id=r.pair_id,original_sample_id=r.original_sample_id,target_label=r.target_label,
            transformation_type=r.transformation_type,original_score=float(original[i,c]),changed_score=float(changed[i,c]),
            delta=float(changed[i,c]-original[i,c])))
    frame=pd.DataFrame(records);summaries=[];controls=[]
    for (label,transformation),g in frame.groupby(['target_label','transformation_type'],sort=True):
        summaries.append(dict(target_label=label,transformation_type=transformation,
            **paired_summary(g.original_score,g.changed_score,g.original_sample_id)))
    # Control-adjusted shifts require the SAME original and target in both arms.
    means=frame.groupby(['target_label','original_sample_id','transformation_type'],sort=True).delta.mean().unstack('transformation_type')
    if 'neutral_control' in means:
        for label in FOCUS:
            if label not in means.index.get_level_values(0):continue
            group=means.loc[label]
            for transformation in TRANSFORMS[:2]:
                if transformation not in group:continue
                paired=group[[transformation,'neutral_control']].dropna()
                if paired.empty:continue
                difference=paired[transformation]-paired.neutral_control
                low,high=mean_interval(difference,seed=stable_seed(label+transformation))
                controls.append(dict(target_label=label,transformation_type=transformation,n_matched_originals=len(paired),
                    mean_control_adjusted_shift=float(difference.mean()),ci_low=low,ci_high=high))
    write_csv(out/'paired_scores.csv',records);write_csv(out/'paired_summary.csv',summaries)
    if controls:write_csv(out/'matched_control_comparison.csv',controls)
    elif (out/'matched_control_comparison.csv').exists():(out/'matched_control_comparison.csv').unlink()
    return dict(status='complete',verified_pairs=len(pairs),original_comments=int(pairs.original_sample_id.nunique()),
        interpretation='Checkpoint sensitivity on selected human-verified pairs only; not proof of the original label-F1 gap or general causal effects',
        uncertainty='Original-comment cluster bootstrap; singleton groups have no interval; no multiplicity correction')


def pending(out,pairs):
    previous=out/'intervention_status.json'
    if previous.exists() and json.loads(previous.read_text())['status']=='complete':
        raise ValueError('Completed intervention results must not be overwritten by pending inputs; use a new output directory')
    state=dict(status='pending',proposed_pairs=int((pairs.status=='proposed').sum()),
               rejected_pairs=int((pairs.status=='rejected').sum()),verified_pairs=0,
               reason='No pairs verified by two independent humans; no model inference or intervention results generated')
    write_json(out/'intervention_status.json',state)
    (out/'intervention_analysis.md').write_text('# Q1 wording interventions\n\n**Pending human verification.**\n\n'+state['reason']+'.\n')
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','bind','run','analyze'])
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--private-dir',type=Path,default=PRIVATE)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT/'interventions')
    parser.add_argument('--pairs',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--vncorenlp-dir',type=Path)
    parser.add_argument('--device',choices=['cpu','cuda'])
    parser.add_argument('--per-label',type=int,default=8)
    args=parser.parse_args()
    if args.per_label<1:parser.error('--per-label must be positive')
    if args.mode=='prepare' and args.pairs:parser.error('--pairs is only for bind/run/analyze')
    out=isolated_output(args.output_dir);private=private_directory(args.private_dir)
    frames,y,p,meta=load_evidence(args.data_dir)
    truth={sample:y[i] for i,sample in enumerate(frames['val'].id)}
    pairs_path=args.pairs or private/'intervention_pairs.csv'
    private_directory(pairs_path.parent)
    if args.mode=='prepare':
        pairs=prepare(frames,y,private,args.per_label);state=pending(out,pairs)
    else:
        if not pairs_path.is_file():raise FileNotFoundError('Run prepare and obtain human-verified intervention pairs first')
        all_pairs=pd.read_csv(pairs_path,dtype=str,keep_default_na=False)
        if args.mode=='bind':
            verified_pairs(all_pairs,truth)
            payload=pd.read_csv(private/'intervention_text_private.csv',dtype=str,keep_default_na=False)
            bound=bind_texts(all_pairs,payload,frames['val']);bound.to_csv(pairs_path,index=False)
            print('Text hashes bound. Changed pairs are proposed and require new independent human verification.');return
        pairs=verified_pairs(all_pairs,truth)
        if pairs.empty:
            state=pending(out,all_pairs)
            write_json(out/'provenance.json',dict(evidence=meta,pairs_sha256=digest_file(pairs_path),raw_text_exported=False,training_performed=False,status='pending'))
            print('Interventions pending: no verified human pairs; inference not attempted.');return
        payload_path=private/'intervention_text_private.csv'
        if not payload_path.is_file():raise FileNotFoundError('Human-reviewed private paraphrase text is required')
        payload=pd.read_csv(payload_path,dtype=str,keep_default_na=False)
        validate_text_bindings(pairs,payload,frames['val'])
        saved={sample:p[i] for i,sample in enumerate(frames['val'].id)}
        if args.mode=='run':
            payload_path=private/'intervention_text_private.csv'
            if not payload_path.is_file():raise FileNotFoundError('Completed private paraphrase text is required')
            payload=pd.read_csv(payload_path,dtype=str,keep_default_na=False)
            original,changed,inference=infer(pairs,payload,frames,y,p,args.checkpoint,args.vncorenlp_dir,args.device)
            np.savez_compressed(out/'verified_predictions.npz',pair_id=pairs.pair_id.to_numpy(dtype=str),
                labels=np.array(LABELS),original_probabilities=original,paraphrase_probabilities=changed)
            write_json(out/'inference_manifest.json',dict(**inference,pairs_sha256=digest_file(pairs_path),
                private_text_sha256=digest_file(payload_path),predictions_sha256=digest_file(out/'verified_predictions.npz'),evidence=meta))
        else:
            manifest_path=out/'inference_manifest.json';prediction_path=out/'verified_predictions.npz'
            if not manifest_path.is_file() or not prediction_path.is_file():raise FileNotFoundError('Run verified frozen-checkpoint inference before analyze')
            manifest=json.loads(manifest_path.read_text())
            if (manifest['pairs_sha256']!=digest_file(pairs_path) or manifest['predictions_sha256']!=digest_file(prediction_path)
                    or manifest['private_text_sha256']!=digest_file(payload_path)):
                raise ValueError('Verified pair/prediction files changed after inference')
            with np.load(prediction_path,allow_pickle=False) as a:
                if set(a.files)!={'pair_id','labels','original_probabilities','paraphrase_probabilities'}:raise ValueError('Unexpected inference NPZ fields')
                np.testing.assert_array_equal(a['pair_id'],pairs.pair_id)
                np.testing.assert_array_equal(a['labels'],LABELS)
                original,changed=a['original_probabilities'].copy(),a['paraphrase_probabilities'].copy()
        verify_original_scores(pairs.original_sample_id,original,saved)
        state=analyze_pairs(pairs,original,changed,out)
        state.update(proposed_pairs=int((all_pairs.status=='proposed').sum()),
                     rejected_pairs=int((all_pairs.status=='rejected').sum()))
        write_json(out/'intervention_status.json',state)
        (out/'intervention_analysis.md').write_text('# Q1 verified wording interventions\n\n'+state['interpretation']+'.\n\n'+state['uncertainty']+'.\n')
    write_json(out/'provenance.json',dict(evidence=meta,pairs_sha256=digest_file(pairs_path),
        raw_text_exported=False,training_performed=False,status=state['status']))
    print('Intervention workflow:',state['status'])


if __name__=='__main__':main()
