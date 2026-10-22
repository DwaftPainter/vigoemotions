"""Colab GPU training for BiGRU and PhoBERT; validation-only midterm evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence

from common import ROOT, LABELS, load_data, targets, scores, environment, write_json
from evaluate import save_evaluation


class BiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2*hidden_dim, 27)

    def forward(self, ids, mask):
        packed = pack_padded_sequence(self.embedding(ids), mask.sum(1).cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.head(self.dropout(torch.cat((hidden[-2],hidden[-1]),dim=1)))


def segmented_texts(frames, model_dir):
    import py_vncorenlp
    segmenter = py_vncorenlp.VnCoreNLP(annotators=['wseg'], save_dir=str(model_dir.resolve()))
    result = {}
    for split in ('train','val'):
        result[split] = [' '.join(segmenter.word_segment(text)) for text in frames[split].normalized]
    return result


def prepare(model_name, config, frames, vncorenlp_dir):
    settings = config[model_name]
    max_length = settings['max_length']
    features, prep_meta = {}, {}
    if model_name == 'gru':
        words = Counter(word for text in frames['train'].normalized for word in text.lower().split())
        vocab = {word:i+2 for i,(word,_) in enumerate(words.most_common(settings['max_vocab']-2))}
        # Vocabulary contains source tokens; keep private, outside public artifacts.
        write_json(ROOT/'data/derived/gru_vocab.json',vocab)
        model = BiGRU(len(vocab)+2,settings['embedding_dim'],settings['hidden_dim'],settings['dropout'])
        for split in ('train','val'):
            seqs = [[vocab.get(w,1) for w in text.lower().split()][:max_length]
                    for text in frames[split].normalized]
            ids = torch.zeros((len(seqs),max_length),dtype=torch.long)
            for i,seq in enumerate(seqs):
                ids[i,:len(seq)] = torch.tensor(seq,dtype=torch.long)
            features[split] = (ids,ids.ne(0).long())
        prep_meta = {'vocab_size':len(vocab)+2,'tokenization':'NFC + lower + whitespace',
            'truncation_rate':{s:float(frames[s].normalized.str.split().str.len().gt(max_length).mean())
                               for s in ('train','val')}}
    else:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        text = segmented_texts(frames,vncorenlp_dir)
        model_args = dict(revision=settings['revision'])
        tokenizer = AutoTokenizer.from_pretrained(settings['checkpoint'],use_fast=False,**model_args)
        model = AutoModelForSequenceClassification.from_pretrained(settings['checkpoint'],
            num_labels=27,problem_type='multi_label_classification',
            id2label=dict(enumerate(LABELS)),label2id={l:i for i,l in enumerate(LABELS)},**model_args)
        truncation = {}
        for split in ('train','val'):
            lengths = [len(tokenizer.encode(t, truncation=False)) for t in text[split]]
            truncation[split] = float(np.mean(np.array(lengths)>max_length))
            encoded = tokenizer(text[split],padding='max_length',truncation=True,
                                max_length=max_length,return_tensors='pt')
            features[split] = (encoded['input_ids'],encoded['attention_mask'])
        prep_meta = {'checkpoint':settings['checkpoint'],'revision':settings['revision'],
                     'tokenization':'VnCoreNLP RDRSegmenter + PhoBERT BPE',
                     'truncation_rate':truncation}
    loaders = {}
    for split,(ids,mask) in features.items():
        dataset = TensorDataset(ids,mask,torch.tensor(targets(frames[split]),dtype=torch.float32))
        loaders[split] = DataLoader(dataset,batch_size=settings['batch_size'],
            shuffle=split=='train',num_workers=0,generator=torch.Generator().manual_seed(config['seed']))
    return model,loaders,prep_meta


def logits(model, model_name, ids, mask):
    if model_name == 'gru':
        return model(ids,mask)
    return model(input_ids=ids,attention_mask=mask).logits


def predict(model, model_name, loader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for ids,mask,_ in loader:
            output = logits(model,model_name,ids.to(device),mask.to(device))
            results.append(torch.sigmoid(output).cpu().numpy())
    return np.concatenate(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',choices=['gru','phobert'],required=True)
    parser.add_argument('--vncorenlp-dir',type=Path,default=ROOT/'data/vncorenlp')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/midterm.json')
    parser.add_argument('--output-dir',type=Path,help='New directory for an isolated experiment')
    parser.add_argument('--checkpoint',type=Path,help='New private checkpoint path for an isolated experiment')
    args = parser.parse_args()
    if bool(args.output_dir) != bool(args.checkpoint):
        parser.error('--output-dir and --checkpoint must be supplied together')
    if args.output_dir:
        if args.checkpoint.exists():
            parser.error('Checkpoint already exists; use a new experiment path')
        args.output_dir.mkdir(parents=True,exist_ok=False)
    out = args.output_dir or ROOT/'reports'
    config = json.loads(args.config.read_text())
    seed = config['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',device,flush=True)
    settings = config[args.model]
    frames,_ = load_data()
    prep_started = time.perf_counter()
    model,loaders,prep_meta = prepare(args.model,config,frames,args.vncorenlp_dir)
    prep_seconds = time.perf_counter()-prep_started
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],
                                 weight_decay=settings.get('weight_decay',0.0))
    loss_fn = nn.BCEWithLogitsLoss()
    history,best_score = [],-1.0
    checkpoint = args.checkpoint or ROOT/'data/checkpoints'/f'{args.model}_best.pt'
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    for epoch in range(1,settings['epochs']+1):
        model.train();total_loss = 0.0
        for ids,mask,y in loaders['train']:
            ids,mask,y = ids.to(device),mask.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(logits(model,args.model,ids,mask),y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            total_loss += loss.item()*len(y)
        p = predict(model,args.model,loaders['val'],device)
        metrics = scores(targets(frames['val']),p)
        history.append(dict(epoch=epoch,loss=total_loss/len(frames['train']),**metrics))
        print(history[-1],flush=True)
        if metrics['macro_f1'] > best_score:
            best_score = metrics['macro_f1']
            torch.save(model.state_dict(),checkpoint)
            best_epoch = epoch
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
    p = predict(model,args.model,loaders['val'],device)
    meta = {'parameter_count':sum(v.numel() for v in model.parameters()),'train_seconds':seconds}
    save_evaluation(args.model,frames,{'val':p},meta,output_dir=out)
    write_json(out/f'{args.model}_run.json',dict(config=config,environment=environment(),
        device=torch.cuda.get_device_name(0) if device.type=='cuda' else str(device),
        best_epoch=best_epoch,history=history,preprocessing=prep_meta,prep_seconds=prep_seconds,
        time_definition='Training loop including per-epoch validation and checkpoint writes; excludes preprocessing/download',
        checkpoint_path=str(checkpoint),
        deterministic_note='Fixed seeds and deterministic cuDNN; exact equality across GPU types not guaranteed',**meta))


if __name__ == '__main__':
    main()
