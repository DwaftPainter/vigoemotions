"""Data-driven research deck using only the current verified Kaggle studies."""
import hashlib
import json

import pandas as pd

from common import ROOT
from import_equal20_results import OUT
from academic_slides import NAVY, TEAL, MUTED, BLUE, AMBER

SPEECH = ['Good morning. Our project recognizes emotions in Vietnamese social-media comments. Because one '
 'comment can express several emotions, this is a multi-label task. The dataset contains twenty '
 'thousand, six hundred and sixty-four comments. We compare three models, with and without an '
 'explicit Neutral output. I will briefly cover the setup, then focus on our results and analysis.',
 'Comments average about one point eight six emotions, and the median length is eleven whitespace '
 'tokens. We keep the supplied split, remove one training copy of a test comment, and fit '
 'vocabulary only on training data. Unicode and whitespace are normalized. Our distribution audit '
 'does not prove the original stratification algorithm, so that requirement still needs '
 'clarification.',
 'The twenty-seven-output model predicts emotions only. Neutral-only comments have an all-zero '
 'emotion target. The twenty-eight-output model adds a learned Neutral label. We preserve mixed '
 'annotations, where Neutral appears with emotions. Both neural settings use sigmoid outputs and '
 'binary cross-entropy because labels are not mutually exclusive. Validation contains ninety-four '
 'Neutral-positive examples.',
 'This diagram shows the three processing routes. TF-IDF uses lexical features and one classifier '
 'per label. BiGRU learns embeddings and reads sequences in both directions. PhoBERT uses '
 'Vietnamese word segmentation and BPE before a pretrained backbone and a new classification head. '
 'The neural routes use independent sigmoid outputs; there is no single-label softmax decision.',
 'We compare TF-IDF with one-versus-rest Logistic Regression, a bidirectional GRU trained from '
 'scratch, and full PhoBERT fine-tuning. Both neural models complete twenty epochs in each '
 'setting. TF-IDF uses a solver limit instead. We share the seed and split, and select checkpoints '
 'by validation Macro-F1 on the same twenty-seven emotions at threshold zero point five.',
 'The table separates full twenty-seven-emotion and twenty-eight-label evaluation. PhoBERT leads '
 'within each block: Macro-F1 is zero point five nine seven four and zero point five seven eight '
 'three, respectively. The additional columns show Micro-F1, Hamming Loss, rankings, selected '
 'epochs, and measured training time. Different label sets mean these two overall F1 scores should '
 'not be directly subtracted.',
 'For a comparison on identical labels, we score only the twenty-seven emotion columns of the '
 'expanded model. PhoBERT then reaches 0.5836, about 1.38 percentage points below the '
 'twenty-seven-output model. TF-IDF is unchanged and BiGRU decreases slightly. This run shows no '
 'improvement on the shared emotions, but one seed cannot establish a general causal conclusion.',
 'PhoBERT identifies 45 of the ninety-four Neutral-positive validation examples. It misses 49 and '
 'produces 67 false positives, giving Neutral F1 of 0.4369. So the additional label is learned, '
 'but still difficult. The twenty-seven-output model has no Neutral classifier; its Neutral F1 is '
 'undefined, not zero. An empty prediction is also not automatically Neutral.',
 'These curves use the same twenty-seven-emotion checkpoint-selection metric for both output '
 'settings. Both neural models completed twenty epochs. The selected PhoBERT checkpoints are at '
 'epochs fourteen and fifteen. Selection is based on validation, not simply the final epoch. '
 'Training loss continued to decrease, but validation loss was not recorded, so we do not claim '
 'confirmed overfitting.',
 'Gratitude is the easiest label for all three models. The weakest label is realization for TF-IDF '
 'and BiGRU, but disapproval for PhoBERT. The bars show selected PhoBERT labels; the table '
 'compares each model’s extremes. Overall F1 hides these differences.',
 'For Q2, each dot represents one emotion under PhoBERT. Its frequency–F1 correlation is only '
 '0.018. The table also shows TF-IDF and BiGRU. These descriptive correlations do not establish '
 'causation; frequency alone cannot explain which labels are difficult.',
 'For Q3, we choose disappointment and sadness because they co-occur most often in training. '
 'Validation contains 122 comments annotated with both. TF-IDF recovers both in two cases, BiGRU '
 'in thirty-seven, and PhoBERT in sixty-three. Joint recall is therefore 1.64, 30.33 and 51.64 '
 'percent. Other predicted labels are allowed, so this does not measure pair precision.',
 'Q4 compares every available text-style group using the same twenty-seven labels. PhoBERT leads '
 'in all four rows. Its largest advantage over BiGRU is teencode: 22.13 percentage points on 307 '
 'examples. These groups use heuristic rules and overlap. Their sample counts differ, so this is '
 'descriptive rather than a controlled style effect.',
 'Q5 keeps each checkpoint fixed and varies only the threshold. The three curves respond '
 'differently. TF-IDF gains about 24.55 percentage points of validation Macro-F1; BiGRU gains '
 '1.86; PhoBERT gains only 0.043. Circles mark the best grid points, not test improvements.',
 'The table compares the default and exploratory best thresholds for every model. Lower thresholds '
 'improve Macro-F1 but increase Hamming Loss in all three cases. Choosing on validation can '
 'overestimate generalization. We keep the main comparison at 0.5 and will lock decoding choices '
 'before evaluating test.',
 'Two examples illustrate the six-case preliminary, AI-assisted review. On the left, a comment '
 'annotated as relief is predicted as Neutral. On the right, a Neutral comment is assigned anger '
 'and annoyance. Implied affect and profanity are possible explanations, not proven causes. We '
 'show IDs and abstract observations, and the interpretations still need team confirmation.',
 'For the final evaluation, we will test whether one threshold suits every emotion. We will freeze '
 'the best neural checkpoint and compare the default threshold, a tuned global threshold, and '
 'per-label thresholds. We select thresholds on validation, lock all choices, then evaluate test '
 'once. We also plan a detailed review of at least twenty false negatives and twenty false '
 'positives.',
 'To conclude, PhoBERT gives the strongest overall results in both settings. Adding Neutral does '
 'not improve shared-emotion F1 in this run, and Neutral prediction still makes substantial '
 'errors. Saved Kaggle predictions and histories support these findings. Our next step is '
 'controlled threshold evaluation. Thank you; we welcome your questions.']

NAMES={'tfidf_lr':'TF-IDF + OvR LR','gru':'BiGRU','phobert':'PhoBERT'}


def table(headers, rows, widths, **kwargs):
    return dict(headers=headers,rows=rows,x=64,top=520,widths=widths,**kwargs)


def load_data():
    comp=pd.read_csv(OUT/'model_comparison.csv')
    if len(comp)!=6 or not comp[comp.model!='tfidf_lr'].epochs_trained.eq(20).all():
        raise ValueError('Expected the six completed equal-20 Kaggle runs')
    for labels in (27,28):
        m=json.loads((OUT/f'evidence/{labels}/manifest.json').read_text())
        for split,checksum in m['split_ids_sha256'].items():
            ids=pd.read_csv(ROOT/f'splits/{split}.csv',dtype=str).id.tolist()
            if hashlib.sha256('\n'.join(ids).encode()).hexdigest()!=checksum:
                raise ValueError('Fixed IDs differ from the verified study')
    return comp


def slide_definitions():
    comp=load_data();r=comp.set_index(['label_count','model'])
    neutral=pd.read_csv(OUT/'neutral_metrics.csv')
    paired=pd.read_csv(OUT/'paired_emotion27_delta.csv')
    label=pd.read_csv(OUT/'analysis/27_emotion27/phobert/tables/phobert_per_label.csv')
    annotation=pd.read_csv(OUT/'evidence/28/annotation_counts.csv').set_index('split')
    style_tables={model:pd.read_csv(OUT/f'analysis/27_emotion27/{model}/tables/{model}_styles.csv').set_index('style')
                  for model in NAMES}
    freq=pd.read_csv(OUT/'frequency_f1.csv').set_index('model')
    pairs=pd.read_csv(OUT/'pair_preservation.csv').set_index('model')
    extremes=pd.read_csv(OUT/'label_extremes.csv').set_index('model')
    threshold=pd.read_csv(OUT/'threshold_sensitivity.csv').set_index('model')
    p27,p28=r.loc[(27,'phobert')],r.loc[(28,'phobert')]
    n=neutral[neutral.model=='PhoBERT'].iloc[0]
    slides=[]
    def add(layout,section,title,subtitle,takeaway,source,**content):
        slides.append(dict(layout=layout,section=section,title=title,subtitle=subtitle,
                           takeaway=takeaway,source=source,**content))

    add('title','MIDTERM','ViGoEmotions','Three models. Two label spaces. One shared 20-epoch protocol.',None,
        'ViGoEmotions / ATI 2026 / Kaggle notebooks 08–09 / Validation only',
        headline='What changes when Neutral becomes\na learned output?',
        stats=[['3','model families'],['20','epochs for every neural run'],['2','label settings: 27 and 28']])

    split_rows=[]
    for split,name,role in [('train','Train','Fit model parameters'),('val','Validation','Select checkpoint / explore thresholds'),('test','Test','Held out from model evaluation')]:
        used=len(pd.read_csv(ROOT/f'splits/{split}.csv'))
        split_rows.append([name,f'{int(annotation.loc[split,"raw_samples"]):,}',f'{used:,}',role])
    split_rows.append(['Total',f'{int(annotation.raw_samples.sum()):,}',f'{sum(len(pd.read_csv(ROOT/f"splits/{s}.csv")) for s in ("train","val","test")):,}','One train–test text overlap removed'])
    selected=pd.concat([label.nlargest(5,'train_count'),label.nsmallest(1,'train_count')])
    add('table_chart','01 / DATA','Dataset, split and emotion frequency',
        '20,664 source comments • fixed supplied partitions • label statistics from training data',
        'Preserve the split; fit text representations on train only.',
        'Sources: current study annotation_counts.csv; fixed split IDs; PhoBERT-27 per-label table. No test model scores.',
        table=table(['Split','Source n','Used n','Role'],split_rows,[125,120,120,340],row_height=58,alignments=[0,2,2,0]),
        bars=dict(labels=selected.label.tolist(),values=selected.train_count.astype(int).tolist(),
                  x=807,top=479,width=409,height=260,label_width=133),
        stats=[dict(x=64,top=207,width=230,number='1.859',label='emotions / train comment'),
               dict(x=330,top=207,width=230,number='11 / 47',label='median tokens / characters')],
        annotations=[dict(value='Most frequent five + least frequent label',x=807,top=520,width=409,size=14,color=MUTED,height=35)])

    add('encoding','02 / TARGETS','27 vs 28 outputs: one additional decision',
        'The label mapping and validation examples stay fixed; Neutral becomes an explicit target.',
        'A missing emotion prediction is not automatically a Neutral prediction.',
        'Sources: config27.json; evidence/28/config.json; annotation_counts.csv. Encoding examples are schematic targets.',
        table=table(['Property','27-emotion model','28-label model'],[
            ['Output space','27 emotion decisions','27 emotions + Neutral'],
            ['Neutral-only target','All 27 emotion entries are zero','Neutral = 1; emotion entries = 0'],
            ['Mixed annotation','Retain the emotion entries','Retain emotions and Neutral'],
            ['Reported full Macro-F1','Average over 27 labels','Average over 28 labels']
        ],[280,436,436],row_height=57,alignments=[0,0,0]))

    add('pipeline','03 / METHODS','Three routes from text to multi-label scores',
        'Model-specific text representations • 27 or 28 independent label decisions',
        'Text representation changes across models; the label decision remains independent for each output.',
        'Sources: frozen run configurations; train_neural.py; neutral_28_labels.py. Neural training: BCEWithLogitsLoss.',
        routes=[
            dict(name='TF-IDF + LR',tag='Lexical baseline',color=BLUE,top=520,nodes=[['Regex tokens','1–2 grams'],['TF-IDF','Fit vocabulary on train'],['OvR classifiers','One LR per label'],['Label scores','Threshold at 0.5']]),
            dict(name='BiGRU',tag='Learn from scratch',color=TEAL,top=400,nodes=[['Token IDs','Lowercase + whitespace'],['Embedding','128 dimensions'],['BiGRU + linear','Forward + backward'],['Sigmoid scores','27 / 28 outputs']]),
            dict(name='PhoBERT',tag='Full fine-tuning',color=NAVY,top=280,nodes=[['Segment + BPE','VnCoreNLP tokenizer'],['PhoBERT backbone','Pretrained Vietnamese'],['Classification head','Train with backbone'],['Sigmoid scores','27 / 28 outputs']])],
        caption='Shared preprocessing: Unicode NFC + whitespace normalization; retain Vietnamese accents and negation.')


    params=lambda key:' / '.join(f'{r.loc[(labels,key)].parameter_count/1e6:.3f}M' for labels in (27,28))
    add('full_table','03 / METHODS','Architectures and training configurations',
        'Shared seed 42 • fixed split • threshold 0.5 • Tesla T4 for neural models',
        'Equal epochs control the training budget; they do not imply equal compute.',
        'Sources: frozen manifests and six run.json files. Parameter counts shown as 27 / 28 outputs; rounded to millions.',
        table=table(['Model','Representation / architecture','Parameters 27 / 28','Batch / learning rate','Budget'],[
            ['TF-IDF + OvR LR','Unigram/bigram; up to 40k features; liblinear, C = 1',params('tfidf_lr'),'Not neural','max_iter = 1,000'],
            ['BiGRU','Train from scratch; embed/hidden 128; length 128',params('gru'),'64 / 0.001','20 epochs'],
            ['PhoBERT','VnCoreNLP + BPE; full fine-tuning; length 256',params('phobert'),'16 / 2e-5','20 epochs']
        ],[185,345,210,230,182],row_height=88,highlights=[(2,0)],alignments=[0,0,2,0,0]),
        caption='Select by validation emotion27 Macro-F1 at 0.5, including for the 28-output model; ties keep the earliest epoch.')

    metrics=[];highlight=[]
    for k,labels in enumerate((27,28)):
        for i,key in enumerate(NAMES):
            a=r.loc[(labels,key)];idx=k*3+i
            seconds=a.train_seconds
            duration=f'{seconds:.2f} s' if seconds<10 else f'{seconds:.1f} s' if seconds<120 else f'{seconds/3600:.2f} h'
            metrics.append([str(labels),NAMES[key],f'{a.macro_f1:.4f}',f'{a.micro_f1:.4f}',f'{a.hamming_loss:.4f}',f'{a.lrap:.4f}',
                            '—' if pd.isna(a.best_epoch) else str(int(a.best_epoch)),duration])
            if key=='phobert':highlight.extend((idx,j) for j in range(1,6))
    add('full_table','04 / RESULTS','PhoBERT leads within each label setting',
        'Validation n = 2,066 • threshold 0.5 • separate full-label evaluations for 27 and 28 outputs',
        'Rank models within each block; the two Macro-F1 blocks evaluate different label sets.',
        'Source: model_comparison.csv. ↑ higher / ↓ lower is better. CPU TF-IDF; T4 neural. Times exclude preprocessing.',
        table=table(['Labels','Model','Macro-F1 ↑','Micro-F1 ↑','Hamming ↓','LRAP ↑','Best ep.','Train time'],metrics,
                    [85,205,145,145,145,145,110,172],row_height=49,group_starts=[3],highlights=highlight,font_size=16),
        caption='Both neural models completed 20 epochs. “Best ep.” identifies the selected checkpoint, not the number of epochs trained.')

    rows=[[a.model,f'{a.macro27:.4f}',f'{a.macro28_projected27:.4f}',f'{a.delta_pp:+.2f}'] for a in paired.itertuples()]
    add('paired','05 / NEUTRAL COMPARISON','Adding Neutral does not raise shared-emotion F1 here',
        'Same 27 emotion columns • same validation IDs • shared checkpoint selection • one seed',
        'PhoBERT changes by −1.38 percentage points on the shared 27-emotion evaluation.',
        'Source: paired_emotion27_delta.csv. Δ = 28-output projection minus 27-output score, in percentage points.',
        table=table(['Model','27-output F1','28 → 27 F1','Δ (pp)'],rows,[225,165,165,150],row_height=70,highlights=[(2,3)],alignments=[0,2,2,2]),
        comparison=[dict(model=a.model,values=[['27 outputs',float(a.macro27),NAVY],['28 → 27',float(a.macro28_projected27),TEAL]]) for a in paired.itertuples()],
        caption='This controls the epoch budget and scoring label space. It does not establish a general causal effect or statistical significance.')

    neutral_rows=[[a.model,f'{a.precision:.4f}',f'{a.recall:.4f}',f'{a.f1:.4f}',str(a.TP),str(a.FP),str(a.FN)] for a in neutral.itertuples()]
    add('neutral','06 / NEUTRAL DETECTION','A learned Neutral output still makes substantial errors',
        '28-label setting only • 94 Neutral-positive validation examples: 75 Neutral-only + 19 mixed',
        'PhoBERT finds 45 of 94 Neutral examples, with 67 false alarms.',
        'Source: neutral_metrics.csv. Threshold 0.5; support 94 for all models. No Neutral metric for the 27-output model.',
        table=table(['Model','Precision','Recall','F1','TP','FP','FN'],neutral_rows,[280,185,185,185,105,105,107],row_height=62,
                    highlights=[(0,1),(2,2),(2,3)],alignments=[0,2,2,2,2,2,2]),
        stats=[dict(x=118,top=252,width=300,number=str(int(n.TP)),label='correct Neutral detections'),
               dict(x=499,top=252,width=300,number=str(int(n.FN)),label='missed Neutral examples',color=AMBER),
               dict(x=880,top=252,width=300,number=str(int(n.FP)),label='false Neutral predictions',color=NAVY)],
        caption='An empty prediction is not a substitute for an explicitly trained Neutral label.')

    charts=[]
    for i,model in enumerate(('gru','phobert')):
        series=[];markers=[]
        for labels,color,dash in [(27,NAVY,None),(28,TEAL,(5,3))]:
            run=json.loads((OUT/f'evidence/{labels}/{model}/run.json').read_text());history=run['history']
            key='macro_f1' if labels==27 else 'selection_macro_f1';best=run['best_epoch']
            series.append(dict(name=f'{labels} outputs · best {best}',x=[h['epoch'] for h in history],y=[float(h[key]) for h in history],color=color,dash=dash))
            markers.append(dict(x=best,y=float(history[best-1][key]),label='',color=color))
        charts.append(dict(title=NAMES[model],series=series,x=64+i*600,y=214,width=552,height=280,
                           xlim=[1,20],ylim=[0,.68],xticks=[1,5,10,15,20],yticks=[0,.2,.4,.6],xlabel='Training epoch',
                           ylabel='Validation emotion27 Macro-F1',markers=markers))
    add('curves','07 / TRAINING DYNAMICS','Twenty epochs trained; the last checkpoint need not win',
        'Both curves in each panel use the same 27-emotion checkpoint-selection metric.',
        'Best PhoBERT checkpoints: epoch 14 for 27 outputs and epoch 15 for 28 outputs.',
        'Sources: evidence/{27,28}/{gru,phobert}/run.json. All histories contain epochs 1–20; no early stopping.',
        charts=charts,caption='Train BCE continued to decrease; validation loss was not recorded. These curves alone do not confirm overfitting.')

    ordered=label.sort_values('f1',ascending=False);subset=pd.concat([ordered.head(3),ordered.tail(3)])
    add('table_chart','08 / Q1','Q1 · Which emotions are easiest and hardest?',
        '27-output study • per-label validation F1 at threshold 0.5 • bars show selected PhoBERT labels',
        'Gratitude ranks highest for every model; the weakest label depends on the model.',
        'Sources: label_extremes.csv; current analysis/27_emotion27/*/tables/*_per_label.csv. Full label tables in report.',
        rq='Q1',
        bars=dict(labels=subset.label.tolist(),values=subset.f1.astype(float).tolist(),x=64,top=479,width=560,height=269,
                  maximum=1,value_format='f1',label_width=157,colors_by_row=[TEAL]*3+[BLUE]*3),
        table=dict(headers=['Model','Highest label / F1','Lowest label / F1'],
                   rows=[[name,f'{a.highest}\n{a.highest_f1:.4f}',f'{a.lowest}\n{a.lowest_f1:.4f}']
                         for name,a in extremes.iterrows()],x=656,top=520,widths=[150,200,210],row_height=76,
                   highlights=[(2,1),(2,2)],alignments=[0,0,0]),
        annotations=[dict(value='PhoBERT: top three and bottom three labels',x=64,top=520,width=560,size=14,color=MUTED,height=35),
                     dict(value='Per-label F1 reveals weaknesses hidden by the overall average. These are single-run validation results.',
                          x=656,top=219,width=560,size=17,color=MUTED,height=65)])

    add('table_chart','09 / Q2','Q2 · Does label frequency predict F1?',
        '27-output study • training counts versus validation per-label F1 • one point per emotion',
        'PhoBERT shows almost no monotonic frequency–F1 association in this run.',
        'Sources: frequency_f1.csv; current PhoBERT-27 per-label table. Descriptive correlation across 27 labels per model.',
        rq='Q2',
        scatter=dict(series=[dict(name='27 emotion labels',x=label.train_count.astype(int).tolist(),y=label.f1.astype(float).tolist(),color=TEAL)],
                     x=64,y=185,width=710,height=340,xlim=[0,3200],ylim=[0,1],xticks=[0,1000,2000,3000],yticks=[0,.25,.5,.75,1],
                     xlabel='Positive training instances',ylabel='PhoBERT validation per-label F1',scatter=True,show_legend=False),
        table=dict(headers=['Model','Spearman ρ','p-value'],
                   rows=[[name,f'{a.spearman_rho:.3f}',f'{a.p_value:.3f}'] for name,a in freq.iterrows()],
                   x=815,top=520,widths=[150,120,131],row_height=66,alignments=[0,2,2],highlights=[(2,1)]),
        annotations=[dict(value='Frequency alone does not explain label difficulty. Association is not causation; one seed was evaluated.',
                          x=815,top=245,width=401,size=17,color=MUTED,height=100)])

    pair_detail=pd.read_csv(OUT/'analysis/27_emotion27/phobert/tables/phobert_pair_preservation_val.csv').sort_values('train_count',ascending=False).iloc[0]
    pair_name=f'{pair_detail.label_a} + {pair_detail.label_b}'
    add('table_chart','10 / Q3','Q3 · How often are both emotions recovered?',
        f'27-output study • threshold 0.5 • {pair_name} (top training pair: {int(pair_detail.train_count):,})',
        f'PhoBERT recovers both emotions in {int(pairs.loc["PhoBERT","joint_TP"])}/{int(pairs.loc["PhoBERT","val_true"])} positive-pair cases; about half are still missed.',
        'Sources: pair_preservation.csv; current 27-output pair tables. Extra predicted labels allowed; this is joint recall, not precision.',
        rq='Q3',
        table=table(['Model','Joint TP','Val. support','Joint recall'],
                    [[name,str(int(a.joint_TP)),str(int(a.val_true)),f'{100*a.joint_recall:.2f}%'] for name,a in pairs.iterrows()],
                    [240,130,165,170],row_height=70,alignments=[0,2,2,2],highlights=[(2,1),(2,3)]),
        bars=dict(labels=['TF-IDF + LR','BiGRU','PhoBERT'],values=pairs.joint_recall.astype(float).tolist(),
                  x=807,top=460,width=409,height=195,label_width=143,maximum=1,value_format='percent'),
        annotations=[dict(value='JOINT RECALL · 0–100% scale',x=807,top=520,width=409,size=14,color=TEAL,bold=True,height=30),
                     dict(value='Joint recall = cases with both labels predicted ÷ cases annotated with both labels.',
                          x=64,top=225,width=1152,size=19,color=NAVY,bold=True,height=55),
                     dict(value='The pair was selected by training co-occurrence, then measured on validation; extra labels do not invalidate a hit.',
                          x=64,top=183,width=1152,size=15,color=MUTED,height=40)])

    style_names={'all':'All comments','emoji_heavy_ge_2':'Emoji-heavy (≥ 2)','short_le_5':'Short (≤ 5 tokens)','teencode':'Teencode'}
    style_rows=[]
    for key,name in style_names.items():
        runs=[style_tables[m].loc[key] for m in NAMES]
        if len({int(a.n) for a in runs})!=1 or any(a.split!='val' for a in runs):
            raise ValueError('Text-style comparisons must share validation groups')
        style_rows.append([name,f'{int(runs[0].n):,}']+[f'{a.macro_f1:.4f}' for a in runs]
                          +[f'{100*(runs[2].macro_f1-runs[1].macro_f1):+.2f}'])
    add('full_table','11 / Q4','Q4 · How does performance vary by text style?',
        '27-output study • validation Macro-F1 over all 27 emotions • threshold 0.5',
        'PhoBERT leads in every style group; its largest gap over BiGRU is teencode (+22.13 pp).',
        'Sources: current analysis/27_emotion27/*/tables/*_styles.csv; neural_style_comparison.csv. Δ = PhoBERT − BiGRU.',
        rq='Q4',
        table=table(['Text-style group','n','TF-IDF + LR','BiGRU','PhoBERT','Δ Pho−GRU (pp)'],style_rows,
                    [240,100,190,180,200,242],row_height=64,alignments=[0,2,2,2,2,2],
                    highlights=[(i,4) for i in range(4)]+[(3,5)]),
        caption='Groups overlap and are heuristic subsets, not a partition. Short = ≤ 5 whitespace tokens; emoji-heavy = ≥ 2 emoji matches; teencode = lexicon match. Group sizes differ; no significance claim.')

    series=[]
    for model,color in [('tfidf_lr',BLUE),('gru',TEAL),('phobert',NAVY)]:
        sweep=pd.read_csv(OUT/f'analysis/27_emotion27/{model}/tables/{model}_thresholds_val.csv')
        series.append(dict(name=NAMES[model],x=sweep.threshold.astype(float).tolist(),y=sweep.macro_f1.astype(float).tolist(),color=color))
    gains=[100*(threshold.loc[name].exploratory_macro-threshold.loc[name].default_macro) for name in NAMES.values()]
    add('threshold','12 / Q5','Q5 · Lower thresholds affect models differently',
        '27-output study • fixed checkpoint per model • validation grid: 0.05–0.80, step 0.05',
        'TF-IDF benefits most in exploratory F1; PhoBERT is almost unchanged. Keep the main comparison at 0.5.',
        'Source: current analysis/27_emotion27/*/tables/*_thresholds_val.csv. Grid 0.05–0.80, step 0.05.',
        rq='Q5',
        chart=dict(series=series,x=64,y=213,width=766,height=314,xlim=[.05,.8],ylim=[0,.7],
                   xticks=[.05,.2,.35,.5,.65,.8],yticks=[0,.2,.4,.6],xlabel='Decision threshold',ylabel='Validation Macro-F1',
                   markers=[dict(x=float(threshold.loc[name].best_exploratory_threshold),y=float(threshold.loc[name].exploratory_macro),color=color)
                            for name,color in zip(NAMES.values(),[BLUE,TEAL,NAVY])]),
        stats=[dict(x=908,top=491-i*120,width=308,number=f'+{gain:.3f} pp',
                    label=f'{name} · τ 0.50 → {threshold.loc[name].best_exploratory_threshold:.2f}',color=color)
               for i,(name,gain,color) in enumerate(zip(NAMES.values(),gains,[BLUE,TEAL,NAVY]))],
        caption='Callouts show the F1 change from 0.5 to the best grid point. Circles mark grid maxima; these are not test improvements.')

    threshold_rows=[]
    for model,name in NAMES.items():
        th=threshold.loc[name];base=r.loc[(27,model)]
        threshold_rows.extend([[name,'Default','0.50',f'{base.macro_f1:.4f}',f'{base.hamming_loss:.4f}','—'],
                               [name,'Exploratory best',f'{th.best_exploratory_threshold:.2f}',f'{th.exploratory_macro:.4f}',
                                f'{th.exploratory_hamming:.4f}',f'{100*(th.exploratory_macro-base.macro_f1):+.3f}']])
    add('full_table','12 / Q5','Q5 · F1 gains come with more label errors',
        '27-output study • same validation examples and checkpoint within each model • Macro-F1-selected grid points',
        'Every best-F1 grid point has higher Hamming Loss; tuning F1 does not optimize every metric.',
        'Sources: model_comparison.csv (27 outputs); threshold_sensitivity.csv. Δ F1 versus each model’s default, in percentage points.',
        rq='Q5',
        table=table(['Model','Threshold setting','τ','Macro-F1 ↑','Hamming ↓','Δ F1 (pp)'],threshold_rows,
                    [220,245,95,200,200,192],row_height=43,group_starts=[2,4],alignments=[0,0,2,2,2,2],
                    highlights=[(i,j) for i in (1,3,5) for j in (2,3,5)]),
        caption='Best = highest Macro-F1 on the explored validation grid, not an independently validated optimum. Main results remain at 0.5. LRAP is cutoff-independent; test generalization is unknown.')

    cases=pd.read_csv(OUT/'phobert28_error_review.csv',dtype={'sample_id':str}).set_index('sample_id')
    fp,fn=cases.loc['30099'],cases.loc['6238']
    add('two_column','13 / ERROR ANALYSIS','Neutral errors: implied affect versus surface cues',
        'Two illustrative cases from the six-case preliminary, AI-assisted review; no raw comments reproduced.',
        'Interpretations are hypotheses from selected cases, not established error causes.',
        'Source: phobert28_error_review.csv; current PhoBERT-28 validation predictions. Team confirmation remains pending.',
        panels=[dict(x=64,heading='False positive / Neutral',id='Validation ID 30099',color=TEAL,
                     truth='Annotated: relief',prediction='Predicted: Neutral',probability=f'P(Neutral) = {fp.neutral_probability:.4f}',
                     hypothesis='Hypothesis: implied emotion in an explanatory comment.'),
                dict(x=664,heading='False negative / Neutral',id='Validation ID 6238',color=AMBER,
                     truth='Annotated: Neutral',prediction='Predicted: anger + annoyance',probability=f'P(Neutral) = {fn.neutral_probability:.5f}',
                     hypothesis='Hypothesis: profanity may resemble negative affect.')],
        caption='Review scope: three confident Neutral false positives and three low-probability Neutral misses, selected deliberately.')

    add('final_plan','14 / FINAL EXPERIMENT','Is one threshold suitable for every emotion?',
        'Planned controlled decoding comparison • no new test results claimed',
        'Hold model weights fixed; change the decoder, then evaluate held-out test once.',
        'Source: course Final RQ1 and current midterm protocol. Thresholds selected strictly on validation predictions.',
        checkpoint='Freeze the best neural checkpoint: current PhoBERT-27',
        strategies=[['Fixed threshold','Use τ = 0.5 for every emotion.'],['Global tuned','Select one shared threshold on validation.'],['Per-label tuned','Select a separate threshold for each emotion.']],
        nodes=[['1. Tune on validation','Compare decoders; review FP / FN'],['2. Lock all choices','Checkpoint + thresholds + protocol'],['3. Evaluate test once','Report scores and limitations']])

    add('conclusions','15 / CONCLUSIONS','What the current study supports',
        'Three evidence-backed observations; a focused next experiment',
        'PhoBERT is the strongest model in each setting; Neutral and decoder design remain open questions.',
        'Sources: current Kaggle 08–09 artifacts; ViGoEmotions (EACL 2026, doi:10.18653/v1/2026.eacl-long.129); PhoBERT (2020).',
        findings=[['PhoBERT leads within both label spaces',f'Macro-F1: {p27.macro_f1:.4f} over 27 emotions; {p28.macro_f1:.4f} over all 28 labels.'],
                  ['Neutral is not a shared-emotion F1 gain here','The 28-output projection is 1.38 pp lower than PhoBERT-27.'],
                  ['Evaluate decoding with a frozen backbone','Explore fixed, global and per-label thresholds without retraining.']],
        limits=['One seed; validation-based selection.','No model test scores or significance claims.','Original stratification algorithm unverified.','Heuristic styles; error review needs confirmation.'])
    if len(slides)!=len(SPEECH):
        raise ValueError('Every slide must have matching speaker notes')
    for slide,speech in zip(slides,SPEECH):
        slide['notes']=speech
    return slides
