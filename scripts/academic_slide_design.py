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
 'For individual emotions, PhoBERT performs best on gratitude and worst on disapproval. The '
 'correlation between training frequency and per-label F1 is approximately 0.018, with a p-value '
 'of 0.928. These descriptive results do not support a simple rule that more frequent labels are '
 'always easier. The report includes every label for all three models.',
 'Our most frequent training pair is disappointment and sadness. PhoBERT predicts both correctly '
 'in 63 of 122 validation cases containing both labels. For text styles, its largest advantage '
 'over BiGRU is in teencode, at about 22.1 percentage points of Macro-F1. Teencode means informal '
 'online spellings and abbreviations. Our style groups use heuristics and can overlap, so this '
 'finding needs cautious interpretation.',
 'We also vary the threshold while keeping the checkpoint fixed. For PhoBERT, the best validation '
 'grid point is 0.35, with Macro-F1 0.5978, compared with 0.5974 at zero point five. The gain is '
 'tiny and Hamming Loss worsens. This is exploratory analysis, not a test improvement; the main '
 'comparison stays at zero point five.',
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
    style=pd.read_csv(OUT/'neural_style_comparison.csv').set_index('style')
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
    add('analysis','08 / Q1–Q2','Performance varies by label, beyond raw frequency',
        'PhoBERT, 27 outputs • per-label validation F1 at threshold 0.5',
        'Gratitude is strongest; disapproval is weakest. Frequency shows little monotonic association in this run.',
        'Sources: per_label_metrics.csv; frequency_f1.csv; current PhoBERT-27 per-label table. Descriptive, single-seed analysis.',
        bars=dict(labels=subset.label.tolist(),values=subset.f1.astype(float).tolist(),x=64,top=479,width=566,height=269,
                  maximum=1,value_format='f1',label_width=157,
                  colors_by_row=[TEAL]*3+[BLUE]*3),
        scatter=dict(series=[dict(name='27 emotion labels',x=label.train_count.astype(int).tolist(),y=label.f1.astype(float).tolist(),color=TEAL)],
                     x=690,y=214,width=526,height=280,xlim=[0,3200],ylim=[0,1],xticks=[0,1000,2000,3000],yticks=[0,.25,.5,.75,1],
                     xlabel='Positive training instances',ylabel='Validation per-label F1',scatter=True,show_legend=False),
        association=f'Spearman ρ = {freq.loc["PhoBERT","spearman_rho"]:.3f}; p = {freq.loc["PhoBERT","p_value"]:.3f}',
        caption='Selected top three and bottom three labels; full 27-label results remain in the report.')

    st=style.loc['teencode'];th=threshold.loc['PhoBERT'];ex=extremes.loc['PhoBERT']
    add('full_table','09 / Q1–Q5','Research questions: a compact evidence checklist',
        'Current 27-output study • frozen checkpoints • validation-only diagnostics',
        'The findings depend on the label, text style, metric and decision threshold.',
        'Sources: label_extremes, frequency_f1, pair_preservation, neural_style_comparison, threshold_sensitivity.csv.',
        table=table(['RQ','What was measured?','Selected result','Interpretation / boundary'],[
            ['Q1','Highest / lowest label F1',f'Gratitude {ex.highest_f1:.4f}; disapproval {ex.lowest_f1:.4f}','PhoBERT-27; full label table in report'],
            ['Q2','Training frequency vs F1',f'Spearman ρ {freq.loc["PhoBERT","spearman_rho"]:.3f}; p {freq.loc["PhoBERT","p_value"]:.3f}','Descriptive association, not causation'],
            ['Q3','Disappointment + sadness',f'PhoBERT: {int(pairs.loc["PhoBERT","joint_TP"])}/122 jointly correct ({pairs.loc["PhoBERT","joint_recall"]*100:.2f}%)','Pair chosen from train: 904 occurrences'],
            ['Q4','PhoBERT advantage by style',f'Teencode: {st.phobert:.4f} vs GRU {st.gru:.4f}; +{st.delta_pp:.2f} pp','Largest gap among heuristic groups; n = 307'],
            ['Q5','Fixed vs explored threshold',f'τ 0.50 → 0.35: Macro-F1 {th.default_macro:.4f} → {th.exploratory_macro:.4f}','Tiny F1 gain; Hamming Loss worsens']
        ],[70,300,435,347],row_height=66,font_size=16,alignments=[0,0,0,0]))

    series=[]
    for model,color in [('tfidf_lr',BLUE),('gru',TEAL),('phobert',NAVY)]:
        sweep=pd.read_csv(OUT/f'analysis/27_emotion27/{model}/tables/{model}_thresholds_val.csv')
        series.append(dict(name=NAMES[model],x=sweep.threshold.astype(float).tolist(),y=sweep.macro_f1.astype(float).tolist(),color=color))
    add('threshold','10 / DECODING','PhoBERT gains little from this global threshold grid',
        '27 outputs • same checkpoint • thresholds explored on validation only',
        'Keep the main comparison at 0.5; test generalization remains unknown.',
        'Source: current analysis/27_emotion27/*/tables/*_thresholds_val.csv. Grid 0.05–0.80, step 0.05.',
        chart=dict(series=series,x=64,y=213,width=766,height=314,xlim=[.05,.8],ylim=[0,.7],
                   xticks=[.05,.2,.35,.5,.65,.8],yticks=[0,.2,.4,.6],xlabel='Decision threshold',ylabel='Validation Macro-F1'),
        stats=[dict(x=908,top=491,width=308,number=f'{th.default_macro:.4f}',label='Macro-F1 at τ = 0.50',color=NAVY),
               dict(x=908,top=371,width=308,number=f'{th.exploratory_macro:.4f}',label='Macro-F1 at τ = 0.35'),
               dict(x=908,top=251,width=308,number=f'+{100*(th.exploratory_macro-th.default_macro):.3f} pp',label='exploratory F1 difference')],
        caption=f'Hamming Loss rises from {p27.hamming_loss:.4f} to {th.exploratory_hamming:.4f}; LRAP is independent of the cutoff.')

    cases=pd.read_csv(OUT/'phobert28_error_review.csv',dtype={'sample_id':str}).set_index('sample_id')
    fp,fn=cases.loc['30099'],cases.loc['6238']
    add('two_column','11 / ERROR ANALYSIS','Neutral errors: implied affect versus surface cues',
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

    add('final_plan','12 / FINAL EXPERIMENT','Is one threshold suitable for every emotion?',
        'Planned controlled decoding comparison • no new test results claimed',
        'Hold model weights fixed; change the decoder, then evaluate held-out test once.',
        'Source: course Final RQ1 and current midterm protocol. Thresholds selected strictly on validation predictions.',
        checkpoint='Freeze the best neural checkpoint: current PhoBERT-27',
        strategies=[['Fixed threshold','Use τ = 0.5 for every emotion.'],['Global tuned','Select one shared threshold on validation.'],['Per-label tuned','Select a separate threshold for each emotion.']],
        nodes=[['1. Tune on validation','Compare decoders; review FP / FN'],['2. Lock all choices','Checkpoint + thresholds + protocol'],['3. Evaluate test once','Report scores and limitations']])

    add('conclusions','13 / CONCLUSIONS','What the current study supports',
        'Three evidence-backed observations; a focused next experiment',
        'PhoBERT is the strongest model in each setting; Neutral and decoder design remain open questions.',
        'Sources: current Kaggle 08–09 artifacts; ViGoEmotions (EACL 2026, doi:10.18653/v1/2026.eacl-long.129); PhoBERT (2020).',
        findings=[['PhoBERT leads within both label spaces',f'Macro-F1: {p27.macro_f1:.4f} over 27 emotions; {p28.macro_f1:.4f} over all 28 labels.'],
                  ['Neutral is not a shared-emotion F1 gain here','The 28-output projection is 1.38 pp lower than PhoBERT-27.'],
                  ['Evaluate decoding with a frozen backbone','Explore fixed, global and per-label thresholds without retraining.']],
        limits=['One seed; validation-based selection.','No model test scores or significance claims.','Original stratification algorithm unverified.','Heuristic styles; error review needs confirmation.'])
    for slide,speech in zip(slides,SPEECH):
        slide['notes']=speech
    return slides
