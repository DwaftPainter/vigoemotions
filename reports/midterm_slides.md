# ViGoEmotions — Midterm research presentation

18 slides • English • validation results from the verified Kaggle 20-epoch studies.

Experiment inputs are read-only. Slide definitions: reports/midterm_slide_design.json.

## 1. ViGoEmotions

**Layout:** title · **Section:** MIDTERM

Three models. Two label spaces. One shared 20-epoch protocol.

What changes when Neutral becomes a learned output?

**3:** model families

**20:** epochs for every neural run

**2:** label settings: 27 and 28

**Source / protocol:** ViGoEmotions / ATI 2026 / Kaggle notebooks 08–09 / Validation only

**Speaker notes:** Good morning. Our project recognizes emotions in Vietnamese social-media comments. Because one comment can express several emotions, this is a multi-label task. The dataset contains twenty thousand, six hundred and sixty-four comments. We compare three models, with and without an explicit Neutral output. I will briefly cover the setup, then focus on our results and analysis.

## 2. Dataset, split and emotion frequency

**Layout:** table_chart · **Section:** 01 / DATA

20,664 source comments • fixed supplied partitions • label statistics from training data

**1.859:** emotions / train comment

**11 / 47:** median tokens / characters

| Split | Source n | Used n | Role |
| --- | --- | --- | --- |
| Train | 16,531 | 16,530 | Fit model parameters |
| Validation | 2,066 | 2,066 | Select checkpoint / explore thresholds |
| Test | 2,067 | 2,067 | Held out from model evaluation |
| Total | 20,664 | 20,663 | One train–test text overlap removed |

**Chart — selected labels:**

- amusement: 2,868
- sadness: 2,785
- annoyance: 2,662
- joy: 1,614
- anger: 1,589
- relief: 635

Most frequent five + least frequent label

**Takeaway:** Preserve the split; fit text representations on train only.

**Source / protocol:** Sources: current study annotation_counts.csv; fixed split IDs; PhoBERT-27 per-label table. No test model scores.

**Speaker notes:** Comments average about one point eight six emotions, and the median length is eleven whitespace tokens. We keep the supplied split, remove one training copy of a test comment, and fit vocabulary only on training data. Unicode and whitespace are normalized. Our distribution audit does not prove the original stratification algorithm, so that requirement still needs clarification.

## 3. 27 vs 28 outputs: one additional decision

**Layout:** encoding · **Section:** 02 / TARGETS

The label mapping and validation examples stay fixed; Neutral becomes an explicit target.

| Property | 27-emotion model | 28-label model |
| --- | --- | --- |
| Output space | 27 emotion decisions | 27 emotions + Neutral |
| Neutral-only target | All 27 emotion entries are zero | Neutral = 1; emotion entries = 0 |
| Mixed annotation | Retain the emotion entries | Retain emotions and Neutral |
| Reported full Macro-F1 | Average over 27 labels | Average over 28 labels |

**Takeaway:** A missing emotion prediction is not automatically a Neutral prediction.

**Source / protocol:** Sources: config27.json; evidence/28/config.json; annotation_counts.csv. Encoding examples are schematic targets.

**Speaker notes:** The twenty-seven-output model predicts emotions only. Neutral-only comments have an all-zero emotion target. The twenty-eight-output model adds a learned Neutral label. We preserve mixed annotations, where Neutral appears with emotions. Both neural settings use sigmoid outputs and binary cross-entropy because labels are not mutually exclusive. Validation contains ninety-four Neutral-positive examples.

## 4. Three routes from text to multi-label scores

**Layout:** pipeline · **Section:** 03 / METHODS

Model-specific text representations • 27 or 28 independent label decisions

**TF-IDF + LR:** Regex tokens (1–2 grams) → TF-IDF (Fit vocabulary on train) → OvR classifiers (One LR per label) → Label scores (Threshold at 0.5)
**BiGRU:** Token IDs (Lowercase + whitespace) → Embedding (128 dimensions) → BiGRU + linear (Forward + backward) → Sigmoid scores (27 / 28 outputs)
**PhoBERT:** Segment + BPE (VnCoreNLP tokenizer) → PhoBERT backbone (Pretrained Vietnamese) → Classification head (Train with backbone) → Sigmoid scores (27 / 28 outputs)

Shared preprocessing: Unicode NFC + whitespace normalization; retain Vietnamese accents and negation.

**Takeaway:** Text representation changes across models; the label decision remains independent for each output.

**Source / protocol:** Sources: frozen run configurations; train_neural.py; neutral_28_labels.py. Neural training: BCEWithLogitsLoss.

**Speaker notes:** This diagram shows the three processing routes. TF-IDF uses lexical features and one classifier per label. BiGRU learns embeddings and reads sequences in both directions. PhoBERT uses Vietnamese word segmentation and BPE before a pretrained backbone and a new classification head. The neural routes use independent sigmoid outputs; there is no single-label softmax decision.

## 5. Architectures and training configurations

**Layout:** full_table · **Section:** 03 / METHODS

Shared seed 42 • fixed split • threshold 0.5 • Tesla T4 for neural models

| Model | Representation / architecture | Parameters 27 / 28 | Batch / learning rate | Budget |
| --- | --- | --- | --- | --- |
| TF-IDF + OvR LR | Unigram/bigram; up to 40k features; liblinear, C = 1 | 0.941M / 0.975M | Not neural | max_iter = 1,000 |
| BiGRU | Train from scratch; embed/hidden 128; length 128 | 2.128M / 2.129M | 64 / 0.001 | 20 epochs |
| PhoBERT | VnCoreNLP + BPE; full fine-tuning; length 256 | 135.019M / 135.020M | 16 / 2e-5 | 20 epochs |

Select by validation emotion27 Macro-F1 at 0.5, including for the 28-output model; ties keep the earliest epoch.

**Takeaway:** Equal epochs control the training budget; they do not imply equal compute.

**Source / protocol:** Sources: frozen manifests and six run.json files. Parameter counts shown as 27 / 28 outputs; rounded to millions.

**Speaker notes:** We compare TF-IDF with one-versus-rest Logistic Regression, a bidirectional GRU trained from scratch, and full PhoBERT fine-tuning. Both neural models complete twenty epochs in each setting. TF-IDF uses a solver limit instead. We share the seed and split, and select checkpoints by validation Macro-F1 on the same twenty-seven emotions at threshold zero point five.

## 6. PhoBERT leads within each label setting

**Layout:** full_table · **Section:** 04 / RESULTS

Validation n = 2,066 • threshold 0.5 • separate full-label evaluations for 27 and 28 outputs

| Labels | Model | Macro-F1 ↑ | Micro-F1 ↑ | Hamming ↓ | LRAP ↑ | Best ep. | Train time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | TF-IDF + OvR LR | 0.2388 | 0.2712 | 0.0606 | 0.6407 | — | 2.75 s |
| 27 | BiGRU | 0.4220 | 0.4590 | 0.0626 | 0.6191 | 17 | 80.3 s |
| 27 | PhoBERT | 0.5974 | 0.5996 | 0.0550 | 0.7419 | 14 | 4.05 h |
| 28 | TF-IDF + OvR LR | 0.2310 | 0.2662 | 0.0601 | 0.6096 | — | 2.52 s |
| 28 | BiGRU | 0.4086 | 0.4483 | 0.0628 | 0.5851 | 20 | 83.6 s |
| 28 | PhoBERT | 0.5783 | 0.5829 | 0.0575 | 0.7077 | 15 | 3.90 h |

Both neural models completed 20 epochs. “Best ep.” identifies the selected checkpoint, not the number of epochs trained.

**Takeaway:** Rank models within each block; the two Macro-F1 blocks evaluate different label sets.

**Source / protocol:** Source: model_comparison.csv. ↑ higher / ↓ lower is better. CPU TF-IDF; T4 neural. Times exclude preprocessing.

**Speaker notes:** The table separates full twenty-seven-emotion and twenty-eight-label evaluation. PhoBERT leads within each block: Macro-F1 is zero point five nine seven four and zero point five seven eight three, respectively. The additional columns show Micro-F1, Hamming Loss, rankings, selected epochs, and measured training time. Different label sets mean these two overall F1 scores should not be directly subtracted.

## 7. Adding Neutral does not raise shared-emotion F1 here

**Layout:** paired · **Section:** 05 / NEUTRAL COMPARISON

Same 27 emotion columns • same validation IDs • shared checkpoint selection • one seed

| Model | 27-output F1 | 28 → 27 F1 | Δ (pp) |
| --- | --- | --- | --- |
| TF-IDF + OvR LR | 0.2388 | 0.2388 | +0.00 |
| BiGRU | 0.4220 | 0.4195 | -0.25 |
| PhoBERT | 0.5974 | 0.5836 | -1.38 |

This controls the epoch budget and scoring label space. It does not establish a general causal effect or statistical significance.

**Takeaway:** PhoBERT changes by −1.38 percentage points on the shared 27-emotion evaluation.

**Source / protocol:** Source: paired_emotion27_delta.csv. Δ = 28-output projection minus 27-output score, in percentage points.

**Speaker notes:** For a comparison on identical labels, we score only the twenty-seven emotion columns of the expanded model. PhoBERT then reaches 0.5836, about 1.38 percentage points below the twenty-seven-output model. TF-IDF is unchanged and BiGRU decreases slightly. This run shows no improvement on the shared emotions, but one seed cannot establish a general causal conclusion.

## 8. A learned Neutral output still makes substantial errors

**Layout:** neutral · **Section:** 06 / NEUTRAL DETECTION

28-label setting only • 94 Neutral-positive validation examples: 75 Neutral-only + 19 mixed

**45:** correct Neutral detections

**49:** missed Neutral examples

**67:** false Neutral predictions

| Model | Precision | Recall | F1 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + OvR LR | 0.5000 | 0.0106 | 0.0208 | 1 | 1 | 93 |
| BiGRU | 0.1778 | 0.0851 | 0.1151 | 8 | 37 | 86 |
| PhoBERT | 0.4018 | 0.4787 | 0.4369 | 45 | 67 | 49 |

An empty prediction is not a substitute for an explicitly trained Neutral label.

**Takeaway:** PhoBERT finds 45 of 94 Neutral examples, with 67 false alarms.

**Source / protocol:** Source: neutral_metrics.csv. Threshold 0.5; support 94 for all models. No Neutral metric for the 27-output model.

**Speaker notes:** PhoBERT identifies 45 of the ninety-four Neutral-positive validation examples. It misses 49 and produces 67 false positives, giving Neutral F1 of 0.4369. So the additional label is learned, but still difficult. The twenty-seven-output model has no Neutral classifier; its Neutral F1 is undefined, not zero. An empty prediction is also not automatically Neutral.

## 9. Twenty epochs trained; the last checkpoint need not win

**Layout:** curves · **Section:** 07 / TRAINING DYNAMICS

Both curves in each panel use the same 27-emotion checkpoint-selection metric.

**Chart:** BiGRU — 27 outputs · best 17; 28 outputs · best 20.
**Chart:** PhoBERT — 27 outputs · best 14; 28 outputs · best 15.

Train BCE continued to decrease; validation loss was not recorded. These curves alone do not confirm overfitting.

**Takeaway:** Best PhoBERT checkpoints: epoch 14 for 27 outputs and epoch 15 for 28 outputs.

**Source / protocol:** Sources: evidence/{27,28}/{gru,phobert}/run.json. All histories contain epochs 1–20; no early stopping.

**Speaker notes:** These curves use the same twenty-seven-emotion checkpoint-selection metric for both output settings. Both neural models completed twenty epochs. The selected PhoBERT checkpoints are at epochs fourteen and fifteen. Selection is based on validation, not simply the final epoch. Training loss continued to decrease, but validation loss was not recorded, so we do not claim confirmed overfitting.

## 10. Q1 · Which emotions are easiest and hardest?

**Layout:** table_chart · **Section:** 08 / Q1

27-output study • per-label validation F1 at threshold 0.5 • bars show selected PhoBERT labels

| Model | Highest label / F1 | Lowest label / F1 |
| --- | --- | --- |
| TF-IDF + OvR LR | gratitude<br>0.6797 | realization<br>0.0000 |
| BiGRU | gratitude<br>0.7701 | realization<br>0.0940 |
| PhoBERT | gratitude<br>0.8370 | disapproval<br>0.3623 |

**Chart — selected labels:**

- gratitude: 0.8370
- embarrassment: 0.7729
- pride: 0.7397
- realization: 0.4641
- excitement: 0.4606
- disapproval: 0.3623

PhoBERT: top three and bottom three labels

Per-label F1 reveals weaknesses hidden by the overall average. These are single-run validation results.

**Takeaway:** Gratitude ranks highest for every model; the weakest label depends on the model.

**Source / protocol:** Sources: label_extremes.csv; current analysis/27_emotion27/*/tables/*_per_label.csv. Full label tables in report.

**Speaker notes:** Gratitude is the easiest label for all three models. The weakest label is realization for TF-IDF and BiGRU, but disapproval for PhoBERT. The bars show selected PhoBERT labels; the table compares each model’s extremes. Overall F1 hides these differences.

## 11. Q2 · Does label frequency predict F1?

**Layout:** table_chart · **Section:** 09 / Q2

27-output study • training counts versus validation per-label F1 • one point per emotion

| Model | Spearman ρ | p-value |
| --- | --- | --- |
| TF-IDF + OvR LR | 0.242 | 0.223 |
| BiGRU | 0.313 | 0.112 |
| PhoBERT | 0.018 | 0.928 |

**Chart:** training frequency versus validation F1 across all 27 emotion labels.

Frequency alone does not explain label difficulty. Association is not causation; one seed was evaluated.

**Takeaway:** PhoBERT shows almost no monotonic frequency–F1 association in this run.

**Source / protocol:** Sources: frequency_f1.csv; current PhoBERT-27 per-label table. Descriptive correlation across 27 labels per model.

**Speaker notes:** For Q2, each dot represents one emotion under PhoBERT. Its frequency–F1 correlation is only 0.018. The table also shows TF-IDF and BiGRU. These descriptive correlations do not establish causation; frequency alone cannot explain which labels are difficult.

## 12. Q3 · How often are both emotions recovered?

**Layout:** table_chart · **Section:** 10 / Q3

27-output study • threshold 0.5 • disappointment + sadness (top training pair: 904)

| Model | Joint TP | Val. support | Joint recall |
| --- | --- | --- | --- |
| TF-IDF + OvR LR | 2 | 122 | 1.64% |
| BiGRU | 37 | 122 | 30.33% |
| PhoBERT | 63 | 122 | 51.64% |

**Chart — selected labels:**

- TF-IDF + LR: 1.64%
- BiGRU: 30.33%
- PhoBERT: 51.64%

JOINT RECALL · 0–100% scale

Joint recall = cases with both labels predicted ÷ cases annotated with both labels.

The pair was selected by training co-occurrence, then measured on validation; extra labels do not invalidate a hit.

**Takeaway:** PhoBERT recovers both emotions in 63/122 positive-pair cases; about half are still missed.

**Source / protocol:** Sources: pair_preservation.csv; current 27-output pair tables. Extra predicted labels allowed; this is joint recall, not precision.

**Speaker notes:** For Q3, we choose disappointment and sadness because they co-occur most often in training. Validation contains 122 comments annotated with both. TF-IDF recovers both in two cases, BiGRU in thirty-seven, and PhoBERT in sixty-three. Joint recall is therefore 1.64, 30.33 and 51.64 percent. Other predicted labels are allowed, so this does not measure pair precision.

## 13. Q4 · How does performance vary by text style?

**Layout:** full_table · **Section:** 11 / Q4

27-output study • validation Macro-F1 over all 27 emotions • threshold 0.5

| Text-style group | n | TF-IDF + LR | BiGRU | PhoBERT | Δ Pho−GRU (pp) |
| --- | --- | --- | --- | --- | --- |
| All comments | 2,066 | 0.2388 | 0.4220 | 0.5974 | +17.54 |
| Emoji-heavy (≥ 2) | 199 | 0.1690 | 0.3506 | 0.4789 | +12.83 |
| Short (≤ 5 tokens) | 343 | 0.3310 | 0.4303 | 0.5654 | +13.51 |
| Teencode | 307 | 0.1671 | 0.3333 | 0.5547 | +22.13 |

Groups overlap and are heuristic subsets, not a partition. Short = ≤ 5 whitespace tokens; emoji-heavy = ≥ 2 emoji matches; teencode = lexicon match. Group sizes differ; no significance claim.

**Takeaway:** PhoBERT leads in every style group; its largest gap over BiGRU is teencode (+22.13 pp).

**Source / protocol:** Sources: current analysis/27_emotion27/*/tables/*_styles.csv; neural_style_comparison.csv. Δ = PhoBERT − BiGRU.

**Speaker notes:** Q4 compares every available text-style group using the same twenty-seven labels. PhoBERT leads in all four rows. Its largest advantage over BiGRU is teencode: 22.13 percentage points on 307 examples. These groups use heuristic rules and overlap. Their sample counts differ, so this is descriptive rather than a controlled style effect.

## 14. Q5 · Lower thresholds affect models differently

**Layout:** threshold · **Section:** 12 / Q5

27-output study • fixed checkpoint per model • validation grid: 0.05–0.80, step 0.05

**+24.555 pp:** TF-IDF + OvR LR · τ 0.50 → 0.15

**+1.860 pp:** BiGRU · τ 0.50 → 0.20

**+0.043 pp:** PhoBERT · τ 0.50 → 0.35

**Chart:** validation Macro-F1 versus decision threshold for TF-IDF, BiGRU and PhoBERT.

Callouts show the F1 change from 0.5 to the best grid point. Circles mark grid maxima; these are not test improvements.

**Takeaway:** TF-IDF benefits most in exploratory F1; PhoBERT is almost unchanged. Keep the main comparison at 0.5.

**Source / protocol:** Source: current analysis/27_emotion27/*/tables/*_thresholds_val.csv. Grid 0.05–0.80, step 0.05.

**Speaker notes:** Q5 keeps each checkpoint fixed and varies only the threshold. The three curves respond differently. TF-IDF gains about 24.55 percentage points of validation Macro-F1; BiGRU gains 1.86; PhoBERT gains only 0.043. Circles mark the best grid points, not test improvements.

## 15. Q5 · F1 gains come with more label errors

**Layout:** full_table · **Section:** 12 / Q5

27-output study • same validation examples and checkpoint within each model • Macro-F1-selected grid points

| Model | Threshold setting | τ | Macro-F1 ↑ | Hamming ↓ | Δ F1 (pp) |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + OvR LR | Default | 0.50 | 0.2388 | 0.0606 | — |
| TF-IDF + OvR LR | Exploratory best | 0.15 | 0.4844 | 0.0867 | +24.555 |
| BiGRU | Default | 0.50 | 0.4220 | 0.0626 | — |
| BiGRU | Exploratory best | 0.20 | 0.4406 | 0.0784 | +1.860 |
| PhoBERT | Default | 0.50 | 0.5974 | 0.0550 | — |
| PhoBERT | Exploratory best | 0.35 | 0.5978 | 0.0587 | +0.043 |

Best = highest Macro-F1 on the explored validation grid, not an independently validated optimum. Main results remain at 0.5. LRAP is cutoff-independent; test generalization is unknown.

**Takeaway:** Every best-F1 grid point has higher Hamming Loss; tuning F1 does not optimize every metric.

**Source / protocol:** Sources: model_comparison.csv (27 outputs); threshold_sensitivity.csv. Δ F1 versus each model’s default, in percentage points.

**Speaker notes:** The table compares the default and exploratory best thresholds for every model. Lower thresholds improve Macro-F1 but increase Hamming Loss in all three cases. Choosing on validation can overestimate generalization. We keep the main comparison at 0.5 and will lock decoding choices before evaluating test.

## 16. Neutral errors: implied affect versus surface cues

**Layout:** two_column · **Section:** 13 / ERROR ANALYSIS

Two illustrative cases from the six-case preliminary, AI-assisted review; no raw comments reproduced.

**False positive / Neutral — Validation ID 30099:** Annotated: relief; Predicted: Neutral; P(Neutral) = 0.9965. Hypothesis: implied emotion in an explanatory comment.
**False negative / Neutral — Validation ID 6238:** Annotated: Neutral; Predicted: anger + annoyance; P(Neutral) = 0.00005. Hypothesis: profanity may resemble negative affect.

Review scope: three confident Neutral false positives and three low-probability Neutral misses, selected deliberately.

**Takeaway:** Interpretations are hypotheses from selected cases, not established error causes.

**Source / protocol:** Source: phobert28_error_review.csv; current PhoBERT-28 validation predictions. Team confirmation remains pending.

**Speaker notes:** Two examples illustrate the six-case preliminary, AI-assisted review. On the left, a comment annotated as relief is predicted as Neutral. On the right, a Neutral comment is assigned anger and annoyance. Implied affect and profanity are possible explanations, not proven causes. We show IDs and abstract observations, and the interpretations still need team confirmation.

## 17. Is one threshold suitable for every emotion?

**Layout:** final_plan · **Section:** 14 / FINAL EXPERIMENT

Planned controlled decoding comparison • no new test results claimed

Freeze the best neural checkpoint: current PhoBERT-27

**Fixed threshold:** Use τ = 0.5 for every emotion.
**Global tuned:** Select one shared threshold on validation.
**Per-label tuned:** Select a separate threshold for each emotion.

1. Tune on validation → 2. Lock all choices → 3. Evaluate test once

**Takeaway:** Hold model weights fixed; change the decoder, then evaluate held-out test once.

**Source / protocol:** Source: course Final RQ1 and current midterm protocol. Thresholds selected strictly on validation predictions.

**Speaker notes:** For the final evaluation, we will test whether one threshold suits every emotion. We will freeze the best neural checkpoint and compare the default threshold, a tuned global threshold, and per-label thresholds. We select thresholds on validation, lock all choices, then evaluate test once. We also plan a detailed review of at least twenty false negatives and twenty false positives.

## 18. What the current study supports

**Layout:** conclusions · **Section:** 15 / CONCLUSIONS

Three evidence-backed observations; a focused next experiment

**PhoBERT leads within both label spaces:** Macro-F1: 0.5974 over 27 emotions; 0.5783 over all 28 labels.
**Neutral is not a shared-emotion F1 gain here:** The 28-output projection is 1.38 pp lower than PhoBERT-27.
**Evaluate decoding with a frozen backbone:** Explore fixed, global and per-label thresholds without retraining.

**Limitations:** One seed; validation-based selection. No model test scores or significance claims. Original stratification algorithm unverified. Heuristic styles; error review needs confirmation.

**Takeaway:** PhoBERT is the strongest model in each setting; Neutral and decoder design remain open questions.

**Source / protocol:** Sources: current Kaggle 08–09 artifacts; ViGoEmotions (EACL 2026, doi:10.18653/v1/2026.eacl-long.129); PhoBERT (2020).

**Speaker notes:** To conclude, PhoBERT gives the strongest overall results in both settings. Adding Neutral does not improve shared-emotion F1 in this run, and Neutral prediction still makes substantial errors. Saved Kaggle predictions and histories support these findings. Our next step is controlled threshold evaluation. Thank you; we welcome your questions.
