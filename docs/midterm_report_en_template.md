# ViGoEmotions — Midterm Report

**Midterm snapshot: Kaggle results from 5 November 2026, executed notebooks 08/09.** All four neural runs completed 20 epochs. Six sets of results were recalculated from the saved validation probabilities in the two output archives. Earlier runs with fewer epochs are excluded from the main tables, figures and conclusions.

**Version note:** This is an English translation of the preserved Vietnamese Midterm report. Final plans and pending work below describe the Midterm stage; completed Final results are reported separately.

**Group information:** Names, student IDs and contributions have not been supplied. Add them to `docs/team_information.md` before submission.

## 1. Task and Scope

The task is multi-label emotion recognition in Vietnamese comments. A comment may express several emotions, so each label is an independent binary decision. We evaluate three models under two encodings: 27 emotion outputs and 28 outputs including Neutral. All model scores in this report are validation results at threshold 0.5, with seed 42. The held-out test set was reserved for the Final evaluation.

The requirements come from the Midterm section of `student-guide.pdf` and Midterm A–E of `02-vigoemotions-vietnamese-emotion.pdf`. They cover EDA, valid splitting and preprocessing, three models, Macro-F1, Micro-F1, per-label F1, Hamming Loss, LRAP, parameter counts, training time, error analysis and a proposed Final research question. The accompanying presentation has 15 slides, supported by executable code and a reproducibility README.

## 2. Dataset and Exploratory Analysis

The source CSVs contain 20,664 comments: 16,531 training, 2,066 validation and 2,067 test examples. One training comment overlapping with held-out text was removed, leaving 16,530 retained training examples. The 27 emotion labels, in canonical ID order, are amusement, excitement, joy, love, desire, optimism, caring, pride, admiration, gratitude, relief, approval, realization, surprise, curiosity, confusion, fear, nervousness, remorse, embarrassment, disappointment, sadness, grief, disgust, anger, annoyance and disapproval. Neutral has ID 27.

Head, mid and tail tiers are defined by training-frequency rank, with nine emotions per tier. These are relative frequency tiers. Amusement is the most frequent label and relief the least frequent, with a frequency ratio of approximately 4.52. Positive emotion labels are sparse: mean training emotion cardinality is 1.859, with a maximum of five emotions. Median text length is 11 whitespace-delimited tokens and 47 characters. Whitespace tokens are distinct from Vietnamese words and BPE tokens.

![Training frequency of the 27 emotions, shared by both Kaggle studies.](figures/label_frequency.png)

![Emotion cardinality, whitespace-token length and character length by split. These distributions are not used to select a model using test performance.](figures/distributions.png)

Disappointment–sadness co-occurs 904 times in training, anger–annoyance 699 times and sadness–grief 620 times. Co-occurrence and phi-correlation matrices are constructed from training labels only. Co-occurrence frequency does not establish causality. Relevant Vietnamese text characteristics include teencode, colloquial expressions, profanity, emojis, HTML and non-standard syntax.

![Training-label correlations in canonical 27-emotion order.](figures/train_correlation.png)

## 3. Splits, Preprocessing and Label Encodings

We retain the supplied split and fit vocabulary and IDF on training data only. Text is normalized to Unicode NFC and whitespace is collapsed while preserving diacritics and negation. TF-IDF uses regex tokenization with unigrams and bigrams; BiGRU uses lowercased, whitespace-tokenized text; PhoBERT uses VnCoreNLP segmentation followed by BPE. No additional annotation-correction rule is introduced.

Notebook 07 passes the integrity checks, confirms the fixed IDs and finds no remaining text overlap after filtering. The maximum label-prevalence difference from training is 1.6324 percentage points for validation and 0.9387 for test. This distribution audit does not establish which stratification algorithm produced the supplied split. Acceptance of the supplied split against the brief therefore remained to be clarified with the instructor. Test data participated only in integrity auditing, not model scoring or threshold selection.

With 27 outputs, `[27]` becomes an all-zero emotion vector, while `[2,27]` retains joy. With 28 outputs, `[27]` is a positive Neutral label and `[2,27]` retains both joy and Neutral. The source CSVs contain 725 Neutral-only comments and 306 mixed-Neutral comments. Validation contains 75 Neutral-only and 19 mixed cases, giving 94 positive Neutral examples. Mixed annotations are retained and their inconsistency is disclosed. We use neither softmax nor a rule enforcing exclusivity between Neutral and emotion labels. An empty prediction is not automatically interpreted as Neutral.

## 4. Kaggle Protocol: 20 Epochs

The two notebooks run independently with the same `configs/kaggle_equal_20ep.json` profile, seed 42, split and checkpoint criterion: validation Macro-F1 on the same 27 emotions at threshold 0.5, with the earliest epoch selected on a tie. Every neural run completes 20 epochs without early stopping. The 28-output study uses `selection_macro_f1` on IDs 0–26; full 28-label metrics are reported separately.

TF-IDF uses `max_features=40000`, unigrams and bigrams, `min_df=2` and sublinear TF, followed by 27 or 28 OvR Logistic Regression classifiers with `liblinear`, `C=1` and `max_iter=1000`. TF-IDF has no epoch budget directly comparable to a neural model. BiGRU is trained from scratch with embedding and hidden sizes of 128, dropout 0.3, maximum length 128, batch size 64 and AdamW learning rate 0.001. PhoBERT-base-v2 is fully fine-tuned with maximum length 256, batch size 16, AdamW learning rate 2e-5 and weight decay 0.01, using revision `86cd7fd4c148980922ac11a2cf5e257f2ba639e1`. Neural models use BCEWithLogitsLoss, independent sigmoids and gradient clipping at 1.0. No scheduler, warmup or AMP is configured.

Both environment manifests record Tesla T4, Python 3.13.16, Torch 2.11.0+cu130, Transformers 4.57.3 and scikit-learn 1.6.1. The code uses one GPU per process even when Kaggle displays T4 x2. One seed across two separate sessions does not measure run-to-run variability.

## 5. Main Validation Results

### 5.1. Three Models with 27 Emotion Outputs

{{metrics27}}

### 5.2. Three Models with 28 Outputs Including Neutral

{{metrics28}}

![Macro-F1 at threshold 0.5. Each panel retains its own label scope; the direct 27/28 difference is not treated as the effect of Neutral.](kaggle_equal20_2026-11-05/figures/model_metrics.png)

PhoBERT performs best within each study on the four aggregate metrics, with lower Hamming Loss indicating better performance. Its 27-label Macro-F1 is 0.597410 and its 28-label Macro-F1 is 0.578341. Macro-F1 and Micro-F1 evaluate thresholded predictions; LRAP evaluates the ranking of continuous scores. Subset accuracy is not the primary metric. No statistical significance test or test-set model score was available at this stage.

### 5.3. Epochs, Parameters and Training Time

{{training}}

TF-IDF training time includes fitting the vectorizer and Logistic Regression classifiers. Neural training time includes training, validation after each epoch and checkpoint writing. Preprocessing and downloading are excluded. These CPU/GPU measurements do not support a direct speed ratio, and equal epoch counts do not imply equal compute. PhoBERT training takes approximately 4.05 hours for 27 outputs and 3.90 hours for 28 outputs. After all 20 epochs, the best checkpoint is loaded: PhoBERT selects epochs 14 and 15, while BiGRU selects epochs 17 and 20, respectively.

![Learning curves from the four 20-epoch histories: training BCE and the common validation emotion27 Macro-F1 checkpoint criterion.](kaggle_equal20_2026-11-05/figures/learning_curves.png)

PhoBERT-27 has F1=0 at epoch 1 and improves later, so an early zero score alone does not establish a training failure. Validation F1 fluctuates after the best epoch while training loss continues to decrease. Validation loss was not recorded; training loss alone does not establish convergence or diagnose overfitting.

### 5.4. Comparison on the Same 27 Emotions

{{projection}}

The 28-output models are projected onto the same IDs 0–26, keeping the same 2,066 validation IDs, targets and threshold 0.5. TF-IDF is unchanged; BiGRU decreases by 0.2490 percentage points and PhoBERT by 1.3830 points. PhoBERT's Hamming Loss and LRAP on the shared 27 emotions are also worse for the 28-output model. This run provides no evidence that adding Neutral improves recognition of the other emotions. A shared epoch budget and checkpoint criterion reduce comparison differences, but one seed does not establish a causal effect or statistical significance.

### 5.5. Neutral Evaluated Separately

{{neutral}}

PhoBERT achieves Neutral F1=0.4369, recovering 45 of 94 positive examples, with 49 FN and 67 FP. Neutral is a learned output but remains difficult. The 27-output models have no Neutral output; their Neutral F1 must not be assigned an artificial value of zero.

## 6. Answers to the Five Midterm Questions

### Q1. Which Labels Have the Highest and Lowest F1?

{{extremes}}

All scores use threshold 0.5. The appendix lists every label. Tied labels are listed where applicable; no label is assumed to be universally easy or difficult across models.

### Q2. How Is Training Frequency Related to F1?

{{frequency}}

Spearman correlation summarizes the association across 27 dependent labels. It does not establish causality or imply that increasing sample counts alone will resolve every difficult emotion.

### Q3. Do the Models Preserve Frequently Co-occurring Emotion Pairs?

{{pairs}}

The disappointment–sadness pair is selected from training data before measuring validation performance. Joint recall is the fraction of examples containing both reference labels for which both are predicted. OvR does not directly model label structure; low joint recall may also reflect missed predictions caused by thresholding.

### Q4. On Which Text Styles Does PhoBERT Outperform BiGRU?

{{styles}}

Among the examined subgroups, teencode shows the largest Macro-F1 difference: +22.13 percentage points. Short comments contain at most five whitespace tokens. Emoji-heavy comments contain at least two emoji characters according to the regex; teencode is identified by a heuristic dictionary. These groups overlap and contain detection noise. The table compares the two current 27-output models; its conclusions should not be transferred to a different study.

### Q5. How Sensitive Are Predictions to the Default Threshold of 0.5?

{{sensitivity}}

The sweep uses probabilities from the current 27-output study and a grid from 0.05 to 0.80 in steps of 0.05. The checkpoint selected at threshold 0.5 remains fixed. This is a post-hoc validation analysis: it does not replace the main-table scores or reselect checkpoints. LRAP is unchanged by thresholding. A validation-optimal threshold does not establish a test improvement and should not be transferred directly to another model or label scope.

## 7. Error Analysis of the Current Study

The qualitative review covers six PhoBERT-28 cases: the three most confident Neutral FP and the three Neutral FN with the lowest Neutral probabilities. Cases are selected from the current study's NPZ rather than reused from earlier runs. The exploratory notes require team confirmation. Original comments are not quoted, and this purposively selected sample is not treated as random.

| sample_id | case_type | observation | hypothesis |
| --- | --- | --- | --- |
| 30099 | Neutral FP | A situational explanation is labeled relief; the model predicts only Neutral. | Emotion implied by an explanation may be difficult to recognize. |
| you003230 | Neutral FP | List-like content is labeled approval and realization; the model predicts only Neutral. | List structure and HTML may obscure emotional cues; an ablation is needed to test this. |
| 15569 | Neutral FP | A short remark is labeled surprise; the model predicts only Neutral. | Surprise may depend on discourse particles and limited context. |
| 6238 | Neutral FN | The reference contains only Neutral; the model predicts anger and annoyance. | Profanity and colloquial language may shift predictions toward negative emotions. |
| 21975 | Neutral FN | The reference contains only Neutral; the model predicts anger. | A proper name or context-dependent phrase may cause confusion; the mechanism is undetermined. |
| 1134 | Neutral FN | The reference contains only Neutral; the model predicts amusement and joy. | Preference-related wording may be interpreted as positive emotion. |

Observations are separated from hypotheses. Implied context, short comments, HTML and profanity suggest directions for further investigation. The accompanying CSV includes reference labels, predictions and Neutral probabilities. Each model also has automatically generated lists of 20 FN and 20 FP candidates in its analysis outputs. Candidate lists do not establish that all cases have been reviewed. The Final requirement for at least 20 FN and 20 FP had not yet been completed at the Midterm stage.

## 8. Proposed Research Question and Final Plan

Is a uniform threshold of 0.5 optimal for every emotion? The planned study uses the strongest Midterm neural backbone; at this stage PhoBERT-27 has the highest emotion27 Macro-F1. The checkpoint is frozen without retraining the backbone when changing the decoder. Fixed 0.5, global tuned and per-label tuned thresholds are compared. Thresholds are selected only on validation data, all choices are locked, and held-out test performance is evaluated once. Analysis will cover frequency tiers, cardinality, threshold profiles, the validation-to-test gap and at least 20 FN and 20 FP. If the study continues with 28 outputs, Neutral and the 27-emotion projection will be reported separately.

## 9. Reproducibility, Sources and Limitations

The two Kaggle archives and executed notebooks 08/09 are the source of truth. `scripts/import_equal20_results.py` checks notebook manifests against archive contents; verifies IDs, targets, label order and metric CSVs against all six NPZs; checks the complete 20-epoch histories and best-epoch criterion; and confirms that the checkpoint is present in each archive. No pickle or checkpoint is loaded, and no training or inference is repeated. Evidence is exported through an allowlist of JSON and ID-only NPZ files.

Raw-file checksums differ in the 28-output run because of CRLF line endings. Each checksum was verified as an exact LF/CRLF variant of the corresponding local source, config or CSV, rather than assuming that two JSON files merely have equivalent meaning. Parsed configurations and split IDs agree between studies, and the environment and GPU information agrees across manifests. `provenance.json` records archive, notebook and extracted-member hashes and each file's line-ending variant. Original manifests are not edited to force a match.

Limitations include one seed, validation-based checkpoint selection and exploratory threshold analysis, heuristic text styles, six qualitative error observations requiring team confirmation, an unverified original stratification algorithm and missing group information. Test performance had not been evaluated at the Midterm stage. Earlier runs with fewer epochs remain historical artifacts and do not support the current main tables or conclusions.

To reproduce the original analysis, run `python scripts/import_equal20_results.py`, `python scripts/build_report.py` and `python scripts/build_study_guide.py`. Import requires the two original archives, executed notebooks and authorized CSVs. The English translation can be rebuilt separately with `python scripts/build_midterm_english.py` from the preserved Vietnamese Markdown and existing figures. Raw text, vocabulary and checkpoints must not be redistributed publicly.

## Appendix A. Per-label F1 for All Three Models

### A.27. 27 Outputs, Neural Models Trained for 20 Epochs

{{per_label27}}

### A.28. 28 Outputs, Neural Models Trained for 20 Epochs

{{per_label28}}

## References

[1] Project 2 ViGoEmotions brief and `student-guide.pdf` in `docs/`, Midterm A–E and Final RQ1.

[2] Hung et al. ViGoEmotions. EACL 2026. DOI: 10.18653/v1/2026.eacl-long.129.

[3] Nguyen and Nguyen. PhoBERT. Findings of EMNLP 2020; VinAIResearch/PhoBERT, model `vinai/phobert-base-v2`.

[4] scikit-learn, PyTorch, Hugging Face Transformers and VnCoreNLP. Versions, revisions and configurations are recorded in the study's manifest and run JSON files.
