# ViGoEmotions — Midterm presentation

## 1. ViGoEmotions: Task and Dataset

- Multi-label emotion recognition in Vietnamese comments.
- 20,664 comments; 27 emotions plus a Neutral state.
- Effective train / validation / test: 16,530 / 2,066 / 2,067.
- Three models; compare 27 and 28 outputs.
- All model scores are validation results.

Speaker notes: Good morning. Our project recognizes emotions in Vietnamese social-media comments. Because one comment can express several emotions, this is a multi-label task. The dataset contains twenty thousand, six hundred and sixty-four comments. We compare three models, with and without an explicit Neutral output. I will briefly cover the setup, then focus on our results and analysis.

## 2. Data Profile and Leakage Prevention

- Mean 1.859 emotions; median 11 tokens / 47 characters.
- Amusement–relief frequency ratio: about 4.52×.
- Fixed split; one overlapping training sample removed.
- Fit vocabulary/IDF on train only; no model test scores.

Speaker notes: Comments average about one point eight six emotions, and the median length is eleven whitespace tokens. We keep the supplied split, remove one training copy of a test comment, and fit vocabulary only on training data. Unicode and whitespace are normalized. Our distribution audit does not prove the original stratification algorithm, so that requirement still needs clarification.

## 3. 27 vs 28 Outputs: What Changes?

- 27 outputs: Neutral-only has an all-zero emotion target.
- 28 outputs: Neutral becomes a separate learned label.
- Keep mixed annotations; no Neutral exclusion rule.
- Validation: 75 Neutral-only + 19 mixed = 94.
- Independent sigmoid outputs with binary cross-entropy.

Speaker notes: The twenty-seven-output model predicts emotions only. Neutral-only comments have an all-zero emotion target. The twenty-eight-output model adds a learned Neutral label. We preserve mixed annotations, where Neutral appears with emotions. Both neural settings use sigmoid outputs and binary cross-entropy because labels are not mutually exclusive. Validation contains ninety-four Neutral-positive examples.

## 4. Models and Shared Training Protocol

- TF-IDF + OvR LR: liblinear; max_iter = 1,000.
- BiGRU from scratch: hidden 128; batch 64; lr 0.001.
- PhoBERT full fine-tuning: batch 16; lr 2e-5.
- Both neural models: 20 epochs per label setting; seed 42.
- Select checkpoints by Macro-F1 on the same 27 emotions.

Speaker notes: We compare TF-IDF with one-versus-rest Logistic Regression, a bidirectional GRU trained from scratch, and full PhoBERT fine-tuning. Both neural models complete twenty epochs in each setting. TF-IDF uses a solver limit instead. We share the seed and split, and select checkpoints by validation Macro-F1 on the same twenty-seven emotions at threshold zero point five.

## 5. Results: 27 Emotions

- TF-IDF Macro-F1: 0.2388.
- BiGRU Macro-F1: 0.4220.
- PhoBERT Macro-F1: 0.5974; Micro-F1: 0.5996.
- PhoBERT Hamming Loss: 0.0550; LRAP: 0.7419.
- Higher F1/LRAP is better; lower Hamming Loss is better.

Speaker notes: For twenty-seven emotions, Macro-F1 is 0.2388 for TF-IDF, 0.4220 for BiGRU, and 0.5974 for PhoBERT. PhoBERT leads on all four overall metrics in this setting. Macro-F1 weights labels equally; Micro-F1 pools decisions across labels. Hamming Loss measures incorrect decisions, while LRAP evaluates probability rankings.

## 6. Results: 28 Labels Including Neutral

- TF-IDF Macro-F1: 0.2310.
- BiGRU Macro-F1: 0.4086.
- PhoBERT Macro-F1: 0.5783; Micro-F1: 0.5829.
- PhoBERT Hamming Loss: 0.0575; LRAP: 0.7077.
- These metrics evaluate all 28 labels.

Speaker notes: For twenty-eight labels, Macro-F1 is 0.2310, 0.4086, and 0.5783, respectively. PhoBERT again performs best within this setting. However, these scores include Neutral. We cannot directly subtract the overall twenty-seven-label and twenty-eight-label scores to measure the benefit of Neutral, because their evaluated label sets differ.

## 7. Does Neutral Help the Shared 27 Emotions?

- PhoBERT with 27 outputs: 0.5974.
- PhoBERT with 28 outputs, scored on 27: 0.5836.
- Difference: -1.38 percentage points.
- No observed improvement on the shared emotions.
- One seed; no causal or statistical conclusion.

Speaker notes: For a comparison on identical labels, we score only the twenty-seven emotion columns of the expanded model. PhoBERT then reaches 0.5836, about 1.38 percentage points below the twenty-seven-output model. TF-IDF is unchanged and BiGRU decreases slightly. This run shows no improvement on the shared emotions, but one seed cannot establish a general causal conclusion.

## 8. Neutral Detection Remains Difficult

- PhoBERT Neutral F1: 0.4369.
- Precision: 0.4018; recall: 0.4787.
- 45 TP / 67 FP / 49 FN; support: 94.
- The 27-output model has no Neutral classifier.
- An empty prediction is not automatically Neutral.

Speaker notes: PhoBERT identifies 45 of the ninety-four Neutral-positive validation examples. It misses 49 and produces 67 false positives, giving Neutral F1 of 0.4369. So the additional label is learned, but still difficult. The twenty-seven-output model has no Neutral classifier; its Neutral F1 is undefined, not zero. An empty prediction is also not automatically Neutral.

## 9. Learning Curves and Best Checkpoints

- All four neural runs completed epochs 1–20.
- Best BiGRU epochs, 27/28: 17 / 20.
- Best PhoBERT epochs, 27/28: 14 / 15.
- Select the best checkpoint, not necessarily the last.
- Training loss alone cannot confirm overfitting.

Speaker notes: These curves cover all twenty epochs. The best PhoBERT checkpoints occur at epochs fourteen and fifteen. We finish training before loading the selected checkpoint. Validation F1 fluctuates after the best epoch even while training loss decreases. Because validation loss was not recorded, we avoid diagnosing confirmed overfitting from training loss alone.

## 10. Q1–Q2: Per-Label Performance and Frequency

- PhoBERT best label: gratitude, F1 0.8370.
- Lowest label: disapproval, F1 0.3623.
- Training frequency vs F1: Spearman rho 0.018.
- p-value 0.928; descriptive evidence only.
- Complete per-label results are in the report.

Speaker notes: For individual emotions, PhoBERT performs best on gratitude and worst on disapproval. The correlation between training frequency and per-label F1 is approximately 0.018, with a p-value of 0.928. These descriptive results do not support a simple rule that more frequent labels are always easier. The report includes every label for all three models.

## 11. Q3–Q4: Label Pairs and Text Styles

- Top training pair: disappointment–sadness (904 cases).
- Both correct: 63/122 validation cases; joint recall 51.64%.
- Largest PhoBERT–BiGRU style gap: teencode.
- Macro-F1 gap: +22.13 percentage points.
- Style groups are heuristic and may overlap.

Speaker notes: Our most frequent training pair is disappointment and sadness. PhoBERT predicts both correctly in 63 of 122 validation cases containing both labels. For text styles, its largest advantage over BiGRU is in teencode, at about 22.1 percentage points of Macro-F1. Teencode means informal online spellings and abbreviations. Our style groups use heuristics and can overlap, so this finding needs cautious interpretation.

## 12. Q5: Does the Default Threshold Matter?

- PhoBERT 27 at 0.5: Macro-F1 0.5974.
- Best validation grid point: 0.35, F1 0.5978.
- Same checkpoint; exploratory validation analysis.
- Very small F1 gain; Hamming Loss increases.
- Main results remain at threshold 0.5.

Speaker notes: We also vary the threshold while keeping the checkpoint fixed. For PhoBERT, the best validation grid point is 0.35, with Macro-F1 0.5978, compared with 0.5974 at zero point five. The gain is tiny and Hamming Loss worsens. This is exploratory analysis, not a test improvement; the main comparison stays at zero point five.

## 13. Error Analysis: New Neutral Predictions

- Six cases: three Neutral false positives and three misses.
- Potential issues: implied emotion, short text, HTML, profanity.
- IDs and abstract observations; no raw comments.
- Preliminary AI-assisted review; team confirmation pending.

Speaker notes: A preliminary, AI-assisted review examines three confident Neutral false positives and three Neutral misses. These examples suggest difficulties with implied emotion, short text, HTML, and profanity. These are hypotheses, not proven causes. The review needs team confirmation. We publish sample IDs and abstract observations instead of the original comments.

## 14. Final Research Question and Next Steps

- Is one threshold suitable for every emotion?
- Freeze the best neural checkpoint and its weights.
- Compare 0.5, global tuned and per-label thresholds.
- Tune on validation; lock choices; evaluate test once.
- Review at least 20 false negatives and 20 false positives.

Speaker notes: For the final evaluation, we will test whether one threshold suits every emotion. We will freeze the best neural checkpoint and compare the default threshold, a tuned global threshold, and per-label thresholds. We select thresholds on validation, lock all choices, then evaluate test once. We also plan a detailed review of at least twenty false negatives and twenty false positives.

## 15. Conclusions and Evidence

- PhoBERT leads within both label settings.
- Neutral does not improve shared-emotion F1 in this run.
- Evidence: Kaggle 08/09, saved predictions and manifests.
- Limits: one seed, validation selection, no test scores.
- Sources: course brief, ViGoEmotions, PhoBERT and libraries.

Speaker notes: To conclude, PhoBERT gives the strongest overall results in both settings. Adding Neutral does not improve shared-emotion F1 in this run, and Neutral prediction still makes substantial errors. Saved Kaggle predictions and histories support these findings. Our next step is controlled threshold evaluation. Thank you; we welcome your questions.


<!-- Optional supplementary slides: 16–17; use for Q&A only. -->

## 16. Supplementary Table A · Model performance

Validation n = 2,066 · threshold 0.5 · full-label scores reported separately

| Labels | Model | Macro-F1 ↑ | Micro-F1 ↑ | Hamming ↓ | LRAP ↑ | Best ep. | Train time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | TF-IDF + OvR LR | 0.2388 | 0.2712 | 0.0606 | 0.6407 | — | 2.75 s |
| 27 | BiGRU | 0.4220 | 0.4590 | 0.0626 | 0.6191 | 17 | 80.3 s |
| 27 | PhoBERT | 0.5974 | 0.5996 | 0.0550 | 0.7419 | 14 | 4.05 h |
| 28 | TF-IDF + OvR LR | 0.2310 | 0.2662 | 0.0601 | 0.6096 | — | 2.52 s |
| 28 | BiGRU | 0.4086 | 0.4483 | 0.0628 | 0.5851 | 20 | 83.6 s |
| 28 | PhoBERT | 0.5783 | 0.5829 | 0.0575 | 0.7077 | 15 | 3.90 h |

Both neural models trained for 20 epochs. Best ep. is the selected checkpoint. Rank models within each label block; full 27-label and 28-label Macro-F1 use different label sets.

Source: current model_comparison.csv. CPU TF-IDF; Tesla T4 neural. Times exclude preprocessing; no test scores.

Speaker notes: Optional backup for Q&A. Main presentation ends at slide 15.

## 17. Supplementary Table B · Pair and text-style analysis

27-output study only · validation · threshold 0.5 · supplementary detail for slide 11

| Model | Joint TP | Support | Recall |
| --- | --- | --- | --- |
| TF-IDF + OvR LR | 2 | 122 | 1.64% |
| BiGRU | 37 | 122 | 30.33% |
| PhoBERT | 63 | 122 | 51.64% |

| Group | n | BiGRU F1 | PhoBERT F1 | Δ (pp) |
| --- | --- | --- | --- | --- |
| All | 2,066 | 0.4220 | 0.5974 | +17.54 |
| Emoji-heavy | 199 | 0.3506 | 0.4789 | +12.83 |
| Short | 343 | 0.4303 | 0.5654 | +13.51 |
| Teencode | 307 | 0.3333 | 0.5547 | +22.13 |

Joint recall counts both emotions predicted among examples annotated with both; extra predicted labels are allowed. Style groups overlap and differ in size; Δ = PhoBERT − BiGRU in Macro-F1 percentage points. One seed; no causal or significance claim.

Sources: current pair_preservation.csv; neural_style_comparison.csv. Pair selected by training co-occurrence; no test metrics.

Speaker notes: Optional backup for Q&A; groups overlap and findings are descriptive.
