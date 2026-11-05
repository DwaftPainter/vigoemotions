# ViGoEmotions — Midterm presentation

## 1. ViGoEmotions: Vietnamese Emotion Recognition

- Multi-label classification of Vietnamese comments.
- Three models; 27 emotions or 28 labels with Neutral.
- Both neural models: 20 epochs in each setting.
- All reported model scores are validation results.

Speaker notes: Good morning, everyone. Our project is ViGoEmotions, a study of emotion recognition in Vietnamese social-media comments. A comment may express several emotions at the same time, so this is a multi-label classification task. We compare three models and two label settings, with and without an explicit Neutral output. Today, we will explain our data, experiments, main findings, and plan for the final evaluation.

## 2. Dataset and Quality Checks

- 20,664 comments; supplied train/validation/test split.
- One train–test text overlap removed from training.
- Effective sizes: 16,530 / 2,066 / 2,067.
- Source annotations: 725 Neutral-only; 306 mixed.

Speaker notes: The dataset contains twenty thousand, six hundred and sixty-four comments. We keep the supplied training, validation, and test partitions. Our audit found one training comment that duplicated a test comment. We removed the training copy, leaving sixteen thousand, five hundred and thirty training samples. We also found comments annotated with both Neutral and an emotion. We report this inconsistency and define how each model handles it, rather than silently changing the annotations.

## 3. Emotion Frequency and Imbalance

- 27 emotions; nine labels per frequency tier.
- Amusement is most frequent; relief is least frequent.
- Maximum/minimum training count: about 4.52×.
- Positive labels are sparse across the output vector.

Speaker notes: This chart shows label frequencies in the training set. We divide the labels into head, middle, and tail groups, with nine labels in each group. Amusement is the most frequent emotion, while relief is the least frequent. The ratio is about four point five. Another challenge is that most positions in a label vector are zero. This is why we focus on F1 and additional multi-label metrics instead of relying on accuracy.

## 4. Label Cardinality and Text Length

- Training mean: 1.859 emotions per comment.
- Maximum: five emotions in one comment.
- Median: 11 whitespace tokens; 47 characters.
- Informal language: slang, teencode, emoji and HTML.

Speaker notes: Label cardinality means the number of positive labels in one comment. In training, the average is about one point eight six emotions, and the maximum is five. The median comment has eleven whitespace tokens and forty-seven characters. A whitespace token is not necessarily a complete Vietnamese word or a PhoBERT token. The comments also contain slang, teencode, emojis, and non-standard syntax, which make emotion recognition more difficult.

## 5. Emotions Often Occur Together

- Disappointment–sadness: 904 training comments.
- Anger–annoyance: 699; sadness–grief: 620.
- Co-occurrence and correlation use training labels.
- Correlation does not establish a causal relationship.

Speaker notes: Some emotions often appear together. The most frequent pair in our training analysis is disappointment and sadness, with nine hundred and four comments. We examine both co-occurrence counts and correlation, because a large count may simply reflect frequent individual labels. These statistics are calculated from training data. Later, we check whether the models can predict both emotions together on validation examples.

## 6. Split Protocol and Leakage Prevention

- Preserve the supplied split and fixed sample IDs.
- Fit vocabulary and IDF on training data only.
- Normalize Unicode and whitespace; preserve negation.
- Distribution audit does not prove stratification.
- No model evaluation or threshold tuning on test.

Speaker notes: We preserve the supplied split and verify sample identities and text overlap. Vocabulary and inverse document frequency are fitted only on training data. Our shared preprocessing normalizes Unicode and whitespace while preserving Vietnamese accents and negation. The split audit describes label distributions, but it does not prove which stratification algorithm originally created the split. That requirement still needs clarification with the lecturer. We have not evaluated model performance or tuned thresholds on the test set.

## 7. Two Label Settings: 27 vs 28

- 27 outputs: Neutral-only becomes an all-zero vector.
- 28 outputs: Neutral is a separate learned label.
- Preserve mixed annotations; no exclusion rule.
- Validation: 75 Neutral-only + 19 mixed = 94.
- Independent sigmoid outputs with binary cross-entropy.

Speaker notes: In the twenty-seven-label setting, the model predicts emotions only. A Neutral-only comment therefore has an all-zero emotion vector. In the twenty-eight-label setting, Neutral becomes an additional output. We preserve comments with mixed annotations and do not force Neutral to exclude other emotions. There are ninety-four Neutral-positive validation examples. Both neural settings use independent sigmoid outputs and binary cross-entropy, because multiple labels can be positive together.

## 8. Three Models, One Experimental Profile

- TF-IDF + OvR LR: liblinear, max_iter = 1,000.
- BiGRU: embedding/hidden 128; batch 64; lr 0.001.
- PhoBERT: full fine-tuning; batch 16; lr 2e-5.
- BiGRU and PhoBERT: 20 epochs for both label sets.
- TF-IDF uses a solver limit, not neural epochs.

Speaker notes: Our first model combines TF-IDF with one-versus-rest Logistic Regression. It is a simple lexical baseline. The second model is a bidirectional GRU. It reads the sequence in both directions and learns its embeddings from scratch. The third is PhoBERT, a pretrained Vietnamese transformer that we fine-tune, including its backbone. Both neural models train for twenty epochs in each label setting. Logistic Regression has a solver iteration limit instead, so we do not describe it as a twenty-epoch neural model.

## 9. Checkpoint Selection and Reproducibility

- Seed 42; threshold 0.5; same split and settings.
- Select by validation Macro-F1 on the 27 emotions.
- Best BiGRU epochs, 27/28: 17 / 20.
- Best PhoBERT epochs, 27/28: 14 / 15.
- Same GPU model and recorded package versions.

Speaker notes: For a clearer comparison, both settings use the same seed, split, and configuration for each model. We select neural checkpoints using validation Macro-F1 on the same twenty-seven emotions, at threshold zero point five. Ties keep the earliest epoch. The best PhoBERT checkpoints are at epochs 14 and 15, but both runs still complete all twenty epochs. We record the environment and verify data and source checksums, including the line-ending differences between the two sessions.

## 10. Results: 27 Emotions

- TF-IDF Macro-F1: 0.2388.
- BiGRU Macro-F1: 0.4220.
- PhoBERT Macro-F1: 0.5974; Micro-F1: 0.5996.
- PhoBERT Hamming Loss: 0.0550; LRAP: 0.7419.
- Higher F1/LRAP is better; lower Hamming Loss is better.

Speaker notes: For twenty-seven emotions, Macro-F1 is 0.2388 for TF-IDF, 0.4220 for BiGRU, and 0.5974 for PhoBERT. PhoBERT performs best on all four overall metrics in this setting. Macro-F1 gives equal weight to each label, while Micro-F1 pools decisions across labels. Hamming Loss measures incorrect label decisions, and LRAP evaluates probability rankings. PhoBERT training took about four hours on a Tesla T4. The report includes parameter counts, training times, and per-label scores for every model.

## 11. Results: 28 Labels Including Neutral

- TF-IDF Macro-F1: 0.2310.
- BiGRU Macro-F1: 0.4086.
- PhoBERT Macro-F1: 0.5783; Micro-F1: 0.5829.
- PhoBERT Hamming Loss: 0.0575; LRAP: 0.7077.
- These metrics include the additional Neutral label.

Speaker notes: With twenty-eight labels, Macro-F1 is 0.2310 for TF-IDF, 0.4086 for BiGRU, and 0.5783 for PhoBERT. PhoBERT again has the strongest overall results within this setting. However, these scores include Neutral. We should not directly subtract the overall twenty-seven-label and twenty-eight-label Macro-F1 values to measure the effect of adding Neutral, because the evaluated label sets are different.

## 12. Does Adding Neutral Help the 27 Emotions?

- PhoBERT with 27 outputs: 0.5974.
- PhoBERT with 28 outputs, scored on 27: 0.5836.
- Difference: -1.38 percentage points.
- No observed improvement on the shared 27 emotions.
- One seed; no causal or statistical conclusion.

Speaker notes: To compare the same task, we remove the Neutral column from the twenty-eight-output predictions and recalculate the metrics on the twenty-seven emotions. PhoBERT then scores 0.5836, compared with 0.5974 for the twenty-seven-output model. This is about 1.38 percentage points lower. TF-IDF is unchanged, and BiGRU also decreases slightly. This run does not show an improvement from adding Neutral on the shared emotions. One seed is not enough for a causal claim or a statistical significance claim.

## 13. Neutral Detection Remains Difficult

- PhoBERT Neutral F1: 0.4369.
- Precision: 0.4018; recall: 0.4787.
- 45 TP / 67 FP / 49 FN; support: 94.
- No Neutral output exists in the 27-label model.
- An empty prediction is not automatically Neutral.

Speaker notes: The additional output does allow the model to predict Neutral directly, but this remains difficult. PhoBERT finds 45 of the 94 true Neutral examples, misses 49, and produces 67 false positives. Its Neutral F1 is 0.4369. The twenty-seven-output model has no Neutral classifier, so its Neutral F1 is undefined rather than zero. Also, an empty prediction should not automatically be interpreted as Neutral.

## 14. Learning Curves Across 20 Epochs

- All four neural runs completed epochs 1–20.
- The best checkpoint can occur before the last epoch.
- PhoBERT 27 starts at F1 = 0, then improves.
- Training loss alone cannot confirm overfitting.

Speaker notes: These curves show training loss and the shared checkpoint-selection metric across all twenty epochs. An interesting example is PhoBERT with twenty-seven outputs: its first-epoch F1 is zero, but it improves substantially later. This shows why we should not judge the entire run from the first epoch. After the best checkpoint, validation F1 fluctuates while training loss continues to decrease. We do not have validation loss, so we avoid claiming confirmed overfitting from these curves alone.

## 15. Q1–Q2: Per-Label F1 and Frequency

- PhoBERT best label: gratitude, F1 0.8370.
- Lowest label: disapproval, F1 0.3623.
- Frequency vs F1: Spearman rho 0.018.
- p-value 0.928; descriptive evidence only.
- Full per-label results are included in the report.

Speaker notes: The first two midterm questions concern individual labels and training frequency. For PhoBERT with twenty-seven outputs, gratitude has the highest F1, approximately 0.837, while disapproval has the lowest, approximately 0.362. The Spearman correlation between training frequency and F1 is only 0.018, with a p-value of 0.928. These results do not support a simple rule that more frequent emotions are always easier to predict. They are descriptive results from this run.

## 16. Q3–Q4: Label Pairs and Text Styles

- Disappointment–sadness: PhoBERT gets both in 63/122 cases.
- Joint recall at 0.5: 51.64%.
- Largest PhoBERT–BiGRU style gap: teencode.
- Macro-F1 gap in this group: +22.13 percentage points.
- Style groups use heuristics and may overlap.

Speaker notes: For label dependencies, there are one hundred and twenty-two validation comments containing both disappointment and sadness. PhoBERT predicts both labels correctly in 63 cases, giving a joint recall of about 51.6 percent. For text styles, the largest PhoBERT advantage over BiGRU is in teencode, at about 22.1 percentage points of Macro-F1. Teencode refers to informal Vietnamese online spelling and abbreviations. Our style groups are heuristic and can overlap, so we interpret this comparison cautiously.

## 17. Q5: Sensitivity to the Decision Threshold

- PhoBERT 27 at 0.5: Macro-F1 0.5974.
- Best validation grid point: 0.35, F1 0.5978.
- Same checkpoint; exploratory validation analysis.
- Very small F1 gain; Hamming Loss increases.
- Main results remain at threshold 0.5.

Speaker notes: Our fifth question examines the default decision threshold. We keep the checkpoint fixed and vary the threshold on validation predictions. For PhoBERT, the best point in our grid is 0.35, with Macro-F1 0.5978, compared with 0.5974 at zero point five. The gain is very small, while Hamming Loss becomes worse. This is exploratory validation analysis, not a test improvement. We keep zero point five for the main results and use this observation to motivate our final research question.

## 18. Error Analysis: New Neutral Predictions

- Six PhoBERT 28 cases: three Neutral FP and three FN.
- Inspect implied emotion, short text, HTML and profanity.
- Publish IDs and abstract observations, not raw comments.
- Preliminary AI-assisted review; team confirmation pending.
- Error candidates are not a completed manual audit.

Speaker notes: A preliminary, AI-assisted review examines six cases from the new twenty-eight-label model: three confident Neutral false positives and three Neutral false negatives. These cases suggest difficulties with implied emotions, short expressions, HTML, and profanity. These are hypotheses from selected examples, not proven causes. The review still needs team confirmation. We retain sample IDs and abstract descriptions without reproducing the original comments, and we distinguish automatically generated error candidates from a completed manual audit.

## 19. Final Research Question and Next Steps

- Is one threshold suitable for every emotion?
- Freeze the best neural checkpoint; keep weights fixed.
- Compare 0.5, global tuned and per-label thresholds.
- Tune on validation; lock choices; evaluate test once.
- Review at least 20 false negatives and 20 false positives.

Speaker notes: For the final evaluation, we will ask whether one threshold is suitable for every emotion. We will freeze the strongest neural checkpoint and compare three decoding strategies: the fixed threshold of zero point five, one tuned global threshold, and separate thresholds for individual labels. Thresholds will be selected on validation data. Only after locking all choices will we evaluate the held-out test set. We also plan a more detailed review of at least twenty false negatives and twenty false positives.

## 20. Conclusions and Evidence

- PhoBERT leads within both label settings.
- Adding Neutral does not improve shared-emotion F1 in this run.
- Evidence: notebooks 08/09, saved predictions and manifests.
- Limits: one seed, validation selection, no test scores.
- Sources: course brief, ViGoEmotions, PhoBERT and libraries.

Speaker notes: Our main finding is that PhoBERT achieves the strongest overall results within both label settings. Adding an explicit Neutral output does not improve the shared twenty-seven-emotion F1 in this run, and Neutral detection still has substantial errors. The results are supported by saved Kaggle predictions, training histories, and provenance checks. Our current limitations are one seed, validation-based selection, and no model test scores. Thank you for listening. We are happy to take questions.
