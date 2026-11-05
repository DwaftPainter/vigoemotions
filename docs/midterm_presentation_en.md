# ViGoEmotions — English Midterm Presentation Script

20 slides; 1,438 spoken words. Target: about 12–15 minutes at a measured pace, including short pauses and handovers. Q&A is additional.

Use this script with reports/midterm_slides.pdf. The numerical results come from the verified Kaggle 20-epoch studies. The Vietnamese report and study guide remain available for preparation.

## Suggested timing

- Slides 1–6: data and preprocessing, about 3–4 minutes.
- Slides 7–9: label settings and methods, about 2 minutes.
- Slides 10–14: results and learning curves, about 3–4 minutes.
- Slides 15–20: analysis, final research question and conclusion, about 4–5 minutes.

## Before presenting

- Add your real group details to the submitted materials; do not read these preparation instructions aloud.
- Confirm the six AI-assisted error-review observations before presenting them as your team’s analysis.
- Rehearse once with a timer. Read key results, rather than every number on a slide.
- Possible handover points: after slide 6 (data), slide 9 (methods), and slide 14 (results). Allocate these sections to your actual team members.

## Slide-by-slide script

### Slide 01 — ViGoEmotions: Vietnamese Emotion Recognition

Good morning, everyone. Our project is ViGoEmotions, a study of emotion recognition in Vietnamese social-media comments. A comment may express several emotions at the same time, so this is a multi-label classification task. We compare three models and two label settings, with and without an explicit Neutral output. Today, we will explain our data, experiments, main findings, and plan for the final evaluation.

### Slide 02 — Dataset and Quality Checks

The dataset contains twenty thousand, six hundred and sixty-four comments. We keep the supplied training, validation, and test partitions. Our audit found one training comment that duplicated a test comment. We removed the training copy, leaving sixteen thousand, five hundred and thirty training samples. We also found comments annotated with both Neutral and an emotion. We report this inconsistency and define how each model handles it, rather than silently changing the annotations.

### Slide 03 — Emotion Frequency and Imbalance

This chart shows label frequencies in the training set. We divide the labels into head, middle, and tail groups, with nine labels in each group. Amusement is the most frequent emotion, while relief is the least frequent. The ratio is about four point five. Another challenge is that most positions in a label vector are zero. This is why we focus on F1 and additional multi-label metrics instead of relying on accuracy.

### Slide 04 — Label Cardinality and Text Length

Label cardinality means the number of positive labels in one comment. In training, the average is about one point eight six emotions, and the maximum is five. The median comment has eleven whitespace tokens and forty-seven characters. A whitespace token is not necessarily a complete Vietnamese word or a PhoBERT token. The comments also contain slang, teencode, emojis, and non-standard syntax, which make emotion recognition more difficult.

### Slide 05 — Emotions Often Occur Together

Some emotions often appear together. The most frequent pair in our training analysis is disappointment and sadness, with nine hundred and four comments. We examine both co-occurrence counts and correlation, because a large count may simply reflect frequent individual labels. These statistics are calculated from training data. Later, we check whether the models can predict both emotions together on validation examples.

### Slide 06 — Split Protocol and Leakage Prevention

We preserve the supplied split and verify sample identities and text overlap. Vocabulary and inverse document frequency are fitted only on training data. Our shared preprocessing normalizes Unicode and whitespace while preserving Vietnamese accents and negation. The split audit describes label distributions, but it does not prove which stratification algorithm originally created the split. That requirement still needs clarification with the lecturer. We have not evaluated model performance or tuned thresholds on the test set.

Optional handover: “That completes our data overview. My teammate will now explain the model settings.”

### Slide 07 — Two Label Settings: 27 vs 28

In the twenty-seven-label setting, the model predicts emotions only. A Neutral-only comment therefore has an all-zero emotion vector. In the twenty-eight-label setting, Neutral becomes an additional output. We preserve comments with mixed annotations and do not force Neutral to exclude other emotions. There are ninety-four Neutral-positive validation examples. Both neural settings use independent sigmoid outputs and binary cross-entropy, because multiple labels can be positive together.

### Slide 08 — Three Models, One Experimental Profile

Our first model combines TF-IDF with one-versus-rest Logistic Regression. It is a simple lexical baseline. The second model is a bidirectional GRU. It reads the sequence in both directions and learns its embeddings from scratch. The third is PhoBERT, a pretrained Vietnamese transformer that we fine-tune, including its backbone. Both neural models train for twenty epochs in each label setting. Logistic Regression has a solver iteration limit instead, so we do not describe it as a twenty-epoch neural model.

### Slide 09 — Checkpoint Selection and Reproducibility

For a clearer comparison, both settings use the same seed, split, and configuration for each model. We select neural checkpoints using validation Macro-F1 on the same twenty-seven emotions, at threshold zero point five. Ties keep the earliest epoch. The best PhoBERT checkpoints are at epochs 14 and 15, but both runs still complete all twenty epochs. We record the environment and verify data and source checksums, including the line-ending differences between the two sessions.

Optional handover: “We have defined the experimental protocol. My teammate will now present the results.”

### Slide 10 — Results: 27 Emotions

For twenty-seven emotions, Macro-F1 is 0.2388 for TF-IDF, 0.4220 for BiGRU, and 0.5974 for PhoBERT. PhoBERT performs best on all four overall metrics in this setting. Macro-F1 gives equal weight to each label, while Micro-F1 pools decisions across labels. Hamming Loss measures incorrect label decisions, and LRAP evaluates probability rankings. PhoBERT training took about four hours on a Tesla T4. The report includes parameter counts, training times, and per-label scores for every model.

### Slide 11 — Results: 28 Labels Including Neutral

With twenty-eight labels, Macro-F1 is 0.2310 for TF-IDF, 0.4086 for BiGRU, and 0.5783 for PhoBERT. PhoBERT again has the strongest overall results within this setting. However, these scores include Neutral. We should not directly subtract the overall twenty-seven-label and twenty-eight-label Macro-F1 values to measure the effect of adding Neutral, because the evaluated label sets are different.

### Slide 12 — Does Adding Neutral Help the 27 Emotions?

To compare the same task, we remove the Neutral column from the twenty-eight-output predictions and recalculate the metrics on the twenty-seven emotions. PhoBERT then scores 0.5836, compared with 0.5974 for the twenty-seven-output model. This is about 1.38 percentage points lower. TF-IDF is unchanged, and BiGRU also decreases slightly. This run does not show an improvement from adding Neutral on the shared emotions. One seed is not enough for a causal claim or a statistical significance claim.

### Slide 13 — Neutral Detection Remains Difficult

The additional output does allow the model to predict Neutral directly, but this remains difficult. PhoBERT finds 45 of the 94 true Neutral examples, misses 49, and produces 67 false positives. Its Neutral F1 is 0.4369. The twenty-seven-output model has no Neutral classifier, so its Neutral F1 is undefined rather than zero. Also, an empty prediction should not automatically be interpreted as Neutral.

### Slide 14 — Learning Curves Across 20 Epochs

These curves show training loss and the shared checkpoint-selection metric across all twenty epochs. An interesting example is PhoBERT with twenty-seven outputs: its first-epoch F1 is zero, but it improves substantially later. This shows why we should not judge the entire run from the first epoch. After the best checkpoint, validation F1 fluctuates while training loss continues to decrease. We do not have validation loss, so we avoid claiming confirmed overfitting from these curves alone.

Optional handover: “We have seen the main results. My teammate will now discuss the midterm questions and next steps.”

### Slide 15 — Q1–Q2: Per-Label F1 and Frequency

The first two midterm questions concern individual labels and training frequency. For PhoBERT with twenty-seven outputs, gratitude has the highest F1, approximately 0.837, while disapproval has the lowest, approximately 0.362. The Spearman correlation between training frequency and F1 is only 0.018, with a p-value of 0.928. These results do not support a simple rule that more frequent emotions are always easier to predict. They are descriptive results from this run.

### Slide 16 — Q3–Q4: Label Pairs and Text Styles

For label dependencies, there are one hundred and twenty-two validation comments containing both disappointment and sadness. PhoBERT predicts both labels correctly in 63 cases, giving a joint recall of about 51.6 percent. For text styles, the largest PhoBERT advantage over BiGRU is in teencode, at about 22.1 percentage points of Macro-F1. Teencode refers to informal Vietnamese online spelling and abbreviations. Our style groups are heuristic and can overlap, so we interpret this comparison cautiously.

### Slide 17 — Q5: Sensitivity to the Decision Threshold

Our fifth question examines the default decision threshold. We keep the checkpoint fixed and vary the threshold on validation predictions. For PhoBERT, the best point in our grid is 0.35, with Macro-F1 0.5978, compared with 0.5974 at zero point five. The gain is very small, while Hamming Loss becomes worse. This is exploratory validation analysis, not a test improvement. We keep zero point five for the main results and use this observation to motivate our final research question.

### Slide 18 — Error Analysis: New Neutral Predictions

A preliminary, AI-assisted review examines six cases from the new twenty-eight-label model: three confident Neutral false positives and three Neutral false negatives. These cases suggest difficulties with implied emotions, short expressions, HTML, and profanity. These are hypotheses from selected examples, not proven causes. The review still needs team confirmation. We retain sample IDs and abstract descriptions without reproducing the original comments, and we distinguish automatically generated error candidates from a completed manual audit.

### Slide 19 — Final Research Question and Next Steps

For the final evaluation, we will ask whether one threshold is suitable for every emotion. We will freeze the strongest neural checkpoint and compare three decoding strategies: the fixed threshold of zero point five, one tuned global threshold, and separate thresholds for individual labels. Thresholds will be selected on validation data. Only after locking all choices will we evaluate the held-out test set. We also plan a more detailed review of at least twenty false negatives and twenty false positives.

### Slide 20 — Conclusions and Evidence

Our main finding is that PhoBERT achieves the strongest overall results within both label settings. Adding an explicit Neutral output does not improve the shared twenty-seven-emotion F1 in this run, and Neutral detection still has substantial errors. The results are supported by saved Kaggle predictions, training histories, and provenance checks. Our current limitations are one seed, validation-based selection, and no model test scores. Thank you for listening. We are happy to take questions.

## Short answers for Q&A

### Why use sigmoid rather than softmax?

Several emotions can be positive together. Independent sigmoid outputs do not force the probabilities to sum to one.

### Why use binary cross-entropy?

Each output is a binary emotion decision. BCEWithLogitsLoss combines the sigmoid calculation with binary cross-entropy in a numerically stable training loss.

### Did both neural models really train for 20 epochs?

Yes. All four neural histories contain epochs 1 through 20. The selected checkpoint can come from an earlier epoch.

### Does adding Neutral help?

Not on the shared 27-emotion F1 in this run. PhoBERT scores 0.5974 with 27 outputs, versus 0.5836 when the 28-output predictions are scored on those same emotions. One seed is insufficient for a general causal claim.

### Why are the 28-label scores 0.5783 and 0.5836 different?

They use the same checkpoint. The first evaluates all 28 labels; the second evaluates only the 27 emotion columns.

### Did you prove that the supplied split was stratified?

No. We verified integrity and described label-distribution differences, but did not verify the original splitting algorithm. Acceptance of the supplied split still needs clarification.

### Are these test results?

No. The model scores are from validation. Test data participated in the data-integrity audit, but was not used to evaluate models or tune thresholds.

### Why did F1 start at zero?

At the first epoch, there were no true-positive label predictions at threshold 0.5. Later epochs improved. That single early result did not establish a failed run.

### Did you use both T4 GPUs?

The code uses one GPU per training process, even though Kaggle lists T4 times two. No distributed training was configured.

### Where do the pretrained weights come from?

PhoBERT-base-v2 is loaded from the VinAI checkpoint. Its backbone is pretrained, while the classification head is initialized for our labels. Fine-tuning updates both. BiGRU and Logistic Regression are trained from scratch.

### Why do some file hashes differ between runs?

We verified that the differences are exact LF versus CRLF line-ending variants of the same local files. Parsed configurations and fixed split IDs match; original manifests were preserved.

### Is the threshold 0.35 now your final choice?

No. It is the best point in an exploratory validation grid for a fixed checkpoint. The main comparison remains at 0.5, and final decoding choices have not been evaluated on test.

## Speaking notes

- TF-IDF: say the letters “T F, I D F”. OvR means “one versus rest”.
- BiGRU: “bidirectional G R U”. BCE: “binary cross-entropy”. LRAP: “label ranking average precision”.
- Read 0.5974 as “zero point five nine seven four”. Keep decimal points in English slides.
- A change of 0.0138 in an F1 score is about 1.38 percentage points; do not call it a 1.38 percent relative change.
- Say “validation results”, “in this run”, and “preliminary review” where appropriate. Do not claim test performance or statistical significance.
