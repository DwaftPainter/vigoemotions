# ViGoEmotions — English Midterm Presentation Script

15 slides; 803 spoken words. Target: about 8–9 minutes, including short pauses and handovers. Finish before the 10-minute limit; leave any remaining time for questions.

Use this script with reports/midterm_slides.pdf. The numerical results come from the verified Kaggle 20-epoch studies. The Vietnamese report and study guide remain available for preparation.

## Suggested timing

- Slides 1–4: compact introduction, data and protocol, about 2 minutes.
- Slides 5–9: results and learning curves, about 3 minutes.
- Slides 10–13: midterm questions and errors, about 2–2.5 minutes.
- Slides 14–15: final research question and conclusion, about 1 minute.

## Before presenting

- Add your real group details to the submitted materials; do not read these preparation instructions aloud.
- Confirm the six AI-assisted error-review observations before presenting them as your team’s analysis.
- Rehearse once with a timer. Read key results, rather than every number on a slide.
- Possible handover points: after slide 4 (setup) and slide 9 (results). Allocate these sections to your actual team members.
- Rehearse at roughly 110–120 words per minute. If behind schedule, shorten the learning-curve and frequency comments; keep the main results and conclusion.

## Slide-by-slide script

### Slide 01 — ViGoEmotions: Task and Dataset

Good morning. Our project recognizes emotions in Vietnamese social-media comments. Because one comment can express several emotions, this is a multi-label task. The dataset contains twenty thousand, six hundred and sixty-four comments. We compare three models, with and without an explicit Neutral output. I will briefly cover the setup, then focus on our results and analysis.

### Slide 02 — Data Profile and Leakage Prevention

Comments average about one point eight six emotions, and the median length is eleven whitespace tokens. We keep the supplied split, remove one training copy of a test comment, and fit vocabulary only on training data. Unicode and whitespace are normalized. Our distribution audit does not prove the original stratification algorithm, so that requirement still needs clarification.

### Slide 03 — 27 vs 28 Outputs: What Changes?

The twenty-seven-output model predicts emotions only. Neutral-only comments have an all-zero emotion target. The twenty-eight-output model adds a learned Neutral label. We preserve mixed annotations, where Neutral appears with emotions. Both neural settings use sigmoid outputs and binary cross-entropy because labels are not mutually exclusive. Validation contains ninety-four Neutral-positive examples.

### Slide 04 — Models and Shared Training Protocol

We compare TF-IDF with one-versus-rest Logistic Regression, a bidirectional GRU trained from scratch, and full PhoBERT fine-tuning. Both neural models complete twenty epochs in each setting. TF-IDF uses a solver limit instead. We share the seed and split, and select checkpoints by validation Macro-F1 on the same twenty-seven emotions at threshold zero point five.

Optional handover: “That is our setup. My teammate will now present the results.”

### Slide 05 — Results: 27 Emotions

For twenty-seven emotions, Macro-F1 is 0.2388 for TF-IDF, 0.4220 for BiGRU, and 0.5974 for PhoBERT. PhoBERT leads on all four overall metrics in this setting. Macro-F1 weights labels equally; Micro-F1 pools decisions across labels. Hamming Loss measures incorrect decisions, while LRAP evaluates probability rankings.

### Slide 06 — Results: 28 Labels Including Neutral

For twenty-eight labels, Macro-F1 is 0.2310, 0.4086, and 0.5783, respectively. PhoBERT again performs best within this setting. However, these scores include Neutral. We cannot directly subtract the overall twenty-seven-label and twenty-eight-label scores to measure the benefit of Neutral, because their evaluated label sets differ.

### Slide 07 — Does Neutral Help the Shared 27 Emotions?

For a comparison on identical labels, we score only the twenty-seven emotion columns of the expanded model. PhoBERT then reaches 0.5836, about 1.38 percentage points below the twenty-seven-output model. TF-IDF is unchanged and BiGRU decreases slightly. This run shows no improvement on the shared emotions, but one seed cannot establish a general causal conclusion.

### Slide 08 — Neutral Detection Remains Difficult

PhoBERT identifies 45 of the ninety-four Neutral-positive validation examples. It misses 49 and produces 67 false positives, giving Neutral F1 of 0.4369. So the additional label is learned, but still difficult. The twenty-seven-output model has no Neutral classifier; its Neutral F1 is undefined, not zero. An empty prediction is also not automatically Neutral.

### Slide 09 — Learning Curves and Best Checkpoints

These curves cover all twenty epochs. The best PhoBERT checkpoints occur at epochs fourteen and fifteen. We finish training before loading the selected checkpoint. Validation F1 fluctuates after the best epoch even while training loss decreases. Because validation loss was not recorded, we avoid diagnosing confirmed overfitting from training loss alone.

Optional handover: “My teammate will now discuss the analysis and next steps.”

### Slide 10 — Q1–Q2: Per-Label Performance and Frequency

For individual emotions, PhoBERT performs best on gratitude and worst on disapproval. The correlation between training frequency and per-label F1 is approximately 0.018, with a p-value of 0.928. These descriptive results do not support a simple rule that more frequent labels are always easier. The report includes every label for all three models.

### Slide 11 — Q3–Q4: Label Pairs and Text Styles

Our most frequent training pair is disappointment and sadness. PhoBERT predicts both correctly in 63 of 122 validation cases containing both labels. For text styles, its largest advantage over BiGRU is in teencode, at about 22.1 percentage points of Macro-F1. Teencode means informal online spellings and abbreviations. Our style groups use heuristics and can overlap, so this finding needs cautious interpretation.

### Slide 12 — Q5: Does the Default Threshold Matter?

We also vary the threshold while keeping the checkpoint fixed. For PhoBERT, the best validation grid point is 0.35, with Macro-F1 0.5978, compared with 0.5974 at zero point five. The gain is tiny and Hamming Loss worsens. This is exploratory analysis, not a test improvement; the main comparison stays at zero point five.

### Slide 13 — Error Analysis: New Neutral Predictions

A preliminary, AI-assisted review examines three confident Neutral false positives and three Neutral misses. These examples suggest difficulties with implied emotion, short text, HTML, and profanity. These are hypotheses, not proven causes. The review needs team confirmation. We publish sample IDs and abstract observations instead of the original comments.

### Slide 14 — Final Research Question and Next Steps

For the final evaluation, we will test whether one threshold suits every emotion. We will freeze the best neural checkpoint and compare the default threshold, a tuned global threshold, and per-label thresholds. We select thresholds on validation, lock all choices, then evaluate test once. We also plan a detailed review of at least twenty false negatives and twenty false positives.

### Slide 15 — Conclusions and Evidence

To conclude, PhoBERT gives the strongest overall results in both settings. Adding Neutral does not improve shared-emotion F1 in this run, and Neutral prediction still makes substantial errors. Saved Kaggle predictions and histories support these findings. Our next step is controlled threshold evaluation. Thank you; we welcome your questions.

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
