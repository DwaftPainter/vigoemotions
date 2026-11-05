# ViGoEmotions — English Midterm Presentation Script

18 slides; 935 spoken words. Target: about 8–9 minutes, including short pauses and handovers. Finish before the 10-minute limit; leave any remaining time for questions.

Use this script with reports/midterm_slides.pdf. The numerical results come from the verified Kaggle 20-epoch studies. The Vietnamese report and study guide remain available for preparation.

## Suggested timing

- Slides 1–5: introduction, data and model setup, about 2 minutes.
- Slides 6–9: results, Neutral comparison and learning curves, about 2 minutes.
- Slides 10–15: dedicated Q1–Q5 analyses, about 3–3.5 minutes.
- Slides 16–18: error analysis, final research question and conclusion, about 1.5 minutes.

## Before presenting

- Add your real group details to the submitted materials; do not read these preparation instructions aloud.
- Confirm the six AI-assisted error-review observations before presenting them as your team’s analysis.
- Rehearse once with a timer. Read key results, rather than every number on a slide.
- Possible handover points: after slide 5 (setup) and slide 9 (results). Allocate these sections to your actual team members.
- Rehearse at roughly 110–120 words per minute. If behind schedule, shorten the learning-curve and frequency comments; keep the main results and conclusion.

## Slide-by-slide script

### Slide 01 — ViGoEmotions

Good morning. Our project recognizes emotions in Vietnamese social-media comments. Because one comment can express several emotions, this is a multi-label task. The dataset contains twenty thousand, six hundred and sixty-four comments. We compare three models, with and without an explicit Neutral output. I will briefly cover the setup, then focus on our results and analysis.

### Slide 02 — Dataset, split and emotion frequency

Comments average about one point eight six emotions, and the median length is eleven whitespace tokens. We keep the supplied split, remove one training copy of a test comment, and fit vocabulary only on training data. Unicode and whitespace are normalized. Our distribution audit does not prove the original stratification algorithm, so that requirement still needs clarification.

### Slide 03 — 27 vs 28 outputs: one additional decision

The twenty-seven-output model predicts emotions only. Neutral-only comments have an all-zero emotion target. The twenty-eight-output model adds a learned Neutral label. We preserve mixed annotations, where Neutral appears with emotions. Both neural settings use sigmoid outputs and binary cross-entropy because labels are not mutually exclusive. Validation contains ninety-four Neutral-positive examples.

### Slide 04 — Three routes from text to multi-label scores

This diagram shows the three processing routes. TF-IDF uses lexical features and one classifier per label. BiGRU learns embeddings and reads sequences in both directions. PhoBERT uses Vietnamese word segmentation and BPE before a pretrained backbone and a new classification head. The neural routes use independent sigmoid outputs; there is no single-label softmax decision.

### Slide 05 — Architectures and training configurations

We compare TF-IDF with one-versus-rest Logistic Regression, a bidirectional GRU trained from scratch, and full PhoBERT fine-tuning. Both neural models complete twenty epochs in each setting. TF-IDF uses a solver limit instead. We share the seed and split, and select checkpoints by validation Macro-F1 on the same twenty-seven emotions at threshold zero point five.

Optional handover: “That is our setup. My teammate will now present the results.”

### Slide 06 — PhoBERT leads within each label setting

The table separates full twenty-seven-emotion and twenty-eight-label evaluation. PhoBERT leads within each block: Macro-F1 is zero point five nine seven four and zero point five seven eight three, respectively. The additional columns show Micro-F1, Hamming Loss, rankings, selected epochs, and measured training time. Different label sets mean these two overall F1 scores should not be directly subtracted.

### Slide 07 — Adding Neutral does not raise shared-emotion F1 here

For a comparison on identical labels, we score only the twenty-seven emotion columns of the expanded model. PhoBERT then reaches 0.5836, about 1.38 percentage points below the twenty-seven-output model. TF-IDF is unchanged and BiGRU decreases slightly. This run shows no improvement on the shared emotions, but one seed cannot establish a general causal conclusion.

### Slide 08 — A learned Neutral output still makes substantial errors

PhoBERT identifies 45 of the ninety-four Neutral-positive validation examples. It misses 49 and produces 67 false positives, giving Neutral F1 of 0.4369. So the additional label is learned, but still difficult. The twenty-seven-output model has no Neutral classifier; its Neutral F1 is undefined, not zero. An empty prediction is also not automatically Neutral.

### Slide 09 — Twenty epochs trained; the last checkpoint need not win

These curves use the same twenty-seven-emotion checkpoint-selection metric for both output settings. Both neural models completed twenty epochs. The selected PhoBERT checkpoints are at epochs fourteen and fifteen. Selection is based on validation, not simply the final epoch. Training loss continued to decrease, but validation loss was not recorded, so we do not claim confirmed overfitting.

Optional handover: “My teammate will now discuss the analysis and next steps.”

### Slide 10 — Q1 · Which emotions are easiest and hardest?

Gratitude is the easiest label for all three models. The weakest label is realization for TF-IDF and BiGRU, but disapproval for PhoBERT. The bars show selected PhoBERT labels; the table compares each model’s extremes. Overall F1 hides these differences.

### Slide 11 — Q2 · Does label frequency predict F1?

For Q2, each dot represents one emotion under PhoBERT. Its frequency–F1 correlation is only 0.018. The table also shows TF-IDF and BiGRU. These descriptive correlations do not establish causation; frequency alone cannot explain which labels are difficult.

### Slide 12 — Q3 · How often are both emotions recovered?

For Q3, we choose disappointment and sadness because they co-occur most often in training. Validation contains 122 comments annotated with both. TF-IDF recovers both in two cases, BiGRU in thirty-seven, and PhoBERT in sixty-three. Joint recall is therefore 1.64, 30.33 and 51.64 percent. Other predicted labels are allowed, so this does not measure pair precision.

### Slide 13 — Q4 · How does performance vary by text style?

Q4 compares every available text-style group using the same twenty-seven labels. PhoBERT leads in all four rows. Its largest advantage over BiGRU is teencode: 22.13 percentage points on 307 examples. These groups use heuristic rules and overlap. Their sample counts differ, so this is descriptive rather than a controlled style effect.

### Slide 14 — Q5 · Lower thresholds affect models differently

Q5 keeps each checkpoint fixed and varies only the threshold. The three curves respond differently. TF-IDF gains about 24.55 percentage points of validation Macro-F1; BiGRU gains 1.86; PhoBERT gains only 0.043. Circles mark the best grid points, not test improvements.

### Slide 15 — Q5 · F1 gains come with more label errors

The table compares the default and exploratory best thresholds for every model. Lower thresholds improve Macro-F1 but increase Hamming Loss in all three cases. Choosing on validation can overestimate generalization. We keep the main comparison at 0.5 and will lock decoding choices before evaluating test.

### Slide 16 — Neutral errors: implied affect versus surface cues

Two examples illustrate the six-case preliminary, AI-assisted review. On the left, a comment annotated as relief is predicted as Neutral. On the right, a Neutral comment is assigned anger and annoyance. Implied affect and profanity are possible explanations, not proven causes. We show IDs and abstract observations, and the interpretations still need team confirmation.

### Slide 17 — Is one threshold suitable for every emotion?

For the final evaluation, we will test whether one threshold suits every emotion. We will freeze the best neural checkpoint and compare the default threshold, a tuned global threshold, and per-label thresholds. We select thresholds on validation, lock all choices, then evaluate test once. We also plan a detailed review of at least twenty false negatives and twenty false positives.

### Slide 18 — What the current study supports

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
