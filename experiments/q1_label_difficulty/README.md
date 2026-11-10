# Q1 label-difficulty study

This isolated study investigates associations and optional controlled interventions for
Gratitude versus Disapproval. It does not replace the official Q1–Q5 results.

H1 (lexical explicitness), H2 (semantic ambiguity), H3 (text characteristics) and
H4 (lexical dependence) are hypotheses. Validation associations cannot establish
their causes. Human annotation and verified intervention results must remain pending
until actual reviews exist.

## Current evidence and scope

| Experiment | Status | Evidence available |
| --- | --- | --- |
| A: quantitative diagnostics | Completed | Saved PhoBERT-27 validation scores and authorized train/val CSVs |
| B: blinded annotation | Pending humans | 71 deterministic items; two independent review packets and adjudication template |
| C: paired wording interventions | Pending humans and inference | 48 proposed slots, blank paraphrases, no intervention scores |

The genuine A report is [q1_analysis.md](../../reports/q1_label_difficulty/q1_analysis.md).
It reproduces Gratitude F1 0.8370 and Disapproval F1 0.3623, with counts and
uncertainty. These are **27-output validation** results at threshold **0.5**, from
the frozen 20-epoch run's selected epoch-14 checkpoint. They are not test results
or results from the 28-output study. TF-IDF/BiGRU training is outside this study.

No source CSV, checkpoint, vocabulary, completed review, or raw source quotation
is published. Private review materials stay in git-ignored `data/`. Do not use
`git add -f` to commit them. Public CSV templates contain headers only.

## Prerequisites and source contracts

Run commands from the repository root using the existing Python environment and
dependencies. Experiments A/B do not require GPU inference. Put authorized source
`id,text,labels` CSVs in `data/train.csv` and `data/val.csv`; alternatively pass
`--data-dir /authorized/location`. A test CSV is neither needed nor read.

The scripts verify CSV bytes against the original manifest (allowing the existing
LF/CRLF contract), fixed train/validation IDs and ordering, canonical label mapping,
saved probability bounds/targets, and the original NPZ/run checksums. They reuse
`common` normalization/targets/style heuristics and the importer's validation
contract rather than maintaining a second label implementation. Train frequency
uses the same retained 16,530 IDs; validation uses 2,066 IDs. The removed train
overlap is not counted again. Missing private data fails clearly.

Sources are `reports/kaggle_equal20_2026-11-05/`, its original manifest/provenance,
`configs/kaggle_equal_20ep.json`, and fixed `splits/train.csv` / `splits/val.csv`.
SHA-256 of inputs and implementation, label order, settings, package versions and
seeds are recorded in the new study's provenance. Existing reports, notebooks,
experimental artifacts, training code and hyperparameters are not rewritten.

## A: reproduce quantitative diagnostics

```bash
.venv/bin/python scripts/analyze_q1_difficulty.py
```

Outputs under `reports/q1_label_difficulty/`:

- `tables/metrics.csv`: full label confusion counts, prevalence, precision,
  recall, F1 and conditional intervals.
- `tables/subgroups.csv`: every length, cardinality, style, punctuation, negation
  and public lexical-probe group, with n, positive/predicted-positive denominators
  and FP/FN separately. Subgroups overlap across families.
- `tables/probabilities.csv`: score quantiles and descriptive proximity bands for
  TP, FP, FN and TN; no threshold optimization.
- `tables/outcome_features.csv`: length/cardinality summaries within each outcome.
- `tables/overlaps.csv`: other-label truth/prediction/unsupported-prediction counts
  within each target outcome; this is not a single-label confusion matrix.
- `tables/train_lexical_probes.csv`: train-only coverage of the hand-written probes.
- `figures/`: score densities and length/cardinality F1 plots.
- `provenance.json` and `q1_analysis.md`: source hashes, paired F1-gap interval,
  methods, findings and unresolved hypotheses.

Wilson intervals use the appropriate binomial denominator. F1 uses 2,000
deterministic bootstrap replicates by default; the between-label gap resamples
the same comments for both labels. Small/undefined F1 intervals are omitted rather
than filled with zeros. The intervals condition on this checkpoint/sample and
exclude training variation and repeated validation selection. No significance
claim or causal interpretation follows from them. The public lexical probes are
unequal, incomplete exploratory proxies; cue absence does not mean implicit emotion.

An independent output tree can be reproduced without changing official artifacts:

```bash
.venv/bin/python scripts/analyze_q1_difficulty.py --output-dir /private/tmp/vigo-q1-reproduction
```

The output guard permits only the new study tree inside the repository, or an
external temporary directory. It refuses to write over official midterm paths.

## B: human annotation and adjudication

Read [annotation_guidelines.md](annotation_guidelines.md) before distributing
packets. Preparation never generates human judgments:

```bash
.venv/bin/python scripts/analyze_q1_annotations.py prepare
.venv/bin/python scripts/analyze_q1_annotations.py analyze
```

`prepare` creates private coordinator mappings, authorized source text, two blank
annotator packets and a blank adjudication packet in `data/q1_label_difficulty/`.
Give each independent annotator only their packet, the review text and guidelines;
do not reveal ground truth, outcomes, model confidence or the other review.
The coordinator mapping and text are SHA-256 locked to the deterministic sample.
Preparation refuses to overwrite actual work; use a new `--private-dir` under
`data/` and a new `--output-dir` for a different design.

After real reviews, run `analyze` again. It validates controlled fields, complete
IDs, reviewer independence and adjudication; a `third_review` adjudicator must be
different from both annotators. It reports raw agreement and Cohen's kappa on
paired completed items. Undefined kappa remains missing. Disagreements require
actual human adjudication, not automatic majority labels. Pending inputs yield a
pending report without invented annotation metrics.

Explicit/implicit recall uses **original ground-truth-positive** TP/FN cases;
FP cases cannot enter that denominator. Outcome-balanced sampling distorts naive
recall, so population estimates use inverse inclusion weights and are withheld
until every planned item is resolved. Outcome-stratified bootstrap intervals are
descriptive and omit finite-population corrections; no interval for groups below
five sampled positives. Human overlap judgments are aggregated separately and
must not be conflated with ground-truth co-occurrence or causal confusion.

## C: optional verified intervention execution

Read [intervention_protocol.md](intervention_protocol.md). Default preparation
creates eight originals per target, with three transformation slots each. It
does not generate paraphrase text, claim equivalence, or load a model:

```bash
.venv/bin/python scripts/evaluate_q1_interventions.py prepare
```

Write proposed paraphrases only in private `intervention_text_private.csv`, then
bind their normalized-text hashes:

```bash
.venv/bin/python scripts/evaluate_q1_interventions.py bind
```

Two distinct humans must verify semantic equivalence and target-label preservation
of the exact bound pair. Set the private pair status to `verified` only after
their independent actual approvals. Editing text resets these approvals on the
next bind and invalidates evaluation until it is reviewed again. Proposed,
rejected, AI-only and unbound pairs cannot enter inference.

To run, obtain `labels27/phobert/best.pt` from the verified private 27-label ZIP
and keep both archive and checkpoint private. The archive must be at the path
recorded in the official provenance. Provide the same VnCoreNLP segmentation
assets and original inference dependencies described in the protocol:

```bash
.venv/bin/python scripts/evaluate_q1_interventions.py run --checkpoint data/checkpoints/phobert_best.pt --vncorenlp-dir data/vncorenlp
.venv/bin/python scripts/evaluate_q1_interventions.py analyze
```

No verified pairs produces a pending state without inference. Verified pairs but
missing assets produce an explicit dependency error. With assets available, the
checkpoint must match the verified ZIP member and original full-27 scores must
reproduce saved predictions within absolute tolerance 1e-4. Do not relax this gate
to obtain results. There is no retraining or threshold tuning.

Public results contain only IDs, scores and aggregates for verified pairs.
Primary shifts/flips average variants within originals before averaging originals;
uncertainty resamples original comments. Control-adjusted estimates require the
same original and target in both arms. Keep proposed/rejected counts visible and
interpret findings as checkpoint sensitivity on selected verified examples, not
an explanation established for the original label-F1 gap.

## Verification

```bash
PYTHONPATH=tests .venv/bin/python -m unittest test_q1_difficulty -v
.venv/bin/python -m unittest discover -s tests -v
```

Tests use synthetic examples only and never create actual human study evidence.
They cover ID/label alignment, canonical metric agreement, threshold boundaries,
subgroup denominators, weighted annotation recall, adjudication, paired shifts,
matched controls, text-bound verification and rejection of unverified inputs.
See `reports/q1_label_difficulty/verification.json` for the final audit record.
