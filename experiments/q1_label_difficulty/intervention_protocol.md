# Controlled wording intervention (H4)

This protocol is ready to execute, but **no paraphrases or human verifications
have been completed by this implementation**. Preparation generates only IDs,
blank text fields and `proposed` metadata. The current public result is pending.

## Paired design

Select up to eight validation ground-truth-positive originals for each target by
deterministic SHA-256 order, independently of model scores. Prepare three proposed
variants per original:

1. `lexical_cue_removal`: remove or replace a hypothesized target cue while retaining
   the intended emotion and propositional meaning.
2. `explicit_to_implicit`: reformulate a direct emotion expression into an indirect
   expression while retaining its intended target. This may be impossible for a
   particular comment; reject the pair rather than force equivalence.
3. `neutral_control`: change emotion-irrelevant surface wording while preserving
   the emotion-bearing wording, meaning and target. This means a **control edit**,
   not changing the target to the Neutral label.

Keep entities, evaluation target, stance/polarity, intensity and substantive events
as constant as possible. Inspect word order, negation, sarcasm, deixis and missing
context. A large semantic rewrite is not an acceptable control. Human verifiers
may reject a variant or mark it uncertain. Do not accept an AI-generated paraphrase
as label-preserving just because its generator says so.

## Private text and human verification

`intervention_pairs.csv` is ID/metadata-only. Original comments remain in
`intervention_originals_private.csv`; edited text belongs in the private
`intervention_text_private.csv` (`paraphrase_id,text`). All stay under ignored
`data/q1_label_difficulty/`. Public template CSVs contain headers only.

After writing proposed text, run `bind`. This computes SHA-256 of the **canonical
NFC/whitespace-normalized** original and paraphrase. Any changed text resets its
status and verifier fields to proposed. Then two distinct independent humans
must each review the exact bound pair, without model scores, and set:

- `verifier_1/2`: stable human pseudonyms; `reviewer_type_1/2=human`;
- `semantic_equivalence_1/2=yes`;
- `label_preserved_1/2=yes`;
- coordinator `status=verified` only after both actual approvals.

An `ai` reviewer, missing approval, conflicting decision, edited text hash or
unverified proposal cannot enter inference. `rejected` pairs are excluded; report
their counts. Retain proposed and rejected records rather than quietly deleting
hard cases. This is attested human verification, not automatic authentication of
the real-world reviewer. Keep B-study reviewers blind to these intervention
packets until their independent B annotations have been frozen.

## Frozen inference contract

Use the verified **27-output** `labels27/phobert/best.pt` from the private Kaggle
ZIP. The script checks the ZIP checksum from official provenance and the extracted
checkpoint byte hash against that member. It loads a state dictionary with
`torch.load(..., weights_only=True)`; it never trains or changes hyperparameters.

It reuses `train_neural.segmented_texts` and `train_neural.predict`, with the frozen
PhoBERT-base-v2 revision, slow tokenizer, VnCoreNLP segmentation, padding/truncation
to 256, batch size 16, sigmoid decoding and threshold 0.5. Originals are inferred
alongside the variants. Their full 27-dimensional scores must reproduce the saved
validation probabilities within absolute tolerance **1e-4**, with zero relative
tolerance, before any intervention result is accepted. A failure is a dependency,
checkpoint or preprocessing mismatch; do not relax the tolerance silently.

Use the original Kaggle environment where practical: Torch 2.11.0+cu130,
Transformers 4.57.3, py-vncorenlp 0.1.4, Java and the same segmentation assets.
CPU/GPU/library differences may fail the reproduction check. No checkpoint is
loaded until verified pairs and private dependencies are available. Missing
dependencies produce errors or a documented pending state, never synthetic scores.

## Estimands and uncertainty

Per target and transformation, report paired score shifts, flips, loss/gain counts
and denominators. The primary mean shift and flip rate first average variants
within an original comment, then average originals equally; bootstrap original
comments (2,000 replicates, seed 42). Pair-weighted descriptive results are also
reported explicitly. Positive-to-negative rates divide by original positive
predictions; negative-to-positive rates use original negative predictions. No
interval is assigned to a singleton original cluster.

Control-adjusted comparisons require the same original and target in both arms.
For each matched original, subtract its mean control shift from its mean treatment
shift, then bootstrap those matched originals. Do not compare unmatched treatments
and controls as if paired. A fixed, frozen checkpoint and within-original controls
support evidence of model sensitivity on the selected examples. They do not by
themselves explain the original population F1 gap or establish a general causal
linguistic effect. Account for selection, human-verification errors, small samples
and multiple transformations when discussing results.

The behavioral-testing motivation follows [CheckList (Ribeiro et al., 2020)](https://aclanthology.org/2020.acl-main.442/).
Unlike label-changing contrast sets, this study requires target-label preservation
verified by humans. Model loading follows the [PyTorch state-dictionary guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
and [Transformers 4.57.3 padding/truncation contract](https://github.com/huggingface/transformers/blob/v4.57.3/docs/source/en/pad_truncation.md).
