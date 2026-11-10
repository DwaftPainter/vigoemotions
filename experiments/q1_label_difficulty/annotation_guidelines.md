# Blinded annotation protocol (H1 and H2)

This is a proposed human study, not completed annotation evidence. Two independent
Vietnamese-proficient annotators review 71 items by default: up to 12 TP, FP and FN
items per target. Gratitude has only 11 FP, so that stratum is a census. Outcomes
are hidden. No TN items are sampled; this study cannot estimate population-wide
human judgments of negatives. Sampling is deterministic (seed 42, SHA-256 ordering).

The coordinator alone may access `data/q1_label_difficulty/annotation_key.csv`.
Give each annotator **only** this guideline, their blank `annotator_N.csv`, and the
authorized `review_text_private.csv` packet. Do not give them source sample IDs,
dataset annotations, saved predictions, probability scores, outcome strata, this
study's diagnostic report, or the other annotator's work before independent review
is locked. Review IDs are opaque pseudonyms, not a cryptographic anonymity guarantee.
The text packet is version-locked by SHA-256; changing it invalidates analysis.

## Unit and definitions

For each `review_id`, judge the **target_label** from the comment alone. Do not
assume that the target is present merely because it is the focus of the item.

- **Gratitude:** appreciation or thankfulness toward a person, entity or event.
- **Disapproval:** an unfavorable evaluation, rejection or negative judgment of a
  behavior, claim, entity or situation. Do not equate every negative feeling with
  disapproval: anger concerns antagonism, annoyance irritation, disgust aversion,
  and disappointment unmet expectations. These distinctions are proposed review
  guidance; difficult boundaries should be marked, not forced.
- **Explicit:** the intended target emotion is directly expressed by the wording
  and its syntactic/pragmatic use. A public phrase such as “cảm ơn” is a cue to
  inspect, not an automatic answer: quotation, sarcasm or negation can reverse it.
- **Implicit:** the target emotion is reasonably conveyed but requires inference
  from the described action, evaluation or situation rather than a direct emotion
  statement. Do not fill in missing conversational context from imagination.
- **Ambiguous:** multiple defensible readings, unclear target, sarcasm that cannot
  be resolved, or insufficient context prevents a reliable presence/explicitness
  judgment. Uncertainty is a valid annotation.
- **Not present:** no sufficient evidence of this target emotion. It is not the
  same as Neutral: other emotions can be present.

## Fields and consistency

1. Set a pseudonymous `annotator_id` consistently for your entire file. The two
   annotators must have different identifiers and work independently.
2. `target_presence`: `yes`, `no`, or `uncertain`.
3. `expression`: `explicit`, `implicit`, `ambiguous`, or `not_present`.
   Presence `no` requires `not_present`; `uncertain` requires `ambiguous`.
   Presence `yes` may use `ambiguous` if presence is clear but explicitness is not.
4. `semantic_overlap`: `none`, `possible`, `shared_cue`, or `uncertain`.
   `possible` means a genuinely difficult boundary with another emotion;
   `shared_cue` means the same wording plausibly conveys both. A simple, clearly
   distinguishable multi-emotion comment is not automatically semantic ambiguity.
5. `overlap_labels`: canonical lower-case labels, semicolon-separated; omit the
   target itself. Use empty for none. With `uncertain`, leave empty if no specific
   alternative can responsibly be named. Use the repository's 27-label mapping in
   `docs/label_mapping.md`; do not add new labels.
6. `rationale_code`: one of `direct_emotion_phrase`, `inference_from_context`,
   `negation_or_target`, `multiple_interpretations`, `insufficient_context`,
   `no_target_signal`. No quotation or free-text comment goes in the CSV.
7. Set `status=complete` only after the actual human review. Blank or `pending`
   means unreviewed; do not replace missing judgments with AI-generated labels.

Any discussion notes or quotations must remain in an authorized private location.
Do not commit raw comments, screenshots of comments or copied rationale text.

## Agreement and adjudication

Freeze both independent files before running agreement analysis. The analysis
reports raw agreement and nominal Cohen κ for expression, presence and overlap
category on paired completed items. Kappa is N/A for a homogeneous pooled category;
agreement of 1 is not sufficient to manufacture κ=1. Agreement describes this
outcome-balanced sample and is not a population estimate. Different target-label
items can refer to the same comment; do not pool them as independent trials.

For disagreements, use the coordinator's `adjudication.csv`. A third reviewer or
a documented consensus discussion must fill actual decisions, `annotator_id`,
`resolution_method=third_review|consensus_discussion`, and `status=complete`.
The adjudicator still does not need model scores. Exact independent agreements
can be resolved as two-annotator agreement; conflicting judgments are never
automatically voted into a fabricated human conclusion. Preserve both originals.

## Analysis denominators

Explicit/implicit **recall** uses only original ground-truth-positive TP and FN
items. FP cases never enter that denominator, even if humans disagree with their
original annotation. Such disagreement is recorded separately and does not modify
official ground truth. A naive TP/(TP+FN) in this balanced review sample is biased
for the full validation population. The coordinator key records inclusion
probabilities; weighted recall uses inverse inclusion weights and is withheld
until **all planned items** are resolved, avoiding an unexamined complete-case
assumption. Intervals resample within outcome strata; no finite-population
correction or training-seed uncertainty is included. Small expression groups
(fewer than five reviewed positives) do not receive bootstrap intervals.

Semantic-overlap tables use reviewed outcome-specific denominators. Neither
co-occurrence nor an overlap judgment alone proves the cause of a model error.
