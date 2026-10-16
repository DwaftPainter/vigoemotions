# Mixed-neutral label ablation (Midterm)

## Research question and annotation issue
Does retaining emotion labels of mixed-neutral training samples improve predictive
performance compared with excluding those samples entirely? Neutral (27) sometimes
co-occurs with emotion labels, creating conflicting annotations under the project's
27-emotion convention. Neutral-only examples have all-zero targets.

## Design
A retains all 16,530 cleaned training rows and removes only the Neutral label.
B excludes exactly 251 mixed-neutral training rows, leaving 16,279 rows.
Both use the same 2,066 validation IDs and target vectors, including all 19 mixed-neutral
validation rows. Their saved prediction archives are checked for exact ID/target equality.
All normalization, TF-IDF settings, OvR Logistic Regression settings, seed 42, and
threshold 0.5 are held constant using configs/midterm.json. Vocabulary and IDF are
fit independently on each training subset. The changed training subset can therefore
also change the learned feature space. No threshold search is performed.
The existing loader reads test data for the established overlap audit only; no test
predictions, metrics, tuning, or selection are performed. Original CSVs are read-only.
The Final research question on threshold optimization remains unchanged.

## Validation results
| Metric | A: emotion priority | B: exclude mixed | B minus A |
|---|---:|---:|---:|
| macro_f1 | 0.23882529 | 0.24041354 | +0.00158825 |
| micro_f1 | 0.27123760 | 0.27290364 | +0.00166604 |
| hamming_loss | 0.06059302 | 0.06046753 | -0.00012549 |
| lrap | 0.64074470 | 0.64021761 | -0.00052709 |
| train_seconds | 3.14540393 | 2.97602575 | -0.16937818 |
| training_samples | 16530.00000000 | 16279.00000000 | -251.00000000 |
| tfidf_features | 34837.00000000 | 34403.00000000 | -434.00000000 |

Removing mixed-neutral training samples **improved validation Macro-F1**
by 0.00158825. Higher F1/LRAP and lower Hamming Loss are better. Differences
are descriptive absolute-scale differences, not relative percentages; CSVs include
both signed B-minus-A values and their absolute magnitudes.

The changes should be assessed metric by metric: an F1 gain can coexist with a
ranking-quality (LRAP) loss. This run does not justify a universal performance claim.

| Label tier or validation subgroup | A Macro-F1 | B Macro-F1 | B minus A |
|---|---:|---:|---:|
| head | 0.25123195 | 0.25562937 | +0.00439742 |
| mid | 0.30708314 | 0.30745048 | +0.00036734 |
| tail | 0.15816077 | 0.15816077 | +0.00000000 |
| mixed_neutral | 0.05185185 | 0.05185185 | +0.00000000 |
| non_mixed_neutral | 0.23923523 | 0.24082265 | +0.00158742 |

See per_label_f1.csv for precision, recall, F1, and support; frequency_tier_comparison.csv
for head/mid/tail metrics; and validation_subgroup_comparison.csv for subgroup metrics.
Tiers are frozen from A's cleaned training frequencies: descending count, ties by
label ID, nine labels per tier. Tier metrics operate on those nine columns only;
tier LRAP ranks within the tier, with the existing all-zero target convention.
Macro-F1 always includes every column in its evaluated group, with zero_division=0.
Training time includes TF-IDF fitting and classifier fitting, excluding data loading,
validation transformation, and prediction. Timing is a single hardware-dependent run.

## Implications and limitations
These results inform whether retaining partially conflicting training annotations is
useful for this baseline. Better predictive performance does not establish which
original annotation is correct. This is one fixed split, seed, model, and threshold;
no significance test or independent test evaluation was performed. Removing rows also
reduces training size and changes label prevalence and learned vocabulary. The optional
random-exclusion control was not run, so annotation effects cannot be separated from
data-volume effects. The 19 mixed-neutral validation cases are supplementary and
are too few for statistically conclusive claims. Validation targets retain the emotion
priority convention, so the comparison is conditional on that evaluation convention.
Cross-environment reproduction may introduce numerical and runtime differences.

## Reproduction
Run `.venv/bin/python scripts/ablate_mixed_neutral.py` from the repository root.
Existing output directories are refused; use `--output reports/ablation_mixed_neutral/rerun_01`
for a new run. Archives contain ordered IDs, targets, continuous probabilities, and
label names. Training-ID manifests, input hashes, configuration, and package versions
are recorded without raw comments, vocabulary, model checkpoints, or text snippets.
