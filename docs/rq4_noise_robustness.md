# Extension 3 — Social-media text noise robustness (RQ4)

**Question.** How do Vietnamese diacritic removal and simulated QWERTY typos change held-out performance of the same frozen PhoBERT-28 checkpoint? The hypothesis is that both perturbations lower test all28 Macro-F1 under the fixed 0.5 decoder. The global and per-label decoders are secondary comparisons. This extension was specified after the completed RQ1 evaluation and is exploratory on the same held-out split.

| Condition | Transformation |
| --- | --- |
| Original | Supplied text after the recorded RQ1 normalization. Comments can already contain natural noise. |
| No diacritics | Remove Vietnamese vowel marks and map đ/Đ to d/D. Preserve case, spaces, punctuation and emoji. |
| QWERTY typo | Each eligible ASCII letter has a 5% replacement probability. Select a neighboring alphabetic key in a fixed staggered QWERTY layout, preserve case and leave accented characters intact. A stable hash of seed 42, sample ID and character position determines each draw. |

Apply noise **after the recorded Unicode/whitespace normalization and before VnCoreNLP and PhoBERT tokenization**. The paired conditions retain all 2,066 validation and 2,067 test IDs, in order, with the original 28-label targets. Include unchanged comments and report the share actually changed to show effective coverage. The two perturbations use their own rules and differ in realized severity. The experiment measures those specific conditions and does not establish equal-severity sensitivity or real-world degradation. Text perturbations can change meaning even though reference labels are held fixed.

Keep epoch 15, preprocessing, model weights and all three RQ1 decoders fixed. Verify the RQ1 pre-test evidence, restored assets and original-text scores before interpreting noise deltas. Run all three conditions with the same RQ4 device and packages on a given split. The original-text control is inferred in the same run and compared against saved RQ1 scores, with absolute tolerance 1e-4 and exact fixed decisions. Agreement for the global and per-label decisions is also recorded. Differences in those decisions do not alter the within-run control used to compute deltas.

## Run

Use the original authorized train/val/test CSVs and intact private 28-label checkpoint archive. Keep the original RQ1 evidence inside the code bundle. See [RQ1 asset and environment instructions](final_execution.md).

On Kaggle, create the code archive with `python scripts/package_rq4.py`. Import [notebook 11](../notebooks/11_kaggle_rq4_noise_robustness.ipynb), attach `vigoemotions_rq4_code.zip`, the original private checkpoint ZIP and the authorized CSVs. Select a GPU. Run through validation and review the coverage and tables. The held-out cell has an explicit `RUN_HELD_OUT` switch. Set it to `True` after validation succeeds and run that cell in the same environment. The notebook stages private inputs in a temporary directory and writes only result archives to Kaggle output.

For an existing local setup with dependencies from `requirements-colab.txt`, Java and restored assets:

```bash
python scripts/rq4_robustness.py verify
python scripts/rq4_robustness.py validate --device cuda
python scripts/rq4_robustness.py test --device cuda
```

Use `--device cpu` on a compatible CPU runtime. Use `--data-dir`, `--private-dir`, and, if needed, `--rq1-dir` to refer to existing assets from a separate Git worktree. RQ4 writes to `reports/rq4_noise/{val,test}/` by default. The test command requires a completed validation manifest with matching configuration, source code, output hashes and runtime. Validation certifies the setup and coverage. The configuration fixes the noise rate before evaluation, with no optimization from either split.

## Read results

- `tables/comparison.csv` contains the three conditions and three decoders, with Macro-F1, Micro-F1, Hamming Loss and LRAP for all28 and emotion27. Each delta is **perturbed minus original**, in percentage points for the same split, scope and decoder.
- `tables/frequency_tiers.csv`, `per_label.csv` and `neutral.csv` show where performance changes. Frequency tiers remain fixed from RQ1 train counts.
- `tables/noise_coverage.csv` records changed comments and edited/eligible characters. For typo coverage, divide the edited characters by eligible ASCII letters.
- `predictions/{original,no_diacritics,qwerty_typo}.npz` saves continuous 28-label scores, binary predictions, label vectors and sample IDs. The prediction files contain no source or perturbed text.
- `figures/macro_f1_delta.{png,pdf}` gives the all28 delta plot in grayscale. `manifest.json` links the results to RQ1 and records runtime details, input and output checksums.

The main result is the test all28 Macro-F1 difference for each perturbation under fixed 0.5. Use secondary decoder and tier tables to describe where the drops differ. Report observed differences for this checkpoint, seed and split.
