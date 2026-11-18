# Final 28-class study — current status

Active design: [protocol v2](../../docs/final_protocol.md), [execution](../../docs/final_execution.md), [config](../../configs/final_thresholds.json).

- Primary: frozen PhoBERT **28 outputs**, including Neutral, epoch **15** from the equal20 run.
- Threshold objective: validation Macro-F1 over **all28**; 28 per-label cutoffs. Additional views: emotion27 projection without re-tuning, and separate Neutral positive-class metrics.
- Source validation NPZ **2,066 × 28**, fixed IDs/targets, original checkpoint selection and train-only emotion tiers verified.
- Saved model/config/tokenizer and VnCoreNLP restored privately under `data/final28/`, with matching hashes.
- **Actual Kaggle validation reproduction and held-out test are complete.** Validation probabilities reproduce with maximum absolute error **0**. All 21 artifacts from `vigoemotions_final28_results.zip` were imported unchanged and verified against the frozen sources, split IDs/targets and current pre-test lock.
- Test all28 Macro-F1: fixed **0.589964**, global **0.591254** (threshold **0.4**), per-label **0.585415**. Report all three methods; test is not used to tune thresholds or change the experiment.
- [Complete English paper, PDF](../final_report.pdf), [Markdown](../final_report.md) and [paper verification](../final_report_verification.json) are available, alongside [Vietnamese working analysis](results_discussion.md), [per-label deltas](analysis/label_deltas.csv), [figures](figures/) and [execution verification](execution_verification.json). All 11 validation/test/gap tables were recomputed during review.
- The user accepted the AI-assisted audit for this report on 17/11. The paper discloses the absence of independent human adjudication. Author/group details, submission packaging and oral defence remain to be completed.
- [Human audit packet and instructions](human_audit/README.md) are prepared: 20 core FN, 20 core FP and 22 supplementary/context cases, 62 unique comments. The editable worksheet with source text is private at `data/final28/human_audit/review_private.csv`; all reviewer/observation fields start blank and completion is still zero.
- [AI-assisted qualitative review](human_audit/ai_assisted_review.md) is complete for all 62 comments at the user's request on 17/11. Its observations, hypotheses and source provenance are separate from the unchanged human worksheet; human confirmation remains pending.

## Reproduce the analysis

```bash
.venv/bin/python scripts/build_final_figures.py
.venv/bin/python scripts/build_final_paper.py
```

This uses only hash-verified, text-free imported artifacts; no GPU or private CSVs are needed. It writes four PNG/PDF figures, `analysis/label_deltas.csv` and figure provenance, preserving the imported NPZ/JSON/CSV files. With authorized private train/val CSVs, `scripts/final_thresholds.py report` can regenerate the metric tables without neural inference. Keep completed-test artifacts and do not rerun `infer-test`.

The paper builder uses the [English template](../../docs/final_paper_template.md), [author metadata](../../docs/final_paper_metadata.json) and [BibTeX](../../docs/final_references.bib). It verifies input hashes and AI/figure provenance, generates all ten tables from derived evidence, and exports the 11-page Markdown/PDF report. Missing identity fields are recorded in verification; no names or student IDs are inferred. Update metadata and rebuild when those details are available.

The frozen config/protocol contain pre-execution status wording; keep their bytes unchanged to preserve the lock. Current execution state is in the artifacts and `execution_verification.json`. The prior validation ZIP uses a different NPZ member order and dependent lock hash; all constituent validation arrays, thresholds and reproduction evidence are unchanged. The test binds the current lock.

Local validation: the seven numeric Vietnamese analysis tables match source CSVs, and the paper's ten tables are generated from verified evidence. Imported artifacts and 141 protected existing files are unchanged. The four reporting-evidence, 14 human-audit and six paper tests pass. The latest full suite has **156 passed / 158 total**, with two pre-existing neural tests erroring because local `torch` is absent; no tests were skipped or weakened. See [paper verification](../final_report_verification.json), the [reporting-stage snapshot](analysis/local_verification.json) and [audit-preparation snapshot](human_audit/local_verification.json). Actual model inference was verified on Kaggle, not repeated locally.

The retired 27-output prototype is [archived](../../archive/final27_prototype_2026-11-11/RETIREMENT.md). Its validation reproduction/gates are not reused for this study. `implementation_verification.json` is the historical pre-Kaggle implementation snapshot; its pending-execution fields describe that earlier stage, not the current status.
