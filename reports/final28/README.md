# Final 28-class study — current status

Active design: [protocol v2](../../docs/final_protocol.md), [execution](../../docs/final_execution.md), [config](../../configs/final_thresholds.json).

- Primary: frozen PhoBERT **28 outputs**, including Neutral, epoch **15** from the equal20 run.
- Threshold objective: validation Macro-F1 over **all28**; 28 per-label cutoffs. Additional views: emotion27 projection without re-tuning, and separate Neutral positive-class metrics.
- Source validation NPZ **2,066 × 28**, fixed IDs/targets, original checkpoint selection and train-only emotion tiers verified.
- Saved model/config/tokenizer and VnCoreNLP restored privately under `data/final28/`, with matching hashes.
- **Actual validation model inference remains pending** because local neural dependencies are absent. No actual fitted Final thresholds, pre-test lock or test-model scores exist yet.
- Human error audit, plots, paper and oral defence remain later tasks. Synthetic tests are implementation checks, not actual-model evidence.

The retired 27-output prototype is [archived](../../archive/final27_prototype_2026-11-11/RETIREMENT.md). Its validation reproduction/gates are not reused for this study. Current implementation checks are recorded in `implementation_verification.json` once verification completes.
