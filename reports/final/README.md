# Final RQ1 — implementation status

**Retired 27-output prototype.** The team has switched to [28-output protocol v2](../../docs/final_protocol.md); current outputs use `reports/final28/`. This directory records the earlier 27-output implementation status only; the original source/status snapshot is [archived](../../archive/final27_prototype_2026-11-11/RETIREMENT.md).

Pipeline and instructions: [final_execution.md](../../docs/final_execution.md). Design: [protocol](../../docs/final_protocol.md). Local checks are recorded in [implementation_verification.json](implementation_verification.json).

- Source validation NPZ (2,066 × 27), original run, split IDs and train-only tiers verified.
- Frozen epoch-14 checkpoint and original VnCoreNLP resources restored privately with matching hashes.
- 23 new synthetic tests pass, including fit/reproduction/lock ordering, tamper rejection, decoder consistency, test-once and technical-resume behavior.
- Full suite: 126/128 pass. Two pre-existing neural tests error because local `torch` is unavailable.
- **Actual PhoBERT validation inference remains pending.** Local `reproduce-val` correctly stops with a dependency error; no verified reproduction, fitted Final thresholds or pre-test lock is claimed.
- **No actual test inference or test-model scoring was performed.**

Use notebook 10 in the original-compatible Kaggle environment to generate actual validation reproduction, fitted thresholds and the lock. The notebook ends before held-out test inference. Human error audit, paper, plots and oral defence remain later tasks in [the checklist](../../tasks/todo.md).
