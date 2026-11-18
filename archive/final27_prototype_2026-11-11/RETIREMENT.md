# Retired Final 27-output prototype

Preserved on 11/11/2026 when the team chose 28 classes including Neutral. This snapshot contains 17 source/config/documentation/status files, with original relative paths and SHA-256 in [snapshot.json](snapshot.json).

- [Original 27-output protocol](docs/final_protocol.md), [config](configs/final_thresholds.json), [pipeline entry](scripts/final_thresholds.py).
- [Original implementation status](reports/final/implementation_verification.json): synthetic workflow tests passed, but actual validation inference, fitted thresholds and held-out test were pending.
- The snapshot is a historical source record, not a standalone installation. Markdown and code paths retain their original workspace meaning; restore mapped files into an independent complete checkout if reproducing the prototype. Do not mix its epoch-14 weights/27-column NPZ/gates with v2.
- Active work uses [28-output protocol v2](../../docs/final_protocol.md), epoch 15, `data/final28/` and `reports/final28/`.

Original Kaggle 27/28 evidence, fixed split IDs and private archives remain unchanged. No raw data, weights or vocabulary were copied into this snapshot.
