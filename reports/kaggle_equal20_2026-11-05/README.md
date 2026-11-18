# Baseline Midterm dùng khi ôn Final

Nguồn là hai Kaggle study 27/28, notebook08/09, seed42; neural đều đủ20 epoch. Tất cả model metrics ở đây là validation tại0.5, không phải test Final. [Report Midterm](../midterm_report.pdf) và [bộ ôn](../../docs/oral_defence/02_protocol_results_vi.md) giải thích ý nghĩa.

- [Model comparison](model_comparison.csv): sáu model, scope, epoch, tham số và thời gian.
- [Cùng27 emotions](paired_emotion27_delta.csv), [Neutral riêng](neutral_metrics.csv), [per-label](per_label_metrics.csv).
- [Q1](label_extremes.csv), [Q2](frequency_f1.csv), [Q3](pair_preservation.csv), [Q4](neural_style_comparison.csv), [Q5](threshold_sensitivity.csv).
- [Evidence](evidence/README.md), [provenance](provenance.json), [verification](verification.json), [figures](figures/).

Phần analysis chi tiết lặp lại và slide snapshot được bỏ trên nhánh ôn. Full source/notebooks và builders ở nhánh `feature/final28-report`, commit `289f2fb`. Giữ CSV/NPZ/run metadata gốc để không trộn với các run ít epoch.
