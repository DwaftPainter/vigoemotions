# Kết quả chính — Kaggle 20 epoch, 05/11/2026

- `model_comparison.csv`: sáu model, metric đủ tập nhãn, epoch, tham số, thời gian.
- `emotion27_comparison.csv`, `paired_emotion27_delta.csv`: đối chiếu cùng 27 cảm xúc.
- `per_label_metrics.csv`, `neutral_metrics.csv`: mọi nhãn và Neutral riêng.
- `label_extremes.csv`, `frequency_f1.csv`, `pair_preservation.csv`, `neural_style_comparison.csv`, `threshold_sensitivity.csv`: Q1–Q5 tính lại trên study 27 mới.
- `analysis/27_emotion27/` và `analysis/28_emotion27/`: evaluator chung trên 27 cột, đúng nguồn NPZ từng study; không ghi đè bảng lịch sử.
- `phobert28_error_review.csv`: sáu ca mới, AI hỗ trợ đọc; nhóm cần xác nhận.
- `evidence/`, `provenance.json`: dữ liệu dự đoán ID-only và kiểm chứng nguồn.
- Báo cáo chính ở `../midterm_report.md`; `figures/`, `slides.json` giữ hình và dữ liệu slide, không tạo thêm bản báo cáo trùng.

Build: `python scripts/import_equal20_results.py`, sau đó `python scripts/build_report.py`. Không train, không chấm test và không dùng metric các lượt ít epoch. Hai ZIP riêng tư gốc không được commit.
