# Lịch sử ViGoEmotions

Kết quả đang dùng nằm ở [README chính](../README.md), [báo cáo hiện tại](../reports/midterm_report.md) và [Kaggle 20 epoch](../reports/kaggle_equal20_2026-11-05/README.md). Thư mục này gom snapshot trước lượt 20 epoch để không trộn vào nội dung thuyết trình.

## Các bản đã chuyển vào đây

- [README pre-midterm](README_PRE_MIDTERM.md).
- [Kiểm chứng Colab cũ](docs/colab_results_check.md), [audit project cũ](docs/project_audit.md), [thiết kế tuning trước đây](docs/phobert_tuning.md).
- [Báo cáo tiến độ 23/10](reports/pre_midterm_progress_2026-10-23.md) / [PDF](reports/pre_midterm_progress_2026-10-23.pdf).
- [Slide pre-midterm 23/10](reports/pre_midterm_slides_2026-10-23.pdf), [speaker notes](reports/pre_midterm_speaker_notes_2026-10-23.md), [evidence và script](reports/pre_midterm_2026-10-23/).
- [Study Neutral 04/11](reports/midterm_2026-11-04/evidence/README.md), [phân tích](reports/midterm_2026-11-04/neutral28_results.md).
- [Lượt chạy notebook local 05/11](reports/local_notebook_run_2026-11-05/README.md), [kiểm tra kỹ thuật giai đoạn trước](reports/midterm_readiness_verification_2026-11-05.json).

## Quy ước lưu trữ

JSON/CSV/NPZ lịch sử giữ nguyên nội dung. Đường dẫn trong manifest, log, PDF và prose ghi lại vị trí tại thời điểm chạy; không sửa những bản ghi đó thành một lượt chạy mới. Các link Markdown được cập nhật theo vị trí lưu trữ mới. Trọng số và ZIP private cũ nằm trong backups/legacy_runs/ (Git ignore); hai ZIP equal20 hiện tại vẫn ở project root theo yêu cầu của importer.

Config, run JSON gốc và notebook 03–06 còn được script/test lịch sử dùng nên giữ nguyên. EDA trong reports/tables và reports/figures cùng audit split 07 vẫn phục vụ báo cáo hiện tại. Không xóa những thư mục này như dữ liệu thừa.
