# Lịch sử ViGoEmotions

Hướng Final nằm ở [protocol](../docs/final_protocol.md) và [checklist](../tasks/todo.md). Baseline đang dùng nằm ở [báo cáo midterm](../reports/midterm_report.md) và [Kaggle 20 epoch](../reports/kaggle_equal20_2026-11-05/README.md). Thư mục này gom workflow đã được thay thế và tài liệu chuẩn bị trước đây.

## Dọn thư mục sau midterm — 11/11/2026

Đã chuyển 13 file, không xóa nội dung nghiên cứu. [Manifest di chuyển](cleanup_2026-11-11.json) ghi đường dẫn cũ/mới, hash trước và sau để đối chiếu hoặc khôi phục.

| Nhóm | Vị trí lưu trữ | Lý do |
| --- | --- | --- |
| Notebook 03 baseline comparison | [03](notebooks/03_baseline_comparison.ipynb) | Workflow baseline cũ, kết quả chính đã thay bằng 08/09 |
| Notebook 04 neural baselines | [Colab](notebooks/04_colab_neural_baselines.ipynb), [Kaggle](notebooks/04_kaggle_neural_baselines.ipynb) | Profile ít epoch/25–20 lịch sử |
| Notebook 05 learning-rate sweep | [Colab](notebooks/05_colab_phobert_tuning.ipynb), [Kaggle](notebooks/05_kaggle_phobert_tuning.ipynb) | Thiết kế prepared, chưa có kết quả; ngoài RQ1 hiện tại |
| Notebook 06 Neutral | [Colab](notebooks/06_colab_neutral_28_labels.ipynb), [Kaggle](notebooks/06_kaggle_neutral_28_labels.ipynb) | Workflow 28 nhãn cũ, đã thay bằng notebook 09 |
| Profile Kaggle 25/20 | [Config](configs/kaggle27_25_20.json) | Chỉ thuộc notebook 04 cũ |
| Worklog/readiness midterm | [Worklog](docs/midterm_worklog.md), [readiness](docs/midterm_readiness_2026-11-05.md) | Nhật ký và checklist của mốc đã hoàn tất |
| Lời dẫn midterm tiếng Việt | [Lời dẫn](docs/midterm_presentation_2026-11-05.md) | Tài liệu trình bày mốc midterm |
| Hướng dẫn workflow phụ | [Kaggle](docs/kaggle_guide.md), [Neutral](docs/neutral_28_labels.md) | Giữ để tra cứu, hướng dẫn hiện hành là equal20/protocol Final |

Notebook và config được chuyển nguyên byte. Các link Markdown được sửa theo vị trí mới; số liệu, code cell, outputs và manifest chạy cũ không được đổi. Notebook 08/09, evidence equal20, báo cáo/slide midterm mới nhất và cẩm nang ôn mô hình vẫn nằm ở vị trí đang được dùng. Các thay đổi đang có trong working tree được giữ nguyên.

Generator `scripts/prepare_kaggle.py` đọc/ghi notebook lịch sử trong `archive/notebooks`; hai packager Colab/Neutral vẫn kèm các nguồn archive tương ứng. Để khôi phục vị trí cũ, dùng mapping trong manifest và kiểm tra file đích trước khi chuyển; các Markdown links hiện trỏ theo cấu trúc archive.

## Các bản đã chuyển vào đây

- [README pre-midterm](README_PRE_MIDTERM.md).
- [Kiểm chứng Colab cũ](docs/colab_results_check.md), [audit project cũ](docs/project_audit.md), [thiết kế tuning trước đây](docs/phobert_tuning.md).
- [Báo cáo tiến độ 23/10](reports/pre_midterm_progress_2026-10-23.md) / [PDF](reports/pre_midterm_progress_2026-10-23.pdf).
- [Slide pre-midterm 23/10](reports/pre_midterm_slides_2026-10-23.pdf), [speaker notes](reports/pre_midterm_speaker_notes_2026-10-23.md), [evidence và script](reports/pre_midterm_2026-10-23/).
- [Study Neutral 04/11](reports/midterm_2026-11-04/evidence/README.md), [phân tích](reports/midterm_2026-11-04/neutral28_results.md).
- [Lượt chạy notebook local 05/11](reports/local_notebook_run_2026-11-05/README.md), [kiểm tra kỹ thuật giai đoạn trước](reports/midterm_readiness_verification_2026-11-05.json).

## Quy ước lưu trữ

JSON/CSV/NPZ lịch sử giữ nguyên nội dung. Đường dẫn trong manifest, log, PDF và prose ghi lại vị trí tại thời điểm chạy; không sửa những bản ghi đó thành một lượt chạy mới. Các link Markdown được cập nhật theo vị trí lưu trữ mới. Trọng số và ZIP private cũ nằm trong backups/legacy_runs/ (Git ignore); hai ZIP equal20 hiện tại vẫn ở project root theo yêu cầu của importer.

Config baseline 4 epoch/Neutral 12 epoch, run JSON gốc, bảng/predictions baseline cũ, ablation và plan tuning vẫn được script/test lịch sử dùng nên giữ nguyên. Chuyển các dependency đó cần thay đổi riêng để bảo toàn workflow; chúng không phải nguồn kết quả Final. EDA trong reports/tables và reports/figures cùng audit split 07 vẫn phục vụ báo cáo hiện tại.
