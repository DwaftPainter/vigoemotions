# Chuẩn bị thuyết trình midterm — chốt nội dung 05/11/2026

> **Phạm vi lịch sử:** Các run/điểm trước lượt Kaggle 20 epoch không còn là kết quả chính. Dùng [README hiện tại](../README.md) và report/slide equal20.

Bộ hiện tại đã cập nhật kết quả dự đoán thêm Neutral từ notebook 06 và ZIP riêng tư. Dùng tài liệu sau cho buổi midterm; các bộ có ngày 23/10 trong `reports/` là snapshot lịch sử.

Đã hoàn tất audit split, tách config/lệnh tái lập, làm sạch notebook/gói chia sẻ, phụ lục per-label và đọc định tính chín ca PhoBERT. Xem [readiness](../docs/midterm_readiness_2026-11-05.md). Còn thiếu thông tin nhóm; cần xác nhận yêu cầu official split và các nhận xét hỗ trợ AI.

## Mở sẵn

1. [20 slide PDF](../reports/midterm_slides.pdf) / [nội dung và ghi chú](../reports/midterm_slides.md).
2. [Lời dẫn từng slide và Q&A](../docs/midterm_presentation_en.md), khoảng 12–15 phút tùy thời lượng được cấp.
3. [Báo cáo PDF](../reports/midterm_report.pdf) / [Markdown](../reports/midterm_report.md), kết quả mới ở mục 5.7.
4. [Bảng metric 28 nhãn](reports/midterm_2026-11-04/evidence/metrics_summary.csv), [Neutral riêng](reports/midterm_2026-11-04/evidence/neutral_analysis.csv), [nguồn và kiểm chứng](reports/midterm_2026-11-04/evidence/README.md).

## Các số cần nhớ

Tất cả số mới là **validation, 2.066 mẫu, seed 42, ngưỡng 0,5**; test chưa được chấm.

| Mô hình | Macro-F1 đủ 28 nhãn | Macro-F1 chiếu về 27 cảm xúc | Neutral F1 |
| --- | --- | --- | --- |
| TF-IDF + OvR LR | 0,231040 | 0,238825 | 0,020833 |
| BiGRU | 0,399227 | 0,410764 | 0,087719 |
| PhoBERT | 0,574653 | 0,579640 | 0,440000 |

Neutral support = **94**: 75 Neutral-only + 19 mixed. PhoBERT có **44 TP / 62 FP / 50 FN**, precision 0,4151, recall 0,4681. BiGRU chọn epoch **10/10**; PhoBERT chọn **11/12** trên Tesla T4.

## Câu chuyện trình bày

| Slide | Ý chính | Nguồn |
| --- | --- | --- |
| 1–6 | Bài toán đa nhãn; audit dữ liệu; EDA 27 cảm xúc; giữ split, chống leakage | [EDA và bảng](../reports/tables/README.md), [báo cáo](../reports/midterm_report.md) |
| 7–12 | Ba baseline gốc 27 nhãn; Q1–Q5, ngưỡng, phong cách, lỗi biên | [Bảng baseline gốc](../reports/tables/model_comparison.csv) |
| 13 | Thêm Neutral thành output riêng; giữ annotation mixed | [Thiết kế notebook 06](../docs/neutral_28_labels.md) |
| 14–15 | Kết quả đủ 28 nhãn và chất lượng Neutral | [Evidence mới](reports/midterm_2026-11-04/evidence/README.md) |
| 16–17 | Phép chiếu 27; khác số epoch; đường học; giới hạn thực thi | [Phân tích kết quả mới](reports/midterm_2026-11-04/neutral28_results.md) |
| 18–20 | RQ Final, việc còn lại và nguồn | [Lời dẫn/Q&A](../docs/midterm_presentation_en.md) |

**Ba lượt kết quả cần phân biệt:** baseline gốc 14/10 có PhoBERT 27 nhãn **0,439320**; lượt chạy lại dùng cho lời thoại 23/10 có **0,467850**, lưu [evidence riêng](reports/pre_midterm_2026-10-23/evidence/model_comparison.csv); study 04/11 có **0,574653 trên 28** và **0,579640 khi chiếu 27**. Study 06 đối chiếu baseline gốc 0,439320. Không trộn các lượt hoặc trừ trực tiếp điểm trên hai tập nhãn khác nhau.

## Những giới hạn cần nói đúng

- PhoBERT mới chạy 12 epoch, baseline gốc 4; quy tắc chọn checkpoint và runtime cũng khác. Chưa thể nói thêm Neutral gây mức tăng +14,0321 điểm phần trăm trên phép chiếu 27.
- Một seed/split, validation đã dùng chọn checkpoint; chưa có kết quả test hay kiểm định ý nghĩa thống kê.
- Lượt nguồn từng lỗi JVM ở cell tìm ví dụ; mã hiện tại đã sửa để đọc NPZ, đã kiểm chứng offline. Không tuyên bố chạy lại toàn bộ training GPU.
- Study 06 là một lượt lr=2e-5; sweep learning rate notebook 05 chưa có kết quả.
- Các phân tích Q1–Q5/style ở slide 8–12 thuộc baseline gốc, chưa phải phân tích style cho study mới.

## Trước khi nộp

- Điền tên, MSSV, phân công nhóm; luyện thời lượng với bộ 20 slide.
- Xác nhận chín ca đọc định tính PhoBERT hỗ trợ AI và việc chấp nhận official split. Notebook 07 đã có bằng chứng phân bố, không chứng nhận thuật toán chia gốc.
- ZIP checkpoint giữ riêng tư. Notebook bản chia sẻ đã xóa output; packager cũng làm sạch tự động. Không kèm lịch sử `.git/` cũ hoặc backup private; xem hướng dẫn tái lập.
- Final: khóa checkpoint và lựa chọn decoder, tune ngưỡng trên validation, chỉ chấm test sau khi chốt. Nếu cần kết luận tác động Neutral, chạy đối chứng 27/28 cùng điều kiện.

Sau khi sửa nguồn nội dung, dựng lại báo cáo/slide bằng `.venv/bin/python scripts/build_report.py`. Phần bổ sung có nguồn Markdown/JSON ở `reports/midterm_2026-11-04/`; script không cần tải checkpoint hoặc train lại. [README chính](../README.md) hướng dẫn chạy project.
