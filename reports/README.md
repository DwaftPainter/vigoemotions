# Đọc kết quả ViGoEmotions

Đây là nơi lưu **kết quả đo, bằng chứng thực nghiệm và bản trình bày**. Bắt đầu từ trang này thay vì mở lần lượt mọi CSV.

Buổi **pre-midterm 23/10/2026**: [báo cáo tiến độ 6 trang](pre_midterm_progress_2026-10-23.pdf), [slide trình bày 12 trang](pre_midterm_slides_2026-10-23.pdf), [lời dẫn từng slide](pre_midterm_speaker_notes_2026-10-23.md) và [nguồn báo cáo sửa được](pre_midterm_progress_2026-10-23.md). Snapshot ngày 22/10 phân biệt kết quả baseline lịch sử với lượt Colab mới đang chạy; mỗi phần có đường dẫn minh chứng. Các tài liệu này không tự cập nhật sau khi có kết quả mới.

## Nên đọc theo thứ tự nào?

1. [Bảng so sánh ba baseline](tables/model_comparison.csv): mô hình nào tốt hơn trên validation, số tham số và thời gian.
2. [Báo cáo PDF](midterm_report.pdf) hoặc [Markdown](midterm_report.md): phương pháp, diễn giải, giới hạn và hướng Final.
3. [Slide PDF](midterm_slides.pdf) hoặc [nội dung/lời dẫn](midterm_slides.md): bản trình bày 18 trang, giới hạn 20 trang.
4. Chọn một câu hỏi trong bảng dưới rồi mở README của nhóm file tương ứng.

| Muốn biết gì? | Đọc ở đâu? |
| --- | --- |
| Dữ liệu có đặc điểm gì? | [Các bảng EDA](tables/README.md), [các hình EDA](figures/README.md) |
| Nhãn nào khó, mô hình bỏ sót ở đâu? | [Bảng theo từng mô hình](tables/README.md), mục per-label và error cases |
| Ngưỡng 0,5 ảnh hưởng thế nào? | [Bảng threshold sweep](tables/README.md), [hình diagnostics](figures/README.md) |
| Làm sao tính lại metric từ đầu ra? | [Xác suất đã lưu và cách đọc NPZ](predictions/README.md) |
| Vì sao bỏ/giữ mixed-neutral? | [Ablation A/B](ablation_mixed_neutral/README.md) |
| Đã tối ưu learning rate chưa? | [Trạng thái tuning PhoBERT](phobert_tuning/README.md) |
| Lượt chạy có tái lập được không? | Các file run JSON bên dưới và [kết quả audit](../docs/project_audit.md) |

## Trạng thái hiện tại

| Nhóm | Trạng thái | Ý nghĩa |
| --- | --- | --- |
| EDA và ba baseline | Đã có kết quả, nhập/kiểm chứng Colab ngày 14/10/2026 | Các bảng/hình/NPZ hiện hữu là kết quả thật; audit 22/10 kiểm tra lại từ xác suất |
| Ablation mixed-neutral | `completed` trong [experiment_config.json](ablation_mixed_neutral/experiment_config.json) | Có hai điều kiện A/B, xác suất và bảng so sánh riêng |
| Sweep learning rate | `prepared` trong [study.json](phobert_tuning/study.json) | Chỉ có thiết kế và ba cấu hình; **chưa có điểm sweep** |
| Logs/history/learning curves bổ sung từ notebook 04 | Code đã có, chưa sinh ở snapshot này | Chỉ xuất hiện sau khi chạy lại notebook 04 mới |
| Audit lỗi thủ công | 10 bình luận TF-IDF của lượt baseline 14/10 đã được diễn giải | CSV 20 FN/20 FP tự sinh của neural chưa chứng minh đã được người đọc |
| Test / thí nghiệm Final | Chưa có dự đoán hay metric test | Không gọi kết quả validation hoặc threshold sweep là kết quả Final |

Các trạng thái trên là snapshot được audit ngày **22/10/2026**. Sau lượt chạy mới, xem run JSON, `study.json` và CSV thực tế; trang README không tự cập nhật trạng thái.

## Cấu trúc thư mục và quy ước tên

```text
reports/
  README.md                   mục lục và quy tắc đọc
  midterm_report.md/.pdf      báo cáo chi tiết, tự sinh
  midterm_slides.md/.pdf      nội dung slide/lời dẫn và bản PDF, tự sinh
  eda_environment.json        môi trường và checksum dữ liệu khi làm EDA
  <model>_run.json            cấu hình, môi trường, thời gian, lịch sử neural
  <model>_analysis.json       tóm tắt thống kê từ evaluator
  tables/                    số liệu dạng CSV
  figures/                   biểu đồ PNG từ code
  predictions/               xác suất validation, không có text
  ablation_mixed_neutral/     một thí nghiệm A/B đã hoàn tất
  phobert_tuning/             thiết kế sweep; kết quả chỉ có sau khi chạy
```

`<model>` là `tfidf_lr` (TF-IDF + OvR Logistic Regression), `gru` (BiGRU) hoặc `phobert` (PhoBERT). `_val` nghĩa là validation; `train_` chỉ thống kê quan hệ nhãn từ train. Thư mục thí nghiệm giữ riêng config/ID/xác suất để tránh trộn với baseline chính.

## Các JSON ở thư mục gốc

| File | Nội dung và cách dùng |
| --- | --- |
| [eda_environment.json](eda_environment.json) | Python, nền tảng, package versions, SHA-256 ba CSV nguồn. Dùng xác nhận EDA và mô hình dựa trên cùng dữ liệu. |
| [tfidf_lr_run.json](tfidf_lr_run.json) | Cấu hình TF-IDF/LR, số feature, tham số, thời gian, môi trường và định nghĩa phạm vi đo. Không có epoch hoặc checkpoint neural. |
| [gru_run.json](gru_run.json) | Cấu hình, GPU, history từng epoch, `best_epoch`, tokenization/truncation, tham số và thời gian BiGRU. |
| [phobert_run.json](phobert_run.json) | Tương tự BiGRU, kèm tên/revision pretrained PhoBERT và tiền xử lý. |
| [tfidf_lr_analysis.json](tfidf_lr_analysis.json), [gru_analysis.json](gru_analysis.json), [phobert_analysis.json](phobert_analysis.json) | Ngưỡng chung tốt nhất trên lưới validation, Macro-F1 tương ứng, Spearman tần suất–F1 và p-value. Là phân tích thăm dò; không thay thế `*_metrics.csv` tại ngưỡng 0,5. |

Run JSON lịch sử chưa có `checkpoint_path`; phiên bản script mới bổ sung trường này. JSON mô tả lượt chạy, **không chứa trọng số** và không chứng minh checkpoint vẫn còn trên máy.

## Đọc metric cho đúng

- Macro/Micro/per-label F1 và LRAP: càng cao càng tốt. Hamming Loss: càng thấp càng tốt. Điểm dùng thang 0–1.
- `delta = 0,01` tương ứng **1 điểm phần trăm**, không phải tăng tương đối 1%.
- Bảng baseline chính dùng threshold **0,5**. Điểm tốt nhất sau sweep đã được chọn trên validation, nên không so như một kết quả test độc lập.
- Cardinality 0 là neutral-only theo quy ước 27 đầu ra. F1 của nhóm này bằng 0 theo `zero_division=0`; LRAP có thể bằng 1 theo quy ước thư viện. Đọc cùng Hamming Loss.
- So thời gian phải đọc `time_definition` và phần cứng trong run JSON. CPU/GPU và phạm vi đo khác nhau không cho phép kết luận tốc độ tuyệt đối chỉ từ bảng chung.

## File nào được sửa tay?

Bảng, hình, NPZ, run/analysis JSON, report và slide là đầu ra của code: sửa số liệu trực tiếp sẽ làm mất khả năng tái lập. Để thay nội dung report/slide, chỉnh `scripts/build_report.py`, rồi build lại. Các nhận xét thủ công cần đối chiếu với ca lỗi của đúng lượt chạy; build report không tự thực hiện audit con người.

Các README này là hướng dẫn viết tay. Sau khi sweep chạy, `neural_error_review.csv` có các cột để nhóm ghi nhận xét thủ công; đó là phần cần tự đọc và điền, không phải metric tự sinh.

Chạy lại EDA/baseline sẽ ghi đè các file cùng tên. Sao lưu **cả nhóm** run JSON, tables, figures, predictions trước khi nhập kết quả mới. Không trộn CSV lượt mới với NPZ lượt cũ. Checkpoint/vocabulary nằm trong `data/`, tải bằng ZIP riêng tư; không đặt vào `reports/` vì gói `--include-results` sẽ mang toàn bộ thư mục này sang Colab.
