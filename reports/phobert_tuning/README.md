# PhoBERT learning-rate sweep — trạng thái prepared

[← Mục lục reports](../README.md) · [Hướng dẫn chạy Colab](../../docs/phobert_tuning.md)

**Snapshot 22/10/2026: chưa chạy sweep GPU.** File cấu hình không phải bằng chứng đã có kết quả. Trạng thái thực tế được ghi trong [study.json](study.json).

## Các file hiện có

| File | Ý nghĩa |
| --- | --- |
| [study.json](study.json) | Thiết kế thí nghiệm, grid, đối chứng, quy tắc chọn cấu hình, `status=prepared`, `test_evaluated=false` |
| [configs/lr_1e-5.json](configs/lr_1e-5.json) | Cấu hình lượt learning rate 1e-5 |
| [configs/lr_2e-5.json](configs/lr_2e-5.json) | Cấu hình đối chứng 2e-5, chạy lại trong cùng runtime |
| [configs/lr_3e-5.json](configs/lr_3e-5.json) | Cấu hình lượt learning rate 3e-5 |

Ba cấu hình chỉ khác learning rate PhoBERT. Các mục TF-IDF/GRU nằm trong bản chụp config chung để truy vết, không có nghĩa sweep chạy lại cả hai mô hình đó. Mỗi lượt PhoBERT bắt đầu từ pretrained revision gốc, không nối tiếp trọng số lượt trước.

## Các file sẽ có sau khi chạy thành công

Các đường dẫn dưới là đầu ra dự kiến, **chưa phải file hiện hữu**:

| Đường dẫn | Ý nghĩa |
| --- | --- |
| `lr_1e-5/`, `lr_2e-5/`, `lr_3e-5/` | Mỗi thư mục có `phobert_run.json`, `phobert_analysis.json`, `tables/`, `figures/`, `predictions/`; quy ước file giống [baseline](../tables/README.md) |
| `comparison.csv` | Metric tại 0,5, epoch tốt nhất, thời gian và delta Macro-F1 so với đối chứng cùng sweep |
| `per_label_comparison.csv` | Chênh lệch F1 từng nhãn giữa mỗi lượt và đối chứng |
| `styles_comparison.csv` | Chênh lệch Macro-F1 theo phong cách giữa mỗi lượt và đối chứng |
| `learning_curves.png` | Train BCE loss và validation Macro-F1 qua epoch cho ba learning rate |
| `neural_error_review.csv` | 20 FN/20 FP ứng viên của lượt được chọn; cột `reviewer`, `observation`, `hypothesis` chờ nhóm tự điền |
| `experiment_summary.md` | Tóm tắt số liệu tự sinh; không thay thế giải thích/đọc lỗi thủ công |
| `study.json` được cập nhật | `status=completed`, `selected_run` và SHA-256 các NPZ |

Trong mỗi lượt: chọn epoch có Macro-F1 validation cao nhất tại 0,5, hòa chọn epoch sớm. Giữa ba lượt: chọn Macro-F1 cao nhất, hòa chọn learning rate nhỏ hơn. Không dùng điểm sau threshold sweep hoặc test để chọn learning rate.

Checkpoint được lưu riêng ở `data/checkpoints/phobert_tuning/<run>/best.pt`; thư mục reports không mang trọng số. `neural_error_review.csv` đã có sẽ được giữ khi chạy summarize; các bảng/summary/plot tự sinh có thể bị tạo lại. Khi build report với sweep completed, script cũng gọi summarize để kiểm tra và cập nhật các đầu ra đó.
