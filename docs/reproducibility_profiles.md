# Profile kết quả chính và lệnh tái lập

Kaggle 08/09 executed và hai ZIP equal20 là source of truth của report/slide/cẩm nang hiện tại.

| Thành phần | 27 nhãn | 28 nhãn |
| --- | --- | --- |
| Profile base | configs/kaggle_equal_20ep.json | Cùng profile; init thêm output Neutral |
| BiGRU chạy / best | 20 / 17 | 20 / 20 |
| PhoBERT chạy / best | 20 / 14 | 20 / 15 |
| Checkpoint selection | emotion27 Macro-F1 val tại 0,5; hòa lấy sớm | Cùng tiêu chí; history dùng selection_macro_f1 |
| TF-IDF/LR | liblinear, C=1, max_iter=1000, seed 42 | Cùng cấu hình, thêm classifier Neutral |

TF-IDF không có epoch neural tương đương. Cùng epoch không có nghĩa cùng compute; batch GRU/PhoBERT là 64/16. Cả bốn lượt neural đã chạy đủ 20 epoch; best checkpoint ở giữa không làm giảm số epoch thực chạy.

Hai manifest cùng Tesla T4, Python 3.13.16, Torch 2.11.0+cu130, Transformers 4.57.3, sklearn 1.6.1. Code chỉ dùng GPU thứ nhất. CSV/source/profile raw hash ở bản 28 có biến thể CRLF; importer xác minh checksum từng file là LF/CRLF của cùng nội dung local, config sau parse và fixed IDs khớp. Giữ nguyên manifest, không sửa hash để vượt kiểm tra. Một seed chưa đo độ biến thiên.

## Kiểm chứng và build báo cáo

```bash
.venv/bin/python scripts/import_equal20_results.py
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/build_study_guide.py
```

Import đọc hai ZIP private tại root, hai executed notebook ở notebooks/ và CSV được cấp quyền. Không nạp model, train, inference lại hoặc chấm test. Tính lại sáu metric từ NPZ, kiểm tra IDs/targets/mapping/probabilities, config/history/best epoch và checkpoint member. Evidence an toàn nằm tại reports/kaggle_equal20_2026-11-05/; provenance lưu SHA-256 nguồn và file đã trích.

Build report mặc định chỉ dùng evidence equal20; thiếu evidence sẽ dừng. Không tự fallback về model scores trong reports/tables hoặc run JSON cũ. Dữ liệu EDA và audit split vẫn dùng cùng CSV/fixed IDs.

## Chạy mới trên Kaggle

Xem [giao thức Kaggle](kaggle_equal_epochs.md). Tạo ZIP code bằng scripts/package_kaggle_equal_epochs.py; notebook 08 chạy 27 nhãn, 09 chạy 28 nhãn. Cùng profile/source/dataset, workspace và ZIP kết quả riêng. Không đổi config/runtime trong study đã init. Các file training cũ có default profile lịch sử khi chạy trực tiếp không tham số; workflow chính truyền profile equal20 rõ ràng, không dựa vào các default đó.

## Các profile trước đây

configs/baseline27_4ep.json, configs/neutral28_12ep_base.json và configs/kaggle_baseline_25_20ep.json cùng artifact cũ chỉ là lịch sử. Không dùng chúng làm epoch/điểm/best checkpoint cho báo cáo chính hiện tại. Không chỉnh metadata cũ để biến thành lượt 20 epoch.
