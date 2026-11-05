# Kaggle là source of truth — cả 27/28 nhãn, neural 20 epoch

Workflow chính gồm **hai notebook độc lập**, dùng chung **một profile** `configs/kaggle_equal_20ep.json`:

| Notebook | Phạm vi | Hoàn tất khi |
| --- | --- | --- |
| `notebooks/08_kaggle_equal_20ep_27_labels.ipynb` | TF-IDF, BiGRU, PhoBERT — 27 nhãn | Đủ 3 model của 27 nhãn |
| `notebooks/09_kaggle_equal_20ep_28_labels.ipynb` | TF-IDF, BiGRU, PhoBERT — 28 nhãn | Đủ 3 model của 28 nhãn |

Hai notebook có thể chạy ở hai session riêng; không cần tải kết quả notebook kia vào để train, report hoặc archive. Notebook 04/06, các profile 10/4, 10/12 và 25/20 cùng điểm số hiện có được giữ làm lịch sử; không dùng làm kết quả của study mới.

## Ngân sách đã chốt

| Mô hình | 27 nhãn | 28 nhãn |
| --- | --- | --- |
| TF-IDF + OvR Logistic Regression | Cùng solver liblinear, C=1, max_iter=1000, seed 42 | Cùng cấu hình solver và TF-IDF |
| BiGRU | Chạy đủ 20 epoch | Chạy đủ 20 epoch |
| PhoBERT | Chạy đủ 20 epoch | Chạy đủ 20 epoch |

TF-IDF không có epoch tương đương neural. Không fit lại 20 lần để gọi thành 20 epoch, không ghi `max_iter=20` như một ngân sách tương đương. Thời gian và số update vẫn khác giữa BiGRU/PhoBERT do batch size 64/16 và kiến trúc; ngang số epoch không có nghĩa ngang compute.

Seed, split, normalization, tokenizer, learning rate, batch và các tham số khác dùng cùng profile giữa 27/28. IDs 0–26 giữ thứ tự cũ; 28 nhãn thêm Neutral và giữ annotation mixed. Chọn checkpoint **cùng validation emotion27 Macro-F1 tại ngưỡng 0,5**, hòa chọn epoch sớm nhất. Với 28 nhãn, history có cả Macro-F1 28 và `selection_macro_f1` trên 27 cảm xúc; Neutral không quyết định checkpoint trong study paired này. Đây là thay đổi có chủ đích so với study Neutral lịch sử chọn bằng Macro-F1 28.

Chạy đủ 20 epoch rồi lấy best checkpoint, không early stopping. Epoch được chọn có thể nhỏ hơn 20; điều đó không làm ngân sách huấn luyện trở thành số epoch của checkpoint. Test không được chấm hoặc dùng chọn cấu hình. Không chạy thêm sweep trong study này.

## Chuẩn bị và chạy

1. Từ project root chạy `.venv/bin/python scripts/package_kaggle_equal_epochs.py` → `vigoemotions_kaggle_equal_20ep.zip`. Gói chỉ có code/config/notebook/test và fixed IDs, không có điểm lịch sử hoặc raw data.
2. Import notebook 08 cho 27 nhãn hoặc notebook 09 cho 28 nhãn vào Kaggle, giữ Private. Attach Dataset code chứa ZIP mới và Dataset Private chứa ba CSV nguồn `id,text,labels`. Không dùng ZIP 04/06 cũ.
3. Bật GPU + Internet. Chạy bootstrap, nối workspace, dependency/GPU/Java, nạp dữ liệu và init. Phải thấy BiGRU=20, PhoBERT=20 cùng tiêu chí `emotion27_macro_f1`.
4. Trong mỗi notebook, chạy ba cell train TF-IDF, BiGRU, PhoBERT cho đúng bộ nhãn đó. Mỗi neural chạy trong process riêng, tránh xung đột JVM giữa model.
5. Cell report chỉ chấp nhận đủ 3 kết quả hợp lệ của notebook hiện tại, không đợi bộ nhãn còn lại. Đọc `comparison.csv`, gồm metric đầy đủ của từng label space và phép chiếu `emotion27_*`. Không lấy Macro-F1 27 trừ trực tiếp Macro-F1 28 để kết luận cải thiện.
6. Chạy cell archive và tải `vigoemotions_kaggle_equal20_27labels_private_results.zip` hoặc `vigoemotions_kaggle_equal20_28labels_private_results.zip` từ thư mục project trong Kaggle Output. ZIP chứa checkpoint/vocabulary, giữ riêng tư. Không copy CSV nguồn vào ZIP này.

Workspace riêng: `/kaggle/working/ViGoEmotions27Equal20` hoặc `/kaggle/working/ViGoEmotions28Equal20`. Study tương ứng ở `reports/neutral_28_labels/kaggle_equal_20ep/labels27_study/` hoặc `labels28_study/`; không ghi đè `reports/tables` lịch sử. Có thể đặt study con mới trong notebook trước init, không đổi profile giữa study.

## Resume và tính toàn vẹn

Sau restart kernel còn filesystem: bỏ qua giải nén, chạy lại cell nối workspace và setup/import/data cần thiết, rồi init. Manifest khớp sẽ giữ model hoàn tất và kiểm chứng lại; config/data/source/GPU/package thay đổi thì dừng. Nếu model dở dang, không hỗ trợ resume optimizer giữa epoch: giữ bản sao và chạy study mới. Không tự xóa model hoặc lấp số liệu thiếu bằng baseline cũ.

TF-IDF 27 nhãn có output/checkpoint riêng; vocabulary GRU và trọng số nằm trong study riêng tư. Các bước validation kiểm tra cùng ID/targets, label order, probability range, config, đủ history 20 epoch và best epoch theo tiêu chí chung. Mỗi ZIP cuối chỉ tạo sau khi đủ ba lượt đúng label scope; không cần sáu lượt trong cùng session.

Sau khi tải ZIP về, có thể giải nén vào namespace tương ứng và chạy `report` trên local để tính lại bảng từ NPZ nếu CSV/source/fixed IDs vẫn khớp. Chế độ report không train và không yêu cầu GPU/package giống Kaggle; mọi lệnh train tiếp vẫn khóa runtime/GPU/package như manifest. Metadata môi trường huấn luyện Kaggle được giữ nguyên.

## Nguồn sự thật sau khi chạy

- `manifest.json`: profile, hash nguồn code/CSV/fixed IDs, môi trường và GPU thực dùng.
- `config27.json`, `labels28/config.json`: hai snapshot cấu hình đầu ra 27/28 từ cùng profile.
- `labels27/<model>/<model>_run.json`, `labels28/models/<model>/run.json`: epoch/history/checkpoint, thời gian và tham số thực đo.
- Các NPZ validation: ID, targets, xác suất liên tục; không có raw text.
- `comparison.csv`, `status.json`: tổng hợp tính lại từ chính ba output của bộ nhãn hiện tại; không nhập điểm lịch sử.
- `logs/`: stdout/stderr các process train neural và pipeline 28 nhãn; output notebook ghi thêm lượt TF-IDF 27.

Khi có cả hai ZIP, kiểm tra profile/source/CSV hash/fixed IDs khớp trước khi ghép bảng so sánh; ghi rõ khác biệt GPU/package giữa hai session. Kết quả từ **lượt Kaggle mới hoàn tất và kiểm chứng** mới được dùng để cập nhật report. Đã nhận đủ notebook 08/09 executed và hai ZIP equal20 ngày 05/11. Sáu NPZ được tính lại độc lập; neural đủ 20 epoch. [Kết quả chính và provenance](../reports/kaggle_equal20_2026-11-05/README.md): PhoBERT 27 Macro-F1 0,597410, PhoBERT 28 Macro-F1 0,578341, phép chiếu emotion27 0,583580. Khác biệt hash nguồn/config/CSV ở lượt 28 được xác minh là LF/CRLF của cùng nội dung, không phải khác dữ liệu. Test chưa chấm.
