### 5.7. Cập nhật 04/11/2026 — dự đoán thêm Neutral, đủ 28 nhãn

**Nguồn và phạm vi.** Đã nhận `vigoemotions_neutral28_private_results.zip` và notebook `06_kaggle_neutral_28_labels.ipynb` có output. Ba mô hình đã huấn luyện xong; kết quả dưới đây chỉ trên 2.066 mẫu validation tại ngưỡng 0,5, seed 42. Test chưa được chấm (`RUN_FINAL_TEST=False`, ZIP không có test predictions/test lock). Đây là thí nghiệm bổ sung, giữ nguyên các artifact baseline 27 nhãn ở mục 5.1–5.6. Các bảng và metadata được trích nguyên tại `reports/midterm_2026-11-04/evidence/`; `provenance.json` ghi SHA-256 và các bước kiểm chứng.

**Hai cách mã hóa.** Baseline 27 nhãn bỏ ID 27: `[27]` thành vector toàn 0 và `[2,27]` chỉ giữ joy. Study 28 nhãn giữ Neutral thành đầu ra riêng: `[27]` có Neutral dương, `[2,27]` có cả joy và Neutral dương. Dùng multi-hot, sigmoid độc lập và BCE; không dùng softmax hoặc luật ép Neutral loại trừ cảm xúc. Có 725 mẫu Neutral-only và 306 mixed-neutral trong CSV gốc. Giữ annotation để đo, không suy đoán nhãn nào sai. Validation có 75 Neutral-only + 19 mixed = 94 mẫu Neutral dương (4,55%).

**Kết quả validation trên đủ 28 nhãn.** Hamming Loss càng thấp càng tốt; các metric còn lại càng cao càng tốt.

| Mô hình | Macro-F1 | Micro-F1 | Hamming Loss | LRAP |
| --- | --- | --- | --- | --- |
| TF-IDF + OvR LR | 0,231040 | 0,266160 | 0,060054 | 0,609645 |
| BiGRU | 0,399227 | 0,459095 | 0,058861 | 0,605071 |
| PhoBERT | 0,574653 | 0,577672 | 0,055127 | 0,712201 |

PhoBERT cao nhất trong study này theo Macro-F1, Micro-F1 và LRAP, đồng thời có Hamming Loss thấp nhất. Đây là xếp hạng trong cùng thiết lập 28 nhãn; chưa có kiểm định ý nghĩa thống kê hoặc kết quả test. BiGRU có F1 cao hơn TF-IDF nhưng LRAP thấp hơn một chút; không khẳng định mọi metric đều tăng.

![Macro-F1 đủ 28 nhãn và F1 Neutral, validation tại 0,5; không đặt metric baseline 27 nhãn vào cùng phép so sánh.](midterm_2026-11-04/neutral28_metrics.png)

**Neutral có được học thực sự không?** Support là 94 cho cả ba mô hình và bao gồm mixed-neutral.

| Mô hình | Precision | Recall | F1 | Dự đoán dương | TP / FP / FN |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + OvR LR | 0,5000 | 0,0106 | 0,0208 | 2 | 1 / 1 / 93 |
| BiGRU | 0,2500 | 0,0532 | 0,0877 | 20 | 5 / 15 / 89 |
| PhoBERT | 0,4151 | 0,4681 | 0,4400 | 106 | 44 / 62 / 50 |

TF-IDF và BiGRU bỏ sót phần lớn Neutral ở 0,5. PhoBERT tìm được 44/94 mẫu Neutral nhưng còn 62 false positives và 50 false negatives. Precision 0,4151 cho thấy đầu ra Neutral còn nhiều dự đoán sai; không trình bày F1 0,4400 như đã giải quyết xong bài toán. Baseline 27 nhãn không có đầu ra Neutral nên không gán Neutral F1 bằng 0 cho baseline đó. Tập dự đoán rỗng trong mô hình 28 nhãn cũng không tự động là Neutral.

**So sánh trên cùng 27 cảm xúc.** Chiếu xác suất mô hình 28 nhãn về IDs 0–26 rồi tính `emotion27_macro_f1`. Đã kiểm tra cùng ID, targets validation và tính lại metric baseline gốc từ NPZ.

| Mô hình | Baseline 27 nhãn gốc | Study 28 chiếu về 27 | Chênh lệch điểm phần trăm |
| --- | --- | --- | --- |
| TF-IDF + OvR LR | 0,238825 | 0,238825 | +0,0000 |
| BiGRU | 0,406002 | 0,410764 | +0,4762 |
| PhoBERT | 0,439320 | 0,579640 | +14,0321 |

Baseline trong bảng là bộ gốc tại `reports/tables/` và `reports/predictions/`, được study 06 dùng làm nguồn so sánh. Lượt chạy lại 27 nhãn dùng cho lời thoại ngày 23/10 có PhoBERT Macro-F1 0,467850, lưu riêng tại `reports/pre_midterm_2026-10-23/evidence/`; nó không phải đối chứng trong bảng này. Không thay lẫn 0,439320 và 0,467850 khi nói về delta.

Delta trên cùng 27 cảm xúc cho biết khác biệt quan sát được, chưa cô lập tác động của thêm Neutral. PhoBERT mới chạy 12 epoch, chọn epoch 11 theo Macro-F1 28 nhãn; baseline gốc chỉ chạy 4 epoch và chọn bằng Macro-F1 27 nhãn. Runtime/package cũng khác. Đặc biệt `study.json.baseline_config` là config sao chép khi init (PhoBERT 12 epoch), không phải bằng chứng baseline gốc đã chạy 12 epoch: phải đọc `reports/phobert_run.json`. Muốn quy nguyên nhân cho Neutral cần chạy đối chứng 27/28 cùng runtime, ngân sách epoch và quy tắc chọn checkpoint đã định trước, tốt hơn với nhiều seed.

Trên cùng 27 cảm xúc, Hamming Loss PhoBERT tăng từ 0,053261 lên 0,055161 dù Macro-F1 tăng; LRAP tăng từ 0,715767 lên 0,732624. Không so trực tiếp Macro-F1/LRAP toàn bộ 27 với toàn bộ 28: tập nhãn và ý nghĩa của Neutral-only đã đổi. Hamming cũng đổi mẫu số khi thêm một đầu ra; dùng phép chiếu 27 khi đối chiếu baseline.

**Cấu hình và đường học từ lượt chạy thật.** Cả hai neural chạy trên Tesla T4; fine-tune toàn bộ PhoBERT, revision cố định, lr=2e-5, batch=16, max_length=256, weight_decay=0,01. BiGRU giữ embedding/hidden 128, batch=64, lr=0,001. Seed 42; chọn checkpoint theo validation Macro-F1 28 nhãn tại 0,5, hòa chọn epoch sớm nhất.

| Mô hình | Epoch chạy / chọn | Tham số | Thời gian train |
| --- | --- | --- | --- |
| TF-IDF + OvR LR | Không áp dụng | 975.464 | 2,59 giây CPU |
| BiGRU | 10 / 10 | 2.128.540 | 34,98 giây T4 |
| PhoBERT | 12 / 11 | 135.019.804 | 8.930,75 giây T4 (148,85 phút) |

Thời gian TF-IDF gồm fit vectorizer + LR, còn neural gồm vòng train, validation từng epoch và ghi checkpoint, không gồm preprocessing. Không dùng bảng này để so tốc độ tuyệt đối giữa CPU và GPU. ZIP riêng tư có checkpoint, tokenizer và vocabulary cho study 28, khác ZIP baseline gốc chỉ có kết quả.

![Train loss và validation Macro-F1 28 nhãn theo epoch; PhoBERT chọn epoch 11, không chọn epoch cuối.](midterm_2026-11-04/neutral28_learning_curves.png)

PhoBERT tăng Macro-F1 từ 0,4753 ở epoch 4 lên 0,5747 ở epoch 11 rồi giảm còn 0,5630 ở epoch 12; train loss vẫn giảm. Chỉ nêu biến động quan sát được: không có validation loss, chưa đủ cơ sở kết luận hội tụ hay overfitting. Đây là một lượt lr=2e-5, không hoàn tất sweep learning rate notebook 05.

**Inference và giới hạn notebook.** Cell inference trên bốn câu tự viết đã xuất kết quả cho cả ba mô hình. Cell code tìm kiểu dự đoán validation tiếp theo (chỉ số cell 24, execution_count 16) dừng với `ValueError: VM is already running, can't set classpath/options` khi nạp lại VnCoreNLP. ZIP không có `validation_prediction_cases.csv`; không khẳng định cell đó chạy xong. Các metric và xác suất validation đã xuất trước lỗi và đã được tính lại độc lập. Bảng `prediction_category_counts.csv` ở evidence được tính trực tiếp từ NPZ tại ngưỡng 0,5; đây là thống kê dự đoán, không phải audit ngữ nghĩa thủ công. Notebook đã lưu có output comment nguồn; giữ bản thực thi riêng tư khi chuẩn bị gói chia sẻ.

**Kết luận dùng khi thuyết trình.** Nhóm đã mở rộng đầu ra thành 28 nhãn và kiểm chứng đầy đủ kết quả validation của ba mô hình. PhoBERT tốt nhất trong study, nhưng Neutral còn khó và phép so sánh với baseline chịu ảnh hưởng số epoch/quy tắc chọn checkpoint. Bước Final vẫn là chọn và khóa một checkpoint, kiểm tra ngưỡng fixed/global/per-label trên validation rồi chấm test sau khi khóa mọi lựa chọn. Nếu dùng 28 nhãn, báo cáo cả metric 28, phép chiếu 27 và Neutral riêng; không tự coi ngưỡng 0,15 của TF-IDF 27 nhãn là ngưỡng của PhoBERT 28 nhãn.
