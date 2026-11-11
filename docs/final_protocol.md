# Protocol Final RQ1 — PhoBERT 28, v2

Quyết định của nhóm ngày 11/11/2026: **dùng 28 class, gồm 27 cảm xúc + Neutral**. Bản này thay thiết kế 27-output; [prototype cũ và hashes](../archive/final27_prototype_2026-11-11/snapshot.json) được giữ để đối chiếu. Cấu hình thực thi: [final_thresholds.json](../configs/final_thresholds.json); [cách chạy](final_execution.md).

**Trạng thái:** pipeline 28 đã triển khai và source validation đã kiểm chứng. Inference PhoBERT thật, fitted thresholds và pre-test lock vẫn phải được tạo bằng đúng checkpoint; chưa có kết quả test Final. Unit tests synthetic không thay thế bằng chứng model thật.

## 1. Câu hỏi và phạm vi

Global hoặc per-label thresholds chọn từ validation có cải thiện test Macro-F1 trên **đủ 28 nhãn** so với 0,5, khi giữ nguyên PhoBERT 28 không? Báo đủ ba decoder, kể cả khi tuned không cải thiện test.

[Đề học phần](02-vigoemotions-vietnamese-emotion.pdf), trang 2–4, viết RQ1 cho 27 cảm xúc. Nhóm quyết định mở rộng output và objective thành 28. Để đối chiếu, mọi decoder có thêm bảng **emotion27** và bảng **Neutral**. Phép chiếu emotion27 dùng cùng cutoff chọn theo objective all28; không gọi đó là một lượt tối ưu global riêng cho 27 hoặc là model train 27 nhãn.

[Paper gốc, mục 5.2](https://arxiv.org/pdf/2602.08371#page=7) cũng dùng 28 outputs. Nhóm kế thừa label space đó, dùng checkpoint midterm equal20 của mình; không gọi là tái lập nguyên cấu hình paper về epochs, loss weighting hoặc learning-rate schedule.

## 2. Model và dữ liệu giữ cố định

| Thành phần | Quyết định |
| --- | --- |
| Backbone | PhoBERT-base-v2, 28 outputs; Neutral là ID 27, 27 emotions giữ IDs 0–26 |
| Checkpoint | Epoch **15** từ đủ 20 epoch train của notebook 09 |
| Selection gốc | Validation **emotion27 Macro-F1 tại 0,5**, hòa giữ epoch sớm; giữ nguyên lịch sử này dù objective tuning mới là all28 |
| Storage | Saved Hugging Face model `pretrained/model.safetensors` + config và saved tokenizer trong ZIP 28 |
| Training cho RQ1 | Giữ nguyên weights, không train thêm khi đổi decoder |
| Split | Retained train 16.530 / validation 2.066 / test 2.067; fixed IDs/order equal20 |
| Preprocessing | NFC/chuẩn hóa khoảng trắng, giữ case/dấu/phủ định; VnCoreNLP word segmentation → saved PhoBERT tokenizer |
| Inference | Local model/tokenizer, slow tokenizer, padding/truncation 256, batch 16, không shuffle, eval mode, sigmoid |
| Seed | 42; một seed, chưa đo biến thiên training |

Checkpoint SHA-256: `30e09d73bd474c6c1a9d2455519af7256ce96ed12576dde9aafcff45a301c835`. Archive/model-config/NPZ/split hashes ghi trong config và provenance gốc. NPZ validation là **2.066 × 28**; Macro-F1 fixed 0,5 là **0,5783409798361765**, emotion27 projection **0,5835797863529424**. Đây là reference midterm, không phải test Final.

Giữ annotation gốc: `[27]` chỉ Neutral dương; `[2,27]` giữ cả joy và Neutral. Không ép Neutral loại trừ cảm xúc, không dùng softmax, argmax, top-k hoặc gán empty prediction thành Neutral. Audit phân bố không chứng minh thuật toán stratification gốc; mô tả đúng nguồn split.

## 3. Decoder và objective đã chốt trước test

Mọi quyết định dương dùng `p >= threshold`. So sánh float64 để scalar/vector cutoff có cùng hành vi boundary; xác suất gốc vẫn được lưu nguyên trong NPZ.

`G = {0,05; 0,10; …; 0,75; 0,80}` — 16 ứng viên, có 0,5.

| Decoder | Quy tắc |
| --- | --- |
| Fixed | Ngưỡng 0,5 cho đủ 28 nhãn |
| Global | Một ngưỡng trong G tối đa hóa **validation Macro-F1 all28**, có Neutral |
| Per-label | **28 ngưỡng**, mỗi ngưỡng tối đa hóa positive-class F1 của nhãn tương ứng trên validation |

Hòa F1 chọn cutoff cao hơn, so giá trị đầy đủ thay vì số làm tròn. Nhãn không có positive validation dùng 0,5, ghi support và fallback. F1 mẫu số 0 → 0. Giữ cùng grid/objective/weights khi chấm test; không chỉnh ngưỡng theo kết quả test. Threshold tuning không phải probability calibration.

Không lấy lại global 0,35 của model 27. Phải tối ưu mới từ NPZ model 28 sau khi checkpoint 28 đã tái tạo validation thành công. Kết luận chỉ áp dụng cho checkpoint/split/grid này; không khẳng định cutoff tối ưu trên mọi số thực.

## 4. Bảng và phân tích bắt buộc

1. **all28, chính:** Macro-F1, Micro-F1, Hamming Loss, LRAP và F1 từng nhãn cho ba decoder, trên validation/test. Global chọn bằng all28 Macro-F1.
2. **emotion27, bổ sung:** cùng ma trận scores/cutoffs bỏ cột Neutral, không re-tune. Scope và label count ghi rõ trong bảng và generalization gap.
3. **Neutral riêng:** positive-class precision/recall/F1, TP/FP/FN, support và cutoff cho mỗi decoder. F1 này không tính trung bình thêm negative class.
4. **Frequency tiers:** giữ membership train-only của 27 emotion labels, chín nhãn mỗi head/mid/tail theo rank giảm dần, hòa theo label ID. [Membership](final_label_tiers.csv) là partition **27 emotions**, không phải partition toàn bộ 28; Neutral có bảng riêng và ghi tier `neutral` trong threshold profile.
5. **Cardinality chính:** `label_cardinality = sum(y[:, :28])`, gồm Neutral; nhóm 0/1/2/3/≥4, metrics trên 28 cột. Nhóm 0 thường rỗng và ghi n=0, metric N/A.
6. **Cardinality bổ sung:** `emotion_cardinality = sum(y[:, :27])`, không gồm Neutral; nhóm 0/1/2/3/≥4, metrics trên 27 cột. Neutral-only có label cardinality 1 nhưng emotion cardinality 0; mixed không bị sửa annotation.
7. **Threshold profile:** đủ 28 nhãn, gồm Neutral, train count, validation support, tier và fallback. Gap là `F1_val − F1_test` của cùng decoder và scope.

Delta bằng strategy trừ fixed trong cùng split/scope/group. Macro giữ nguyên tập nhãn đã khai báo, `zero_division=0`; subgroup ghi n và support. LRAP bất biến khi chỉ thay cutoffs trong cùng scope, nhưng all28 và emotion27 là hai ranking tasks khác nhau.

## 5. Audit lỗi do người thực hiện

Audit chính: **per-label decoder trên đủ 28 nhãn**, dù nó thắng hay thua test. Người thật đọc ≥20 FN và ≥20 FP, đơn vị `(sample_id,label_id)`; cho xem cả ba decoder trên cùng ca. Neutral FP/FN và mixed annotation là các nhóm cần xem trong phần diễn giải.

Giữ quy tắc hash đã ghi trong config: sắp theo `42|FN_or_FP|sample_id|label_id`, ưu tiên comment khác nhau trước rồi bổ sung cặp chưa chọn. Không đủ lỗi thì audit tất cả và báo thiếu. Thêm năm ca true cardinality all28 ≥4; đối chiếu thêm emotion-only cardinality. Ba cặp emotion đồng xuất hiện chọn chỉ từ train 27 emotions, rồi lấy ca test theo quy tắc hash trong config.

Raw text chỉ đọc private. Ghi reviewer thật, true/pred/score/cutoff/cardinality và nhận xét trừu tượng; báo số cặp và comment duy nhất. Không coi danh sách tự sinh/AI là human audit hoàn tất, không suy tỷ lệ lỗi toàn tập từ mẫu chọn có chủ đích.

## 6. Thứ tự thực thi và khóa trước test

1. `verify`: source run 28, NPZ, hashes, IDs/targets và train-only tiers; không đọc CSV test.
2. `restore`: kiểm ZIP 28 rồi trích model/config/tokenizer/VnCoreNLP vào **data/final28/**; không dùng checkpoint 27 cũ trong data/final/.
3. `reproduce-val`: frozen model 28 inference đủ validation; absolute tolerance **1e-4**, relative 0 và fixed decisions giống hệt NPZ gốc. Lệch thì tìm lỗi môi trường/preprocessing, không tự nới tolerance.
4. `fit`: sau reproduction PASS, chọn thresholds từ **NPZ nguồn 28**. Lưu global + 28 per-label thresholds, source hashes và evidence trong **reports/final28/**.
5. `lock`: bind protocol/config/implementation, checkpoint, NPZ nguồn, thresholds, reproduction và split IDs. Code/protocol thay đổi làm lock stale.
6. `infer-test`: kiểm lock/assets/runtime trước đọc held-out text/targets; lưu continuous scores trước decoding. Một lượt scores dùng cho cả ba decoder và cả hai scopes.
7. `report`: tái đọc saved predictions để tái sinh metric tables, không inference lại. Lỗi kỹ thuật giữ attempt record; chỉ resume cùng lựa chọn và dùng lại scores đã lưu nếu có.

Prediction NPZ có IDs, labels, true/probability vectors, ba pred matrices, `label_cardinality` và `emotion_cardinality`. Không public raw comments, weights hoặc vocabulary. Notebook 10 dừng sau validation/lock, không tự chấm test.

## 7. Giới hạn và oral defence

Validation đã dùng chọn checkpoint và có phân tích midterm, nên đây không phải preregistration trước mọi quan sát. Một seed chưa đủ kết luận ổn định/có ý nghĩa thống kê; group nhỏ cần đọc support. Chuyển 28 là quyết định thiết kế của nhóm trước test, không phải bằng chứng thêm Neutral làm tốt hơn mọi model 27.

Câu trình bày: “Chúng em dùng 28 independent sigmoid outputs, gồm Neutral và giữ mixed labels. Chúng em cố định checkpoint epoch 15, tối ưu cutoffs trên validation Macro-F1 của đủ 28 nhãn, khóa trước test và báo thêm emotion27 projection cùng Neutral riêng.”
