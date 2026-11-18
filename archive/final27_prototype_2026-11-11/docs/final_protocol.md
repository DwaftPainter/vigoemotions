# Protocol Final RQ1 — PhoBERT 27, v1

Ngày ghi quyết định: 11/11/2026. Mục tiêu là hoàn thiện Final theo [đề ViGoEmotions, trang 2–4](02-vigoemotions-vietnamese-emotion.pdf). Cấu hình chi tiết: [final_thresholds.json](../configs/final_thresholds.json).

**Trạng thái:** đã xác định thiết kế và có [pipeline thực thi](final_execution.md), với kiểm thử synthetic cho decoder và các gate. Inference validation bằng PhoBERT thật, fitted thresholds và pre-test lock vẫn phải hoàn tất trên môi trường neural trước test. Việc có code/protocol không đồng nghĩa đã đủ điều kiện chấm test.

## 1. Câu hỏi và phép so sánh

Trên cùng một PhoBERT 27 nhãn đã cố định, global hoặc per-label thresholds chọn bằng validation có cải thiện test Macro-F1 so với ngưỡng 0,5 không? Tập trung mức thay đổi, nhóm nhãn được lợi/chịu thiệt và khả năng khái quát hóa.

Biến thay đổi duy nhất là decoder. Cố định weights, tokenizer, preprocessing, split, label order và xác suất đầu vào. Báo cáo đủ ba decoder, kể cả khi kết quả tuned kém hơn fixed; không chọn lại thiết kế sau khi đọc test.

Kết luận chỉ áp dụng cho checkpoint/split và lưới ứng viên này, không chứng minh ngưỡng tối ưu trên mọi giá trị thực hay mọi model. Đây là threshold tuning theo F1, không phải probability calibration.

**Đối chiếu paper gốc:** [ViGoEmotions, arXiv v2, mục 5.2](https://arxiv.org/pdf/2602.08371#page=7) dùng 28 outputs (27 cảm xúc + Neutral), 12 epoch, AdamW LR 5e-5, warmup/linear decay và BCEWithLogits có positive-class weights. Protocol này là nghiên cứu RQ1 theo đề học phần trên 27 outputs, kế thừa checkpoint midterm của nhóm; cần mô tả rõ các khác biệt khi báo cáo, không gọi là tái lập nguyên cấu hình paper. Điểm giữa hai nghiên cứu chỉ so trực tiếp khi thống nhất tập nhãn, split và cách đánh giá.

## 2. Model, dữ liệu và inference giữ cố định

| Thành phần | Quyết định |
| --- | --- |
| Backbone | PhoBERT-base-v2, 27 emotion outputs, revision theo config equal20 |
| Weights | `labels27/phobert/best.pt` trong ZIP private 27; epoch 14 được chọn từ đủ 20 epoch train |
| Tiêu chí chọn checkpoint gốc | Validation Macro-F1 trên 27 cảm xúc tại 0,5; hòa giữ epoch sớm |
| Training cho RQ1 | Giữ nguyên checkpoint hiện có; không train thêm khi đổi decoder |
| Split | Retained train 16.530, validation 2.066, test 2.067; giữ IDs và thứ tự equal20 |
| Preprocessing | NFC/chuẩn hóa khoảng trắng; giữ case/dấu/phủ định; VnCoreNLP word segmentation → PhoBERT BPE |
| Tokenizer/inference | Revision cố định, slow tokenizer, padding/truncation 256, batch 16, không shuffle, eval mode, sigmoid |
| Neutral | Bỏ ID 27 khỏi emotion targets; Neutral-only → 27 số 0; mixed giữ các emotion labels |
| Seed | 42 như baseline; một seed, không đo được biến thiên training |

Checkpoint SHA-256: `41216d8245a7bd5b98db85054a326f291547effeac9e63addf9ad0ab5be4d178`. ZIP và ordered split ID hashes nằm trong config, khớp provenance/manifest equal20. Checkpoint được hash trực tiếp trong ZIP, chưa được load để inference trong công việc ghi protocol này.

Giữ split được cấp và lịch sử loại overlap; audit phân bố không chứng minh thuật toán stratification gốc. Nếu giảng viên yêu cầu split khác, cần sửa protocol trước test và xác định lại ảnh hưởng tới baseline, không âm thầm thay dữ liệu.

## 3. Quy tắc tìm ngưỡng

Mọi quyết định dương dùng `p >= threshold`. Cùng lưới cho global và per-label:

Implementation chuyển scores sang float64 khi so sánh để scalar/vector thresholds có cùng hành vi boundary; giá trị/dtype xác suất gốc vẫn được lưu nguyên trong NPZ.

`G = {0,05; 0,10; 0,15; …; 0,75; 0,80}` — 16 ứng viên, có 0,5.

| Decoder | Cách chọn | Dữ liệu để chọn |
| --- | --- | --- |
| Fixed | `tau = 0,5` cho cả 27 nhãn | Không tuning |
| Global | Ngưỡng trong G tối đa hóa Macro-F1 của 27 nhãn | Chỉ validation |
| Per-label | Với từng nhãn c, ngưỡng trong G tối đa hóa F1 của nhãn c | Chỉ validation |

- F1 dùng `2TP / (2TP + FP + FN)`; mẫu số 0 → 0 để nhất quán với baseline. Macro trung bình F1 của đúng tập nhãn đang đánh giá.
- Nếu cùng objective F1, chọn ngưỡng **cao hơn** như sweep midterm; so trên giá trị tính đầy đủ, không dựa số làm tròn trong bảng.
- Riêng nhãn không có positive validation: dùng **0,5**, đánh dấu fallback và support; không dùng test để tìm thay thế.
- Grid giữ nguyên sweep midterm để có phép so sánh nhất quán, hạn chế tìm kiếm thêm trên validation đã dùng chọn checkpoint. Grid hữu hạn có thể bỏ lỡ cutoff tốt hơn ngoài các ứng viên.
- Không dùng argmax, top-k, bắt buộc ít nhất một nhãn hay postprocessing Neutral; các quy tắc đó sẽ thay đổi decoder ngoài RQ1 đã chốt.

Global 0,35 và Macro-F1 khoảng 0,597843 hiện có là kết quả **thăm dò cũ**; vẫn cần tái tạo và ghi fitted artifact Final. Không điền trước ngưỡng tối ưu cho per-label. File config giữ giá trị fitted là `null` cho tới lúc chạy bước này.

## 4. Những bảng và hình phải sinh

Metric chính là test Macro-F1; đồng thời báo Micro-F1, F1 của đủ 27 nhãn và Hamming Loss trên validation/test cho từng decoder. Nếu giữ LRAP từ midterm, tính từ cùng continuous scores nên ba decoder phải có LRAP giống nhau.

1. **Overall:** ba decoder × validation/test; delta bằng strategy trừ fixed trong cùng split. Hamming Loss giảm là tốt, F1/LRAP tăng là tốt.
2. **Frequency tiers:** lấy positive count chỉ từ 16.530 train IDs; sắp giảm dần, hòa theo ID nhãn tăng dần. Head hạng 1–9, mid 10–18, tail 19–27. [Bảng membership đã tính](final_label_tiers.csv) có đủ 27 nhãn. Đây là chia theo hạng tần suất, không tuyên bố các nhóm cách nhau bởi một ngưỡng hiếm phổ quát.
3. **Tier metrics:** lấy chín cột nhãn của từng tier trên mọi comment; Macro-F1 trung bình đúng chín nhãn, Micro/Hamming trên cùng ma trận con.
4. **Cardinality:** phân nhóm theo tổng 27 ground-truth emotions: 1, 2, 3, ≥4; thêm nhóm 0 để giữ Neutral-only. Mỗi nhóm tính metrics trên đủ 27 cột, `zero_division=0`; không tự bỏ các nhãn vắng support. Nhóm rỗng có n=0, metric N/A.
5. **Threshold profile:** đủ 27 nhãn, ngưỡng, train count, validation support và tier.
6. **Generalization gap:** Macro/Micro-F1 validation trừ test theo từng decoder, cùng support để đọc nguy cơ overfit. Không coi mọi gap là bằng chứng nguyên nhân do threshold.

Các bảng subgroup ghi n, positive support và tập nhãn dùng trong metric; phân tích per-label có TP/FP/FN để giải thích đánh đổi. Không so các Macro-F1 có tập nhãn khác nhau như cùng một đại lượng.

## 5. Error audit do người thực hiện

Chọn trước **per-label decoder** làm decoder chính để audit, không phụ thuộc nó thắng hay thua trên test. Mỗi ca vẫn trình bày outputs fixed/global/per-label để thấy lỗi được sửa, giữ lại hoặc phát sinh.

- Đọc tối thiểu **20 FN + 20 FP**, đơn vị là cặp `(sample_id, label_id)` của per-label trên test sau khi đã đánh giá bằng thresholds khóa.
- Chọn theo hash ổn định của `42|FN_or_FP|sample_id|label_id`, sắp tăng dần, hòa theo sample ID/label ID. Duyệt trước một cặp mỗi comment, rồi bổ sung cặp chưa chọn nếu cần để đủ 20. Nếu tổng lỗi không đủ, đọc tất cả và báo số thiếu.
- Bổ sung năm comment có true cardinality ≥4 theo hash `42|high_cardinality|sample_id`; ít hơn năm thì đọc toàn bộ. Không coi các ca này là FN/FP nếu chúng không có lỗi.
- Chọn ba cặp nhãn đồng xuất hiện nhiều nhất **trong train**, hòa theo IDs tăng dần. Với mỗi cặp, lấy tối đa năm comment test có cả hai true labels hoặc cả hai per-label predicted labels, theo hash `42|pair|a|b|sample_id`; loại trùng comment giữa các cặp. Đây là ca để đọc về đồng xuất hiện/prediction mismatch, chưa tự nó chứng minh semantic confusion.
- Thành viên thật đọc raw text private; ghi reviewer, nhãn thật, dự đoán, score, threshold, cardinality và nhận xét khái quát. Không public nguyên văn bình luận.
- Báo số cặp và số comment duy nhất, phần trùng giữa các nhóm; không suy tỷ lệ lỗi dân số từ tập audit có chủ đích này. Giả thuyết về ngôn ngữ/annotation phải được gọi đúng là giả thuyết.

## 6. Điều kiện khóa trước test và đầu ra

Thực hiện theo thứ tự:

1. Kiểm hash ZIP/checkpoint/NPZ, source config và ordered split IDs; giữ canonical 27-label order.
2. Khôi phục checkpoint, inference toàn bộ validation; so với NPZ nguồn trong absolute tolerance **1e-4**, relative tolerance 0. Kiểm ID/targets/labels/shape và fixed Macro-F1 ≈0,597410. Nếu lệch, tìm nguyên nhân môi trường/preprocessing; không tự nới tolerance.
3. Fit ngưỡng từ **NPZ nguồn đã lưu**, sau khi đã chứng minh checkpoint khớp, để tuning có cùng nguồn xác suất tái lập. Lưu `reports/final/thresholds.json` gồm một global scalar, 27 ngưỡng theo label order, objective/grid/tie-break/fallback và source hashes.
4. Tạo `reports/final/pre_test_lock.json` ghi hash của protocol, config, tiers, implementation, checkpoint, validation NPZ, thresholds và split IDs. Kiểm thử decoder trước test.
5. Inference test đúng một lượt bằng checkpoint đó, lưu continuous probabilities trước binarization; dùng lại một ma trận scores cho cả ba decoder.
6. Sinh bảng/hình/audit. Có thể tái đọc saved probabilities để tái tạo kết quả; không đổi thresholds sau khi xem test. Job lỗi kỹ thuật phải giữ evidence và chỉ tiếp tục với thiết kế không đổi.

NPZ/CSV validation và test phải có `sample_id`, `true_label_vector`, `probability_vector`, `pred_fixed`, `pred_global`, `pred_per_label`, `label_cardinality`, cùng label order và liên kết tới frozen thresholds. Giữ checkpoint/vocabulary/raw text private.

## 7. Giới hạn và câu trả lời khi bảo vệ

Validation đã dùng để chọn epoch và thăm dò global threshold trước khi ghi protocol này; đây không phải preregistration trước mọi quan sát dữ liệu. Protocol ghi các quyết định trước test và hạn chế lựa chọn hậu nghiệm. Một seed không đủ khẳng định tính ổn định qua training; tail ở đây là nhóm hạng thấp hơn và có thể vẫn còn hàng trăm mẫu train.

Nếu per-label tăng validation nhưng giảm test, báo đúng đánh đổi và gap. Nếu fixed tốt nhất test, kết luận chỉ rằng tuning theo protocol này chưa cho thấy cải thiện trên held-out split này. Không dùng test để điều chỉnh rồi gọi đó là cùng thí nghiệm RQ1.

Câu trình bày ngắn: “Chúng em giữ nguyên PhoBERT 27 và checkpoint epoch 14, chỉ thay decoder. Ngưỡng được tối ưu trên validation, khóa trước test; cùng xác suất test được dùng cho cả ba chiến lược. Chúng em phân tích kết quả theo tần suất nhãn, cardinality và lỗi thực tế.”
