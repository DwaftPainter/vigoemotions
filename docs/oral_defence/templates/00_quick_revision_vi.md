# ViGoEmotions — Tờ ôn nhanh oral defence

**Phạm vi:** Final RQ1 về ngưỡng dự đoán và RQ4 về độ bền với nhiễu văn bản trên PhoBERT 28 nhãn gồm Neutral. Đọc tài liệu này trước, rồi xem các tài liệu 01–04 để ôn sâu hơn. Các bảng dùng kết quả đã kiểm chứng.

## 1. Bài nói trong 60 giây

Chúng em nghiên cứu nhận diện nhiều cảm xúc trong một bình luận tiếng Việt. Midterm so sánh TF-IDF + Logistic Regression, BiGRU và PhoBERT. Với RQ1, nhóm dùng PhoBERT 28 đầu ra ở epoch 15 để so sánh ngưỡng cố định 0.5, ngưỡng chung và ngưỡng riêng từng nhãn. Trên test, ngưỡng chung 0.4 tăng Macro-F1 nhẹ, còn ngưỡng riêng giảm điểm dù tốt nhất trên validation. RQ4 giữ nguyên model và ngưỡng, rồi tạo nhiễu trên cùng bình luận. Ở ngưỡng 0.5, bỏ dấu làm Macro-F1 giảm {{rq4_diacritics_drop_pp}} điểm phần trăm và lỗi gõ làm giảm {{rq4_typo_drop_pp}} điểm. Các kết quả mô tả một checkpoint và một split.

**English opening:** RQ1 compares three decision rules on scores from a frozen 28-output PhoBERT checkpoint. Global tuning gives a small observed test gain, while per-label tuning loses its validation advantage. RQ4 uses the same checkpoint and thresholds to compare original text with two types of simulated noise. At the fixed 0.5 threshold, diacritic removal and keyboard typos reduce test Macro-F1 by {{rq4_diacritics_drop_pp}} and {{rq4_typo_drop_pp}} percentage points, respectively.

## 2. Bảy số phải nhớ

| Nội dung | Giá trị |
| --- | --- |
| Train sau lọc overlap | 16,530; nguồn ban đầu 16,531 |
| Validation / test | 2,066 / 2,067 |
| Số outputs | 28 = 27 emotions + Neutral, ID 27 |
| Training Midterm / checkpoint dùng Final | 20 epoch / epoch 15 |
| Seed | 42; chưa đo biến thiên nhiều seed |
| Global cutoff | {{global_threshold}} |
| Neutral per-label cutoff | {{neutral_threshold}} |

## 3. Kết quả chính RQ1 và RQ4

{{primary_table}}

Δ được tính theo **test strategy − test fixed**, đơn vị điểm phần trăm. Global tăng {{global_gain_pp}} pp; per-label đổi {{per_label_gain_pp}} pp. Không đọc chênh lệch 0.001289 thành 0.1289 trên thang F1.

Fixed có Hamming Loss thấp nhất. LRAP giống nhau giữa ba decoder vì thứ hạng scores không đổi. Không chọn lại decoder, ngưỡng hoặc checkpoint dựa trên test.

**RQ4 ở fixed 0.5:** Macro-F1 test all28 đi từ {{test_fixed_macro}} trên văn bản gốc xuống {{rq4_diacritics_macro}} khi bỏ dấu và {{rq4_typo_macro}} khi tạo lỗi gõ QWERTY. Nhóm so từng dạng nhiễu với văn bản gốc của cùng 2,067 mẫu. Hai phép biến đổi có mức độ khác nhau. Xem bảng đủ ba decoder trong [tài liệu 02](02_protocol_results_vi.md).

## 4. Những chỗ dễ trả lời nhầm

- Đây là **multi-label**: sigmoid độc lập, không phải softmax hoặc chọn một lớp bằng argmax.
- “28 class” trong trao đổi nghĩa là 28 đầu ra nhị phân, không phải bài toán 28 lớp loại trừ nhau.
- Checkpoint epoch 15 được chọn trước đây bằng **emotion27 Macro-F1 tại 0.5**. Final tuning dùng **all28 Macro-F1**. Hai tiêu chí khác phạm vi và đều phải nói rõ.
- Emotion27 projection là bỏ cột Neutral của cùng model28 và cùng cutoffs; không phải model27 huấn luyện riêng hoặc ngưỡng global27 tune riêng.
- `[27]` là Neutral dương; `[2,27]` giữ joy và Neutral. Không ép exclusivity, không đổi dự đoán rỗng thành Neutral.
- Tiers chia 27 emotions theo train, chín nhãn mỗi nhóm. Neutral báo riêng.
- Gap = validation − test. Gap âm có nghĩa test cao hơn validation trong lượt đo này; không tự suy ra leakage.
- Không có bằng chứng thống kê cho mức tăng nhỏ, không chứng minh nguyên nhân từ vài ca lỗi.

## 5. Bằng chứng mở ngay khi bị hỏi

| Cần chứng minh | File |
| --- | --- |
| Protocol / objective / Neutral | configs/final_thresholds.json; docs/final_protocol.md |
| Ngưỡng đã fit | reports/final28/thresholds.json |
| Reproduction max error = 0 | reports/final28/validation_reproduction.json |
| Bind trước test / completed attempt | reports/final28/pre_test_lock.json; test_attempt.json |
| Scores và ba decoder | reports/final28/predictions/test.npz |
| Metric / hỗ trợ từng nhóm | reports/final28/tables/ |
| Quan sát lỗi | reports/final28/human_audit/cases.csv và notes trừu tượng |
| RQ4 và số mẫu thực sự đổi | reports/rq4_noise/test/tables/comparison.csv và noise_coverage.csv |

Protocol/config là snapshot trước khi chạy, nên một số status còn ghi pending. Trạng thái đã thực hiện nằm ở reproduction, fitted thresholds, test attempt và bảng kết quả; không sửa snapshot để đổi chữ status.

## 6. Trước khi vào phòng

Tự nói được RQ, một dòng dữ liệu giả lập, sigmoid/BCE, cách tính F1 và một giới hạn. Mở sẵn report Final và CSV kết quả ở máy offline. Mỗi người chỉ nhận công việc mình thực sự làm; chưa biết thì nêu điều chưa biết và chỉ cách kiểm tra.
