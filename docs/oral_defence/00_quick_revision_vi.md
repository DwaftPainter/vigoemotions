# ViGoEmotions — Tờ ôn nhanh oral defence

**Phạm vi:** Final RQ1, PhoBERT 28 nhãn gồm Neutral. Đọc tài liệu này trước; khi bị hỏi sâu, chuyển sang các tài liệu 01–04. Số liệu dưới đây lấy từ artifacts đã hoàn thành, không phải kết quả chạy mới.

## 1. Bài nói trong 60 giây

Chúng em nghiên cứu nhận diện nhiều cảm xúc trong một bình luận tiếng Việt. Midterm so sánh TF-IDF + Logistic Regression, BiGRU và PhoBERT. Final cố định checkpoint PhoBERT 28 outputs, epoch 15, rồi so sánh fixed 0.5, một ngưỡng global và 28 ngưỡng per-label. Mọi ngưỡng được chọn trên validation và khóa trước test. Trên test, global 0.4 tăng Macro-F1 nhẹ; per-label tốt nhất trên validation nhưng giảm trên test. Kết luận là tối ưu cutoff linh hoạt hơn không bảo đảm tổng quát hóa tốt hơn. Nghiên cứu có một seed và một checkpoint nên chưa kết luận ý nghĩa thống kê.

**English opening:** We freeze a 28-output PhoBERT checkpoint and compare three decision rules on the same scores. Validation-tuned per-label thresholds improve validation F1 but do not transfer to test, while a global threshold gives a small observed gain.

## 2. Bảy số phải nhớ

| Nội dung | Giá trị |
| --- | --- |
| Train sau lọc overlap | 16,530; nguồn ban đầu 16,531 |
| Validation / test | 2,066 / 2,067 |
| Số outputs | 28 = 27 emotions + Neutral, ID 27 |
| Training Midterm / checkpoint dùng Final | 20 epoch / epoch 15 |
| Seed | 42; chưa đo biến thiên nhiều seed |
| Global cutoff | 0.40 |
| Neutral per-label cutoff | 0.75 |

## 3. Bảng kết quả chính all28

| Decoder | Val Macro | Test Macro | Test Micro | Hamming | Δ Macro pp |
| --- | --- | --- | --- | --- | --- |
| Fixed 0.5 | 0.578341 | 0.589964 | 0.597350 | 0.055135 | +0.0000 |
| Global | 0.582941 | 0.591254 | 0.598233 | 0.057347 | +0.1289 |
| Per-label | 0.595555 | 0.585415 | 0.594666 | 0.058297 | -0.4549 |

Δ được tính theo **test strategy − test fixed**, đơn vị điểm phần trăm. Global tăng +0.1289 pp; per-label đổi -0.4549 pp. Không đọc chênh lệch 0.001289 thành 0.1289 trên thang F1.

Fixed có Hamming Loss thấp nhất. LRAP giống nhau giữa ba decoder vì thứ hạng scores không đổi. Không chọn lại decoder, ngưỡng hoặc checkpoint dựa trên test.

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

Protocol/config là snapshot trước khi chạy, nên một số status còn ghi pending. Trạng thái đã thực hiện nằm ở reproduction, fitted thresholds, test attempt và bảng kết quả; không sửa snapshot để đổi chữ status.

## 6. Trước khi vào phòng

Tự nói được RQ, một dòng dữ liệu giả lập, sigmoid/BCE, cách tính F1 và một giới hạn. Mở sẵn report Final và CSV kết quả ở máy offline. Mỗi người chỉ nhận công việc mình thực sự làm; chưa biết thì nêu điều chưa biết và chỉ cách kiểm tra.
