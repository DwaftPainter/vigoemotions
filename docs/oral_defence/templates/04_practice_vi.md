# ViGoEmotions — Bài tính tay và mock defence

Mọi số trong bài tập 1–3 là **ví dụ tự tạo để học**, không phải dataset hoặc kết quả mô hình của nhóm. Phần mock dùng kết quả thật ở tài liệu 02. Lịch luyện dưới đây không phải thời lượng oral do giảng viên thông báo.

## 1. Tính Macro-F1, Micro-F1 và Hamming bằng tay

Chỉ xét ba nhãn A, B, C và bốn mẫu:

| Mẫu | Ground truth | Prediction tại 0.5 |
| --- | --- | --- |
| s1 | [1,0,1] | [1,0,0] |
| s2 | [0,1,0] | [0,1,1] |
| s3 | [1,1,0] | [1,0,0] |
| s4 | [0,0,0] | [0,0,0] |

Trước khi đọc đáp án: viết TP/FP/FN mỗi nhãn; tính per-label F1, Macro-F1, Micro-F1, Hamming và subset accuracy. Giải thích tại sao hai cách lấy trung bình khác nhau.

**Đáp án:** A có TP=2, FP=0, FN=0, F1=1. B có TP=1, FP=0, FN=1, F1=2/3. C có TP=0, FP=1, FN=1, F1=0. Macro-F1=(1+2/3+0)/3=5/9≈0.555556. Cộng counts: TP=3, FP=1, FN=2; Micro-F1=6/(6+1+2)=2/3≈0.666667. Hamming=(1+2)/(4×3)=0.25. Chỉ s4 khớp toàn bộ nhãn nên subset accuracy=1/4=0.25.

**Câu hỏi tiếp:** Nếu dự đoán tất cả bằng 0 thì Hamming có thể trông thấp trong dữ liệu nhiều nhãn âm, nhưng recall/F1 của positives sẽ thế nào? Không suy “đúng nhiều quyết định âm” thành “tìm cảm xúc tốt”.

## 2. Trace ba decoder trên một vector giả lập

Thứ tự bốn nhãn giả lập là joy, sadness, gratitude, Neutral:

```text
scores    = [0.62, 0.41, 0.08, 0.56]
fixed     = 0.50
global    = 0.40
per_label = [0.65, 0.35, 0.05, 0.75]
```

Hãy viết output từng decoder theo p>=cutoff, giải thích vì sao có thể có Neutral cùng emotion, và trả lời cutoff có làm thay đổi scores không.

**Đáp án:** Fixed cho [1,0,0,1]; global cho [1,1,0,1]; per-label cho [0,1,1,0]. Đây là vector bốn nhãn minh họa, không phải bốn cutoff thật trích từ model28. Nếu score đúng bằng cutoff thì output dương. Cutoff không cập nhật weights hoặc score.

Không thể tính FP/FN/F1 của ví dụ này nếu chưa cho ground truth. Không nhìn một score cao rồi khẳng định prediction đúng.

## 3. Hiểu gradient của BCE

Cho một output có p=0.8. Nếu y=1, loss=-log(0.8)≈0.2231. Nếu y=0, loss=-log(0.2)≈1.6094. Với một hạng không trọng số, gradient theo logit là p−y: lần lượt -0.2 và +0.8.

Gradient descent trừ learning_rate×gradient: gradient âm làm logit tăng, gradient dương làm logit giảm. Với mean loss trên batch và nhãn có thêm hệ số chuẩn hóa. Không dùng gradient của loss như thể đó là ngưỡng quyết định cuối cùng.

**Hỏi thêm:** Khi chạy inference với `model.eval()` và `no_grad`, optimizer có cập nhật weights không? Không. RQ1 đổi ngưỡng trên cùng điểm số. RQ4 đổi đầu vào rồi lấy điểm số mới bằng cùng checkpoint.

## 4. Đọc một kết quả thật trong 90 giây

Chọn một dòng trong `reports/final28/tables/test_per_label.csv` và đối chiếu `threshold_profile.csv`. Tự nói label/support, TP/FP/FN/F1, cutoff fixed/global/per-label, và điều gì thay đổi. Các CSV chỉ có derived outputs, không cần mở raw comments.

Ví dụ nervousness: fixed TP=41, FP=26, FN=55; per-label TP=51, FP=81, FN=45. Tính F1 fixed = 82/163≈0.503067; per-label = 102/228≈0.447368. Recall tăng từ 41/96 lên 51/96 nhưng F1 giảm khoảng 5.5699 pp vì FP tăng mạnh. Phân biệt tính toán quan sát với giả thuyết nguyên nhân ngôn ngữ.

Tiếp theo đọc một tier và một cardinality group. Hỏi lại: tier phân nhóm nhãn hay phân nhóm comment? N trong bảng là số comment hay số positives? Bảng hiện đang all28 hay emotion27?

### Tính chênh lệch RQ4 và giải thích đối chứng

Dùng Macro-F1 test all28 ở fixed 0.5. Từ văn bản gốc {{test_fixed_macro}}, bỏ dấu còn {{rq4_diacritics_macro}} và lỗi gõ QWERTY còn {{rq4_typo_macro}}. Lấy điểm từng điều kiện trừ điểm gốc và nhân 100. Đáp án là giảm {{rq4_diacritics_drop_pp}} và {{rq4_typo_drop_pp}} điểm phần trăm.

Người hỏi đổi tình huống: “Nếu nhóm hạ ngưỡng riêng cho mỗi loại nhiễu thì so sánh còn trả lời cùng câu hỏi không?” Giải thích rằng kết quả khi đó gộp ảnh hưởng của đổi văn bản và đổi ngưỡng. Sau đó nêu một lý do không thể suy rộng mức giảm thành so sánh của hai dạng nhiễu ở cùng độ nặng.

## 5. Mock defence buổi 1 — kiến thức và pipeline

Thời gian gợi ý 20–30 phút mỗi người. Một người hỏi, một người trả lời, người còn lại nếu có chỉ ghi lỗi; đổi vai sau mỗi lượt. Chưa cần thuộc đáp án tiếng Anh dài.

- Nói bài mở đầu 60 giây, không đọc report.
- Vẽ pipeline từ text tới target/logits/probabilities/decisions. Ghi kích thước một batch.
- Giải thích sigmoid, BCE, full fine-tuning, eval mode và frozen Final checkpoint.
- Làm bài tập metric và decoding; người hỏi đổi một target hoặc cutoff rồi yêu cầu tính lại.
- Mở đúng function trong code map, chỉ đầu vào/đầu ra và một điều kiện bảo vệ.

Chấm mỗi mục 0–2: 0 là chưa giải thích được; 1 là đúng ý chính nhưng cần nhắc; 2 là tự giải thích và xử lý ví dụ đổi. Mục nào chưa đạt 2 thì ôn lại, không chỉ sửa câu trả lời thuộc lòng.

## 6. Mock defence buổi 2 — kết quả và phản biện

Thời gian gợi ý 20–30 phút mỗi người; đây là bài tập luyện, không phải rubric chính thức.

- Trình bày kết quả all28 và một kết quả không thuận lợi cho giả thuyết ban đầu.
- Trả lời ba phản biện liên tiếp: vì sao 28; vì sao epoch15; vì sao không tune lại test để cứu per-label.
- Giải thích gap âm, tail decline và precision/recall trade-off của Neutral.
- Trình bày RQ4 trong 30 giây: đầu vào đổi ở đâu, đối chứng giữ gì, kết quả chính và một giới hạn.
- Nêu một điều artifacts chứng minh được và một điều chưa chứng minh được.
- Mỗi người chỉ ra đóng góp thật của mình, cách đã kiểm tra và điều còn chưa chắc chắn.

Người hỏi chọn 5 trong 34 câu Q&A, gồm ít nhất một câu RQ4. Mỗi câu thử một đáp án 20–40 giây rồi xử lý một câu hỏi đổi điều kiện. Có thể viết hoặc vẽ sơ đồ khi luyện. Quy định sử dụng tài liệu khi thi theo thông báo của giảng viên.

## 7. Phiếu tự đánh giá

| Mục cần tự kiểm | Mức hiện tại / việc sửa |
| --- | --- |
| Nói đúng multi-label, 28 và mixed Neutral | Điền sau buổi luyện |
| Trace một batch và training step | Điền sau buổi luyện |
| Tính được Macro/Micro/Hamming và decoding | Điền sau buổi luyện |
| Nhớ đúng epoch criterion và tuning objective | Điền sau buổi luyện |
| Đọc đúng scope, support, pp và gap | Điền sau buổi luyện |
| Phân biệt đổi ngưỡng ở RQ1 với đổi đầu vào ở RQ4 | Điền sau buổi luyện |
| Phản biện mà không thêm claim vượt evidence | Điền sau buổi luyện |
| Mở đúng file để chứng minh đóng góp | Điền sau buổi luyện |

## 8. Checklist offline trước buổi bảo vệ

Tải các PDF ôn và report Final xuống máy. Mở CSV/JSON bằng công cụ đọc offline; không cần GPU/Internet để giải thích kết quả. Chuẩn bị một ví dụ tự tạo, bảng metric và code map. Không chạy lại training hoặc infer-test trong buổi ôn để tạo thêm kết quả.

Xác nhận với thông báo thật về thời gian, ngôn ngữ, thứ tự thành viên và tài liệu được phép dùng. Không tự coi lịch cuối tháng hoặc thời lượng mock trong tài liệu là lịch chính thức.
