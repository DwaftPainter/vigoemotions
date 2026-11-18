# ViGoEmotions — Kiến thức nền để bảo vệ

Tài liệu giải thích kiến thức dùng trong dự án. Các câu minh họa tự tạo, không phải raw comments của dataset. Cấu hình và kết quả cụ thể đọc ở tài liệu 02.

## 1. Đầu vào, target và tensor

Một dòng là một bình luận tiếng Việt và một tập nhãn. Multi-label cho phép nhiều nhãn cùng dương: câu tự tạo “Mình vui và biết ơn vì được giúp” có thể mang joy và gratitude. Target là vector multi-hot: một số vị trí bằng 1, các vị trí còn lại bằng 0.

Trong Final, một batch có B mẫu tạo logits kích thước B × 28. Target cũng B × 28. PhoBERT nhận token IDs và attention mask kích thước B × 256 với padding/truncation của cấu hình này. IDs là chỉ số token, không phải xác suất. Attention mask phân biệt token thật và padding; nó không phải nhãn cảm xúc.

Một câu có thể có nhiều nhãn, nên xác suất 28 outputs không bắt buộc cộng thành 1. Multi-class khác ở chỗ chỉ chọn một lớp trong các lớp loại trừ nhau.

## 2. Logit, sigmoid và BCE

Logit z là điểm thực do model tạo. Sigmoid chuyển riêng từng logit thành p trong khoảng 0–1:

```text
p = 1 / (1 + exp(-z))
loss(y,p) = -y*log(p) - (1-y)*log(1-p)
prediction = (p >= cutoff)
```

Nếu y=1, loss phạt p nhỏ; nếu y=0, loss phạt p lớn. Với một hạng loss không có trọng số, đạo hàm theo logit là p−y; khi lấy mean trên nhiều mẫu/nhãn còn có hệ số chuẩn hóa. Loss hướng mô hình tăng score cho nhãn dương và giảm score cho nhãn âm.

BCEWithLogitsLoss nhận **logits**, gộp sigmoid và BCE theo cách ổn định số. Không sigmoid trước rồi đưa vào BCEWithLogitsLoss. Khi inference mới lấy sigmoid để lưu scores và áp cutoff. Trong training của nhóm, BCE không có positive-class weights; đừng mô tả thành weighted BCE của paper gốc.

Outputs được tối ưu từng nhãn nhưng neural backbone dùng chung biểu diễn. Vì vậy, “independent sigmoid” không có nghĩa các cảm xúc trong thực tế độc lập thống kê hoặc model không học được bất cứ tín hiệu liên quan nào.

## 3. Tiền xử lý và tránh leakage

NFC giúp các chuỗi Unicode có cách biểu diễn dấu nhất quán; gom whitespace loại khoảng trắng thừa. Không tự xóa phủ định, dấu tiếng Việt hoặc emoji vì chúng có thể mang tín hiệu cảm xúc.

TF-IDF fit vocabulary/IDF trên train; BiGRU học vocabulary/embedding từ train. PhoBERT dùng tokenizer pretrained và VnCoreNLP word segmentation. Không fit IDF hoặc vocabulary mới trên validation/test. Validation dùng chọn checkpoint và thresholds; test chỉ dùng đánh giá đã khóa.

Train nguồn có một mẫu trùng văn bản held-out bị loại. Kiểm tra overlap và phân phối nhãn là bằng chứng kiểm dữ liệu, nhưng không chứng minh thuật toán stratification gốc của supplied split. Không tự chia lại dữ liệu sau khi thấy test score.

## 4. TF-IDF + One-vs-Rest Logistic Regression

TF mô tả mức xuất hiện của token/ngram trong văn bản. IDF giảm trọng số của ngram xuất hiện ở quá nhiều văn bản. Với smooth IDF thường dùng trong pipeline này, ý tưởng là log((1+N)/(1+df))+1; sublinear TF thay số lần xuất hiện dương bằng 1+log(tf). Unigram và bigram giúp baseline dùng từ hoặc cặp từ gần nhau.

Vector sparse x được đưa vào một classifier Logistic Regression cho mỗi nhãn: z_c = w_c·x+b_c. OvR nghĩa là nhiều bài toán nhị phân dùng chung input features. Đây là baseline có khả năng học từ khóa/ngram, không phải chỉ đoán theo nhãn phổ biến.

Ưu điểm: nhanh, dễ kiểm tra, hợp làm mốc so sánh. Giới hạn: khó biểu diễn ngữ cảnh dài, mỉa mai hoặc thay đổi ý nghĩa theo câu. Không gọi TF-IDF có “20 epoch giống neural”: budget của solver và epoch neural không tương đương.

## 5. BiGRU học từ đầu

Text được lower-case, tách whitespace và đổi token thành ID. Embedding học một vector cho mỗi token. GRU dùng cổng để cập nhật trạng thái theo chuỗi; bản bidirectional đọc cả chiều thuận và nghịch. Hai trạng thái cuối được nối, qua dropout rồi linear head tạo logits.

Ví dụ kích thước: B × L token IDs → B × L × d embeddings → B × 2h biểu diễn câu → B × C logits. Trong cấu hình nhóm, embedding d=128 và hidden h=128; C là 27 hoặc 28 theo study. Packed sequences giúp GRU xử lý độ dài thật thay vì xem padding như nội dung.

BiGRU khai thác thứ tự tốt hơn bag-of-ngrams nhưng học biểu diễn từ tập train của dự án. Không nói nó dùng pretrained embedding nếu code không có bước đó.

## 6. PhoBERT và full fine-tuning

PhoBERT đã học biểu diễn tiếng Việt từ pretraining. Transformer self-attention kết hợp thông tin giữa các token để tạo biểu diễn theo ngữ cảnh. Word segmentation tạo dạng đầu vào phù hợp với PhoBERT; BPE tiếp tục chia thành subword units. Một từ tiếng Việt, một token whitespace và một BPE token không phải cùng một đơn vị.

Nhóm dùng PhoBERT-base-v2, fine-tune cả backbone và head trên task multi-label. Full fine-tuning khác frozen feature extraction: ở Midterm, backbone có gradient và được optimizer cập nhật. **Frozen checkpoint trong Final** nghĩa là dừng cập nhật weights đã chọn khi chỉ so decoder; không có nghĩa backbone bị đóng băng trong suốt training Midterm.

Checkpoint revision được pin để xác định model artifact. Pretrained weights là điểm khởi đầu, không phải weights của model sau fine-tuning. Saved checkpoint, tokenizer và preprocessing phải đi cùng nhau khi tái tạo inference.

## 7. Một training step diễn ra thế nào?

```text
batch -> zero_grad -> forward logits -> BCEWithLogitsLoss
-> backward -> gradient clipping -> optimizer.step
end of epoch -> eval + no_grad -> validation scores -> select checkpoint
```

Trong code thực tế, zero_grad được gọi trước forward/loss của batch. Backward tính gradient, optimizer.step mới cập nhật tham số. Gradient clipping giới hạn norm để giảm rủi ro gradient quá lớn; dropout ngẫu nhiên tắt một số activations lúc train. eval mode tắt dropout và no_grad không lưu graph gradient khi dự đoán.

Batch là một nhóm mẫu trong một bước; epoch là một lượt qua train. Learning rate điều khiển độ lớn bước cập nhật; weight decay điều chuẩn tham số. Training 20 epoch rồi chọn best epoch khác dừng sớm ở epoch 15. Nhóm đã train đủ 20 và dùng best checkpoint; Final không train thêm.

## 8. Metric: phải nói được tính gì và đánh đổi gì

| Khái niệm | Công thức / ý nghĩa |
| --- | --- |
| TP | Nhãn thật dương và dự đoán dương |
| FP | Nhãn thật âm nhưng dự đoán dương |
| FN | Nhãn thật dương nhưng dự đoán âm |
| Precision | TP / (TP+FP): bao nhiêu dự đoán dương là đúng |
| Recall | TP / (TP+FN): tìm lại được bao nhiêu nhãn dương |
| F1 | 2TP / (2TP+FP+FN) |
| Macro-F1 | Tính F1 mỗi nhãn rồi lấy trung bình trên label scope đã khai báo |
| Micro-F1 | Cộng TP/FP/FN của mọi nhãn rồi mới tính F1 |
| Hamming Loss | (FP+FN) / (N×C), mỗi cặp mẫu–nhãn là một quyết định |
| LRAP | Đánh giá nhãn đúng được xếp cao tới đâu trong scores của mỗi mẫu |
| Subset accuracy | Chỉ đúng khi toàn bộ tập nhãn dự đoán khớp hoàn toàn |

Macro cho mỗi nhãn trọng số như nhau; micro bị chi phối nhiều hơn bởi các nhãn có nhiều positives. Hamming có thể nhỏ trong dữ liệu sparse dù bỏ sót nhiều positives, nên phải đọc cùng F1. F1 là trung bình điều hòa precision/recall, không phải lấy trung bình cộng hai số này.

Không có positive support có thể khiến F1 không xác định theo công thức; study đặt zero_division=0. Nhóm cardinality không có mẫu thì báo n=0 và N/A, không tự tạo score. Neutral được chấm positive-class F1; đừng tính macro binary gồm cả lớp âm rồi gọi là F1 của Neutral.

## 9. Threshold tuning không phải probability calibration

Đổi cutoff chỉ đổi decision từ cùng scores. Probability calibration, theo nghĩa dự báo xác suất phù hợp tần suất thực nghiệm, thường cần kiểm tra khác; nghiên cứu này không chứng minh điều đó bằng sweep F1.

Hạ threshold làm tập dự đoán dương rộng hơn hoặc giữ nguyên trên cùng scores: TP/FP không giảm, FN không tăng. Recall không giảm nhưng precision và F1 có thể tăng hoặc giảm. Tăng threshold có hướng ngược lại. LRAP giữ nguyên khi chỉ đổi cutoff vì thứ hạng scores không đổi.

Per-label có 28 lựa chọn ngưỡng thay vì một scalar nên linh hoạt hơn trên validation. Tính linh hoạt có thể chọn theo nhiễu của split, nhất là support ít; đây là một giải thích khả dĩ, chưa phải cơ chế đã được kiểm chứng nhân quả trong dự án.
