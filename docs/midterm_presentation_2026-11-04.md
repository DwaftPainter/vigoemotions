# Lời dẫn midterm — hoàn thiện ngày 05/11/2026, giữ kết quả Neutral 04/11

Dùng cùng [20 slide PDF](../reports/midterm_slides.pdf) và [báo cáo PDF](../reports/midterm_report.pdf). Dự kiến 12–15 phút, điều chỉnh theo thời lượng giảng viên cấp; dành khoảng 4–5 phút cho slide 13–17 về kết quả mới. [Bảng nguồn và kiểm chứng](../reports/midterm_2026-11-04/evidence/README.md) giúp trả lời khi được hỏi số liệu lấy từ đâu. Điền tên, MSSV và phân công nhóm trước khi nộp.

## Lời dẫn từng slide

### 1. Bài toán và cập nhật chính

“Nhóm thực hiện nhận diện cảm xúc đa nhãn trên bình luận tiếng Việt: một câu có thể vừa vui vừa buồn, nên không ép mô hình chỉ chọn một lớp. Nhóm đã có ba baseline 27 cảm xúc và vừa hoàn tất thí nghiệm mở rộng thành 28 nhãn có Neutral. Trong study mới, PhoBERT đạt Macro-F1 0,5747 trên validation; phần sau sẽ giải thích cả kết quả lẫn giới hạn của phép so sánh.”

### 2. Dữ liệu và chất lượng

“Dữ liệu có 20.664 mẫu chia sẵn thành train, validation và test. Nhóm phát hiện một văn bản train trùng test, nên loại bản train và còn 16.530 mẫu huấn luyện. Một bất nhất cần chú ý là 306 mẫu có cả Neutral và cảm xúc. Nhóm ghi nhận rõ hai cách mã hóa thay vì tự kết luận annotation nào sai.”

### 3. Phân bố 27 cảm xúc

“Đây là phân bố trên train của 27 cảm xúc, chưa cộng Neutral. Amusement phổ biến nhất, relief ít nhất; chênh lệch khoảng 4,52 lần. Nhóm chia ba tier theo thứ hạng train để phân tích. Vấn đề đáng chú ý là số nhãn dương trên mỗi câu khá thưa so với số đầu ra, nên chỉ nhìn tỷ lệ quyết định đúng có thể che mất việc bỏ sót nhãn.”

### 4. Cardinality và độ dài

“Trung bình một mẫu train có 1,859 cảm xúc, nhiều nhất 5. Median độ dài là 11 token tách bằng khoảng trắng. Đây không phải số từ tiếng Việt hay số BPE token; mỗi mô hình có tokenizer và giới hạn độ dài riêng. Các số này mô tả 27 cảm xúc; nếu tính Neutral là đầu ra thứ 28 thì cách đếm cardinality thay đổi.”

### 5. Các nhãn thường đi cùng

“Cặp disappointment–sadness xuất hiện cùng nhau 904 lần trên train, anger–annoyance là 699. Nhóm tính cả đồng xuất hiện và tương quan phi, vì một cặp xuất hiện nhiều có thể do từng nhãn vốn phổ biến. Những thống kê này chỉ xây từ train và không thể hiện quan hệ nhân quả.”

### 6. Chia tập và preprocessing

“Nhóm giữ split được cấp, fit từ vựng và IDF chỉ trên train. Chuẩn hóa Unicode và khoảng trắng nhưng giữ dấu, phủ định và emoji ở bước chung. PhoBERT có tách từ VnCoreNLP trước BPE. Notebook 07 xác nhận fixed IDs khớp và không overlap sau xử lý; lệch prevalence lớn nhất là 1,6324 điểm phần trăm ở validation, 0,9387 ở test. Đây là audit chất lượng phân bố, chưa xác nhận thuật toán chia gốc. Test chưa được dùng để chọn mô hình hay ngưỡng.”

### 7. Ba baseline gốc

“Ba baseline là TF-IDF cộng Logistic Regression theo One-vs-Rest, BiGRU học từ đầu và PhoBERT pretrained được fine-tune toàn bộ. Cấu hình gốc chạy BiGRU 10 epoch, PhoBERT 4 epoch. Neural có các logits độc lập, dùng BCE rồi sigmoid, phù hợp bài toán đa nhãn. Checkpoint được chọn trên validation Macro-F1 tại ngưỡng 0,5.”

### 8. Kết quả baseline 27 nhãn

“Slide này và các phân tích tới slide 12 dùng bộ baseline gốc 14/10: Macro-F1 lần lượt 0,2388, 0,4060 và 0,4393. Đây là số lịch sử để đọc các phân tích Q1–Q5 đã có. Nhóm cũng có một lượt chạy lại 27 nhãn dùng trong buổi 23/10 với PhoBERT 0,4679, lưu riêng; không thay lẫn hai lượt khi tính delta.”

### 9. Nhãn tốt và khó

“Với TF-IDF gốc tại 0,5, gratitude có F1 cao nhất khoảng 0,6797, còn realization bằng 0. Tương quan tần suất với F1 chưa đủ mạnh để nói chỉ cần nhiều dữ liệu là nhãn sẽ dễ. Nhận xét này riêng cho TF-IDF và ngưỡng đang xét, không tự chuyển thành kết luận cho PhoBERT 28 nhãn.”

### 10. Độ nhạy của ngưỡng

“Giữ nguyên xác suất TF-IDF và khảo sát ngưỡng trên validation, ngưỡng 0,15 cho Macro-F1 0,4844 thay vì 0,2388. Tuy nhiên Hamming Loss tăng, nghĩa là có đánh đổi. Đây là khảo sát trên chính validation dùng chọn ngưỡng, chưa phải điểm test và chưa chứng minh 0,15 phù hợp với mô hình khác.”

### 11. Khả năng giữ cặp nhãn

“Trong 122 mẫu validation có cả disappointment và sadness, TF-IDF tại 0,5 chỉ đoán đủ cặp ở 2 mẫu. Con số này cho thấy bỏ sót đồng thời, nhưng chưa tách được vấn đề ngưỡng khỏi khả năng học phụ thuộc nhãn. Cặp được chọn từ train trước rồi mới đánh giá trên validation.”

### 12. Phong cách và lỗi biên

“Nhóm phân tích các nhóm ngắn, nhiều emoji và teencode bằng heuristic; chúng có thể chồng lấn. Baseline PhoBERT hơn BiGRU nhiều nhất ở nhóm teencode trong các nhóm khảo sát. Nhóm đã đọc thủ công 10 bình luận lỗi TF-IDF gốc; danh sách lỗi neural tự sinh chưa có nghĩa đã được đọc. Những phân tích này không phải phân tích style mới của study 28.”

### 13. Neutral là đầu ra riêng

“Ở baseline cũ, Neutral-only trở thành vector cảm xúc toàn 0. Study mới giữ ID 27 thành một nhãn học được. Ví dụ nhãn `[2,27]` giữ cả joy và Neutral, nhất quán với annotation hiện có dù có mâu thuẫn với guideline loại trừ. Validation có 75 Neutral-only và 19 mixed, tổng cộng 94 Neutral dương. Nhóm chưa áp luật decoder buộc Neutral loại trừ cảm xúc.”

### 14. Kết quả study 28 nhãn

“Ba mô hình đã chạy xong trên cùng 2.066 mẫu validation. Macro-F1 trên đủ 28 nhãn là 0,2310 cho TF-IDF, 0,3992 cho BiGRU và 0,5747 cho PhoBERT. PhoBERT đứng đầu trong study theo cả F1, LRAP và Hamming. Nhóm đã tính lại số liệu từ xác suất đã lưu. Các số này chưa cho biết hiệu quả trên test và không so trực tiếp với Macro-F1 trên 27 nhãn.”

### 15. Chất lượng nhận diện Neutral

“Neutral F1 của ba mô hình lần lượt 0,0208, 0,0877 và 0,4400. Với PhoBERT, trong 94 Neutral thật, mô hình tìm được 44 và bỏ sót 50; ngoài ra có 62 dự đoán Neutral sai. Vì vậy Neutral đã có thể được dự đoán trực tiếp nhưng vẫn còn khó. Một tập dự đoán rỗng không đồng nghĩa Neutral, và baseline không có đầu ra Neutral thì metric Neutral của nó không xác định.”

### 16. So sánh trên cùng 27 cảm xúc

“Để giữ cùng tập nhãn, nhóm lấy 27 cột cảm xúc từ đầu ra mới rồi tính lại. TF-IDF gần như giữ nguyên, BiGRU từ 0,4060 lên 0,4108, PhoBERT từ 0,4393 lên 0,5796. Tuy nhiên PhoBERT mới được chạy 12 epoch thay vì 4, checkpoint cũng chọn theo tập nhãn khác và runtime khác. Đây là khác biệt quan sát được, chưa phải bằng chứng thêm Neutral gây ra mức tăng đó. Trên phép chiếu 27, Hamming của PhoBERT còn tăng nhẹ dù F1 tăng.”

### 17. Đường học và thực nghiệm đã hoàn tất đến đâu

“PhoBERT chọn epoch 11 với Macro-F1 28 nhãn 0,5747; epoch 12 giảm còn 0,5630 dù train loss tiếp tục giảm. Chúng ta chưa có validation loss nên không kết luận overfitting chỉ từ đây. Study này chạy một learning rate 2e-5, chưa phải sweep notebook 05. Lượt nguồn từng lỗi JVM ở cell tìm ví dụ; bản mã ngày 05/11 đã dùng xác suất lưu sẵn để tránh nạp lại JVM. Không chạy lại training GPU; metric giữ nguyên.”

### 18. Câu hỏi Final

“Nhóm muốn kiểm tra một ngưỡng chung 0,5 có phù hợp cho mọi nhãn không. Chọn và khóa một checkpoint neural, giữ trọng số rồi so sánh ngưỡng cố định, ngưỡng chung tune và ngưỡng riêng từng nhãn trên validation. Sau khi khóa mọi lựa chọn mới đánh giá test. Nếu dùng 28 nhãn, cần đọc thêm Neutral riêng và phép chiếu 27 để các kết luận có cùng ý nghĩa.”

### 19. Việc còn lại

“Nhóm đã có checkpoint, tokenizer và vocabulary của study 28 trong ZIP riêng tư. Đã bổ sung chín ca đọc lỗi PhoBERT hỗ trợ AI và phụ lục per-label, nhóm cần xác nhận. Còn thiếu thông tin nhóm và xác nhận yêu cầu official split. Nếu muốn kết luận tác động riêng của Neutral thì cần đối chứng 27/28 cùng điều kiện; sweep learning rate vẫn chưa có kết quả. Notebook/gói ZIP chia sẻ đã xóa output, không kèm lịch sử Git cũ có dữ liệu nguồn. Các giới hạn này được ghi trong báo cáo thay vì trình bày như đã hoàn tất.”

### 20. Nguồn và tái lập

“Đề môn học, paper dataset, PhoBERT và các thư viện có trong danh mục tham khảo. Seed, cấu hình, revision, môi trường và checksum được lưu cùng kết quả. Những điểm trình bày là số chạy của nhóm trên validation; điểm trong paper không được dùng thay kết quả của nhóm.”

## Câu hỏi dễ gặp

| Câu hỏi | Câu trả lời cần giữ đúng |
| --- | --- |
| Thêm Neutral làm PhoBERT tốt hơn 14 điểm? | Chưa chứng minh. +14,0321 điểm phần trăm là delta quan sát trên cùng 27 cảm xúc so với baseline gốc; còn khác ngân sách 4/12 epoch, quy tắc chọn checkpoint và runtime. |
| Vì sao 0,5747 khác 0,5796? | 0,5747 tính cả 28 nhãn; 0,5796 chỉ chiếu về 27 cảm xúc, bỏ cột Neutral khi chấm. Cả hai dùng cùng checkpoint mới. |
| Neutral support là 75 hay 94? | 75 Neutral-only + 19 mixed = 94 mẫu có Neutral thật trên validation. |
| Sao có Neutral đi cùng cảm xúc? | Annotation có mixed. Study giữ nguyên để đo, công bố bất nhất và chưa đặt luật sửa nhãn/decoder loại trừ. |
| Baseline là 0,4393 hay 0,4679? | Hai lượt khác nhau: artifact gốc 14/10 là 0,4393, lượt chạy lại dùng ngày 23/10 là 0,4679. Study 06 đối chiếu bộ gốc 0,4393. |
| Đã tune learning rate chưa? | Study 06 dùng lr=2e-5; chạy 12 epoch không hoàn tất grid learning rate của notebook 05. |
| Lỗi notebook có làm metric sai không? | Lượt nguồn từng lỗi JVM sau khi xuất metric; đã sửa riêng cell đọc ca bằng NPZ và kiểm chứng offline. Không gọi đó là chạy lại training GPU; metric giữ nguyên. |
| Kết quả đã qua test chưa? | Chưa. Test chỉ tham gia audit dữ liệu/trùng lặp; `RUN_FINAL_TEST=False`, chưa có metric test mô hình trong ZIP. |
| Có bảo đảm kết quả lặp lại y hệt không? | Có seed/config/revision và artifact để kiểm chứng; không bảo đảm bitwise giữa GPU/package khác nhau. Một seed chưa đo được độ biến thiên. |

## Mở sẵn để trình bày

1. [Slide PDF](../reports/midterm_slides.pdf).
2. [Bảng 28 nhãn](../reports/midterm_2026-11-04/evidence/metrics_summary.csv) và [Neutral riêng](../reports/midterm_2026-11-04/evidence/neutral_analysis.csv).
3. [Báo cáo](../reports/midterm_report.pdf), phần 5.7; [nguồn phần cập nhật](../reports/midterm_2026-11-04/neutral28_results.md).
4. [Kiểm chứng](../reports/midterm_2026-11-04/evidence/provenance.json) và [run PhoBERT mới](../reports/midterm_2026-11-04/evidence/models/phobert/run.json).
