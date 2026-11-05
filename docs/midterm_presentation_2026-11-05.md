# Lời dẫn midterm — kết quả Kaggle 20 epoch

> **Ôn tập bằng tiếng Việt:** Slide chính đã chuyển sang tiếng Anh. Khi thuyết trình, dùng [script tiếng Anh và Q&A](midterm_presentation_en.md) hoặc [PDF luyện nói](midterm_presentation_en.pdf). Bản này giữ để nhóm ôn nội dung.

Dùng [20 slide PDF](../reports/midterm_slides.pdf) và [report](../reports/midterm_report.pdf). Dự kiến 12–15 phút tùy thời lượng giảng viên. Mỗi số model lấy từ hai ZIP 08/09 equal20; không live train trong buổi thuyết trình. Tên/MSSV/phân công còn cần bổ sung.

## Slide 1. ViGoEmotions | Midterm 20 epoch

- Nhận diện cảm xúc tiếng Việt đa nhãn.
- Ba model × hai bộ nhãn: 27 và 28 gồm Neutral.
- Kaggle 08/09 đã hoàn tất; neural đều 20 epoch.
- PhoBERT 27 Macro-F1 0.5974; 28: 0.5783.

Lời dẫn: “Nhóm nghiên cứu nhận diện đa nhãn cảm xúc trên bình luận tiếng Việt. Bài trình bày dùng sáu kết quả Kaggle mới, trong đó BiGRU và PhoBERT đều chạy đủ 20 epoch ở cả hai bộ nhãn. Mọi điểm là validation; test được giữ cho Final.”

## Slide 2. Dữ liệu & chất lượng

- 20.664 bình luận; split gốc 80/10/10.
- Train 16.531 → 16.530 sau loại một overlap.
- Validation 2.066; test 2.067, chưa chấm model.
- 725 Neutral-only và 306 mixed trong nguồn.

Lời dẫn: “Dữ liệu có 20.664 mẫu chia sẵn. Nhóm loại một mẫu train trùng văn bản test và còn 16.530 mẫu train. Nguồn có cả Neutral-only và Neutral đi cùng cảm xúc, nên nhóm công bố quy tắc mã hóa trước khi so sánh.”

## Slide 3. Phân bố cảm xúc

- 27 cảm xúc, chín nhãn mỗi tier theo train.
- Amusement nhiều nhất; relief ít nhất.
- Chênh lệch max/min 4.52 lần.
- Nhãn dương thưa; không dùng accuracy chính.

Lời dẫn: “Nhóm chia head, mid và tail theo thứ hạng tần suất trên train, chín nhãn mỗi tier. Nhãn dương còn thưa so với số đầu ra, vì vậy một mô hình dự đoán toàn 0 có thể có Hamming Loss thấp nhưng F1 bằng 0.”

## Slide 4. Cardinality & độ dài

- Train trung bình 1,859 cảm xúc, tối đa 5.
- Median: 11 token khoảng trắng; 47 ký tự.
- Whitespace token khác từ và BPE token.
- Khẩu ngữ, teencode, emoji, HTML.

Lời dẫn: “Một mẫu train trung bình có khoảng 1,859 cảm xúc. Median độ dài là 11 token khoảng trắng, khác số từ tiếng Việt và BPE token. Vì thế mỗi pipeline dùng cách tokenize phù hợp và ghi tỷ lệ truncation riêng.”

## Slide 5. Các nhãn đi cùng nhau

- Train: disappointment–sadness 904 mẫu.
- Anger–annoyance 699; sadness–grief 620.
- Co-occurrence và phi chỉ tính từ train.
- Tương quan không chứng minh nhân quả.

Lời dẫn: “Disappointment và sadness là cặp đồng xuất hiện nhiều nhất trong các cặp train được khảo sát. Nhóm xem cả số lần đồng xuất hiện và tương quan phi, vì đếm lớn có thể đến từ tần suất riêng của hai nhãn.”

## Slide 6. Split & tránh leakage

- Giữ split được cấp; audit 07 toàn vẹn PASS.
- Fixed IDs khớp, không overlap sau xử lý.
- Fit vocabulary/IDF chỉ trên train.
- Prevalence audit chưa chứng minh stratification.
- Test giữ cho Final.

Lời dẫn: “Nhóm giữ split được cấp và kiểm tra fixed IDs, overlap và prevalence bằng notebook 07. Chỉ train được dùng để học vocabulary và IDF. Audit cho bằng chứng phân bố, chưa xác nhận được thuật toán stratification gốc; nhóm cần làm rõ với giảng viên.”

## Slide 7. 27 hay 28 đầu ra?

- 27: Neutral-only là vector cảm xúc toàn 0.
- 28: Neutral là output ID 27 học được.
- Giữ mixed annotation; không ép loại trừ.
- Validation có 94 Neutral dương.
- Sigmoid độc lập + BCE; không softmax.

Lời dẫn: “Bản 27 coi Neutral-only là vector cảm xúc toàn 0. Bản 28 học Neutral qua một output riêng và giữ mixed annotation. Do các nhãn không loại trừ nhau, nhóm dùng sigmoid và BCE, không dùng softmax hay ép Neutral loại trừ cảm xúc.”

## Slide 8. Ba mô hình cùng profile

- TF-IDF + OvR LR: liblinear, max_iter=1000.
- BiGRU từ đầu: embed/hidden 128; batch 64.
- PhoBERT full fine-tune: lr 2e-5; batch 16.
- BiGRU/PhoBERT đều đủ 20 epoch ở cả hai bản.
- TF-IDF không có epoch neural tương đương.

Lời dẫn: “TF-IDF là baseline từ vựng với OvR Logistic Regression. BiGRU học embedding và trọng số từ đầu, còn PhoBERT full fine-tuning backbone pretrained. Cả hai neural có cùng ngân sách 20 epoch; TF-IDF dùng cùng solver và max_iter, không có epoch neural tương đương.”

## Slide 9. Checkpoint & tính tái lập

- Seed 42; ngưỡng 0,5; cùng split và profile.
- Chọn bằng emotion27 Macro-F1; hòa lấy sớm.
- Best GRU 27/28: 17/20; PhoBERT: 14/15.
- Cùng Tesla T4 và phiên bản package manifest.
- Hash khác do LF/CRLF đã được đối chiếu.

Lời dẫn: “Hai study cùng seed, split, profile và tiêu chí chọn checkpoint trên 27 cảm xúc tại ngưỡng 0,5. Nhóm chạy hết 20 epoch rồi nạp best, nên best ở epoch 14 hoặc 15 không có nghĩa chỉ train từng đó. Các chênh lệch checksum đã được đối chiếu là LF/CRLF.”

## Slide 10. Kết quả 27 cảm xúc

- TF-IDF Macro-F1 0.2388.
- BiGRU Macro-F1 0.4220.
- PhoBERT Macro-F1 0.5974; Micro 0.5996.
- PhoBERT Hamming 0.0550; LRAP 0.7419.
- Per-label, tham số và thời gian trong report.

Lời dẫn: “Với 27 cảm xúc, PhoBERT đạt Macro-F1 0.5974, cao nhất trong ba model của study. Report có thêm Micro-F1, Hamming Loss, LRAP, F1 từng nhãn và chi phí; không dùng một metric trung bình để nói model tốt nhất cho mọi nhãn.”

## Slide 11. Kết quả 28 nhãn

- TF-IDF Macro-F1 0.2310.
- BiGRU Macro-F1 0.4086.
- PhoBERT Macro-F1 0.5783; Micro 0.5829.
- PhoBERT Hamming 0.0575; LRAP 0.7077.
- Metric này tính đủ 28 nhãn, gồm Neutral.

Lời dẫn: “Với đủ 28 nhãn, PhoBERT đạt Macro-F1 0.5783 và tiếp tục đứng đầu trong study này. Đây là metric có thêm Neutral, nên không lấy trực tiếp số này trừ Macro-F1 27 để kết luận tác động của thêm nhãn.”

## Slide 12. Cùng 27 cảm xúc: Neutral có giúp?

- PhoBERT 27: 0.5974; bản 28 chiếu 27: 0.5836.
- Delta PhoBERT -1.38 điểm phần trăm.
- Delta BiGRU -0.25 điểm; TF-IDF không đổi.
- Lượt này chưa cho thấy lợi ích trên 27 cảm xúc.
- Một seed; không kết luận nhân quả/thống kê.

Lời dẫn: “Để so cùng tập nhãn, nhóm chấm lại 27 cột cảm xúc của bản 28. PhoBERT đạt 0.5836, thấp hơn bản 27 khoảng 1.38 điểm phần trăm. Lượt một seed này chưa cho thấy lợi ích trên 27 cảm xúc; chưa có kết luận nhân quả hoặc kiểm định thống kê.”

## Slide 13. Neutral riêng còn khó

- PhoBERT Neutral F1 0.4369.
- Precision 0.4018; recall 0.4787.
- 45 TP, 67 FP, 49 FN; support 94.
- Model 27 không có output Neutral.
- Dự đoán rỗng không tự động là Neutral.

Lời dẫn: “Neutral F1 của PhoBERT là 0.4369. Mô hình tìm được 45 trong 94 mẫu Neutral thật, còn 49 bỏ sót và 67 dự đoán sai. Vì model 27 không có output Neutral nên metric Neutral của nó không xác định, không tự gán bằng 0.”

## Slide 14. Đường học đủ 20 epoch

- Bốn history neural đều đủ epoch 1–20.
- Best checkpoint có thể ở giữa lịch sử.
- Epoch 1 PhoBERT 27 F1=0; sau đó cải thiện.
- Thiếu val loss: chưa chẩn đoán overfitting.

Lời dẫn: “Bốn đường học có đủ 20 epoch. PhoBERT 27 bắt đầu với F1 bằng 0 nhưng sau đó cải thiện, minh họa vì sao không kết luận thất bại từ epoch đầu. F1 sau best dao động dù train loss giảm; thiếu validation loss nên nhóm không chẩn đoán overfitting chỉ từ biểu đồ này.”

## Slide 15. Q1–Q2 | Nhãn và tần suất

- PhoBERT cao nhất: gratitude, F1 0.8370.
- Thấp nhất: disapproval, F1 0.3623.
- PhoBERT rho 0.018; p 0.928.
- Phụ lục có F1 mọi nhãn cho cả ba model.
- Tương quan mô tả, không suy ra nguyên nhân.

Lời dẫn: “Ở PhoBERT 27, gratitude có F1 cao nhất và disapproval thấp nhất. Spearman giữa tần suất train và F1 chỉ khoảng 0.018, nên dữ liệu hiện tại không hỗ trợ nhận xét đơn giản rằng nhãn càng phổ biến càng dễ. Phụ lục có kết quả của mọi nhãn và mọi model.”

## Slide 16. Q3–Q4 | Cặp nhãn và phong cách

- Disappointment–sadness: PhoBERT đúng đủ 63/122 ca.
- Joint recall 51.64% tại 0,5.
- PhoBERT hơn GRU nhiều nhất: teencode.
- Delta +22.13 điểm Macro-F1 trong nhóm.
- Style heuristic chồng lấn; chỉ dùng lượt 27 mới.

Lời dẫn: “PhoBERT dự đoán đủ disappointment–sadness ở 63 trên 122 ca validation có cả hai nhãn. Với phong cách, teencode có mức hơn BiGRU lớn nhất trong các nhóm khảo sát, khoảng 22.13 điểm Macro-F1. Các style là heuristic, có chồng lấn và nhiễu.”

## Slide 17. Q5 | Ngưỡng quyết định

- PhoBERT tại 0,5: Macro-F1 0.5974.
- Lưới val tốt nhất 0.35: 0.5978.
- Khảo sát lại từ NPZ mới; giữ checkpoint.
- Không thay điểm chính hoặc xem là điểm test.
- Động lực cho RQ Final về decoder.

Lời dẫn: “Trên checkpoint PhoBERT 27 cố định, lưới ngưỡng validation cho điểm cao nhất ở 0.35, Macro-F1 0.5978. Mức tăng so với 0,5 rất nhỏ và Hamming Loss xấu hơn. Đây là khảo sát hậu nghiệm, bảng chính vẫn giữ 0,5 và chưa có kết quả test.”

## Slide 18. Đọc lỗi từ kết quả mới

- Sáu ca PhoBERT 28: ba Neutral FP, ba FN.
- Khó: cảm xúc hàm ý, câu ngắn, HTML, từ tục.
- Bảng giữ ID/nhãn/xác suất, không có raw text.
- Nhận xét có hỗ trợ AI; nhóm cần xác nhận.
- Ứng viên lỗi tự sinh chưa là audit thủ công.

Lời dẫn: “Nhóm đọc sáu ca Neutral FP/FN của model mới. Câu ngắn, cảm xúc hàm ý, HTML và từ tục gợi ra hướng điều tra, nhưng đây là giả thuyết từ các ca có chủ đích. Phần đọc có hỗ trợ AI, nhóm cần xác nhận; danh sách lỗi tự sinh không phải audit thủ công đã hoàn tất.”

## Slide 19. RQ Final & việc cần chốt

- Khóa checkpoint neural; không retrain decoder.
- So fixed 0,5 / global tuned / per-label tuned.
- Tune validation → khóa → chấm test một lần.
- Đọc ít nhất 20 FN/20 FP; phân tích tier/cardinality.
- Bổ sung tên/MSSV/phân công, xác nhận split.

Lời dẫn: “Final sẽ kiểm tra một ngưỡng chung có phù hợp cho mọi nhãn không. Nhóm khóa checkpoint neural, giữ nguyên backbone, chọn global hoặc per-label thresholds trên validation rồi mới chấm test sau khi khóa lựa chọn. Trước khi nộp cần bổ sung thông tin nhóm và xác nhận giới hạn split.”

## Slide 20. Nguồn và kết luận

- PhoBERT tốt nhất trong từng study mới.
- Thêm Neutral chưa giúp F1 trên 27 cảm xúc.
- Nguồn: hai ZIP và notebook 08/09 executed.
- Đã kiểm chứng sáu NPZ, history và checkpoint.
- Đề môn, paper ViGoEmotions, PhoBERT, thư viện.

Lời dẫn: “Kết quả chính cho thấy PhoBERT đứng đầu trong từng study, còn thêm Neutral chưa cải thiện emotion27 ở lượt này. Mỗi kết luận có NPZ, history và provenance từ hai ZIP Kaggle. Nhóm công bố rõ một seed, validation dùng chọn checkpoint và test chưa chấm.”

## Câu hỏi cần trả lời chính xác

| Câu hỏi | Câu trả lời |
| --- | --- |
| Vì sao cùng 20 epoch nhưng best 14/15? | Chạy đủ ngân sách rồi nạp checkpoint có emotion27 Macro-F1 validation cao nhất; best không nhất thiết là epoch cuối. |
| Thêm Neutral có giúp không? | Trong lượt này, PhoBERT 28 chiếu 27 đạt 0,583580, thấp hơn PhoBERT 27 đạt 0,597410. Chưa thấy lợi ích trên 27 cảm xúc; một seed chưa cho kết luận nhân quả/thống kê. |
| Sao 0,578341 và 0,583580 khác nhau? | Cùng checkpoint PhoBERT 28: một số chấm đủ 28 nhãn, số còn lại chấm 27 cột cảm xúc. |
| Neutral F1 bao nhiêu? | 0,4369; 45 TP, 67 FP, 49 FN trên 94 mẫu validation có Neutral. Model 27 không có Neutral output. |
| Vì sao source/data hash khác? | CRLF ở lượt 28; từng hash đã đối chiếu chính xác với biến thể LF/CRLF của cùng nội dung local. Config đã parse và fixed IDs khớp; manifest gốc giữ nguyên. |
| Split đã stratified chưa? | Có audit toàn vẹn/prevalence, chưa xác minh thuật toán split gốc; còn xác nhận việc chấp nhận official split. |
| Epoch đầu F1=0 có lỗi không? | PhoBERT 27 epoch 1 chưa có true positive ở 0,5; về sau đạt best 0,597410. Không kết luận toàn lượt lỗi từ epoch đầu. |
| Ngưỡng PhoBERT 0,35 có phải kết quả chính? | Không. Đó là khảo sát hậu nghiệm validation trên checkpoint cố định; bảng chính vẫn tại 0,5. |
| Đã chấm test/tune learning rate chưa? | Chưa; 20 epoch của một learning rate không phải grid search. Test giữ cho Final sau khi khóa mọi lựa chọn. |
| Model có chạy hai GPU không? | Kaggle hiển thị T4 x2 nhưng code dùng một GPU mỗi process, không có phân tán/DataParallel. |

Mở sẵn model_comparison.csv, paired_emotion27_delta.csv, neutral_metrics.csv, bốn run JSON và provenance trong reports/kaggle_equal20_2026-11-05/ để trả lời nguồn số liệu. Các nhận xét đọc lỗi có hỗ trợ AI cần nhóm xác nhận.
