# ViGoEmotions — Nội dung thuyết trình giữa kỳ

## 1. ViGoEmotions | Midterm

- Nhận diện đa nhãn: một bình luận có thể mang nhiều cảm xúc.
- 27 cảm xúc; neutral là vector 0.
- Bằng chứng hiện có: ba mô hình trên validation.

Ghi chú thuyết trình: Mở đầu bằng bài toán, phạm vi và trạng thái thực nghiệm; không trình bày neural như đã chạy nếu chưa có số liệu.

## 2. Dữ liệu & kiểm tra chất lượng

- 20.664 mẫu; split gốc 16.531 / 2.066 / 2.067.
- Loại 1 mẫu train trùng test → train 16.530.
- 306 mẫu mixed-neutral; 1 mẫu lặp nhãn.
- Giữ dữ liệu nguồn riêng; chỉ xuất số liệu và ID.

Ghi chú thuyết trình: Giải thích lỗi đếm mixed-neutral trong notebook cũ và quy ước 27 output.

## 3. Phân bố 27 nhãn

- 9 nhãn mỗi tier theo tần suất train.
- Amusement cao nhất; relief thấp nhất.
- Chênh lệch max/min: 4.52 lần.
- Nhãn dương thưa hơn vấn đề chênh lệch giữa lớp.

Ghi chú thuyết trình: Tier tương đối; không nói tail chỉ có rất ít mẫu. Bảng 27 nhãn đầy đủ có trong báo cáo.

## 4. Cardinality & độ dài

- Train: 1.859 cảm xúc/mẫu.
- Cardinality tối đa 5; có neutral-only.
- Median: 11 token khoảng trắng, 47 ký tự.
- Whitespace token ≠ từ tiếng Việt ≠ BPE token.

Ghi chú thuyết trình: Nói rõ đơn vị token và vì sao phải đo tỷ lệ truncation riêng khi chạy neural.

## 5. Các cảm xúc thường đi cùng nhau

- Disappointment–sadness: 904 mẫu train.
- Anger–annoyance: 699.
- Sadness–grief: 620.
- Ma trận chỉ xây trên train; phi không hàm ý nhân quả.

Ghi chú thuyết trình: Đếm đồng xuất hiện và tương quan trả lời hai câu hỏi khác nhau.

## 6. Split & preprocessing

- Giữ split đã cấp; fit vocab/IDF trên train.
- NFC + gom khoảng trắng; giữ dấu và phủ định.
- GRU: whitespace; PhoBERT: VnCoreNLP + BPE.
- Có audit prevalence; chưa chứng minh split gốc được stratified.
- Test giữ cho Final; midterm phân tích validation.

Ghi chú thuyết trình: Phân biệt dữ liệu mô tả test với sử dụng test chọn mô hình. Nêu giới hạn nguồn thuật toán split một cách rõ ràng.

## 7. Ba baseline theo đề

- TF-IDF (1–2 grams) + OvR Logistic Regression.
- BiGRU: embedding 128, hidden 128 mỗi hướng, 10 epoch.
- PhoBERT base-v2: revision cố định, 4 epoch.
- Neural: 27 logits + BCE; sigmoid khi dự đoán.
- Chọn checkpoint theo validation Macro-F1 ở 0,5.

Ghi chú thuyết trình: Không dùng softmax. Nêu trạng thái Colab của hai neural model.

## 8. Kết quả baseline | validation

- TF-IDF Macro-F1: 0.2388; Micro-F1: 0.2712.
- Hamming Loss: 0.0606; LRAP: 0.6407.
- 940,626 hệ số + bias; train 2.52 giây CPU.
- BiGRU Macro-F1: 0.4060; PhoBERT: 0.4393.
- Subset accuracy không là metric chính.

Ghi chú thuyết trình: Thời gian TF-IDF gồm vectorizer + LR, không so tốc độ tuyệt đối với GPU. LRAP trên mẫu all-zero có quy ước đặc biệt.

## 9. Nhãn nào tốt, nhãn nào khó?

- Gratitude F1 0,6797; embarrassment 0,6311; pride 0,5500.
- Realization F1 0; disapproval 0,0160; disappointment 0,0280.
- Tần suất–F1: Spearman rho 0.242, p 0.223.
- Tần suất cao chưa đủ; disappointment là ví dụ.

Ghi chú thuyết trình: Các số ở đây chỉ nói về TF-IDF 0,5. Không khái quát các kết luận riêng của TF-IDF sang PhoBERT.

## 10. Ngưỡng 0,5 ảnh hưởng mạnh

- Khảo sát 0,05–0,80 trên validation.
- Ngưỡng 0.15: Macro-F1 0.4844.
- Tăng 24.55 điểm phần trăm.
- Hamming tăng lên 0.0867.
- Chưa chứng minh cải thiện trên test.

Ghi chú thuyết trình: LRAP không đổi khi đổi threshold. Tăng F1 không đồng nghĩa mọi metric cùng tốt hơn.

## 11. Baseline giữ được cặp nhãn?

- Disappointment–sadness: chỉ 2/122 ca validation đúng đủ cặp.
- Joint recall = 1,64% tại ngưỡng 0,5.
- OvR không có cơ chế học cấu trúc nhãn trực tiếp.
- Cần phân biệt phụ thuộc nhãn với vấn đề ngưỡng bỏ sót.

Ghi chú thuyết trình: Cặp được chọn từ train trước, rồi đo trên validation; không xây ma trận test.

## 12. Phong cách & lỗi biên

- TF-IDF Macro-F1: short 0,3310; emoji-heavy 0,1690; teencode 0.1671.
- Emoji bị bỏ bởi word-token TF-IDF; giả thuyết cho FN amusement.
- Câu hỏi giúp đỡ có thể bị FP curiosity.
- Có 20 FN + 20 FP; đã đọc thủ công 10 ca khác nhau.
- Q4 PhoBERT vs GRU: teencode có delta cao nhất: 9.41 điểm phần trăm.

Ghi chú thuyết trình: Style groups chồng lấn và có heuristic nhiễu. Không chép comment thô lên slide; mã ca lỗi trong báo cáo giúp kiểm tra nội bộ.

## 13. RQ Final: một ngưỡng có đủ?

- Chọn neural backbone tốt nhất từ midterm; giữ nguyên trọng số.
- So sánh fixed 0,5 / global tuned / per-label tuned.
- Tune validation → khóa ngưỡng → test đúng một lần.
- Phân tích theo tier/cardinality, profile ngưỡng và val–test gap.

Ghi chú thuyết trình: Bằng chứng TF-IDF tạo động lực, nhưng đề yêu cầu Final dùng neural architecture tốt nhất.

## 14. Việc cần chốt trước khi nộp

- Đã kiểm chứng kết quả Colab; lưu riêng checkpoint và vocab.
- Đối chiếu ID-to-label với metadata nguồn; xác nhận split protocol.
- Đọc error analysis neural; bổ sung tên/MSSV/phân công nhóm.
- Gói nộp: PDF, slide, code, README, config, ID, bảng/hình, probabilities.

Ghi chú thuyết trình: Nêu trạng thái thật và công việc cụ thể còn lại. Nếu đã chạy Colab, chuyển trọng tâm sang rà soát nội dung.

## 15. Nguồn & tái lập

- Đề ViGoEmotions và student-guide: tiêu chí đánh giá.
- Hung et al., EACL 2026: doi.org/10.18653/v1/2026.eacl-long.129.
- PhoBERT: Nguyen & Nguyen, 2020; vinai/phobert-base-v2.
- scikit-learn, Transformers, VnCoreNLP: dẫn nguồn trong báo cáo.
- Seed 42; package versions, checksum, hardware lưu trong run JSON.

Ghi chú thuyết trình: Các thư viện và pretrained checkpoint phải được trích dẫn. Điểm paper không phải điểm chạy của nhóm.
