# ViGoEmotions — Nội dung thuyết trình giữa kỳ

## 1. ViGoEmotions | Midterm 04/11

- Nhận diện đa nhãn cảm xúc tiếng Việt.
- Baseline 27 cảm xúc và study mới 28 nhãn có Neutral.
- Đã chạy đủ ba mô hình; kết quả trên validation.
- PhoBERT 28 nhãn: Macro-F1 0,5747; Neutral F1 0,4400.

Ghi chú thuyết trình: Tách nguồn baseline gốc 14/10, lượt chạy lại 23/10 và study 28 nhãn 04/11. Test giữ cho Final.

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

## 8. Baseline 27 nhãn | lượt gốc 14/10

- TF-IDF Macro-F1: 0.2388; Micro-F1: 0.2712.
- Hamming Loss: 0.0606; LRAP: 0.6407.
- 940,626 hệ số + bias; train 2.52 giây CPU.
- BiGRU Macro-F1: 0.4060; PhoBERT: 0.4393.
- Subset accuracy không là metric chính.

Ghi chú thuyết trình: Các slide 8–12 dùng baseline gốc. Kết quả mới 28 nhãn ở slide 13–17. Thời gian TF-IDF gồm vectorizer + LR, không so tốc độ tuyệt đối với GPU. LRAP trên mẫu all-zero có quy ước đặc biệt.

## 9. Nhãn nào tốt, nhãn nào khó?

- Cao nhất: gratitude, F1 0.6797.
- Thấp nhất: realization, F1 0.0000.
- Tần suất–F1: Spearman rho 0.242, p 0.223.
- Xem đầy đủ 27 nhãn; có thể có nhãn đồng hạng.

Ghi chú thuyết trình: Các số ở đây chỉ nói về TF-IDF 0,5. Không khái quát các kết luận riêng của TF-IDF sang PhoBERT.

## 10. Ngưỡng 0,5 ảnh hưởng mạnh

- Khảo sát 0,05–0,80 trên validation.
- Ngưỡng 0.15: Macro-F1 0.4844.
- Delta +24.55 điểm phần trăm.
- Hamming tăng: 0.0867.
- Chưa chứng minh cải thiện trên test.

Ghi chú thuyết trình: LRAP không đổi khi đổi threshold. Tăng F1 không đồng nghĩa mọi metric cùng tốt hơn.

## 11. Baseline giữ được cặp nhãn?

- Disappointment–sadness: 2/122 ca validation đúng đủ cặp.
- Joint recall = 1.64% tại ngưỡng 0,5.
- OvR không có cơ chế học cấu trúc nhãn trực tiếp.
- Cần phân biệt phụ thuộc nhãn với vấn đề ngưỡng bỏ sót.

Ghi chú thuyết trình: Cặp được chọn từ train trước, rồi đo trên validation; không xây ma trận test.

## 12. Phong cách & lỗi biên

- TF-IDF Macro-F1: short 0.3310; emoji-heavy 0.1690; teencode 0.1671.
- Emoji bị bỏ bởi word-token TF-IDF; giả thuyết cho FN amusement.
- Câu hỏi giúp đỡ có thể bị FP curiosity.
- Audit thủ công 10 ca thuộc lượt baseline 14/10/2026.
- Q4 PhoBERT vs GRU: teencode có delta cao nhất: +9.41 điểm phần trăm.

Ghi chú thuyết trình: Style groups chồng lấn và có heuristic nhiễu. Sau khi chạy lại cần kiểm tra ca lỗi mới. Không chép comment thô lên slide.

## 13. Thêm Neutral thành đầu ra thứ 28

- IDs 0–26: cảm xúc; ID 27: Neutral riêng.
- Giữ annotation: [27] → Neutral; [2,27] → joy + Neutral.
- Validation: 75 Neutral-only + 19 mixed = 94 Neutral dương.
- Sigmoid + BCE; ngưỡng 0,5; không ép loại trừ Neutral.

Ghi chú thuyết trình: Baseline cũ bỏ ID 27; study mới giữ cả nhãn mixed. Đây là quyết định đo lường trước bất nhất annotation, không khẳng định đã sửa nhãn. Nguồn: annotation_counts.csv và config.json trong evidence ngày 04/11.

## 14. Kết quả mới | 28 nhãn, validation

- Macro-F1: TF-IDF 0,2310; BiGRU 0,3992; PhoBERT 0,5747.
- PhoBERT: Micro-F1 0,5777; Hamming 0,0551; LRAP 0,7122.
- Đủ 2.066 mẫu; seed 42; test chưa chấm.
- PhoBERT cao nhất trong study 28 nhãn này.

Ghi chú thuyết trình: Đọc bảng metric đúng tập 28 nhãn. Không lấy 0,5747 trừ trực tiếp baseline 27 nhãn để kết luận cải thiện do Neutral. Nguồn: metrics_summary.csv đã tính lại từ ba NPZ.

## 15. Neutral vẫn là nhãn khó

- Neutral F1: 0,0208 / 0,0877 / 0,4400 (TF-IDF / BiGRU / PhoBERT).
- PhoBERT: precision 0,4151; recall 0,4681; support 94.
- Dự đoán 106 ca: 44 TP, 62 FP; còn 50 FN.
- Tập dự đoán rỗng không tự động là Neutral.

Ghi chú thuyết trình: PhoBERT tìm được gần một nửa Neutral thật nhưng precision còn thấp. Baseline 27 không có output Neutral: không tự gán Neutral F1=0. Nguồn: neutral_analysis.csv và verification.csv.

## 16. So sánh công bằng trên 27 cảm xúc

- Macro-F1 cũ → mới chiếu về 27: TF-IDF 0,2388 → 0,2388.
- BiGRU: 0,4060 → 0,4108; PhoBERT: 0,4393 → 0,5796.
- PhoBERT cũ 4 epoch; mới 12 epoch, chọn epoch 11.
- Delta chưa tách tác động Neutral, epoch và runtime.

Ghi chú thuyết trình: Đối chứng ở đây là baseline gốc 14/10 tại reports/tables, không phải lượt chạy lại 23/10 có PhoBERT 0,4679. Checkpoint cũng được chọn theo tập nhãn khác. Hamming trên phép chiếu 27 tăng 0,0533 → 0,0552 dù F1 tăng. Muốn kết luận nhân quả cần đối chứng cùng điều kiện.

## 17. Đường học & trạng thái thực nghiệm

- PhoBERT: 12 epoch; chọn 11; Macro-F1 28 nhãn 0,5747.
- Epoch 12 giảm còn 0,5630; chưa kết luận overfitting.
- BiGRU chọn 10/10; neural chạy Tesla T4.
- lr=2e-5 là một lượt; sweep 05 vẫn chưa có kết quả.

Ghi chú thuyết trình: Đường học có đủ history; train loss không thay validation loss. Cell tìm ví dụ validation lỗi JVM sau khi đã xuất metric; điểm số đã được tính lại từ NPZ. Giữ notebook output riêng tư. Nguồn: models/*/run.json và provenance.json.

## 18. RQ Final: một ngưỡng có đủ?

- Chọn neural backbone tốt nhất từ midterm; giữ nguyên trọng số.
- So sánh fixed 0,5 / global tuned / per-label tuned.
- Tune validation → khóa ngưỡng → test đúng một lần.
- Phân tích theo tier/cardinality, profile ngưỡng và val–test gap.

Ghi chú thuyết trình: Bằng chứng TF-IDF tạo động lực, nhưng đề yêu cầu Final dùng neural architecture tốt nhất.

## 19. Việc cần chốt trước khi nộp

- Study 28 đã có checkpoint/tokenizer/vocab trong ZIP private.
- Cell tìm ví dụ validation lỗi JVM; metric đã kiểm chứng riêng.
- Đọc lỗi neural; điền tên/MSSV/phân công; xác nhận split.
- Sweep 05 chưa chạy; cần đối chứng 27/28 cùng điều kiện.
- Giữ notebook output có comment nguồn riêng tư.

Ghi chú thuyết trình: Đây là giới hạn và việc còn lại, không phủ nhận kết quả đã kiểm chứng. Không cần train lại để trình bày số đo hiện có.

## 20. Nguồn & tái lập

- Đề ViGoEmotions và student-guide: tiêu chí đánh giá.
- Hung et al., EACL 2026: doi.org/10.18653/v1/2026.eacl-long.129.
- PhoBERT: Nguyen & Nguyen, 2020; vinai/phobert-base-v2.
- scikit-learn, Transformers, VnCoreNLP: dẫn nguồn trong báo cáo.
- Seed 42; package versions, checksum, hardware lưu trong run JSON.

Ghi chú thuyết trình: Các thư viện và pretrained checkpoint phải được trích dẫn. Điểm paper không phải điểm chạy của nhóm.
