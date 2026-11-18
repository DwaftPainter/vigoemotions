# ViGoEmotions — Câu hỏi oral defence và đáp án gợi ý

Tự trả lời trước khi đọc đáp án. Mỗi câu có một mốc tiếng Anh để luyện nói; không cần học thuộc nguyên văn. Khi giảng viên đổi ví dụ, phải suy luận từ pipeline và số liệu.

## A. Bài toán và dữ liệu

### 1. Nhóm đang giải quyết bài toán gì?

Nhận diện một hoặc nhiều cảm xúc chi tiết trong bình luận tiếng Việt. Output là vector nhị phân 28 chiều, có Neutral. Input không chỉ quyết định positive/negative sentiment.

**English:** This is fine-grained multi-label emotion recognition. A comment can receive several emotion labels, so we predict 28 binary outputs, including Neutral.

### 2. Vì sao không dùng softmax?

Softmax chuẩn hóa tổng outputs thành 1 và thường đi với các lớp loại trừ nhau. Ở đây joy và gratitude có thể cùng đúng, nên dùng sigmoid từng output và BCE.

**English:** The labels are not mutually exclusive. Independent sigmoid outputs let several emotions be positive for the same comment.

### 3. Tại sao chọn 28 khi đề hỏi 27 emotions?

Đây là quyết định mở rộng label space của nhóm để học Neutral trực tiếp, phù hợp không gian nhãn nguồn. Không nói thêm Neutral chắc chắn tốt hơn. Báo all28 chính, emotion27 projection và Neutral riêng để phân biệt phạm vi.

**English:** We explicitly model Neutral as the 28th output. We disclose the extension and also report the 27-emotion projection without retuning.

### 4. Mixed Neutral có bị sửa không?

Không. `[2,27]` giữ joy và Neutral. Không ép độc quyền, không xóa nhãn hoặc dùng empty prediction thành Neutral. Phân biệt chính sách xử lý annotation với đánh giá sự hợp lý ngữ nghĩa của annotation.

**English:** We preserve the supplied mixed annotations. No exclusivity or empty-prediction-to-Neutral rule is applied.

### 5. Làm sao tránh leakage? Có chắc split stratified không?

Fit vocabulary/IDF và weights trên train, chọn epoch/cutoffs trên validation, khóa trước test; bỏ một train overlap với held-out. Audit phân phối và hashes không chứng minh thuật toán stratification gốc. Không tự khẳng định đã tự chạy iterative stratification.

**English:** We keep fixed split IDs and separate fitting, selection and final evaluation. The integrity audit does not prove the original stratification algorithm.

## B. Mô hình và huấn luyện

### 6. Tại sao có ba mô hình ở Midterm?

TF-IDF + LR cho mốc lexical sparse; BiGRU kiểm khả năng học chuỗi từ train; PhoBERT dùng biểu diễn pretrained theo ngữ cảnh. So kết quả trong cùng label scope và split, không coi số model càng nhiều là thiết kế càng tốt.

**English:** The models compare lexical features, a recurrent representation learned from scratch, and a pretrained contextual representation.

### 7. Full fine-tuning khác frozen checkpoint ở đâu?

Midterm cập nhật backbone và head. Final dùng lại checkpoint đã học và chỉ đổi cutoff. Không có mâu thuẫn: frozen mô tả giai đoạn Final, không mô tả cách train Midterm.

**English:** We fine-tuned the full model during the Midterm. We then froze that trained checkpoint for the Final decoder comparison.

### 8. Vì sao cần VnCoreNLP và BPE?

VnCoreNLP word-segment tiếng Việt trước khi tokenizer PhoBERT chia subwords. Giữ đúng preprocessing/tokenizer của checkpoint để tránh đổi distribution đầu vào lúc inference. Không gọi segmentation là dịch văn bản.

**English:** Word segmentation prepares Vietnamese text for the saved PhoBERT tokenizer, which then produces subword IDs.

### 9. Model học ở dòng nào? Có cần sigmoid trước loss không?

Backward tính gradients; optimizer.step cập nhật weights. BCEWithLogitsLoss nhận logits, không sigmoid trước loss này. Inference dùng sigmoid để lấy scores.

**English:** Backpropagation computes gradients and the optimizer updates the weights. BCEWithLogitsLoss already incorporates the sigmoid computation.

### 10. Có train thêm vì paper gốc dùng số epoch khác không?

Không thay weights trong study RQ1 đã khóa. Paper gốc và nhóm có cấu hình khác; phải mô tả khác biệt thay vì gọi là tái lập nguyên setup. Nếu train thêm sau này, đó là study mới chọn bằng validation, không được dùng test hiện tại để tối ưu vòng lặp.

**English:** RQ1 isolates decision thresholds, so we do not retrain the backbone. Additional training would require a separate experimental design.

### 11. Đã train 20 hay 15 epoch? Vì sao chọn epoch 15?

Đã train đủ 20. PhoBERT28 chọn best epoch 15 bằng validation emotion27 Macro-F1 tại 0.5, tie chọn sớm hơn. Final objective cutoff là all28, không viết lại tiêu chí chọn epoch.

**English:** Training completed 20 epochs. Epoch 15 was selected using validation Macro-F1 over the 27 emotions at threshold 0.5.

## C. Metrics và threshold

### 12. Vì sao Macro-F1 là metric chính?

Macro cho mỗi nhãn trọng số như nhau, phù hợp quan tâm cả cảm xúc ít positives. Nó vẫn nhạy với support nhỏ và zero-division convention. Micro bổ sung cái nhìn từ tổng số TP/FP/FN, Hamming và LRAP đo khía cạnh khác.

**English:** Macro-F1 weights labels equally. We also report Micro-F1, Hamming Loss and LRAP because no single metric captures every trade-off.

### 13. Một Hamming Loss nhỏ có đủ chứng minh tốt không?

Không. Trên dữ liệu sparse, dự đoán nhiều nhãn âm có thể tạo Hamming thấp nhưng bỏ sót positives. Phải đọc F1, support và counts cùng nhau.

**English:** A low Hamming Loss can coexist with poor positive-label recall in sparse multi-label data.

### 14. Global và per-label khác gì?

Global chọn một scalar tối đa hóa Macro-F1 all28 trên validation. Per-label chọn cutoff cho F1 từng nhãn. Chúng dùng cùng model/scores; không thêm lần training.

**English:** Global tuning selects one cutoff for all labels. Per-label tuning selects 28 cutoffs, using the same validation scores and frozen model.

### 15. Vì sao per-label không thấp hơn global trên validation grid?

Với F1 tính độc lập từng nhãn và Macro là trung bình, vector chọn tốt nhất từng nhãn có thể đạt ít nhất score của mọi scalar cùng grid trên các nhãn có support. Fallback cho nhãn không có positives dùng 0.5 và F1 của nhãn đó bằng 0 theo convention, nên không tạo lợi ích ảo. Đây là tính chất **trong validation**, không phải bảo đảm cho test.

**English:** Independent label-wise maximization can match or exceed any shared grid cutoff on validation. That property does not guarantee better test performance.

### 16. Hạ threshold có làm precision/recall tăng cùng nhau không?

Trên cùng scores, hạ cutoff không giảm số dự đoán dương; recall không giảm nhưng precision/F1 có thể đổi theo hai hướng. Nervousness là ví dụ recall tăng nhưng FP tăng đủ mạnh để F1 giảm.

**English:** Lowering a threshold can recover positives and introduce false positives. Recall cannot decrease on fixed scores, but precision and F1 may worsen.

### 17. Tại sao LRAP của ba decoder giống nhau?

Chỉ đổi cutoff, không đổi continuous scores hoặc thứ hạng nhãn. LRAP là ranking metric, không tính từ các ma trận binary đã threshold.

**English:** The score rankings are unchanged, so the ranking metric is unchanged. Threshold tuning only changes binary decisions.

### 18. Có phải đã calibration xác suất không?

Study chọn decision cutoffs để tối ưu F1. Chưa kiểm chứng calibration theo nghĩa xác suất dự đoán phù hợp tần suất thực nghiệm; không dùng hai khái niệm thay nhau.

**English:** We optimize decision thresholds for F1. We do not establish probability calibration from this experiment.

### 19. Nếu p đúng bằng cutoff, dự đoán gì? Vì sao có tie-break?

Dương vì dùng >=. Tie-break chọn cutoff cao hơn khi computed F1 bằng nhau để quyết định tái lập được; không chọn theo bảng đã làm tròn. Đây là convention đã ghi trước test, không phải định luật cutoff cao luôn tốt hơn.

**English:** Equality is positive. Exact objective ties use the higher threshold under the frozen protocol.

## D. Kết quả và phản biện

### 20. Kết luận chính của Final?

Test all28 Macro-F1: fixed 0.589964, global 0.591254, per-label 0.585415. Global gain +0.1289 pp nhỏ; per-label đổi -0.4549 pp dù validation tốt hơn. Không có lợi ích đồng đều từ tuning linh hoạt hơn.

**English:** The global threshold gives a small observed gain. Per-label tuning improves validation F1 but reduces test F1 in this study.

### 21. Global cao nhất test, vậy chốt dùng global từ test được không?

Global 0.4 được chọn bằng validation, không phải quét test. Báo thứ tự test là kết quả của so sánh đã định trước. Nếu quyết định triển khai theo chi phí lỗi, cần objective/validation riêng; không biến test hiện tại thành tuning set.

**English:** The global cutoff was chosen on validation. Its test ranking is an observation, not a new test-based tuning rule.

### 22. Có chứng minh per-label overfit vì tail ít data không?

Có quan sát validation advantage không transfer và tail giảm mạnh nhất, phù hợp giả thuyết nhạy với estimation noise. Chưa có multi-seed/ablation để chứng minh cơ chế nhân quả. Dùng “suggests” hoặc “is consistent with”, không “proves”.

**English:** The pattern is consistent with threshold-selection instability, but one split does not establish its cause.

### 23. Gap âm có vấn đề không?

Gap được định nghĩa val−test. Âm nghĩa test cao hơn validation trong run này; có thể do sampling/difficulty differences. Nó không tự động chứng minh leakage hoặc model tốt hơn trong tương lai.

**English:** A negative gap means test F1 exceeds validation F1 in this run. It is not, by itself, evidence of leakage.

### 24. Neutral có được cải thiện nhờ per-label không?

Test Neutral F1 fixed khoảng 0.433628, per-label khoảng 0.422111. Cutoff 0.75 tăng precision nhưng giảm recall/F1. Không lấy validation Neutral gain để nói test đã tốt hơn.

**English:** The higher Neutral cutoff improves precision but reduces test recall and F1.

### 25. Cardinality lấy từ đâu? Emotion27 zero-cardinality nghĩa gì?

Từ ground truth. All28 gồm Neutral; emotion27 bỏ Neutral. Có 80 test comments Neutral-only nên emotion-cardinality=0, nhưng label-cardinality=1. Không gọi chúng là không có annotation.

**English:** Cardinality is computed from reference labels. Neutral-only comments have zero emotion cardinality but positive all28 cardinality.

### 26. Audit 62 ca chứng minh annotation sai bao nhiêu phần trăm?

Không suy được tỷ lệ population từ mẫu chọn có chủ đích. Notes là quan sát/giả thuyết chưa có independent adjudication; không đổi targets hoặc metric. Cần phân biệt cặp lỗi và số comments, lỗi giữ nguyên và lỗi mới sau tuning.

**English:** The qualitative sample is purposive and unadjudicated. It generates hypotheses rather than an estimate of annotation error prevalence.

### 27. Có chắc đã test đúng một lần?

Artifacts ghi một completed attempt, shared score matrix và lock bindings. Các kiểm tra xác minh records được giữ và tính nhất quán của chúng; không độc lập chứng minh mọi hoạt động ở notebook session trước đó. Không chạy lại inference để luyện oral.

**English:** The retained artifacts record one completed attempt bound to the lock. Hash checks establish consistency, not a complete history of every interactive action.

### 28. Nếu muốn làm tốt hơn, extension nào hợp lý?

Sau RQ1 có thể thử dependency modeling, imbalance-aware loss, noise robustness hoặc nhiều seed. Mỗi hướng cần hypothesis, controls, validation selection và đánh giá phù hợp. Các hướng đó chưa phải kết quả đã làm của paper này.

**English:** Future work could test label dependencies, imbalance-aware losses, robustness or multiple seeds under a separate controlled protocol.

### 29. Điểm của nhóm có so trực tiếp với paper gốc được không?

Chỉ so trực tiếp khi cùng label scope, split, preprocessing và metric protocol. Nhóm kế thừa 28-output label space nhưng training config/checkpoint criterion khác; không gọi điểm thấp/cao hơn là thắng/thua công bằng nếu các điều kiện chưa khớp.

**English:** A direct benchmark comparison requires matching evaluation conditions. Our study reuses the label space but does not reproduce the original training setup exactly.

### 30. Chứng minh đóng góp cá nhân thế nào?

Chỉ ra một phần mình thật sự làm, source/commit hoặc artifact liên quan, và cách mình xác minh nó. Tự trace một sample giả lập, tính metric hoặc giải thích nếu đổi cutoff thì chuyện gì xảy ra. Không dùng câu trả lời mẫu để nhận thay công việc của người khác.

**English:** I would show the specific component I completed, its evidence and the checks I performed, then explain a practical example myself.
