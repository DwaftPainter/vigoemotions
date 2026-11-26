# ViGoEmotions — Protocol và cách đọc kết quả Final

## 1. Câu hỏi và thiết kế đã làm

RQ1 hỏi ngưỡng chung và ngưỡng riêng từng nhãn chọn trên validation có cải thiện test Macro-F1 **all28** so với fixed 0.5 trên cùng checkpoint PhoBERT không. Đề gốc nêu 27 cảm xúc. Nhóm mở rộng thành 28 đầu ra gồm Neutral và báo thêm phép chiếu emotion27. RQ4 hỏi hiệu năng thay đổi thế nào khi thêm nhiễu vào cùng các bình luận.

Midterm đã huấn luyện ba mô hình ở cả 27/28 outputs. Final giữ checkpoint PhoBERT-28 epoch 15 từ run 20 epoch. Tiêu chí chọn checkpoint gốc là emotion27 Macro-F1 tại 0.5, tie chọn epoch sớm hơn. Không thay lịch sử đó thành chọn epoch bằng all28.

{{midterm_table}}

Điểm trong bảng trên là **validation tại 0.5**, không phải test Final. Không so trực tiếp Macro-F1 27 và 28 để kết luận tác dụng của Neutral. Khi chiếu model28 về cùng 27 emotions, PhoBERT có Macro-F1 validation 0.583580; model27 riêng là 0.597410. Đây là một run, không chứng minh hiệu ứng nhân quả của thêm output.

## 2. Từ bình luận tới ba tập dự đoán

```text
private comment -> NFC + whitespace -> VnCoreNLP -> saved BPE tokenizer
-> saved PhoBERT-28 checkpoint -> 28 logits -> sigmoid -> one score matrix
-> fixed cutoff / global cutoff / per-label cutoffs -> evaluation
```

Max length 256, batch inference 16, giữ thứ tự IDs, eval mode. Validation có 2,066 mẫu; test có 2,067 mẫu. Model/tokenizer/revision và preprocessing phải khớp với run nguồn.

Reproduce-val đã chạy inference thật và đạt max absolute difference = **0** so với probabilities nguồn; tolerance protocol là 1e-4, rtol=0 và fixed decisions phải giống. Đây là bằng chứng tái tạo validation, không phải bằng chứng model đúng về ngữ nghĩa cho từng comment.

## 3. Chọn ngưỡng và khóa trước test

Grid là 16 giá trị từ 0.05 đến 0.80, bước 0.05. Fixed là 0.5; global tối đa hóa Macro-F1 all28; per-label tối đa hóa F1 nhị phân của từng nhãn. Khi F1 tính đầy đủ bằng nhau, chọn cutoff cao hơn. Nhãn không có positive validation dùng 0.5; run này không dùng fallback.

Cutoff global đã chọn là {{global_threshold}}. Có đủ 28 cutoffs per-label, gồm Neutral {{neutral_threshold}}. Quyết định dương dùng **p >= cutoff**, không phải p > cutoff. Code đưa scores/cutoffs về cùng precision float64 khi so sánh để scalar/vector không khác hành vi ở boundary.

```text
verify -> restore -> reproduce-val -> fit -> lock -> infer-test -> report
```

Lock bind config, protocol, code, source scores, reproduction và thresholds. Test artifact ghi completed và dùng một shared score matrix cho cả ba decoder. Hashes chứng minh liên kết/nhất quán của artifacts được giữ, không độc lập chứng minh mọi thao tác trong session tương tác trước đó.

Kết quả RQ1 được phân tích từ đầu ra đã lưu, với ngưỡng và checkpoint đã khóa. RQ4 có lượt inference riêng để so sánh đầu vào gốc với đầu vào nhiễu trên cùng split. Trong nhánh ôn, các bảng được dựng từ kết quả đã kiểm chứng.

## 4. Kết quả all28 và câu trả lời cho RQ

{{primary_table}}

Global tăng test Macro-F1 {{global_gain_pp}} điểm phần trăm. Per-label tăng validation tới {{val_per_label_macro}}, nhưng test đổi {{per_label_gain_pp}} điểm phần trăm so với fixed. Như vậy, validation-optimal per-label không đem lại lợi ích test trong lượt đo này. Fixed có Hamming Loss tốt nhất, dù global có F1 cao hơn.

Không gọi mức tăng global nhỏ là “có ý nghĩa thống kê”; chưa có confidence interval hoặc significance test. Không chọn lại chiến lược chính dựa trên thứ tự test. Tất cả ba kết quả đều được báo cáo.

![Macro-F1 validation và test của ba decoder; cùng checkpoint và label scope all28.](../../reports/final28/figures/validation_test.png)

## 5. Gap validation–test

{{gap_table}}

Gap = validation − test, đơn vị pp trong bảng. Per-label có validation cao hơn test khoảng 1.0140 pp; fixed/global có gap âm. Đây là bằng chứng quan sát về transfer, chưa tách được ảnh hưởng của nhiễu chọn ngưỡng, khác biệt phân phối và độ khó của hai split.

## 6. Frequency tiers và cardinality

Tiers lấy từ retained train, xếp giảm theo positive count, hòa theo label ID; chín emotions mỗi tier. Neutral không nằm trong tier. Bảng dưới tính Macro-F1 trên chín cột của từng tier, không phải chia các comment thành ba nhóm độc lập.

{{tier_table}}

Per-label tail giảm khoảng 1.3654 pp; head tăng khoảng 0.4885 pp. Không khái quát rằng mọi nhãn hiếm đều giảm hoặc tần suất là nguyên nhân duy nhất.

Cardinality dùng **ground truth**, không phải số nhãn model dự đoán. Bảng chính all28 gồm cả Neutral:

{{cardinality_table}}

All28 không có mẫu cardinality=0. Emotion27 projection có 80 mẫu cardinality=0 vì bỏ Neutral của các mẫu Neutral-only. Hai định nghĩa không được trộn. Nhóm >=4 có 66 mẫu, nên cần đọc support trước khi so mức chênh lệch.

## 7. Neutral và emotion27 projection

{{neutral_table}}

Per-label tăng Neutral cutoff từ 0.5 lên {{neutral_threshold}}, precision tăng nhưng recall/F1 giảm trên test. Reference Neutral positives là 116. `[27]` giữ Neutral; `[2,27]` giữ cả joy và Neutral. Không áp quy tắc loại trừ sau sigmoid.

{{emotion27_table}}

Projection dùng 27 cột đầu của cùng scores và cùng cutoffs đã chọn theo all28. Không tune lại cutoff cho emotion27 và không gọi projection là model27 độc lập.

## 8. Đọc lỗi và giới hạn

Packet có 20 FN và 20 FP chính, hai Neutral errors, năm high-cardinality cases và 15 ca cho ba cặp chọn theo train: tổng 62 comments khác nhau. Trong 40 cặp chính, cả 20 FN cũng đã sai ở fixed; 15 FP giữ từ fixed và năm FP xuất hiện khi dùng per-label. Một phần lớn lỗi trong mẫu không chỉ do đổi cutoff.

Ba cặp train thường gặp là disappointment–sadness (904), anger–annoyance (699), sadness–grief (620). Ca được chọn để đọc lỗi, không phải mẫu ngẫu nhiên để suy tỷ lệ lỗi ngữ nghĩa toàn test. Có nhận xét định tính nhưng chưa có independent adjudication; không đổi reference labels hoặc tính lại metric từ nhận xét.

Ví dụ từ counts: nervousness ở fixed có TP=41, FP=26, FN=55; per-label có TP=51, FP=81, FN=45. Recall tăng nhưng FP tăng mạnh, làm F1 giảm. Đây là đánh đổi quyết định từ scores đã lưu; không xác định được cơ chế ngôn ngữ chỉ bằng các counts này.

Một seed, một checkpoint, một grid và một split hạn chế khả năng khái quát. Validation đã được dùng để chọn epoch và khảo sát trong Midterm trước khi chốt RQ1. Các hướng đồ thị nhãn, thay loss và chạy nhiều seed vẫn là việc có thể nghiên cứu thêm.

## 9. RQ4 — Độ bền với nhiễu văn bản

So sánh ba phiên bản của cùng 2,066 mẫu validation và 2,067 mẫu test. Mỗi phiên bản giữ nguyên ID, thứ tự, nhãn, checkpoint epoch 15, tokenizer và ba bộ ngưỡng RQ1. Văn bản gốc là đối chứng chạy cùng lượt inference. Quyết định trên đối chứng khớp RQ1 ở cả ba bộ ngưỡng.

- **Bỏ dấu:** bỏ dấu trên nguyên âm tiếng Việt, đổi đ/Đ thành d/D và giữ chữ hoa/thường, dấu câu, emoji. Phép này làm thay đổi {{rq4_diacritics_changed}} trong 2,067 bình luận test.
- **Lỗi gõ QWERTY:** mỗi chữ cái ASCII đủ điều kiện có xác suất 5% bị thay bằng phím chữ lân cận. Seed 42, ID mẫu và vị trí ký tự quyết định phép thay. Có {{rq4_typo_changed}} bình luận test thay đổi.

Nhiễu được tạo sau chuẩn hóa NFC và khoảng trắng, trước VnCoreNLP và BPE. Bảng dùng toàn bộ mẫu test, gồm cả mẫu mà phép biến đổi giữ nguyên. Cột Δ lấy điểm của điều kiện trừ điểm văn bản gốc ở cùng decoder và nhân 100.

{{rq4_table}}

![RQ4: thay đổi Macro-F1 test all28 của ba decoder so với văn bản gốc.](../../reports/rq4_noise/test/figures/macro_f1_delta.png)

Ở fixed 0.5, bỏ dấu giảm {{rq4_diacritics_drop_pp}} pp và lỗi gõ giảm {{rq4_typo_drop_pp}} pp. Tail giảm nhiều nhất trong ba tiers ở cả hai điều kiện, lần lượt {{rq4_diacritics_tail_drop_pp}} và {{rq4_typo_tail_drop_pp}} pp. Hai mức giảm mô tả cách tạo nhiễu đã chọn, vì mức độ và số mẫu bị sửa khác nhau. RQ4 được thiết kế sau khi có kết quả RQ1, dùng lại cùng held-out split. Giữ nhãn gốc cả khi phép sửa có thể làm câu đổi nghĩa là một giới hạn.

Mở bảng [điểm theo điều kiện](../../reports/rq4_noise/test/tables/comparison.csv), [mức thay đổi đầu vào](../../reports/rq4_noise/test/tables/noise_coverage.csv) hoặc [biên bản kiểm chứng val và test](../../reports/rq4_noise/verification.json) khi cần giải thích nguồn số liệu.

## 10. Bản đồ code để mở khi được hỏi

{{code_map}}

Với RQ4, xem `prepare_conditions` trong [rq4_noise.py](https://github.com/DwaftPainter/vigoemotions/blob/cb0ea5c/scripts/rq4_noise.py) và `run_split` trong [rq4_robustness.py](https://github.com/DwaftPainter/vigoemotions/blob/cb0ea5c/scripts/rq4_robustness.py). Tài liệu này dùng bảng RQ4 đã kiểm chứng từ nhánh `submission/ati-final-rq4`.

## Phụ lục: 28 cutoffs đã fit

{{threshold_table}}
