# ViGoEmotions — Protocol và cách đọc kết quả Final

## 1. Câu hỏi và thiết kế đã làm

RQ chính: global/per-label cutoff chọn trên validation có cải thiện test Macro-F1 **all28** so với fixed 0.5 trên cùng frozen PhoBERT checkpoint không? Đề gốc nêu 27 emotions; nhóm mở rộng thành 28 gồm Neutral và báo emotion27 projection bổ sung.

Midterm đã huấn luyện ba mô hình ở cả 27/28 outputs. Final giữ checkpoint PhoBERT-28 epoch 15 từ run 20 epoch. Tiêu chí chọn checkpoint gốc là emotion27 Macro-F1 tại 0.5, tie chọn epoch sớm hơn. Không thay lịch sử đó thành chọn epoch bằng all28.

| Mô hình | Outputs | Val Macro-F1 | Best epoch |
| --- | --- | --- | --- |
| TF-IDF + LR | 27 | 0.238825 | N/A |
| BiGRU | 27 | 0.421976 | 17 |
| PhoBERT | 27 | 0.597410 | 14 |
| TF-IDF + LR | 28 | 0.231040 | N/A |
| BiGRU | 28 | 0.408615 | 20 |
| PhoBERT | 28 | 0.578341 | 15 |

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

Cutoff global đã chọn là 0.40. Có đủ 28 cutoffs per-label, gồm Neutral 0.75. Quyết định dương dùng **p >= cutoff**, không phải p > cutoff. Code đưa scores/cutoffs về cùng precision float64 khi so sánh để scalar/vector không khác hành vi ở boundary.

```text
verify -> restore -> reproduce-val -> fit -> lock -> infer-test -> report
```

Lock bind config, protocol, code, source scores, reproduction và thresholds. Test artifact ghi completed và dùng một shared score matrix cho cả ba decoder. Hashes chứng minh liên kết/nhất quán của artifacts được giữ, không độc lập chứng minh mọi thao tác trong session tương tác trước đó.

Sau test, chỉ đọc saved outputs để phân tích. Không đổi cutoff, thêm epoch, chọn checkpoint hay quay lại test để tìm cấu hình tốt hơn. Trong nhánh ôn, chỉ cần dựng notes/PDF và đọc evidence.

## 4. Kết quả all28 và câu trả lời cho RQ

| Decoder | Val Macro | Test Macro | Test Micro | Hamming | Δ Macro pp |
| --- | --- | --- | --- | --- | --- |
| Fixed 0.5 | 0.578341 | 0.589964 | 0.597350 | 0.055135 | +0.0000 |
| Global | 0.582941 | 0.591254 | 0.598233 | 0.057347 | +0.1289 |
| Per-label | 0.595555 | 0.585415 | 0.594666 | 0.058297 | -0.4549 |

Global tăng test Macro-F1 +0.1289 điểm phần trăm. Per-label tăng validation tới 0.595555, nhưng test đổi -0.4549 điểm phần trăm so với fixed. Như vậy, validation-optimal per-label không đem lại lợi ích test trong lượt đo này. Fixed có Hamming Loss tốt nhất, dù global có F1 cao hơn.

Không gọi mức tăng global nhỏ là “có ý nghĩa thống kê”; chưa có confidence interval hoặc significance test. Không chọn lại chiến lược chính dựa trên thứ tự test. Tất cả ba kết quả đều được báo cáo.

![Macro-F1 validation và test của ba decoder; cùng checkpoint và label scope all28.](../../reports/final28/figures/validation_test.png)

## 5. Gap validation–test

| Decoder | Val−test Macro pp |
| --- | --- |
| Fixed 0.5 | -1.1623 |
| Global | -0.8312 |
| Per-label | +1.0140 |

Gap = validation − test, đơn vị pp trong bảng. Per-label có validation cao hơn test khoảng 1.0140 pp; fixed/global có gap âm. Đây là bằng chứng quan sát về transfer, chưa tách được ảnh hưởng của nhiễu chọn ngưỡng, khác biệt phân phối và độ khó của hai split.

## 6. Frequency tiers và cardinality

Tiers lấy từ retained train, xếp giảm theo positive count, hòa theo label ID; chín emotions mỗi tier. Neutral không nằm trong tier. Bảng dưới tính Macro-F1 trên chín cột của từng tier, không phải chia các comment thành ba nhóm độc lập.

| Tier | Fixed Macro | Global Macro | Per-label Macro | PL Δ pp |
| --- | --- | --- | --- | --- |
| head | 0.578569 | 0.584597 | 0.583454 | +0.4885 |
| mid | 0.644118 | 0.642867 | 0.640013 | -0.4105 |
| tail | 0.564577 | 0.563558 | 0.550923 | -1.3654 |

Per-label tail giảm khoảng 1.3654 pp; head tăng khoảng 0.4885 pp. Không khái quát rằng mọi nhãn hiếm đều giảm hoặc tần suất là nguyên nhân duy nhất.

Cardinality dùng **ground truth**, không phải số nhãn model dự đoán. Bảng chính all28 gồm cả Neutral:

| Card. all28 | N | Fixed Macro | Global Macro | Per-label Macro |
| --- | --- | --- | --- | --- |
| 0 | 0 | N/A | N/A | N/A |
| 1 | 695 | 0.459668 | 0.446783 | 0.449004 |
| 2 | 946 | 0.597258 | 0.603888 | 0.593143 |
| 3 | 360 | 0.603230 | 0.610477 | 0.609866 |
| >=4 | 66 | 0.584676 | 0.583418 | 0.572108 |

All28 không có mẫu cardinality=0. Emotion27 projection có 80 mẫu cardinality=0 vì bỏ Neutral của các mẫu Neutral-only. Hai định nghĩa không được trộn. Nhóm >=4 có 66 mẫu, nên cần đọc support trước khi so mức chênh lệch.

## 7. Neutral và emotion27 projection

| Decoder | Cutoff | TP | FP | FN | Neutral F1 |
| --- | --- | --- | --- | --- | --- |
| Fixed 0.5 | 0.50 | 49 | 61 | 67 | 0.433628 |
| Global | 0.40 | 51 | 67 | 65 | 0.435897 |
| Per-label | 0.75 | 42 | 41 | 74 | 0.422111 |

Per-label tăng Neutral cutoff từ 0.5 lên 0.75, precision tăng nhưng recall/F1 giảm trên test. Reference Neutral positives là 116. `[27]` giữ Neutral; `[2,27]` giữ cả joy và Neutral. Không áp quy tắc loại trừ sau sigmoid.

| Decoder | Test Macro27 | Test Micro27 | Hamming27 |
| --- | --- | --- | --- |
| Fixed 0.5 | 0.595755 | 0.602156 | 0.054884 |
| Global | 0.597007 | 0.602965 | 0.057105 |
| Per-label | 0.591463 | 0.598892 | 0.058396 |

Projection dùng 27 cột đầu của cùng scores và cùng cutoffs đã chọn theo all28. Không tune lại cutoff cho emotion27 và không gọi projection là model27 độc lập.

## 8. Đọc lỗi và giới hạn

Packet có 20 FN và 20 FP chính, hai Neutral errors, năm high-cardinality cases và 15 ca cho ba cặp chọn theo train: tổng 62 comments khác nhau. Trong 40 cặp chính, cả 20 FN cũng đã sai ở fixed; 15 FP giữ từ fixed và năm FP xuất hiện khi dùng per-label. Một phần lớn lỗi trong mẫu không chỉ do đổi cutoff.

Ba cặp train thường gặp là disappointment–sadness (904), anger–annoyance (699), sadness–grief (620). Ca được chọn để đọc lỗi, không phải mẫu ngẫu nhiên để suy tỷ lệ lỗi ngữ nghĩa toàn test. Có nhận xét định tính nhưng chưa có independent adjudication; không đổi reference labels hoặc tính lại metric từ nhận xét.

Ví dụ từ counts: nervousness ở fixed có TP=41, FP=26, FN=55; per-label có TP=51, FP=81, FN=45. Recall tăng nhưng FP tăng mạnh, làm F1 giảm. Đây là đánh đổi quyết định từ scores đã lưu; không xác định được cơ chế ngôn ngữ chỉ bằng các counts này.

Một seed, một checkpoint, một grid và một split hạn chế khả năng khái quát. Validation đã dùng chọn epoch và khảo sát Midterm; protocol Final không phải preregistration trước mọi quan sát. Graph/loss/noise robustness/multi-seed là extensions, chưa được thực hiện trong kết quả này.

## 9. Bản đồ code để mở khi được hỏi

| File | Vị trí | Ý cần hiểu |
| --- | --- | --- |
| [common.py](../../scripts/common.py) | scores, dòng 71 | Macro/Micro, Hamming, LRAP |
| [train_tfidf.py](../../scripts/train_tfidf.py) | main, dòng 16 | Fit TF-IDF + OvR LR |
| [train_neural.py](../../scripts/train_neural.py) | BiGRU, dòng 19 | Embedding, GRU hai hướng, linear head |
| [neutral_28_labels.py](../../scripts/neutral_28_labels.py) | train_neural28, dòng 254 | Full fine-tuning và checkpoint epoch 15 |
| [final_threshold_core.py](../../scripts/final_threshold_core.py) | fit_thresholds, dòng 40 | Grid, tie-break và fallback |
| [final_threshold_core.py](../../scripts/final_threshold_core.py) | decode, dòng 23 | Scalar/vector cutoffs, so sánh >= |
| [final_study.py](../../scripts/final_study.py) | verify_lock, dòng 212 | Kiểm source và lock bindings |
| [final_inference.py](../../scripts/final_inference.py) | predict_checkpoint, dòng 92 | Saved model/tokenizer, eval và sigmoid |

Đọc source để giải thích đầu vào/đầu ra, điều kiện và quyết định. Nhánh ôn giữ code gốc và decoder tests; workflow đầy đủ, report builders và nghiên cứu lịch sử nằm ở nhánh nguồn `feature/final28-report`.

## Phụ lục: 28 cutoffs đã fit

| Nhãn | Cutoff | Val positives | Train positives | Tier |
| --- | --- | --- | --- | --- |
| amusement | 0.65 | 327 | 2868 | head |
| excitement | 0.70 | 91 | 824 | mid |
| joy | 0.55 | 168 | 1614 | head |
| love | 0.20 | 148 | 1175 | head |
| desire | 0.35 | 102 | 762 | tail |
| optimism | 0.55 | 122 | 1046 | mid |
| caring | 0.50 | 131 | 1062 | head |
| pride | 0.65 | 87 | 774 | mid |
| admiration | 0.50 | 107 | 883 | mid |
| gratitude | 0.45 | 96 | 765 | mid |
| relief | 0.80 | 58 | 635 | tail |
| approval | 0.40 | 119 | 916 | mid |
| realization | 0.35 | 96 | 687 | tail |
| surprise | 0.25 | 87 | 651 | tail |
| curiosity | 0.55 | 112 | 840 | mid |
| confusion | 0.30 | 89 | 676 | tail |
| fear | 0.35 | 117 | 745 | tail |
| nervousness | 0.10 | 109 | 728 | tail |
| remorse | 0.10 | 96 | 707 | tail |
| embarrassment | 0.50 | 137 | 885 | mid |
| disappointment | 0.45 | 211 | 1522 | head |
| sadness | 0.55 | 354 | 2785 | head |
| grief | 0.40 | 97 | 747 | tail |
| disgust | 0.50 | 164 | 1206 | head |
| anger | 0.20 | 185 | 1589 | head |
| annoyance | 0.05 | 331 | 2662 | head |
| disapproval | 0.45 | 123 | 969 | mid |
| neutral | 0.75 | 94 | 821 | neutral |
