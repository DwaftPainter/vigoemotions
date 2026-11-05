# ViGoEmotions — Báo cáo giữa kỳ

**Kết quả chính: Kaggle 05/11/2026, notebook 08/09 executed.** Cả bốn lượt neural đã chạy đủ 20 epoch. Sáu kết quả được tính lại từ xác suất validation trong hai ZIP đầu ra. Không dùng kết quả của các lượt ít epoch trong bảng, biểu đồ hoặc kết luận báo cáo này.

**Thông tin nhóm:** Chưa được cung cấp; bổ sung họ tên, MSSV và phân công trong docs/team_information.md trước khi nộp.

## 1. Bài toán và phạm vi

Nhận diện cảm xúc đa nhãn trên bình luận tiếng Việt. Một mẫu có thể có nhiều cảm xúc, nên mỗi nhãn là một quyết định nhị phân độc lập. Nhóm đánh giá ba mô hình trên hai cách mã hóa: 27 cảm xúc và 28 nhãn gồm Neutral. Mọi điểm mô hình trong báo cáo là validation, ngưỡng 0,5, seed 42; test được giữ cho Final.

Nguồn yêu cầu: student-guide.pdf, mục Midterm; 02-vigoemotions-vietnamese-emotion.pdf, mục Midterm A–E. Cần EDA, split/preprocessing hợp lệ, ba mô hình, Macro/Micro-F1, per-label F1, Hamming Loss, LRAP, tham số/thời gian, phân tích lỗi và RQ Final. Bộ slide có 20 trang; code và README tái lập đi cùng kết quả.

## 2. Dữ liệu và EDA

CSV nguồn có 20.664 bình luận: train 16.531, validation 2.066, test 2.067. Loại 1 mẫu train trùng văn bản held-out; train thực dùng là 16,530. Mapping 27 cảm xúc theo đúng thứ tự ID: amusement, excitement, joy, love, desire, optimism, caring, pride, admiration, gratitude, relief, approval, realization, surprise, curiosity, confusion, fear, nervousness, remorse, embarrassment, disappointment, sadness, grief, disgust, anger, annoyance, disapproval. Neutral có ID 27.

Phân nhóm head/mid/tail theo thứ hạng tần suất trên train, chín nhãn mỗi nhóm; đây là tier tương đối. Amusement có tần suất lớn nhất, relief ít nhất, chênh lệch khoảng 4.52 lần. Nhãn dương thưa: cardinality train trung bình 1.859, tối đa 5 cảm xúc. Median độ dài là 11 token khoảng trắng và 47 ký tự. Whitespace token không đồng nghĩa từ tiếng Việt hoặc BPE token.

![Phân bố 27 cảm xúc trên train; cùng dữ liệu với hai study Kaggle.](figures/label_frequency.png)

![Cardinality, token khoảng trắng và ký tự theo split; không dùng để chọn mô hình bằng test.](figures/distributions.png)

Disappointment–sadness đồng xuất hiện 904 lần trên train, anger–annoyance 699 và sadness–grief 620. Ma trận co-occurrence và tương quan phi chỉ xây từ train; tần suất đồng xuất hiện không chứng minh quan hệ nhân quả. Các đặc điểm tiếng Việt cần chú ý gồm teencode, khẩu ngữ, từ tục, emoji, HTML và cú pháp phi chuẩn.

![Tương quan nhãn trên train, giữ thứ tự mapping 27 cảm xúc.](figures/train_correlation.png)

## 3. Split, preprocessing và hai cách mã hóa

Giữ split được cấp, fit vocabulary và IDF chỉ trên train; chuẩn hóa NFC/gom khoảng trắng, giữ dấu và phủ định. TF-IDF dùng regex unigram/bigram, BiGRU dùng lowercase/whitespace, PhoBERT dùng VnCoreNLP rồi BPE. Không thêm luật sửa annotation.

Notebook 07 xác nhận toàn vẹn PASS, fixed IDs khớp và không còn overlap sau lọc. Lệch prevalence lớn nhất so với train là 1.6324 điểm phần trăm ở validation và 0.9387 ở test. Audit phân bố không chứng minh thuật toán stratification gốc; việc chấp nhận official split theo đề còn cần xác nhận với giảng viên. Test chỉ tham gia audit, không chấm mô hình/chọn ngưỡng.

Với 27 đầu ra, [27] thành vector cảm xúc toàn 0; [2,27] giữ joy. Với 28 đầu ra, [27] là Neutral dương; [2,27] giữ cả joy và Neutral. CSV nguồn có 725 Neutral-only và 306 mixed-neutral; validation có 75 Neutral-only + 19 mixed = 94 Neutral dương. Giữ annotation mixed và công bố bất nhất; không dùng softmax hoặc ép Neutral loại trừ cảm xúc. Dự đoán rỗng không tự động đồng nghĩa Neutral.

## 4. Giao thức Kaggle 20 epoch

Hai notebook chạy độc lập, cùng profile configs/kaggle_equal_20ep.json, seed 42, cùng split và tiêu chí chọn checkpoint: emotion27 Macro-F1 validation tại 0,5; hòa chọn epoch sớm nhất. Tất cả neural chạy đủ 20 epoch, không early stopping. 28 nhãn dùng selection_macro_f1 trên IDs 0–26; metric 28 nhãn được báo cáo riêng.

TF-IDF: max_features=40.000, unigram/bigram, min_df=2, sublinear_tf; 27 hoặc 28 Logistic Regression OvR, liblinear, C=1, max_iter=1000. TF-IDF không có epoch tương đương neural. BiGRU: train từ đầu, embedding/hidden 128, dropout 0,3, max_length=128, batch=64, AdamW lr=0,001. PhoBERT-base-v2: full fine-tuning, max_length=256, batch=16, AdamW lr=2e-5, weight_decay=0,01, revision 86cd7fd4c148980922ac11a2cf5e257f2ba639e1. Neural dùng BCEWithLogitsLoss, sigmoid độc lập và gradient clipping 1,0; không có scheduler/warmup/AMP được cấu hình.

Hai môi trường manifest cùng Tesla T4, Python 3.13.16, Torch 2.11.0+cu130, Transformers 4.57.3, scikit-learn 1.6.1. Code dùng một GPU mỗi process dù Kaggle hiển thị T4 x2. Một seed và hai session riêng chưa đo độ biến thiên.

## 5. Kết quả chính trên validation

### 5.1. Ba mô hình, 27 cảm xúc

| model | macro_f1 | micro_f1 | hamming_loss | lrap |
| --- | --- | --- | --- | --- |
| TF-IDF + OvR LR | 0.2388 | 0.2712 | 0.0606 | 0.6407 |
| BiGRU | 0.4220 | 0.4590 | 0.0626 | 0.6191 |
| PhoBERT | 0.5974 | 0.5996 | 0.0550 | 0.7419 |

### 5.2. Ba mô hình, 28 nhãn gồm Neutral

| model | macro_f1 | micro_f1 | hamming_loss | lrap |
| --- | --- | --- | --- | --- |
| TF-IDF + OvR LR | 0.2310 | 0.2662 | 0.0601 | 0.6096 |
| BiGRU | 0.4086 | 0.4483 | 0.0628 | 0.5851 |
| PhoBERT | 0.5783 | 0.5829 | 0.0575 | 0.7077 |

![Macro-F1 tại 0,5; mỗi panel giữ riêng tập nhãn và không dùng chênh lệch trực tiếp 27/28 như hiệu ứng Neutral.](kaggle_equal20_2026-11-05/figures/model_metrics.png)

PhoBERT cao nhất trong mỗi study theo bốn metric tổng hợp (Hamming Loss thấp hơn là tốt hơn). Macro-F1 27 của PhoBERT là 0.597410; Macro-F1 28 là 0.578341. Macro-F1 và Micro-F1 đánh giá dự đoán sau ngưỡng; LRAP đánh giá thứ hạng xác suất. Subset accuracy không dùng làm metric chính. Chưa có kiểm định ý nghĩa thống kê hoặc metric test.

### 5.3. Epoch, tham số và thời gian

| model | label_count | epochs_trained | best_epoch | parameter_count | train_seconds | device |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + OvR LR | 27 | Không áp dụng | Không áp dụng | 940626 | 2.7524 | cpu |
| BiGRU | 27 | 20 | 17 | 2128283 | 80.3307 | Tesla T4 |
| PhoBERT | 27 | 20 | 14 | 135019035 | 14589.6182 | Tesla T4 |
| TF-IDF + OvR LR | 28 | Không áp dụng | Không áp dụng | 975464 | 2.5247 | cpu |
| BiGRU | 28 | 20 | 20 | 2128540 | 83.6053 | Tesla T4 |
| PhoBERT | 28 | 20 | 15 | 135019804 | 14057.7186 | Tesla T4 |

Thời gian TF-IDF gồm fit vectorizer và LR, neural gồm train, validation từng epoch và ghi checkpoint; không gồm preprocessing/download. Không so tỷ lệ tốc độ CPU/GPU hoặc xem cùng epoch là cùng compute. PhoBERT 27 train khoảng 4.05 giờ, 28 khoảng 3.90 giờ. Chạy đủ 20 rồi nạp best: epoch chọn của PhoBERT là 14 và 15; BiGRU chọn 17 và 20.

![Đường học từ bốn history 20 epoch: train BCE và tiêu chí chọn checkpoint chung emotion27 Macro-F1.](kaggle_equal20_2026-11-05/figures/learning_curves.png)

PhoBERT 27 có F1=0 ở epoch 1 nhưng về sau cải thiện, nên không kết luận lỗi từ epoch đầu. F1 validation biến động sau best trong khi train loss tiếp tục giảm. Không có validation loss; không tuyên bố hội tụ hoặc chẩn đoán overfitting chỉ từ đường loss train.

### 5.4. So sánh cùng 27 cảm xúc

| model | macro27 | macro28_projected27 | delta_pp | hamming27 | hamming28_projected27 |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + OvR LR | 0.2388 | 0.2388 | 0.0000 | 0.0606 | 0.0606 |
| BiGRU | 0.4220 | 0.4195 | -0.2490 | 0.0626 | 0.0629 |
| PhoBERT | 0.5974 | 0.5836 | -1.3830 | 0.0550 | 0.0575 |

Chiếu model 28 về cùng IDs 0–26, giữ cùng 2.066 ID/targets và ngưỡng 0,5. Kết quả: TF-IDF không đổi; BiGRU giảm 0.2490 điểm phần trăm; PhoBERT giảm 1.3830 điểm phần trăm. Với PhoBERT, Hamming Loss và LRAP trên 27 cảm xúc cũng kém hơn ở bản 28. Lượt đo này không cung cấp bằng chứng thêm Neutral cải thiện 27 cảm xúc. Cùng ngân sách và checkpoint criterion đã giảm nhiễu so sánh; vẫn chưa suy ra hiệu ứng nhân quả hay ý nghĩa thống kê từ một seed.

### 5.5. Neutral riêng

| model | precision | recall | f1 | support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + OvR LR | 0.5000 | 0.0106 | 0.0208 | 94 | 1 | 1 | 93 |
| BiGRU | 0.1778 | 0.0851 | 0.1151 | 94 | 8 | 37 | 86 |
| PhoBERT | 0.4018 | 0.4787 | 0.4369 | 94 | 45 | 67 | 49 |

PhoBERT có Neutral F1=0.4369, tìm được 45/94 mẫu, còn 49 FN và 67 FP. Neutral đã có đầu ra học được nhưng còn khó. Model 27 không có đầu ra Neutral: không tự gán Neutral F1=0 cho model đó.

## 6. Trả lời năm câu hỏi midterm bằng study 27 nhãn mới

### Q1. Nhãn nào có F1 cao/thấp nhất?

| model | highest | highest_f1 | lowest | lowest_f1 |
| --- | --- | --- | --- | --- |
| TF-IDF + OvR LR | gratitude | 0.6797 | realization | 0.0000 |
| BiGRU | gratitude | 0.7701 | realization | 0.0940 |
| PhoBERT | gratitude | 0.8370 | disapproval | 0.3623 |

Tất cả tại 0,5; xem phụ lục đủ từng nhãn. Các nhãn đồng hạng được liệt kê, không khái quát một nhãn dễ/khó cho mọi model.

### Q2. Tần suất train liên hệ với F1 ra sao?

| model | spearman_rho | p_value |
| --- | --- | --- |
| TF-IDF + OvR LR | 0.2424 | 0.2232 |
| BiGRU | 0.3132 | 0.1117 |
| PhoBERT | 0.0183 | 0.9278 |

Spearman mô tả tương quan trên 27 nhãn; các nhãn phụ thuộc nhau. Không suy ra quan hệ nhân quả hoặc rằng chỉ tăng số mẫu sẽ giải quyết mọi nhãn khó.

### Q3. Model giữ được cặp thường đồng xuất hiện không?

| model | val_true | joint_TP | joint_recall |
| --- | --- | --- | --- |
| TF-IDF + OvR LR | 122 | 2 | 0.0164 |
| BiGRU | 122 | 37 | 0.3033 |
| PhoBERT | 122 | 63 | 0.5164 |

Cặp disappointment–sadness chọn từ train trước khi đo validation. Joint recall là tỷ lệ mẫu có cả hai nhãn thật được đoán đủ cả hai. OvR không mô hình hóa cấu trúc nhãn trực tiếp; tỷ lệ thấp còn có thể liên quan ngưỡng bỏ sót.

### Q4. PhoBERT hơn BiGRU ở phong cách nào?

| style | n | gru | phobert | delta_pp |
| --- | --- | --- | --- | --- |
| all | 2066 | 0.4220 | 0.5974 | 17.5434 |
| emoji_heavy_ge_2 | 199 | 0.3506 | 0.4789 | 12.8265 |
| short_le_5 | 343 | 0.4303 | 0.5654 | 13.5113 |
| teencode | 307 | 0.3333 | 0.5547 | 22.1348 |

Trong các nhóm khảo sát, teencode có chênh lệch Macro-F1 lớn nhất: +22.13 điểm phần trăm. Nhóm short có ≤5 token khoảng trắng, emoji-heavy có ≥2 ký tự emoji theo regex và teencode theo từ điển heuristic; các nhóm chồng lấn và có nhiễu. Bảng dùng hai model 27 nhãn mới, không chuyển kết luận sang study khác.

### Q5. Nhạy với ngưỡng 0,5 đến đâu?

| model | default_macro | best_exploratory_threshold | exploratory_macro | exploratory_hamming |
| --- | --- | --- | --- | --- |
| TF-IDF + OvR LR | 0.2388 | 0.1500 | 0.4844 | 0.0867 |
| BiGRU | 0.4220 | 0.2000 | 0.4406 | 0.0784 |
| PhoBERT | 0.5974 | 0.3500 | 0.5978 | 0.0587 |

Khảo sát lại trên xác suất của study 27 mới, lưới 0,05–0,80 bước 0,05. Giữ checkpoint đã chọn tại 0,5; sweep chỉ là phân tích hậu nghiệm trên validation, không thay điểm bảng chính hay chọn lại checkpoint. LRAP không đổi khi đổi ngưỡng. Ngưỡng tối ưu trên validation chưa chứng minh cải thiện test, không áp thẳng cho model/tập nhãn khác.

## 7. Phân tích lỗi trên lượt mới

Đọc sáu ca PhoBERT 28: ba Neutral FP tự tin nhất và ba Neutral FN có xác suất thấp nhất. Chọn theo NPZ của lượt mới; không dùng lại các ca và kết luận dự đoán từ run cũ. Có hỗ trợ AI, nhóm cần xác nhận. Không xuất nguyên văn bình luận và không coi các ca chọn có chủ đích là mẫu ngẫu nhiên.

| sample_id | case_type | observation | hypothesis |
| --- | --- | --- | --- |
| 30099 | Neutral FP | Lời giải thích tình huống được gán relief; model chỉ dự đoán Neutral. | Cảm xúc hàm ý trong giải thích tình huống có thể khó nhận diện. |
| you003230 | Neutral FP | Nội dung liệt kê được gán approval và realization; model chỉ dự đoán Neutral. | Dạng liệt kê và HTML có thể che tín hiệu cảm xúc; cần ablation để kiểm chứng. |
| 15569 | Neutral FP | Lời bình ngắn được gán surprise; model chỉ dự đoán Neutral. | Sắc thái ngạc nhiên phụ thuộc tiểu từ/ngữ cảnh ngắn. |
| 6238 | Neutral FN | Ground truth chỉ Neutral; model dự đoán anger và annoyance. | Từ tục và khẩu ngữ có thể kéo dự đoán sang cảm xúc tiêu cực. |
| 21975 | Neutral FN | Ground truth chỉ Neutral; model dự đoán anger. | Tên riêng hoặc cụm từ theo ngữ cảnh có thể gây nhầm; chưa xác định cơ chế. |
| 1134 | Neutral FN | Ground truth chỉ Neutral; model dự đoán amusement và joy. | Từ mang nghĩa ưa thích có thể bị đọc thành cảm xúc tích cực. |

Quan sát được tách với giả thuyết: ngữ cảnh hàm ý, câu ngắn, HTML và từ tục là hướng cần kiểm chứng bằng phân tích tiếp. CSV đi kèm có ground truth, dự đoán và xác suất Neutral. Mỗi model còn có 20 ứng viên FN và 20 FP tự sinh trong analysis; đó là danh sách để đọc, không phải bằng chứng tất cả đã được nhóm audit. Yêu cầu Final ít nhất 20 FN/20 FP vẫn chưa hoàn tất.

## 8. RQ và kế hoạch Final

Một ngưỡng chung 0,5 có tối ưu cho mọi nhãn không? Dùng neural backbone tốt nhất của midterm; hiện PhoBERT 27 có emotion27 Macro-F1 cao nhất. Khóa checkpoint, không retrain backbone khi đổi decoder; so sánh fixed 0,5, global tuned và per-label tuned. Chọn ngưỡng chỉ trên validation; khóa mọi lựa chọn rồi chấm held-out test đúng một lần. Phân tích theo tier/cardinality, profile ngưỡng, khoảng cách validation–test và ít nhất 20 FN/20 FP. Nếu tiếp tục dùng 28 nhãn, báo cáo riêng Neutral và phép chiếu 27.

## 9. Tái lập, nguồn và giới hạn

Hai ZIP Kaggle và notebook executed 08/09 là source of truth. scripts/import_equal20_results.py kiểm chứng manifest notebook khớp ZIP, sáu NPZ khớp ID/target/label order và metric CSV, history đủ 20, epoch best đúng tiêu chí và checkpoint có trong archive. Không nạp pickle/checkpoint, không train hay inference lại. Evidence được xuất bằng allowlist gồm JSON và NPZ ID-only.

Checksum file thô ở lượt 28 khác vì CRLF: đã xác minh từng checksum là biến thể LF/CRLF chính xác của source/config/CSV local. Không chỉ giả định hai file JSON có cùng nghĩa. Cấu hình sau parse và split IDs khớp giữa hai lượt; môi trường và GPU cùng thông tin manifest. provenance.json lưu checksum ZIP, notebook, thành viên đã trích và loại biến thể từng file; không sửa manifest gốc để ép khớp.

Giới hạn: một seed, checkpoint và khảo sát ngưỡng dùng validation; test chưa chấm; style heuristic; sáu ca lỗi có hỗ trợ AI cần nhóm xác nhận; chưa xác minh thuật toán split gốc và chưa có thông tin nhóm. Kết quả ít epoch được lưu như lịch sử, không là bằng chứng cho các bảng/kết luận chính hiện tại.

Lệnh tái tạo: python scripts/import_equal20_results.py; python scripts/build_report.py; python scripts/build_study_guide.py. Cần hai ZIP/notebook executed gốc và CSV được cấp quyền để import; các PDF/tổng hợp chỉ cần evidence đã kiểm chứng và EDA. Không phân phối raw text, vocabulary hoặc checkpoint công khai.

## Phụ lục A. F1 mọi nhãn của ba model

### A.27. 27 nhãn, neural 20 epoch

| label | tfidf_lr | gru | phobert |
| --- | --- | --- | --- |
| amusement | 0.2089 | 0.5822 | 0.6435 |
| excitement | 0.0426 | 0.1406 | 0.4606 |
| joy | 0.2525 | 0.4722 | 0.5321 |
| love | 0.2983 | 0.5311 | 0.5833 |
| desire | 0.0561 | 0.3333 | 0.5370 |
| optimism | 0.3311 | 0.4369 | 0.6719 |
| caring | 0.3436 | 0.4786 | 0.6397 |
| pride | 0.5500 | 0.5942 | 0.7397 |
| admiration | 0.1552 | 0.3490 | 0.5882 |
| gratitude | 0.6797 | 0.7701 | 0.8370 |
| relief | 0.0656 | 0.3011 | 0.5785 |
| approval | 0.1955 | 0.4142 | 0.6552 |
| realization | 0.0000 | 0.0940 | 0.4641 |
| surprise | 0.0449 | 0.2162 | 0.5087 |
| curiosity | 0.1626 | 0.4590 | 0.6266 |
| confusion | 0.0645 | 0.2615 | 0.4916 |
| fear | 0.4106 | 0.6732 | 0.6971 |
| nervousness | 0.1176 | 0.3420 | 0.5388 |
| remorse | 0.4032 | 0.5868 | 0.7192 |
| embarrassment | 0.6311 | 0.5764 | 0.7729 |
| disappointment | 0.0280 | 0.3452 | 0.4988 |
| sadness | 0.4068 | 0.5677 | 0.6819 |
| grief | 0.2609 | 0.4286 | 0.6736 |
| disgust | 0.1341 | 0.2979 | 0.4765 |
| anger | 0.2833 | 0.4489 | 0.5762 |
| annoyance | 0.3056 | 0.5095 | 0.5751 |
| disapproval | 0.0160 | 0.1829 | 0.3623 |

### A.28. 28 nhãn, neural 20 epoch

| label | tfidf_lr | gru | phobert |
| --- | --- | --- | --- |
| amusement | 0.2089 | 0.5855 | 0.6196 |
| excitement | 0.0426 | 0.1789 | 0.4192 |
| joy | 0.2525 | 0.4370 | 0.5378 |
| love | 0.2983 | 0.5551 | 0.6351 |
| desire | 0.0561 | 0.2937 | 0.5027 |
| optimism | 0.3311 | 0.4360 | 0.6692 |
| caring | 0.3436 | 0.4909 | 0.6486 |
| pride | 0.5500 | 0.6061 | 0.7135 |
| admiration | 0.1552 | 0.3506 | 0.6275 |
| gratitude | 0.6797 | 0.7317 | 0.8261 |
| relief | 0.0656 | 0.2588 | 0.5781 |
| approval | 0.1955 | 0.3905 | 0.6491 |
| realization | 0.0000 | 0.1618 | 0.4615 |
| surprise | 0.0449 | 0.2419 | 0.4505 |
| curiosity | 0.1626 | 0.4407 | 0.5975 |
| confusion | 0.0645 | 0.2540 | 0.4787 |
| fear | 0.4106 | 0.6244 | 0.7296 |
| nervousness | 0.1176 | 0.2825 | 0.4712 |
| remorse | 0.4032 | 0.5405 | 0.7059 |
| embarrassment | 0.6311 | 0.6009 | 0.7589 |
| disappointment | 0.0280 | 0.3626 | 0.4918 |
| sadness | 0.4068 | 0.5744 | 0.6616 |
| grief | 0.2609 | 0.4472 | 0.5756 |
| disgust | 0.1341 | 0.4136 | 0.4895 |
| anger | 0.2833 | 0.4419 | 0.5540 |
| annoyance | 0.3056 | 0.4791 | 0.5277 |
| disapproval | 0.0160 | 0.1458 | 0.3760 |
| neutral | 0.0208 | 0.1151 | 0.4369 |

## Tài liệu tham khảo

[1] Đề Project 2 ViGoEmotions và student-guide.pdf trong docs/, Midterm A–E và RQ1 Final.

[2] Hung et al., ViGoEmotions, EACL 2026, DOI 10.18653/v1/2026.eacl-long.129.

[3] Nguyen và Nguyen, PhoBERT, Findings of EMNLP 2020; VinAIResearch/PhoBERT, model vinai/phobert-base-v2.

[4] scikit-learn, PyTorch, Hugging Face Transformers và VnCoreNLP; phiên bản, revision và cấu hình theo manifest/run JSON trong evidence của study này.
