# Các bảng CSV

[← Mục lục reports](../README.md)

## Bắt đầu với hai bảng tổng hợp

| File | Câu hỏi trả lời | Cách đọc |
| --- | --- | --- |
| [model_comparison.csv](model_comparison.csv) | Ba baseline khác nhau thế nào? | Một dòng/mô hình, cùng validation và threshold 0,5; Macro/Micro-F1, Hamming, LRAP, tham số, thời gian. Tạo bởi `build_report.py`. |
| [neural_style_comparison.csv](neural_style_comparison.csv) | PhoBERT hơn/kém BiGRU ở phong cách nào? | Một dòng/style, `delta_macro_f1 = phobert − gru`. `all` là toàn validation; các nhóm còn lại có thể chồng lấn. |

## EDA: hiểu dữ liệu trước khi đọc điểm mô hình

Các bảng dưới được sinh bởi `scripts/analyze_data.py` từ dữ liệu đã chuẩn hóa và loại train trùng held-out.

| File | Nội dung / ý nghĩa |
| --- | --- |
| [dataset_summary.csv](dataset_summary.csv) | Một dòng/split: số mẫu, cardinality trung bình/tối đa, neutral-only, mixed-neutral và số hàng có nhãn lặp. Train ở đây đã còn 16.530 mẫu. |
| [label_frequency.csv](label_frequency.csv) | Đủ 27 ID/tên nhãn, count/prevalence từng split, tier head/mid/tail và drift. Prevalence là tỷ lệ 0–1; drift `_pp` đã tính bằng điểm phần trăm. Tier chỉ xếp theo train, 9 nhãn/tier. |
| [cardinality.csv](cardinality.csv) | Số mẫu theo số cảm xúc dương 0–5 ở mỗi split. Không phải bảng chất lượng dự đoán. |
| [text_lengths.csv](text_lengths.csv) | Mean, median, p95, p99, min/max theo token khoảng trắng và ký tự. Không phải số từ VnCoreNLP hay BPE token. |
| [text_styles.csv](text_styles.csv) | Count và proportion của `all`, `short_le_5`, `emoji_heavy_ge_2`, `teencode` theo split. Đây là heuristic, không phải phân loại phong cách đã gán nhãn thủ công. |
| [train_cooccurrence.csv](train_cooccurrence.csv) | Ma trận 27×27 số lần nhãn cùng xuất hiện trên train; đường chéo là tần suất nhãn. |
| [train_correlation.csv](train_correlation.csv) | Ma trận 27×27 Pearson/phi giữa biến nhãn nhị phân trên train; có giá trị âm/dương, không suy ra nhân quả. |
| [train_top_pairs.csv](train_top_pairs.csv) | **Toàn bộ 351 cặp** không lặp, sắp theo count giảm dần, kèm phi. Tên file có “top” nhưng không chỉ chứa top 10; dùng các dòng đầu để trình bày. |

Hai ma trận CSV có cột đầu chứa tên hàng. Đọc bằng `pd.read_csv(path, index_col=0)` để nhận đúng ma trận 27×27; cột `Unnamed: 0` không phải nhãn thứ 28.

## Bảy loại bảng cho mỗi mô hình

Các bảng này được sinh bởi `scripts/evaluate.py` từ xác suất validation của đúng mô hình. Mỗi dòng dưới liên kết đủ ba file.

| Loại | TF-IDF | BiGRU | PhoBERT | Nội dung / ý nghĩa |
| --- | --- | --- | --- | --- |
| Metrics | [tfidf_lr_metrics.csv](tfidf_lr_metrics.csv) | [gru_metrics.csv](gru_metrics.csv) | [phobert_metrics.csv](phobert_metrics.csv) | Một dòng validation tại 0,5; metric tổng thể, số tham số và thời gian. |
| Per-label | [tfidf_lr_per_label.csv](tfidf_lr_per_label.csv) | [gru_per_label.csv](gru_per_label.csv) | [phobert_per_label.csv](phobert_per_label.csv) | 27 dòng: precision, recall, F1, support validation và train count từng cảm xúc. |
| Styles | [tfidf_lr_styles.csv](tfidf_lr_styles.csv) | [gru_styles.csv](gru_styles.csv) | [phobert_styles.csv](phobert_styles.csv) | Metric theo bốn nhóm phong cách; `n` là số mẫu trong nhóm. |
| Cardinality | [tfidf_lr_cardinality_val.csv](tfidf_lr_cardinality_val.csv) | [gru_cardinality_val.csv](gru_cardinality_val.csv) | [phobert_cardinality_val.csv](phobert_cardinality_val.csv) | Metric theo số nhãn **thật** 0/1/2/3/≥4, không theo số nhãn mô hình dự đoán. |
| Pair preservation | [tfidf_lr_pair_preservation_val.csv](tfidf_lr_pair_preservation_val.csv) | [gru_pair_preservation_val.csv](gru_pair_preservation_val.csv) | [phobert_pair_preservation_val.csv](phobert_pair_preservation_val.csv) | 10 cặp chọn từ train, sau đó đo trên validation; joint recall = số ca đúng đủ cặp / số ca thật có cặp. |
| Thresholds | [tfidf_lr_thresholds_val.csv](tfidf_lr_thresholds_val.csv) | [gru_thresholds_val.csv](gru_thresholds_val.csv) | [phobert_thresholds_val.csv](phobert_thresholds_val.csv) | 16 ngưỡng 0,05–0,80, bước 0,05, dùng cùng xác suất; không retrain. LRAP không đổi khi đổi threshold. |
| Error cases | [tfidf_lr_error_cases_val.csv](tfidf_lr_error_cases_val.csv) | [gru_error_cases_val.csv](gru_error_cases_val.csv) | [phobert_error_cases_val.csv](phobert_error_cases_val.csv) | 20 FN có xác suất thấp nhất và 20 FP cao nhất tại 0,5; mỗi dòng là **một cặp sample–label lỗi**, không nhất thiết 40 bình luận khác nhau. |

Trong error cases: `FN` là bỏ sót nhãn thật; `FP` là dự đoán thêm nhãn. `true_labels`/`predicted_labels` ngăn cách bằng dấu `;`; `styles` có thể có nhiều nhóm. Các file không chứa text gốc. Danh sách này ưu tiên lỗi tự tin và không phải mẫu ngẫu nhiên hay bằng chứng đã audit thủ công.

## File chỉ xuất hiện sau lần chạy mới

Notebook 04 mới sẽ tạo `gru_history.csv` và `phobert_history.csv`: một dòng/epoch gồm train `loss`, metric validation và `selected_checkpoint`. Snapshot hiện tại chưa có hai file này; lịch sử cũ đã nằm trong `../gru_run.json` và `../phobert_run.json`.

`selected_checkpoint=True` đánh dấu epoch tốt nhất theo Macro-F1 validation tại 0,5. Có thể xảy ra epoch cuối không phải checkpoint tốt nhất.
