# Neutral trong kết quả chính Kaggle 20 epoch

Dùng notebook 09 và profile configs/kaggle_equal_20ep.json; BiGRU/PhoBERT đều 20 epoch, best lần lượt 20 và 15. Nguồn là notebook executed 09 và ZIP vigoemotions_kaggle_equal20_28labels_private_results.zip. [Báo cáo](../reports/midterm_report.md), [bằng chứng](../reports/kaggle_equal20_2026-11-05/README.md), [cách chạy](kaggle_equal_epochs.md).

27 cảm xúc giữ IDs 0–26; Neutral là output 27. [27] → chỉ Neutral dương, [2,27] giữ cả joy và Neutral. CSV nguồn có 725 Neutral-only và 306 mixed; validation có 94 Neutral dương. Giữ annotation mixed, không dùng softmax hoặc decoder ép loại trừ. Dự đoán rỗng không tự động là Neutral.

Checkpoint 28 chọn bằng emotion27 Macro-F1 tại 0,5, hòa lấy sớm; không chọn theo Macro-F1 đủ 28. Cả 20 epoch vẫn chạy xong trước khi nạp best. Ngưỡng chính 0,5; test chưa chấm.

| model | precision | recall | f1 | support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + OvR LR | 0.5000 | 0.0106 | 0.0208 | 94 | 1 | 1 | 93 |
| BiGRU | 0.1778 | 0.0851 | 0.1151 | 94 | 8 | 37 | 86 |
| PhoBERT | 0.4018 | 0.4787 | 0.4369 | 94 | 45 | 67 | 49 |

Macro-F1 đủ 28: TF-IDF 0,231040; BiGRU 0,408615; PhoBERT 0,578341. PhoBERT 28 chiếu về cùng 27 cảm xúc đạt 0,583580, thấp hơn bản 27 đạt 0,597410. Chưa có bằng chứng thêm Neutral cải thiện emotion27 trong lượt một seed này. Không so trực tiếp Macro-F1 27 với Macro-F1 đủ 28 để tính hiệu ứng Neutral.

Sáu ca Neutral FP/FN mới được đọc với hỗ trợ AI, giữ ID và diễn giải trừu tượng tại reports/kaggle_equal20_2026-11-05/phobert28_error_review.csv; nhóm cần xác nhận. Checkpoint/tokenizer/vocab nằm trong ZIP private, không đưa lên repo công khai.

Notebook 06 và study 12 epoch trước đây là lịch sử, không phải nguồn điểm hiện tại.
