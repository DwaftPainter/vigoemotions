# Các biểu đồ PNG

[← Mục lục reports](../README.md)

| File | Xem gì? | Bảng tương ứng |
| --- | --- | --- |
| [label_frequency.png](label_frequency.png) | Tần suất đủ 27 nhãn trên train, màu theo head/mid/tail | [label_frequency.csv](../tables/label_frequency.csv) |
| [distributions.png](distributions.png) | Ba phân bố: cardinality, token khoảng trắng, ký tự; overlay các split | [cardinality.csv](../tables/cardinality.csv), [text_lengths.csv](../tables/text_lengths.csv) |
| [train_cooccurrence.png](train_cooccurrence.png) | Ma trận số lần đồng xuất hiện; count cao có thể do nhãn vốn phổ biến | [train_cooccurrence.csv](../tables/train_cooccurrence.csv) |
| [train_correlation.png](train_correlation.png) | Ma trận tương quan phi; chỉ tính trên train | [train_correlation.csv](../tables/train_correlation.csv) |
| [tfidf_lr_diagnostics.png](tfidf_lr_diagnostics.png) | Trái: F1 theo threshold; phải: tần suất train và F1 từng nhãn TF-IDF tại 0,5 | Các bảng `tfidf_lr_thresholds_val.csv` và `tfidf_lr_per_label.csv` trong [tables](../tables/README.md) |
| [gru_diagnostics.png](gru_diagnostics.png) | Cùng hai phân tích trên cho BiGRU | Các bảng `gru_*` trong [tables](../tables/README.md) |
| [phobert_diagnostics.png](phobert_diagnostics.png) | Cùng hai phân tích trên cho PhoBERT | Các bảng `phobert_*` trong [tables](../tables/README.md) |

EDA do `analyze_data.py` tạo; diagnostics do `evaluate.py` tạo. CSV là nguồn để đọc giá trị chính xác; ảnh là cách trình bày cùng thông tin.

`neural_learning_curves.png` sẽ có sau khi chạy notebook 04 mới: train BCE loss và validation Macro-F1 theo epoch cho hai neural baseline. **Chưa có file này trong snapshot hiện tại.** History cũ vẫn đọc được trong run JSON. Biểu đồ không có validation loss và không tự chứng minh mô hình hội tụ hay overfit.

Hình ablation nằm riêng ở [ablation_mixed_neutral/comparison.png](../ablation_mixed_neutral/comparison.png); learning curves của sweep sẽ nằm trong thư mục [phobert_tuning](../phobert_tuning/README.md), không thay hình baseline.
