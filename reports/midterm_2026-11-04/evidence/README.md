# Nguồn kết quả Neutral 28 nhãn — kiểm chứng 04/11/2026

Trích từ `vigoemotions_neutral28_private_results.zip`, đối chiếu notebook `notebooks/06_kaggle_neutral_28_labels.ipynb`. Hash ZIP, notebook và từng file trích nguyên nằm trong [provenance.json](provenance.json). Chỉ lưu CSV tổng hợp và metadata; checkpoint, vocabulary, probabilities và output comment nguồn vẫn ở nguồn riêng tư.

| Muốn kiểm tra | File |
| --- | --- |
| Metric đủ 28 nhãn và phép chiếu 27 | [metrics_summary.csv](metrics_summary.csv) |
| Neutral precision/recall/F1, support, số dự đoán dương | [neutral_analysis.csv](neutral_analysis.csv) |
| Đếm Neutral-only/mixed theo split, trước loại overlap | [annotation_counts.csv](annotation_counts.csv) |
| Cả 28 nhãn theo từng mô hình | [per_class_metrics.csv](per_class_metrics.csv) |
| Baseline gốc và study mới, điều kiện khác nhau | [comparison_27_vs_28.csv](comparison_27_vs_28.csv) |
| Delta từng cảm xúc trên cùng 27 nhãn | [emotion27_per_class_comparison.csv](emotion27_per_class_comparison.csv) |
| Config và môi trường của study | [config.json](config.json), [study.json](study.json) |
| Cấu hình thật, best epoch, thời gian và tham số | [TF-IDF run](models/tfidf_lr/run.json), [BiGRU run](models/gru/run.json), [PhoBERT run](models/phobert/run.json) |
| Đường học theo epoch | [BiGRU history](models/gru/history.csv), [PhoBERT history](models/phobert/history.csv) |
| Kiểm chứng lại metric và Neutral TP/FP/FN | [verification.csv](verification.csv) |
| Số mẫu cho năm kiểu dự đoán, tính trực tiếp từ NPZ | [prediction_category_counts.csv](prediction_category_counts.csv) |

Đã tính lại cả bốn metric trên 28 và 27 nhãn, toàn bộ precision/recall/F1/support từng nhãn và Neutral TP/FP/FN. Sai số metric tổng hợp lớn nhất dưới 1e-12. Ba NPZ cùng 2.066 ID validation, đúng thứ tự 28 nhãn và targets từ CSV local. Checksum ba CSV nguồn và ID các split khớp study; train còn 16.530 sau loại một overlap. Baseline gốc được kiểm tra lại ID/targets/metric từ `reports/predictions/*_val.npz`; không huấn luyện lại.

Hai bảng `verification.csv` và `prediction_category_counts.csv` được tính thêm trong lần cập nhật docs, không có sẵn trong ZIP. Với bảng category, năm nhóm loại trừ nhau: empty; Neutral-only; một cảm xúc không Neutral; nhiều cảm xúc không Neutral; Neutral + ít nhất một cảm xúc. Mỗi mô hình có tổng 2.066 mẫu. Có 31 ca PhoBERT dự đoán mixed; đó là đầu ra mô hình, không phải 31 ca annotation mixed (validation chỉ có 19 annotation mixed).

**Không trộn baseline:** nguồn đối chiếu trong ZIP là bộ gốc (PhoBERT 0,439320, 4 epoch), không phải [lượt chạy lại 23/10](../../pre_midterm_2026-10-23/evidence/model_comparison.csv) (0,467850). `study.json.baseline_config` là config tại init có 12 epoch; đối chiếu thời lượng baseline thật bằng [reports/phobert_run.json](../../phobert_run.json). File comparison giữ nguyên đường dẫn runtime `/kaggle/working/...` để bảo toàn nguồn; bảng trong thư mục này là bản sao local.

**Giới hạn thực thi:** code cell 24 (execution_count 16) tìm ví dụ validation lỗi JVM/VnCoreNLP. ZIP không có `validation_prediction_cases.csv`; bảng đếm bổ sung ở đây không khẳng định cell đó chạy thành công. Inference bốn câu tự viết và huấn luyện/xuất metric đã có output. Test không được chấm. Notebook nguồn có output văn bản dataset, cần giữ private; thư mục evidence không sao chép các output đó.

[Phần diễn giải đưa vào báo cáo](../neutral28_results.md) • [Lời dẫn và Q&A](../../../docs/midterm_presentation_2026-11-04.md).
