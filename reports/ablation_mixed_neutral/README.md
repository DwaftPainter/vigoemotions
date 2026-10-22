# Ablation mixed-neutral — một thí nghiệm đã hoàn tất

[← Mục lục reports](../README.md)

Đọc [experiment_summary.md](experiment_summary.md) trước để hiểu câu hỏi, thiết kế và giới hạn. Đây là thí nghiệm **TF-IDF tại threshold 0,5**, tách riêng khỏi ba baseline chính.

- **A:** giữ 16.530 mẫu train đã loại trùng; bỏ ID neutral, giữ nhãn cảm xúc.
- **B:** bỏ thêm 251 mẫu mixed-neutral khỏi train, còn 16.279 mẫu.
- Cả hai dùng cùng 2.066 mẫu validation và cùng ground truth. Không chọn cấu hình bằng test.

| File | Nội dung / ý nghĩa |
| --- | --- |
| [experiment_summary.md](experiment_summary.md) | Báo cáo tự sinh: câu hỏi nghiên cứu, thiết kế, bảng kết quả, diễn giải và giới hạn |
| [experiment_config.json](experiment_config.json) | Trạng thái `completed`, cấu hình, môi trường/checksum dữ liệu, điều kiện A/B, tier và phạm vi đo |
| [comparison.csv](comparison.csv) | Metric toàn validation, thời gian, số mẫu train và feature cho A/B; `B_minus_A` có dấu, `absolute_difference` chỉ độ lớn |
| [comparison.png](comparison.png) | Biểu đồ bốn metric; Hamming càng thấp càng tốt, ba metric khác càng cao càng tốt |
| [per_label_f1.csv](per_label_f1.csv) | Precision/recall/F1, count train và delta từng nhãn |
| [frequency_tier_comparison.csv](frequency_tier_comparison.csv) | Metric theo head/mid/tail; tier cố định theo train A, không chia lại theo B |
| [validation_subgroup_comparison.csv](validation_subgroup_comparison.csv) | Tách 19 mẫu validation mixed-neutral và 2.047 mẫu còn lại; nhóm 19 mẫu quá nhỏ để kết luận mạnh |
| [A_training_ids.csv](A_training_ids.csv) | ID train thật dùng cho A, không có text |
| [B_training_ids.csv](B_training_ids.csv) | ID train thật dùng cho B; đối chiếu để xác nhận chỉ loại đúng mixed-neutral |
| [A_val.npz](A_val.npz) | ID, ground truth, xác suất và thứ tự nhãn của A |
| [B_val.npz](B_val.npz) | Cùng cấu trúc và ID/ground truth như A; xác suất thuộc mô hình B |

Hai NPZ có bốn key `sample_id`, `true_label_vector`, `probability_vector`, `labels`; không có key cardinality riêng như NPZ baseline. Có thể tính cardinality bằng `true_label_vector.sum(axis=1)`.

Macro-F1 của B tăng khoảng **0,00159** (0,159 điểm phần trăm) so với A, nhưng LRAP giảm nhẹ. Chưa chạy random-exclusion control nên chưa tách được ảnh hưởng mâu thuẫn nhãn khỏi ảnh hưởng giảm dữ liệu/vocabulary. Một seed/split/model không chứng minh cải thiện có ý nghĩa thống kê.

Chạy lại bằng `scripts/ablate_mixed_neutral.py --output reports/ablation_mixed_neutral/rerun_01` với thư mục chưa tồn tại. Giữ toàn bộ nhóm file cùng nhau; không ghi đè kết quả lịch sử hoặc thay baseline chính bằng kết quả B mà không đổi giao thức rõ ràng.
