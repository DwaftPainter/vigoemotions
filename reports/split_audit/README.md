# Audit split ViGoEmotions — bằng chứng phân bố

Trạng thái toàn vẹn: **PASS**. Train/validation/test thực nghiệm: **16530 / 2066 / 2067**. Loại 1 mẫu train trùng heldout theo pipeline; không sửa CSV nguồn hoặc fixed split IDs. Các kiểm tra cụ thể ở `integrity_checks.csv` và `leakage_audit.csv`.

Trên 27 cảm xúc, chênh lệch prevalence tuyệt đối lớn nhất so với train: validation **1.6324 điểm phần trăm** (joy); test **0.9387 điểm phần trăm** (relief). Đây là tỷ lệ nhãn dương trên số mẫu, không phải chênh lệch count thô. Đọc thêm mức biến động tương đối và support trong `label_prevalence.csv`.

Neutral có 821 / 94 / 116 mẫu dương trên train/val/test; gồm Neutral-only và mixed, không coi mixed là Neutral-only. Cardinality theo cả hai cách mã hóa ở `cardinality_distribution.csv`; khoảng cách TV ở `cardinality_total_variation.csv`. Cặp nhãn được chọn trên train, chỉ đối chiếu validation trong `train_selected_pairs_validation.csv`; không tạo ma trận nhãn test.

**Diễn giải dùng trong báo cáo:** Nhóm giữ split được cấp, kiểm tra toàn vẹn và định lượng độ lệch phân bố đa nhãn bằng notebook audit tái chạy được. Những bảng này là bằng chứng về chất lượng phân bố, không xác nhận thuật toán tạo split gốc là iterative stratification. Không dùng điểm mô hình/test để lựa chọn hoặc đổi split. Mức chấp nhận phân bố và việc dùng split được cấp cần đối chiếu yêu cầu giảng viên; notebook không tự gán chứng nhận “đạt stratification”.

**Tái chạy:** mở `notebooks/07_split_distribution_audit.ipynb`, kernel project, Run All. Output mặc định tại `reports/split_audit/`. Không train mô hình hoặc tải dependency qua mạng. Chỉ có thống kê tổng hợp, ID và metadata; không có bình luận nguồn. Kết quả là chẩn đoán mô tả; không dùng ngưỡng phân bố đặt sau khi xem số liệu để công bố PASS.

Nguồn yêu cầu: `docs/02-vigoemotions-vietnamese-emotion.pdf` (Midterm A; Reproducibility Rules), `docs/student-guide.pdf` (Midterm; Submission Package). `provenance.json` lưu môi trường và checksum. Các file ở đây độc lập với bảng baseline và báo cáo midterm hiện hữu.
