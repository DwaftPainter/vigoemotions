# Trạng thái midterm — kết quả chính Kaggle 20 epoch

| Hạng mục | Bằng chứng / trạng thái |
| --- | --- |
| Training | Hai notebook 08/09 executed, mỗi bản 12 cell code có output, không error; sáu model hoàn tất |
| Ngân sách neural | BiGRU/PhoBERT đều đủ 20 epoch ở cả hai bộ nhãn; best GRU 17/20, PhoBERT 14/15 |
| Kết quả chính | reports/kaggle_equal20_2026-11-05/model_comparison.csv; report/slide/cẩm nang dùng nguồn này |
| Kiểm chứng | Tính lại sáu NPZ khớp CSV; cùng ID/targets/label order, config/history/checkpoint; provenance ghi hash |
| LF/CRLF | Chênh hash source/profile/CSV ở 28 được chứng minh bằng biến thể xuống dòng; không sửa manifest gốc |
| EDA và split | Cùng dữ liệu; audit 07 PASS toàn vẹn và fixed IDs; stratification gốc chưa xác minh |
| Q1–Q5 | Đã tính lại per-label, frequency–F1, pair preservation, style delta và threshold sensitivity từ NPZ mới |
| Phân tích lỗi | Sáu ca PhoBERT 28 mới, AI hỗ trợ; nhóm cần xác nhận. Danh sách tự sinh không được gọi là đã audit thủ công |
| Test | Chưa chấm mô hình; giữ cho Final |
| Thông tin nhóm | Cần bổ sung tên/MSSV/phân công trong team_information.md |
| Learning rate / Final decoder | Chưa chạy grid learning rate hoặc Final test; không là kết quả của lượt 20 epoch |

Hai executed notebook và ZIP nguồn được giữ nguyên. Importer không train, không inference lại, không nạp pickle, không xuất raw text/trọng số. Bảng model ít epoch, các báo cáo theo ngày cũ và output notebook 01–07 chỉ là lịch sử. Các artifact lịch sử không bị chỉnh thành số mới.

Trước buổi trình bày: điền thông tin nhóm; nhóm xác nhận phần đọc lỗi; chuẩn bị giải thích giới hạn official split, một seed và chưa có test. Thuyết trình bằng tiếng Anh: dùng docs/midterm_presentation_en.md (có bản PDF) và reports/midterm_slides.pdf. Lời dẫn tiếng Việt giữ để ôn tập.

## Kiểm tra bộ tài liệu hiện tại

21 kiểm thử liên quan đều qua (14 workflow, 5 import/provenance, 2 report). Report PDF 10 trang, slide 15 trang, cẩm nang 22 trang và 16 đoạn trích mã đã được kiểm tra. Hai notebook executed giữ nguyên SHA-256 và output; các thành viên evidence trích từ ZIP giữ nguyên byte. Đã kiểm tra link và xem các trang PDF đại diện. Chi tiết tại [verification.json](../reports/kaggle_equal20_2026-11-05/verification.json).

Presentation hiện tại có 18 slide 16:9, 11 bảng, sơ đồ và biểu đồ vector; script tiếng Anh 935 từ. Q1–Q4 có slide riêng; Q5 có hai slide để giữ bảng metric và đường threshold dễ đọc. Đã xem từng slide, chạy 15 kiểm thử presentation/import/report và đối chiếu checksum của toàn bộ evidence cùng hai notebook executed. Q3 joint TP/support/recall và Q5 F1/Hamming được tính lại từ NPZ; Q4 được đối chiếu bảng style của cả ba model. [Kiểm chứng presentation hiện tại](../reports/midterm_presentation_verification.json) tách riêng; các file trong thư mục evidence không bị sửa. Bản định nghĩa slide mới ở `reports/midterm_slide_design.json`.
