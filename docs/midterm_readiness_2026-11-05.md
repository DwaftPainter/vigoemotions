# Kiểm tra trước khi chốt midterm — 05/11/2026

| Mục | Trạng thái và bằng chứng |
| --- | --- |
| Audit split | Đã chạy notebook 07; toàn vẹn PASS, fixed IDs khớp; thống kê phân bố ở `reports/split_audit/` |
| Yêu cầu stratification | Có bằng chứng mô tả để bảo vệ việc giữ split; chưa xác nhận thuật toán gốc hoặc quyết định chấp nhận của giảng viên. Không tự gắn chứng nhận “stratified” |
| Cấu hình từng lượt | Đã tách profile baseline 27/4 epoch, Neutral 28/12 epoch, Kaggle 25/20; xem `reproducibility_profiles.md` |
| Notebook và package | Bản hiện tại xóa output/execution metadata; packager lọc private artifact và làm sạch notebook khi ghi ZIP |
| Xác suất dự đoán | Có NPZ validation baseline 27 và cả ba model study 28, chỉ ID/nhãn/xác suất |
| Per-label F1 | Phụ lục A của report có đủ mọi nhãn cho cả ba model, tách baseline 27 và study 28 |
| Phân tích lỗi | Giữ 10 ca TF-IDF gốc; bổ sung đọc định tính 9 ca PhoBERT 28 có hỗ trợ AI, cần nhóm xác nhận |
| Cell tìm ví dụ lỗi JVM | Đã sửa để đọc xác suất validation lưu sẵn, không nạp lại JVM; đã kiểm chứng offline với fixture và cả ba NPZ thật. Không chạy lại toàn bộ training GPU |
| Tên/MSSV/phân công | CHƯA CUNG CẤP — bổ sung `team_information.md` và trang đầu report/slide trước khi nộp |
| Sweep learning rate/test | Chưa chạy; không yêu cầu hoàn tất cho midterm. Test tiếp tục giữ cho Final |

Các điểm số mới không thay đổi. Không sửa dữ liệu nguồn hoặc fixed IDs, không đổi split, không train lại và không chấm test. Đã lưu config và bằng chứng riêng để không trộn baseline gốc, lượt chạy lại hoặc study Neutral.

Nguồn yêu cầu: `student-guide.pdf`, mục Midterm/Submission Package/Academic Integrity; `02-vigoemotions-vietnamese-emotion.pdf`, Midterm A–E và Reproducibility Rules. Yêu cầu tối thiểu 20 FN/20 FP, ba decoder và metric test thuộc Final. Yêu cầu ít nhất ba seed chỉ áp dụng paper track.

Gói công khai dùng archive đã sanitize, không kèm `.git/` hoặc backup riêng tư. Lịch sử cũ còn output nên chưa thể coi toàn bộ lịch sử Git là gói chia sẻ sạch. Không thêm kết quả hoặc thông tin nhóm chưa có nguồn.


## Kiểm chứng kỹ thuật

61 test case được kiểm chứng qua hai môi trường: 59 test qua ở Python 3.14 của project; hai smoke test tiny neural được chạy bổ sung offline trên CPU trong môi trường Python 3.12 đã có Torch/Transformers và đều qua. Không gọi đây là một lần full suite xanh trong cùng môi trường; không bỏ/skip test. Audit số liệu EDA/baseline/ablation qua. Đã kiểm tra gói ZIP, PDF 19 trang, slide 20 trang và 18 đoạn trích code cẩm nang. Chi tiết ở `reports/midterm_readiness_verification_2026-11-05.json`.
