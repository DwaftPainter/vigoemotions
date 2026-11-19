# Điền thông tin trước khi nộp report

Bản làm việc: [final_report_ati.docx](../reports/final_report_ati.docx). Được dựng từ đúng [template giảng viên](ATI_Final_Report_Template.docx); bản phân tích cũ `final_report.*` vẫn giữ riêng. Nhánh: `submission/ati-final`.

## Những ô để bạn điền

| Vị trí | Nội dung còn thiếu |
| --- | --- |
| Cover | `[G00]`, `[CLASS CODE]`, họ tên và MSSV của đủ thành viên; thêm/bớt hàng nếu số thành viên khác ba |
| Cover và Appendix B | Đánh dấu `*` đúng người đại diện nộp |
| Running header | Thay `[PREFIX]` bằng `<CLASS>_G<NN>_P02` ở header của phần thân/phụ lục |
| Appendix A | `[CHECKPOINT LINK, IF REQUIRED]`: link riêng đã cấp quyền cho giảng viên nếu cần |
| Appendix A | `[FINAL EXECUTED NOTEBOOK PATH]`: đường dẫn notebook Final đã chạy, giữ outputs, trong gói nộp |
| Appendix B | MSSV, họ tên, nhiệm vụ, tỷ lệ đóng góp số nguyên của mọi thành viên; tổng 100, cả nhóm thống nhất |
| Appendix C | Công cụ/phiên bản, mục đích, các sections/files thực tế bị tác động; điền theo việc nhóm đã sử dụng |
| Tên file khi đóng gói | `<PREFIX>_report.docx` trong `<PREFIX>/report/` |

## Kiểm sau khi điền

Mở bằng Word, cập nhật toàn bộ mục lục và các fields (Update entire table), rồi lưu DOCX. Kiểm lại phần thân Sections 1–7 vẫn 15–20 trang, không tính cover/abstract/TOC/references/appendices. Bản render kiểm tra hiện có 17 trang thân; Word hoặc nội dung bổ sung có thể làm phân trang thay đổi. PDF kèm theo chỉ để đọc và kiểm bố cục, file nộp là DOCX.

Report giữ nguyên các kết quả đã có; không train, inference hoặc fit lại thresholds để tạo báo cáo. Thông tin chưa biết còn trong dấu ngoặc vuông; numeric citations `[1]`–`[10]` là trích dẫn hợp lệ, không xóa khi tìm placeholders.

## Nguồn và dựng lại

- Nội dung: [ati_report_content.md](ati_report_content.md); số liệu được lấy từ saved evidence bởi [build_ati_report.py](../scripts/build_ati_report.py).
- Kiểm chứng: [final_report_ati_verification.json](../reports/final_report_ati_verification.json).
- Dựng bản mới: `.venv/bin/python scripts/build_ati_report.py`. Lệnh này tạo lại DOCX, vì vậy hãy lưu bản Word đã điền sang tên nộp riêng trước khi chạy lại.
- Sau khi render DOCX bằng Word/LibreOffice ra PDF, có thể cache mục lục native từ bố cục đó bằng `.venv/bin/python scripts/build_ati_report.py --cache-toc-from-pdf /path/to/rendered.pdf`, rồi render/kiểm lại. Phần thân dùng số trang bắt đầu từ 1; mục lục vẫn có fields để Word cập nhật về sau.

Chưa đóng ZIP nộp, chưa thay thông tin chưa được cung cấp và chưa xác nhận deadline/oral.
