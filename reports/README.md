# Đọc kết quả ViGoEmotions

**Nguồn chính hiện tại là Kaggle 20 epoch, notebook 08/09 executed.**

1. [Báo cáo](midterm_report.pdf) / [Markdown](midterm_report.md).
2. [15 slide chính + 2 bảng bổ trợ tiếng Anh](midterm_slides.pdf) / [speaker notes](midterm_slides.md).
3. [Kết quả, Q1–Q5 và evidence mới](kaggle_equal20_2026-11-05/README.md).
4. [Script tiếng Anh + Q&A](../docs/midterm_presentation_en.md) / [PDF](../docs/midterm_presentation_en.pdf).

`kaggle_equal20_2026-11-05/model_comparison.csv` có đủ sáu model và metric đúng tập nhãn; `paired_emotion27_delta.csv` dùng cùng 27 cảm xúc; `neutral_metrics.csv` chỉ Neutral. `provenance.json` ghi checksum, môi trường và đối chiếu LF/CRLF. Evidence chỉ chứa ID/nhãn/xác suất/metadata, không có raw text hoặc trọng số.

Các bảng EDA trong `tables/label_frequency.csv`, `dataset_summary.csv`, `text_lengths.csv`, train co-occurrence/correlation và hình EDA vẫn hợp lệ cho cùng dữ liệu. `split_audit/` là bằng chứng phân bố/toàn vẹn, không chứng nhận thuật toán stratification gốc.

Snapshot `midterm_2026-11-04/`, `pre_midterm*`, lượt local và các tài liệu theo ngày trước đây đã chuyển vào [archive/reports/](../archive/README.md). Các bảng **model** trong tables/ và run JSON ở reports root vẫn phục vụ script/test lịch sử. Không dùng điểm trong các nguồn đó cho báo cáo hiện tại. EDA dùng chung vẫn giữ nguyên; toàn bộ phân tích model hiện tại ở kaggle_equal20_2026-11-05/.

Lệnh build chính: `python scripts/import_equal20_results.py`, sau đó `python scripts/build_report.py`. Không train lại, không chấm test và không ghi đè artifact lịch sử.

Redesign hiện tại chỉ đổi hình thức của 15 slide đã chốt, đối chiếu nguyên văn với snapshot `kaggle_equal20_2026-11-05/slides.json`. Hai trang 16–17 là bảng dự phòng cho Q&A, không nằm trong script nói có tính giờ. Build riêng presentation bằng `python scripts/build_equal20_presentation.py`; bố cục ở [midterm_slide_design.json](midterm_slide_design.json), kiểm chứng bảo toàn nội dung ở [midterm_presentation_verification.json](midterm_presentation_verification.json).
