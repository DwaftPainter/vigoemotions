# Bộ ôn Final oral defence

Đọc theo thứ tự 00–04; mỗi tài liệu có Markdown và PDF cùng tên.

| Bộ | Markdown | PDF |
| --- | --- | --- |
| Tra cứu nhanh | [00](00_quick_revision_vi.md) | [PDF](00_quick_revision_vi.pdf) |
| Kiến thức nền | [01](01_foundations_vi.md) | [PDF](01_foundations_vi.pdf) |
| Protocol và kết quả | [02](02_protocol_results_vi.md) | [PDF](02_protocol_results_vi.pdf) |
| 34 câu Q&A, có đáp án tiếng Anh ngắn | [03](03_questions_answers_vi.md) | [PDF](03_questions_answers_vi.pdf) |
| Tính tay và mock defence | [04](04_practice_vi.md) | [PDF](04_practice_vi.pdf) |

Số liệu RQ1 và RQ4 lấy từ CSV/JSON đã kiểm chứng. Ví dụ tính tay dùng dữ liệu giả lập. Report Midterm cung cấp bối cảnh baseline.

Chỉnh nội dung ở `templates/`, rồi chạy `.venv/bin/python scripts/build_oral_defence.py` từ project root. [Verification](verification.json) ghi nguồn, SHA và số trang. [Source manifest](source_manifest.json) kiểm file gốc còn giữ; [cleanup manifest](cleanup_manifest.json) liệt kê file bỏ khỏi nhánh ôn, phục hồi được từ nhánh nguồn tại `289f2fb`.

[README chính](../../README.md) có lộ trình 30 phút, 2 giờ hoặc hai buổi và liên kết tới report Final đầy đủ trên `submission/ati-final-rq4`.
