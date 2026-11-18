# Đọc error analysis khi ôn oral defence

Packet có62 comments:20 core FN,20 core FP,2 Neutral errors,5 high-cardinality và15 ca cho ba cặp chọn từ train. [Case index](cases.csv), [sampling design](sampling_design.json) và [review notes](ai_assisted_review.md) cung cấp bằng chứng text-free.

[Summary nhận xét](ai_assisted_summary.json) và [provenance](ai_assisted_provenance.json) được giữ nguyên từ nguồn. Human worksheet chưa có xác nhận; [summary](summary.json) ghi số hoàn tất bằng0. Đây là quan sát định tính chưa có independent adjudication, không phải reference labels mới hoặc ước lượng annotation-error prevalence.

Khi bị hỏi, phân biệt sample-label pair và comment; FN/FP giữ từ fixed và mới sau per-label; observation và hypothesis. Không đổi targets, metric hay cutoffs từ các notes. Raw comments/worksheets private không được đưa vào tài liệu ôn hoặc repo public.

Hướng dẫn chuẩn bị/kiểm worksheet và công cụ vận hành đầy đủ nằm ở `feature/final28-report`, commit `289f2fb`. Nhánh ôn chỉ đọc packet đã giữ, không tạo audit mới.
