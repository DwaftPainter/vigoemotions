# Kế hoạch Final và oral defence — hướng 28 class

Cập nhật 19/11/2026: **28 outputs gồm Neutral**, checkpoint PhoBERT epoch 15. Validation/test và AI-assisted audit đã hoàn tất; [bản nháp báo cáo PDF](../reports/final_report.pdf) / [Markdown](../reports/final_report.md) cần chuyển sang template nộp mới. Nhóm chốt dùng AI-assisted audit và công bố giới hạn chưa có independent human adjudication. Deadline chính thức chưa rõ; dự kiến cuối tháng 11. Mục tiêu nội bộ: sẵn sàng 27/11, dự phòng 28–30/11.

## Quy định nộp mới

[Submission Guide](../docs/ATI_Submission_Guide.pdf) và [Word template](../docs/ATI_Final_Report_Template.docx) mới được bổ sung, xem [checklist nguồn và khoảng cách hiện tại](../docs/submission_requirements.md). Nộp một ZIP `<CLASS>_G<NN>_P02.zip` với một root cùng PREFIX, gồm slides/, code/, report/. Slides là đúng bản Midterm tiếng Anh PDF đã trình chiếu; không có slides Final. Report nộp DOCX theo template, phần thân Sections 1–7 dài 15–20 trang, có Appendix A/B/C, phần đóng góp tổng 100 và kê khai công cụ GenAI. Giữ outputs của executed notebooks. Bản report 11 trang hiện tại là bản nháp tham khảo, chưa phải bản nộp đạt chuẩn mới.

## Mục tiêu và các quyết định

Câu hỏi chính: global/per-label thresholds tối ưu trên validation có cải thiện test Macro-F1 **all28** so với fixed 0,5, khi weights không đổi? Báo thêm emotion27 projection và Neutral riêng. Phép chiếu dùng cùng thresholds đã chọn bằng all28 objective; không phải model/train/tuning 27 riêng.

- [Protocol v2](../docs/final_protocol.md), [config](../configs/final_thresholds.json), [execution guide](../docs/final_execution.md).
- [Đề](../docs/02-vigoemotions-vietnamese-emotion.pdf), trang 2–4 và [student guide](../docs/student-guide.pdf) là căn cứ yêu cầu; đề RQ1 viết cho 27 emotions, nhóm mở rộng thành 28 và phải trình bày rõ khác biệt.
- [Prototype 27 đã lưu](../archive/final27_prototype_2026-11-11/snapshot.json); evidence equal20 và raw/private data không đổi.
- Primary weights: saved model/tokenizer trong ZIP 28, epoch 15 sau 20 epoch. Tiêu chí chọn checkpoint gốc là emotion27 Macro-F1 tại 0,5; giữ nguyên lịch sử đó.
- Fixed/global/per-label cùng scores 28; grid 0,05–0,80 bước 0,05, hòa chọn cao hơn, thiếu positive validation fallback 0,5. Giữ mixed Neutral annotations, không ép loại trừ.
- Tiers giữ 27 emotions, 9 nhãn mỗi nhóm theo retained train; Neutral báo riêng. Cardinality all28 và emotion27 phải tách tên.
- Validation dùng để chọn ngưỡng; khóa trước test; notebook 10 không tự chạy test.

## Đầu việc và tiêu chí hoàn thành

### T1 — Protocol và phân công

**Đầu ra:** protocol/config v2, train-only tiers, phân công thật.

- [x] Ghi rõ 28-label objective, epoch 15, label order, Neutral policy, grid/ties/fallback và error audit.
- [x] Tách báo cáo all28/emotion27/Neutral; ghi rõ liên hệ với RQ1 27 emotions trong đề.
- [ ] Điền tên/MSSV/phân công và xác nhận lịch/thời lượng oral khi giảng viên thông báo.

**Xác minh:** đối chiếu đề và provenance; thành viên đọc được quyết định và giới hạn. **Phụ thuộc:** không. **Quy mô:** nhỏ, tài liệu/cấu hình.

### T2 — Khôi phục model 28 và tái tạo validation

**Đầu ra:** private local model/tokenizer/VnCoreNLP, actual validation reproduction report.

- [x] Có code kiểm hash ZIP/config/safetensors/tokenizer, giữ canonical 28 outputs và saved preprocessing.
- [x] Source NPZ 2.066 × 28, fixed Macro-F1 ≈0,578341, fixed IDs/targets và tiers đã kiểm chứng local.
- [x] `restore` model 28 thật vào `data/final28/`, khớp hashes và saved 28-label mapping; giữ source weights 27 cũ riêng.
- [x] Actual Kaggle `reproduce-val`: sai lệch xác suất tối đa 0, fixed decisions giống hệt, ghi runtime/asset hashes.

**Xác minh:** actual checkpoint phải chạy, mock tests không thay thế proof; không tự nới tolerance. **Phụ thuộc:** T1. **Quy mô:** module inference/study và evidence sinh tự động.

### T3 — Fit và khóa ngưỡng validation

**Đầu ra:** global scalar + 28 cutoffs, bảng/projection/Neutral metrics, pre-test lock.

- [x] Có thuật toán và test cho objective all28, ties/fallback/boundary/shape; gate chặn fit khi thiếu actual reproduction.
- [x] `fit` sau T2 PASS, lưu global 0,4 và đủ 28 ngưỡng/support/train count/tier/fallback.
- [x] Review validation, có lock bind config/protocol/implementation/thresholds/source/reproduction; test liên kết đúng lock hiện tại.

**Xác minh:** all28 và emotion27 có scope đúng, Neutral F1 tính positive label; thử thay nguồn/ngưỡng/runtime phải bị phát hiện. **Phụ thuộc:** T2. **Quy mô:** core/study/CLI cùng output.

**Checkpoint 1 — hoàn tất:** model 28 tái tạo đúng validation, ngưỡng 28 đã fit và khóa; có actual execution evidence.

### T4 — Held-out test một lượt

**Đầu ra:** continuous test scores trước decoding và ba pred matrices 28 cột.

- [x] Test artifact ghi `completed`; 2.067 mẫu đúng IDs/targets/checksum nguồn, cùng assets/runtime với validation và bind lock hiện tại.
- [x] Lưu IDs/true/probability vectors/labels, pred_fixed/global/per_label, label_cardinality (28), emotion_cardinality (27).
- [x] Cả hai scopes dùng cùng test scores; không re-tune hoặc inference thêm khi import/review/report.

**Xác minh:** pred tái tạo từ saved scores + thresholds; NPZ/lock/runtime/CSV nguồn đã đối chiếu. ZIP ghi một completed attempt, không độc lập chứng minh mọi hoạt động session trước đó. **Phụ thuộc:** Checkpoint 1. **Trạng thái:** actual execution và local artifact audit hoàn tất; giữ nguyên completed-test gate.

### T5 — Bảng, hình và trả lời RQ

- [x] Metric validation/test và delta theo decoder cho all28; emotion27 projection, Neutral riêng, tier và hai cardinality views; 11 bảng đã tính lại và kiểm chứng.
- [x] Threshold profile đủ 28 nhãn, bốn hình PNG/PDF và generalization gap theo scope; support rõ, nhóm rỗng N/A.
- [x] Results/Discussion ghi global +0,1289 pp test Macro-F1, per-label −0,4549 pp, Hamming tốt nhất ở fixed; giới hạn một seed/repeated validation selection.

**Xác minh:** metric tables được tái sinh từ saved outputs và đối chiếu NPZ; `build_final_figures.py` kiểm imported hashes/completed-test bindings trước khi vẽ. **Phụ thuộc:** T3/T4. **Đầu ra:** CSV/figures/RQ analysis đã có trong `reports/final28/`; chưa thay thế T6 human audit.

### T6 — Qualitative audit, dùng AI-assisted theo quyết định nhóm 17/11

- [x] Chuẩn bị packet theo hash: 20 core FN + 20 core FP, 2 Neutral errors, 5 high-cardinality, 15 cases cho ba cặp chọn từ train; tổng 62 bình luận. [Hướng dẫn](../reports/final28/human_audit/README.md), worksheet private và validator đã có; không tự điền nhận xét/reviewer.
- [x] AI-assisted review 17/11 theo yêu cầu: đọc nội dung 62 ca, ghi quan sát và giả thuyết riêng, phân biệt lỗi giữ nguyên/mới do threshold, đánh dấu ca cần adjudication. [Kết quả](../reports/final28/human_audit/ai_assisted_review.md) không thay thế human confirmation.
- [x] AI-assisted audit đã bao phủ 20 core FN + 20 core FP, Neutral/mixed, high-cardinality và train-selected pairs; so sánh ba decoder trên cùng ca.
- [x] Ghi nguồn nhận xét là AI assistant, IDs/labels/scores/cutoffs/cardinality, quan sát tách giả thuyết; raw text giữ private và human worksheet không đổi.

**Xác minh:** lỗi khớp targets/predictions, không đếm trùng cặp để đủ quota, ghi rõ chưa có independent human adjudication. Human confirmation thực tế 0/62; không được trình bày nhận xét AI thành xác nhận của thành viên. **Phụ thuộc:** T4/T5. **Đầu ra:** AI-assisted audit CSV và thảo luận, được nhóm chốt dùng cho paper.

**Checkpoint 2 — hoàn tất ở phạm vi nhóm đã chốt:** đủ evidence/test/breakdowns/AI-assisted audit và công bố giới hạn; không mở tuning theo test để thay đổi kết luận.

### T7 — Paper

**Đầu ra:** DOCX theo ATI_Final_Report_Template.docx, phần thân 15–20 trang, baseline Midterm và Final RQ1, đủ References/Appendix A reproducibility/B contributions/C GenAI. **Phụ thuộc:** T5/T6 và metadata thật. **Xác minh:** số liệu truy về artifact, đúng template/styles, Word body pages và TOC; không dùng PDF 11 trang cũ làm proof đạt length.

**Bản nháp 18/11:** đã có paper tiếng Anh PDF 11 trang và DOCX, 10 bảng sinh từ evidence, ba hình, sáu references và hai phụ lục; cần chuyển cấu trúc/định dạng theo guide mới ngày 19/11, chưa hoàn tất bản nộp. `scripts/build_final_paper.py` tạo Markdown/PDF/verification từ nguồn đã kiểm hash; thông tin tác giả/nhóm/lớp còn chờ điền trong `docs/final_paper_metadata.json`.

**Chuyển template 19/11:** [final_report_ati.docx](../reports/final_report_ati.docx) tích hợp sáu baseline, Final RQ1 và kết quả RQ4 ở mục 5.3. Bản render có 18 trang thân, bốn hình, tám phương trình và Appendix A/B/C. [Thông tin chưa biết](../docs/ati_report_fill_in.md) để nhóm điền; chưa xác nhận sẵn sàng nộp.

### T8 — Gói tái lập

**Đầu ra:** một ZIP theo PREFIX và đúng slides/code/report layout, notebooks đã chạy giữ outputs, predictions mọi model/condition, code README và results/environment.md. Code ZIP `vigoemotions_final28_code.zip` là bundle chạy Kaggle, chưa phải submission archive. Packaging hiện xóa outputs nên cần workflow đóng gói riêng. **Phụ thuộc:** T7, metadata và deck Midterm đúng nguồn. **Xác minh:** unzip/layout, exclusions, repro paths và Word requirements; không inference lại study đã hoàn thành.

### T9 — Oral defence

**Đầu ra:** notes/Q&A và hai lượt mock defence; không tạo slides Final theo submission guide mới. Mỗi thành viên giải thích được sigmoid/BCE, vì sao 28 outputs, Neutral mixed, model epoch 15 được chọn bằng emotion27 trước đây, objective threshold all28 hiện tại, hai cardinality definitions và threshold overfitting. **Phụ thuộc:** ôn từ T1, chốt số sau T5–T7. **Xác minh:** nói đúng scope/metric/support, chỉ đúng evidence, bấm giờ và phản biện chéo.

**Checkpoint 3:** DOCX đúng template và body length, slides Midterm đúng bản đã trình chiếu, ZIP đúng PREFIX/layout, metadata/contributions đầy đủ và đã luyện oral; hạn theo thông báo giảng viên.

## Lịch đề xuất

| Mốc | Công việc |
| --- | --- |
| 11–16/11 | T1–T3: migrate 28, actual validation reproduction, fit và lock |
| 17–20/11 | T4–T6: test, breakdowns và human audit |
| 21–24/11 | Chuyển T7 sang Word template, chuẩn bị T8 và notes oral T9 |
| 25–27/11 | Chốt package và hai mock defence |
| 28–30/11 | Dự phòng kỹ thuật/format theo lịch giảng viên; giữ nguyên quyết định đã chấm test |

## Rủi ro và phạm vi bổ sung

Runtime không khớp phải xử lý trước test, không nới gate. V1 checkpoint/NPZ 27 phải bị từ chối bởi v2, không trộn artifact. Đề gốc RQ27 cần mô tả trung thực việc mở rộng28; projection không được gọi là tối ưu global27 độc lập.

Q1 Gratitude–Disapproval hiện là nghiên cứu bổ sung từ **model27**, không được diễn giải thành chẩn đoán model28. B/C vẫn chờ người annotation/verification thật. Loss mới, dependency model, noise robustness và nhiều seed chỉ làm sau RQ chính; paper-track mới có yêu cầu ≥3 seed. Không tự suy tên/số thành viên hay deadline chính thức.
