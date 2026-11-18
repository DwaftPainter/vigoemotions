# Checklist Final và oral defence — 28 class

Cập nhật 18/11/2026: **PhoBERT 28 outputs (gồm Neutral), epoch 15**. Validation/test và AI-assisted audit đã hoàn tất; [paper tiếng Anh PDF](../reports/final_report.pdf) / [Markdown](../reports/final_report.md). Deadline dự kiến cuối tháng 11, mục tiêu nội bộ 27/11. [Protocol v2](../docs/final_protocol.md), [plan](plan.md), [execution](../docs/final_execution.md).

## Đã hoàn tất: validation và khóa trước test

- [x] Lưu prototype 27 cùng hashes vào [archive](../archive/final27_prototype_2026-11-11/snapshot.json).
- [x] Chuyển config/source run/NPZ/model loader/tokenizer sang đúng checkpoint 28 epoch 15.
- [x] Source validation 2.066 × 28, targets/IDs/tiers đã kiểm chứng; giữ Neutral-only và mixed annotations.
- [x] Triển khai objective global all28, vector 28 cutoffs, emotion27 projection và Neutral metrics riêng.
- [x] Tách label_cardinality (28) và emotion_cardinality (27), bảo vệ fit/lock/test gates.
- [x] Khôi phục model/tokenizer/VnCoreNLP 28 thật vào `data/final28/`, đối chiếu hash model/config và 28-label mapping.
- [x] Actual `reproduce-val` trên Kaggle: full probabilities khớp nguồn, max absolute error 0, fixed decisions giống hệt.
- [x] `fit`: global 0,4 và đủ 28 per-label thresholds; all28/emotion27/Neutral tables và profile đã review.
- [x] Có lock bind protocol/config/source/implementation/thresholds; test artifact liên kết đúng lock hiện tại.
- [x] **Checkpoint 1:** có actual model reproduction và ngưỡng khóa, không chỉ synthetic tests.

## Test và phân tích

- [x] Test artifact ghi `completed` với 2.067 mẫu và cùng assets/CUDA runtime đã verified; không inference lại khi import/report.
- [x] Continuous scores và decoded NPZ đủ 28 outputs, IDs/targets và hai cardinality fields, hashes khớp.
- [x] Tính lại 11 overall/per-label/tier/cardinality/Neutral/gap tables, đối chiếu đúng scope.
- [x] Head/mid/tail cho 27 emotions; Neutral riêng; cardinality all28/emotion27 tách rõ.
- [x] Bốn hình PNG/PDF: validation–test, threshold profile, delta nhãn, delta tier/cardinality; script tái sinh và provenance.
- [x] Viết kết luận RQ từ kết quả thật: global +0,1289 pp test Macro-F1, per-label −0,4549 pp; không tune lại bằng test.
- [x] Chuẩn bị [human audit packet](../reports/final28/human_audit/README.md): 20 core FN, 20 core FP, Neutral FN/FP, 5 high-cardinality và 15 train-selected-pair cases; 62 dòng/62 bình luận, worksheet raw text private, chưa có review hoàn tất.
- [x] Theo yêu cầu 17/11, AI assistant đọc đủ 62 ca và lưu [AI-assisted audit](../reports/final28/human_audit/ai_assisted_review.md), nhận xét từng ca/provenance riêng; giữ nguyên human worksheet và không tính thành human reviews.
- [x] **Checkpoint 2:** đủ kết quả/breakdowns và AI-assisted audit theo quyết định nhóm 17/11; giữ giới hạn chưa có independent human adjudication trong báo cáo.

Human confirmation thực tế vẫn là 0/62; không đánh dấu worksheet của người đã hoàn tất. Theo yêu cầu nhóm, phần audit của paper dùng bản AI-assisted đã công bố rõ nguồn; xác nhận bởi người là phần bổ sung nếu nhóm thực hiện sau.

## Paper, tái lập và oral

- [x] Paper tiếng Anh và PDF 11 trang đủ mục, references, limitations, AI-assisted failure cases và phụ lục đủ 28 nhãn; đã kiểm tra PDF thật.
- [x] Bản nháp Results/Discussion nêu RQ27 và mở rộng28, epoch selection emotion27 khác threshold objective all28, một seed và giới hạn thống kê.
- [x] README/setup/run/report, code/config/split IDs/outputs/runtime metadata và paper builder đầy đủ trong repo; private data/weights không public. Archive nộp cuối cùng còn đóng gói sau.
- [x] Chạy tests liên quan và regression suite: 156/158 pass, gồm sáu paper tests; hai neural tests cũ lỗi do local thiếu torch, đã ghi rõ; không skip hoặc hạ kiểm tra.
- [ ] Điền tên/MSSV/phân công; xác nhận deadline/thời lượng/ngôn ngữ oral.
- [ ] Deck Final và appendix, Q&A về metrics/threshold/leakage/Neutral/cardinality/overfit.
- [ ] Hai mock defence có bấm giờ và phản biện chéo; mỗi người hiểu pipeline chung và phần đóng góp.
- [ ] **Checkpoint 3:** package/PDF/slide/README thống nhất, sẵn sàng trước 27/11 theo lịch nội bộ.

## Sau phần bắt buộc

- [ ] Nếu dùng Q1 Gratitude–Disapproval, ghi đúng nguồn model27; chưa có B/C completed do người.
- [ ] Chỉ mở loss/dependency/robustness/multiple-seed studies khi RQ chính đã hoàn tất.
