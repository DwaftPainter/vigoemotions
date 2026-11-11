# Checklist Final và oral defence — 28 class

Cập nhật 11/11/2026 theo quyết định nhóm: **PhoBERT 28 outputs (gồm Neutral), epoch 15**. Deadline dự kiến cuối tháng 11, mục tiêu nội bộ 27/11. [Protocol v2](../docs/final_protocol.md), [plan](plan.md), [execution](../docs/final_execution.md).

## Làm ngay: actual validation trước test

- [x] Lưu prototype 27 cùng hashes vào [archive](../archive/final27_prototype_2026-11-11/snapshot.json).
- [x] Chuyển config/source run/NPZ/model loader/tokenizer sang đúng checkpoint 28 epoch 15.
- [x] Source validation 2.066 × 28, targets/IDs/tiers đã kiểm chứng; giữ Neutral-only và mixed annotations.
- [x] Triển khai objective global all28, vector 28 cutoffs, emotion27 projection và Neutral metrics riêng.
- [x] Tách label_cardinality (28) và emotion_cardinality (27), bảo vệ fit/lock/test gates.
- [x] Khôi phục model/tokenizer/VnCoreNLP 28 thật vào `data/final28/`, đối chiếu hash model/config và 28-label mapping.
- [ ] Chạy notebook 10 hoặc CLI `reproduce-val` trên môi trường neural, giữ tolerance 1e-4 và fixed decisions giống nguồn.
- [ ] Sau actual reproduction PASS: `fit`, đọc all28/emotion27/Neutral tables và threshold profile.
- [ ] `lock` protocol/config/source/implementation/thresholds trước khi đọc held-out để đánh giá.
- [ ] **Checkpoint 1:** có actual model reproduction và ngưỡng khóa; tests synthetic không đủ.

## Test và phân tích

- [ ] `infer-test` đúng một lượt với 2.067 mẫu và cùng assets/runtime đã verified.
- [ ] Lưu continuous scores trước decode; NPZ đủ 28 outputs và hai cardinality fields.
- [ ] Đối chiếu overall/per-label F1, Micro-F1, Hamming, LRAP, delta/gap trong đúng scope.
- [ ] Head/mid/tail cho 27 emotions; Neutral riêng; cardinality all28/emotion27 tách rõ.
- [ ] Vẽ threshold profile và delta theo nhãn/tier/cardinality.
- [ ] Người thật audit ≥20 FN + ≥20 FP, có Neutral/mixed/high-cardinality cases và emotion pairs.
- [ ] **Checkpoint 2:** trả lời RQ theo kết quả thật, không tune lại bằng test.

## Paper, tái lập và oral

- [ ] Report paper-style và PDF đủ mục, references, limitations và failure cases.
- [ ] Nêu rõ đề RQ27 và mở rộng28 của nhóm; epoch selection gốc emotion27 khác threshold objective all28.
- [ ] README/setup/run/report, code/config/split IDs/outputs/runtime metadata đầy đủ; private data/weights không public.
- [ ] Kiểm thử liên quan và regression suite; báo dependency errors thật, không skip để có xanh.
- [ ] Điền tên/MSSV/phân công; xác nhận deadline/thời lượng/ngôn ngữ oral.
- [ ] Deck Final và appendix, Q&A về metrics/threshold/leakage/Neutral/cardinality/overfit.
- [ ] Hai mock defence có bấm giờ và phản biện chéo; mỗi người hiểu pipeline chung và phần đóng góp.
- [ ] **Checkpoint 3:** package/PDF/slide/README thống nhất, sẵn sàng trước 27/11 theo lịch nội bộ.

## Sau phần bắt buộc

- [ ] Nếu dùng Q1 Gratitude–Disapproval, ghi đúng nguồn model27; chưa có B/C completed do người.
- [ ] Chỉ mở loss/dependency/robustness/multiple-seed studies khi RQ chính đã hoàn tất.
