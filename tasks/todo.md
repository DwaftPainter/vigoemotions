# Checklist Final và oral defence

Cập nhật 11/11/2026. Deadline chính thức chưa rõ; dự kiến cuối tháng 11 theo người dùng. Mục tiêu nội bộ: sẵn sàng 27/11, dự phòng 28–30/11.

Chi tiết, nguồn yêu cầu và tiêu chí xác minh: [plan.md](plan.md). Các mục Final bên dưới chưa được đánh dấu hoàn tất chỉ vì đã có baseline midterm.

## Việc cần làm ngay — hoàn tất validation trước khi chấm test

Hướng làm việc hiện tại: PhoBERT **27 nhãn**, checkpoint **epoch 14** của lượt train 20 epoch; giữ trọng số cho thí nghiệm ba decoder.

1. **T1 — Protocol/config Final đã ghi:** xem [protocol v1](../docs/final_protocol.md), [config](../configs/final_thresholds.json) và [tiers](../docs/final_label_tiers.csv). Bước tiếp theo là khôi phục model, kiểm chứng validation và fit ngưỡng.
2. **T2 — Khôi phục checkpoint:** kiểm hash từ ZIP private, chuẩn bị đúng preprocessing và chứng minh inference validation khớp NPZ nguồn 2.066 × 27.
3. **T3 — Hoàn thiện tuning validation:** dùng saved probabilities cho fixed 0,5, global tuned và per-label tuned; lưu bảng metric, ngưỡng global và vector 27 ngưỡng.
4. **Checkpoint 1 — Khóa cấu hình/ngưỡng:** kiểm nhãn/IDs, tái tạo metric, lưu hashes; sau đó mới chuyển sang inference test.

Đầu ra của đợt này: protocol/config, bằng chứng khớp checkpoint/validation, bảng so sánh ba decoder trên validation và thresholds đã khóa. Protocol/config/tiers đã có; inference validation, fitted thresholds, bảng Final và pre-test lock vẫn chưa hoàn tất.

## 11–16/11 — Chốt thiết kế, khôi phục model, tuning validation

- [x] **T1, phần thiết kế:** Ghi protocol RQ1: PhoBERT 27, checkpoint epoch 14, không retrain khi đổi decoder.
- [x] Ghi split IDs/label order, grid/objective/tie-break/fallback và cách chọn audit trong protocol/config v1; pre-test lock sẽ tạo sau khi fit.
- [x] Xuất train count và membership head/mid/tail theo config equal20, trước test.
- [ ] Điền tên/MSSV/phân công thực tế; cập nhật lịch/feedback/thời lượng oral khi giảng viên thông báo.
- [ ] **T2:** Khôi phục checkpoint private, kiểm chứng hash và tái tạo đủ validation probabilities khớp evidence.
- [ ] **T3:** Chạy fixed 0,5, global tuned và per-label tuned chỉ trên validation.
- [ ] Lưu global threshold + đủ 27 per-label thresholds và bảng validation.
- [ ] Kiểm thử boundary/tie/fallback/ID-label alignment và tách tuning khỏi test.
- [ ] **Checkpoint 1:** thresholds/config/checkpoint/split đã khóa, sẵn sàng test.

## 17–20/11 — Test và evidence RQ1

- [ ] **T4:** Inference test bằng frozen checkpoint đúng một lượt sau khi khóa lựa chọn.
- [ ] Lưu validation/test probabilities cùng IDs, true vectors, label order và cardinality.
- [ ] Lưu `pred_fixed`, `pred_global`, `pred_per_label`; kiểm tra cả ba dùng cùng scores/targets.
- [ ] **T5:** Sinh bảng validation/test: Macro-F1, Micro-F1, Hamming Loss, per-label F1 của ba decoder.
- [ ] Sinh delta theo head/mid/tail và cardinality 1/2/3/≥4; thêm nhóm 0 để không mất Neutral-only.
- [ ] Vẽ threshold profile 27 nhãn và bảng `F1_val − F1_test` cho từng decoder.
- [ ] **T6:** Thành viên đọc thật ít nhất 20 FN + 20 FP, ghi nhận xét và người review.
- [ ] Có ca cardinality cao và nhãn hay co-occur/bị dự đoán nhầm; đối chiếu thay đổi lỗi khi đổi ngưỡng.
- [ ] **Checkpoint 2:** đủ kết quả và audit; viết câu trả lời RQ1 kèm giới hạn, không tune tiếp theo test.

## 21–24/11 — Paper và gói tái lập

- [ ] **T7:** Hoàn thiện paper-style report, trích dẫn và PDF; có discussion, limitations, failure cases.
- [ ] Kiểm từng số/bảng/hình với artifacts sinh từ code; tách baseline midterm với kết quả Final.
- [ ] **T8:** README có đường chạy đầy đủ và đường tái sinh bảng từ saved probabilities.
- [ ] Gói đủ code/notebook, split IDs, configs, validation/test outputs, bảng/hình và hardware/versions/seeds/revision.
- [ ] Kiểm tra package public không chứa raw restricted text/vocabulary/checkpoint private; có thông tin nhóm.
- [ ] Tái sinh bảng từ probabilities ở output mới; chạy focused checks và regression suite liên quan.
- [ ] **T9:** Draft deck Final + appendix và Q&A dựa trên kết quả thật.

## 25–27/11 — Oral defence và chốt nộp

- [ ] Mỗi người giải thích được sigmoid/BCE, Macro/Micro-F1, Hamming, LRAP và ba decoder.
- [ ] Trả lời được vì sao khóa backbone, cách chống leakage, cách chia tiers/cardinality và nguy cơ threshold overfitting.
- [ ] Giải thích được Neutral/mixed annotations, official split và giới hạn một seed.
- [ ] Mở được một FN, một FP, một ca nhiều cảm xúc và chỉ ra evidence cho nhận xét.
- [ ] Mỗi người nói rõ phần đóng góp và AI assistance; hiểu pipeline chung.
- [ ] Mock defence 1: bấm giờ, sửa slide/mạch nói theo thời lượng được thông báo.
- [ ] Mock defence 2: phản biện chéo, sửa câu trả lời chưa có bằng chứng.
- [ ] PDF/slide/README/package thống nhất; có bản backup offline, kiểm tra link và cách nộp.
- [ ] **Checkpoint 3:** sẵn sàng nộp và bảo vệ trước 27/11 theo lịch nội bộ.

## Chỉ làm thêm sau khi RQ1 đã hoàn tất

- [ ] Quyết định có đưa Q1 Gratitude–Disapproval A vào appendix hay không.
- [ ] Nếu tiếp tục B/C: làm annotation/adjudication và xác minh intervention thật; hiện B là 0/71, C chưa có cặp verified.
- [ ] Chỉ thêm loss/dependency/robustness/multiple seeds nếu còn thời gian và có mục tiêu rõ; ba seed cần nếu được yêu cầu paper-track.

Không bắt đầu inference test chỉ để xem điểm; không coi danh sách FP/FN tự sinh là human audit đã xong.
