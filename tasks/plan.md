# Kế hoạch Final và oral defence — hướng 28 class

Cập nhật 11/11/2026 theo quyết định của nhóm: **28 outputs gồm Neutral**, checkpoint PhoBERT epoch 15. Deadline chính thức chưa rõ; dự kiến cuối tháng 11. Mục tiêu nội bộ: sẵn sàng 27/11, dự phòng 28–30/11.

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
- [ ] `reproduce-val` trên môi trường neural: toàn bộ scores lệch ≤1e-4, fixed decisions giống hệt, ghi runtime/asset hashes.

**Xác minh:** actual checkpoint phải chạy, mock tests không thay thế proof; không tự nới tolerance. **Phụ thuộc:** T1. **Quy mô:** module inference/study và evidence sinh tự động.

### T3 — Fit và khóa ngưỡng validation

**Đầu ra:** global scalar + 28 cutoffs, bảng/projection/Neutral metrics, pre-test lock.

- [x] Có thuật toán và test cho objective all28, ties/fallback/boundary/shape; gate chặn fit khi thiếu actual reproduction.
- [ ] Chạy `fit` sau T2 PASS, lưu đủ 28 ngưỡng/support/train count/tier/fallback.
- [ ] Review validation tables, chạy `lock` trước test; bind config/protocol/implementation/thresholds/source/reproduction.

**Xác minh:** all28 và emotion27 có scope đúng, Neutral F1 tính positive label; thử thay nguồn/ngưỡng/runtime phải bị phát hiện. **Phụ thuộc:** T2. **Quy mô:** core/study/CLI cùng output.

**Checkpoint 1:** model 28 tái tạo đúng validation, ngưỡng 28 đã fit và khóa. Chưa đạt chỉ vì code/tests chạy được.

### T4 — Held-out test một lượt

**Đầu ra:** continuous test scores trước decoding và ba pred matrices 28 cột.

- [ ] Kiểm lock/assets/runtime trước đọc test text/targets; inference 2.067 mẫu đúng thứ tự.
- [ ] Lưu IDs/true/probability vectors/labels, pred_fixed/global/per_label, label_cardinality (28), emotion_cardinality (27).
- [ ] Cả hai scopes dùng cùng test scores; không re-tune, không lặp inference sau một lượt hoàn tất.

**Xác minh:** tái tạo pred từ saved scores + thresholds; technical resume giữ cùng lựa chọn và dùng lại raw scores nếu có. **Phụ thuộc:** Checkpoint 1. **Quy mô:** đường inference/test gate đã có; actual execution còn chờ.

### T5 — Bảng, hình và trả lời RQ

- [ ] Metric validation/test và delta theo decoder cho all28; emotion27 projection, Neutral riêng, tier và hai cardinality views.
- [ ] Threshold profile đủ 28 nhãn, hình delta và generalization gap theo scope; support rõ, nhóm rỗng N/A.
- [ ] Kết luận đúng kết quả dù tuned không cải thiện test; ghi giới hạn một seed và repeated validation selection.

**Xác minh:** `report` tái sinh metric tables từ saved outputs; số liệu đối chiếu NPZ. **Phụ thuộc:** T3/T4. **Đầu ra:** CSV/figures/RQ analysis; code bảng đã có, hình/thảo luận còn làm.

### T6 — Human error audit

- [ ] Người thật đọc ≥20 FN + ≥20 FP trên 28-label per-label decoder; Neutral/mixed cases được xem và ghi rõ.
- [ ] Có ca cardinality cao, các emotion pairs chọn từ train, so sánh lỗi giữa ba decoder.
- [ ] Ghi reviewer, IDs/label/score/cutoff/cardinality, quan sát tách giả thuyết; raw text giữ private.

**Xác minh:** lỗi khớp targets/predictions, không đếm trùng cặp để đủ quota. **Phụ thuộc:** T4/T5. **Đầu ra:** audit CSV và thảo luận do người xác nhận.

**Checkpoint 2:** đủ evidence/test/breakdowns/human audit. Không mở tuning theo test để thay đổi kết luận.

### T7 — Paper

**Đầu ra:** report Markdown/PDF, figures và references. Có abstract/introduction/data/method/experiments/results/discussion/limitations/conclusion. Mô tả 28-label extension và baseline selection gốc; so điểm cùng scope/split. **Phụ thuộc:** outline từ T1, chốt sau T5/T6. **Xác minh:** mỗi số truy về artifact, đọc PDF thật.

### T8 — Gói tái lập

**Đầu ra:** README/code/notebook/config/split IDs/probabilities/tables/figures/runtime metadata. Code ZIP hiện là `vigoemotions_final28_code.zip`; model/data riêng tư. **Phụ thuộc:** T5–T7. **Xác minh:** unit/regression tests và đường `report`; không ghi actual inference PASS nếu chưa chạy.

### T9 — Oral defence

**Đầu ra:** slide Final, notes/Q&A, hai lượt mock defence. Mỗi thành viên giải thích được sigmoid/BCE, vì sao 28 outputs, Neutral mixed, model epoch 15 được chọn bằng emotion27 trước đây, objective threshold all28 hiện tại, hai cardinality definitions và threshold overfitting. **Phụ thuộc:** ôn từ T1, chốt số sau T5–T7. **Xác minh:** nói đúng scope/metric/support, chỉ đúng evidence, bấm giờ và phản biện chéo.

**Checkpoint 3:** report/package nhất quán, người thật đã audit và luyện bảo vệ, sẵn sàng nộp.

## Lịch đề xuất

| Mốc | Công việc |
| --- | --- |
| 11–16/11 | T1–T3: migrate 28, actual validation reproduction, fit và lock |
| 17–20/11 | T4–T6: test, breakdowns và human audit |
| 21–24/11 | T7–T8, deck T9 |
| 25–27/11 | Chốt package và hai mock defence |
| 28–30/11 | Dự phòng kỹ thuật/format theo lịch giảng viên; giữ nguyên quyết định đã chấm test |

## Rủi ro và phạm vi bổ sung

Runtime không khớp phải xử lý trước test, không nới gate. V1 checkpoint/NPZ 27 phải bị từ chối bởi v2, không trộn artifact. Đề gốc RQ27 cần mô tả trung thực việc mở rộng28; projection không được gọi là tối ưu global27 độc lập.

Q1 Gratitude–Disapproval hiện là nghiên cứu bổ sung từ **model27**, không được diễn giải thành chẩn đoán model28. B/C vẫn chờ người annotation/verification thật. Loss mới, dependency model, noise robustness và nhiều seed chỉ làm sau RQ chính; paper-track mới có yêu cầu ≥3 seed. Không tự suy tên/số thành viên hay deadline chính thức.
