# Kế hoạch Final và oral defence — ViGoEmotions

Cập nhật: 11/11/2026. Midterm đã hoàn tất theo xác nhận của người dùng. Deadline chưa chính thức; người dùng dự kiến cuối tháng 11. Mục tiêu nội bộ: sẵn sàng ngày 27/11, dự phòng 28–30/11; điều chỉnh khi có lịch chính thức.

## Mục tiêu và căn cứ

Hoàn thành RQ1: **ngưỡng chung 0,5 có tối ưu cho cả 27 cảm xúc hay không?** So sánh fixed, global tuned và per-label tuned trên cùng một checkpoint neural, tối ưu trên validation, khóa lựa chọn rồi đánh giá held-out test một lần. Kết quả không cần tăng mới có giá trị; phải trả lời được RQ bằng thí nghiệm có kiểm soát.

Nguồn yêu cầu:

- [Đề ViGoEmotions](../docs/02-vigoemotions-vietnamese-emotion.pdf), trang 2–4: RQ1, phân tích bắt buộc, error audit, deliverables và extensions.
- [Student guide](../docs/student-guide.pdf), trang 1–2: Final chiếm 60%, gồm paper-style report, viva và reproducibility. Slides nộp nếu được yêu cầu.
- [Báo cáo midterm](../reports/midterm_report.md), [profile tái lập](../docs/reproducibility_profiles.md), [provenance](../reports/kaggle_equal20_2026-11-05/provenance.json).

## Điểm xuất phát đã kiểm tra trong repo

| Hạng mục | Hiện trạng | Cách dùng cho Final |
| --- | --- | --- |
| Ba mô hình, hai bộ nhãn | Có kết quả validation của sáu model equal20 | Dùng làm baseline và lý do chọn backbone |
| PhoBERT 27 | Macro-F1 0,597410 tại 0,5; checkpoint tốt nhất epoch 14 sau 20 epoch train | Chọn làm model chính cho RQ1 |
| Dữ liệu/checkpoint | Có ba CSV local và ZIP private 27; provenance cũ ghi đã kiểm chứng checkpoint | Kiểm chứng lại checkpoint khi khôi phục; giữ private |
| Xác suất validation | Có NPZ 2.066 × 27 cùng IDs, targets, labels | Tuning decoder dùng lại xác suất này |
| Global sweep cũ | Lưới 0,05–0,80, bước 0,05; tốt nhất 0,35, Macro-F1 0,597843 | Bằng chứng thăm dò; chưa thay cho thí nghiệm Final đầy đủ |
| Test | Provenance hiện ghi chưa chấm mô hình | Chỉ inference sau khi khóa protocol/ngưỡng |
| Tier nhãn | Profile equal20 có head hạng 1–9, mid 10–18, tail 19–27 theo tần suất train; hòa theo ID nhãn | Tái sử dụng, xuất membership trước test |
| Error audit | Sáu ca Neutral có hỗ trợ AI; có danh sách FP/FN tự sinh | Chưa đáp ứng audit Final ít nhất 20 FN + 20 FP |
| Q1 Gratitude–Disapproval | A có báo cáo; B: 0/71 mục được giải quyết; C chưa có cặp được người xác minh | Nghiên cứu bổ sung, không thay thế RQ1 |
| Thông tin nhóm | Tên/MSSV/phân công vẫn để trống trong tài liệu | Điền theo người dùng, không suy từ Git |

Đây là kiểm kê tài liệu/artifact, không phải xác nhận đã chạy lại toàn bộ workflow hay đã có kết quả Final. Working tree đang có thay đổi midterm/Q1; kế hoạch này không chỉnh các phần đó.

## Thiết kế đề xuất cần ghi rõ trước khi test

- Giữ PhoBERT 27 epoch 14, tokenizer/revision/segmentation/preprocessing, label order và split equal20. Không retrain khi thay decoder. Backbone và xác suất phải giống nhau cho ba chiến lược.
- Fixed: mọi nhãn dùng 0,5. Global: một ngưỡng tối đa hóa validation Macro-F1. Per-label: mỗi nhãn tối đa hóa validation F1 riêng.
- Đề xuất dùng lại lưới 0,05–0,80 bước 0,05 cho cả global và per-label để giữ nhất quán với sweep midterm. Đây là lựa chọn thiết kế, không phải yêu cầu bắt buộc của đề. Nếu đổi lưới, ghi lý do và khóa trước test.
- Quy tắc đề xuất: dự đoán dương khi `p >= threshold`; cùng F1 chọn ngưỡng cao hơn như sweep cũ; nhãn không có positive trên validation dùng 0,5 và đánh dấu rõ. Lưu lưới, objective, tie-break và fallback trong config.
- Head/mid/tail chỉ theo retained train (16.530 mẫu), không dùng test để chia tier. Cardinality chỉ tính 27 cảm xúc; báo cáo nhóm 1, 2, 3, ≥4 theo đề, thêm nhóm 0 cho Neutral-only để không bỏ mẫu.
- Khóa config, checkpoint hash, label order, split IDs, 1 global threshold và 27 per-label thresholds trước inference test. Một lượt xác suất test dùng cho cả ba decoder; đọc lại xác suất đã lưu để tái tạo bảng không phải lượt tuning mới.
- Không chọn lại decoder/checkpoint/ngưỡng theo test. Có thể đọc test sau đánh giá để phân tích lỗi, nhưng mọi đề xuất sửa tiếp phải nằm ở limitations/future work.
- LRAP là kiểm tra phụ có thể giữ từ midterm; dùng cùng xác suất liên tục thì LRAP phải giống nhau giữa ba decoder. Threshold tuning không phải probability calibration.
- Không mặc định thêm Neutral vào output hoặc suy empty prediction thành Neutral. Nếu trình bày study 28, tách rõ khỏi kết quả RQ1 trên 27 cảm xúc.

## Các đầu việc và tiêu chí hoàn thành

### T1 — Chốt protocol và phân công

**Đầu ra:** `docs/final_protocol.md`, `configs/final_thresholds.json`; cập nhật thông tin nhóm khi có dữ liệu thật.

**Tiến độ:** đã có [protocol v1](../docs/final_protocol.md), [config](../configs/final_thresholds.json) và [train-only tiers](../docs/final_label_tiers.csv). ZIP/checkpoint member hash và ordered split IDs đã đối chiếu nguồn. Inference validation, fitted thresholds và pre-test lock chưa hoàn tất; phần phân công/review nhóm vẫn cần thành viên thực hiện.

- [ ] Ghi model/checkpoint, split, grid, objective, tie-break, fallback, tiers và các metrics sẽ báo cáo.
- [ ] Có bảng nhãn → train count → tier, cùng lịch/phụ trách; ghi rõ deadline/thời lượng oral còn chờ xác nhận.
- [ ] Ghi trước cách chọn mẫu audit: decoder chính nào, đếm theo cặp comment–label, cách xử lý comment trùng và nhóm cardinality cao.

**Xác minh:** đối chiếu đề trang 2–3 và config equal20; người phụ trách thí nghiệm cùng một thành viên đọc protocol. **Phụ thuộc:** không. **Quy mô:** nhỏ; 2–3 tài liệu cấu hình.

### T2 — Khôi phục đúng checkpoint và chứng minh inference khớp

**Đầu ra dự kiến:** phần inference dùng lại code hiện có trong `scripts/final_thresholds.py`, kiểm thử liên quan và manifest mới tại `reports/final/`. Checkpoint/segmentation assets nằm private.

- [ ] Checkpoint trích từ ZIP private khớp member/hash đã kiểm chứng; khóa revision và preprocessing.
- [ ] Chạy lại validation bằng checkpoint đó; ID/label/target/order khớp NPZ nguồn và xác suất khớp sai số đã ghi trước, đề xuất absolute tolerance 1e-4 như study Q1 hiện có.
- [ ] Nếu lệch, tìm khác biệt môi trường/tokenizer/segmentation trước khi tiếp tục; không âm thầm nới tolerance.

**Xác minh:** so sánh toàn bộ ma trận validation và tái tính fixed Macro-F1 ≈0,597410; ghi môi trường thực chạy. **Phụ thuộc:** T1. **Quy mô:** vừa; khoảng 3 file nguồn/kiểm thử, chưa tính output sinh tự động.

### T3 — Hoàn thiện ba decoder trên validation

**Đầu ra:** global scalar, vector 27 ngưỡng, bảng validation và file khóa ngưỡng trong `reports/final/`.

- [ ] Fixed/global/per-label dùng cùng NPZ, tìm ngưỡng đúng objective và đúng tie-break/fallback đã định.
- [ ] Lưu đủ nhãn/ngưỡng/train support/validation support; sinh bảng bốn metrics bắt buộc và per-label F1, có delta so với fixed.
- [ ] Khóa thresholds/config/hash trước mọi inference test; không đọc test probabilities/metrics trong hàm tuning.

**Xác minh:** kiểm thử trường hợp `p == threshold`, hòa F1, nhãn không positive, sai thứ tự nhãn; tái tạo fixed và sweep 0,35 cũ nếu dùng cùng grid. Với tối ưu độc lập đầy đủ trên grid chứa 0,5, validation Macro-F1 per-label không được thấp hơn global do cùng objective macro. **Phụ thuộc:** T1–T2. **Quy mô:** vừa; khoảng 3 file nguồn/kiểm thử.

**Checkpoint 1:** protocol có thể đọc lại; inference validation khớp; ngưỡng đã lưu và khóa. Đây là điểm sẵn sàng chấm test.

### T4 — Inference test một lần và lưu đầu ra đủ để tái lập

**Đầu ra:** NPZ/CSV validation và test trong `reports/final/predictions/`, cùng inference/evaluation manifest.

- [ ] Chạy frozen checkpoint trên test sau Checkpoint 1, lưu xác suất liên tục trước binarization; không retrain/tune theo test.
- [ ] Lưu `sample_id`, `true_label_vector`, `probability_vector`, `pred_fixed`, `pred_global`, `pred_per_label`, `label_cardinality`; kèm canonical label order và liên kết tới frozen thresholds.
- [ ] Cả ba decoder dùng cùng test IDs/targets/probabilities; đủ 2.067 mẫu theo split hiện có, 27 cột, không raw text.

**Xác minh:** kiểm tra shape/bounds/hash/order và tái tạo cả ba ma trận dự đoán từ probabilities + thresholds. Nếu job lỗi kỹ thuật, lưu lỗi và nguyên nhân; không dùng kết quả một phần để điều chỉnh thiết kế. **Phụ thuộc:** T3/Checkpoint 1. **Quy mô:** nhỏ; mở rộng script và kiểm thử của T2–T3.

### T5 — Sinh toàn bộ bảng và hình trả lời RQ1

**Đầu ra:** `scripts/analyze_final.py`, kiểm thử và bảng/hình được sinh trong `reports/final/`.

- [ ] Macro-F1, Micro-F1, Hamming Loss và F1 của cả 27 nhãn cho ba decoder trên validation/test; delta so với fixed và gap `F1_val − F1_test` theo từng decoder.
- [ ] Phân tích head/mid/tail và cardinality 1/2/3/≥4, thêm 0; ghi rõ số nhãn/số mẫu/support và quy ước nhãn không support.
- [ ] Hình threshold profile 27 nhãn, delta theo nhãn/tier/cardinality, bảng generalization gap; kết luận nêu cả cải thiện lẫn đánh đổi.

**Xác minh:** tính lại metric trực tiếp từ NPZ; tiers phân hoạch đủ 27 nhãn, cardinality phân hoạch đủ từng split; nếu giữ LRAP thì ba decoder khớp nhau. Có thể thêm paired bootstrap mô tả nếu còn thời gian, không coi là đo biến thiên nhiều seed. **Phụ thuộc:** T4. **Quy mô:** vừa; 2 file nguồn/kiểm thử và output sinh tự động.

### T6 — Audit lỗi do thành viên thực hiện

**Đầu ra:** `reports/final/error_audit.csv` chỉ chứa IDs/nhãn/quan sát khái quát và `docs/final_error_analysis.md`; việc đọc raw text diễn ra private.

- [ ] Người thật đọc và xác nhận ít nhất 20 FN + 20 FP theo cách chọn đã chốt; ghi decoder, sample ID, label, score, threshold, cardinality, người review và nhận xét.
- [ ] Có ca nhiều cảm xúc, nhãn hay đồng xuất hiện hoặc bị dự đoán nhầm; so sánh những FP/FN được sửa hoặc phát sinh khi đổi decoder.
- [ ] Phân biệt quan sát với giả thuyết, ghi thiên lệch chọn mẫu; tránh đếm một cặp comment–label nhiều lần để đủ quota. Ưu tiên nhiều comment khác nhau khi khả thi, công khai số comment duy nhất.

**Xác minh:** đối chiếu từng FP/FN với targets và decoder output; thành viên xác nhận thật, không coi danh sách tự sinh/AI diễn giải là audit đã hoàn thành. **Phụ thuộc:** T4, dùng bảng T5 khi có. **Quy mô:** nhỏ về code, cần thời gian đọc thủ công.

**Checkpoint 2:** đủ evidence RQ1, bảng/hình và audit lỗi; ghi kết luận dù per-label không cải thiện test. Không mở nghiên cứu mới để chạy theo điểm test.

### T7 — Viết report theo dạng paper

**Đầu ra:** `reports/final_report.md` và PDF; script build report nếu cần.

- [ ] Có Abstract, Introduction/RQ, Related Work, Data, Method, Experimental Setup, Results, Discussion, Limitations, Conclusion và References.
- [ ] Đưa baseline midterm làm bối cảnh; kết quả chính là so sánh decoder cùng checkpoint. Mọi số/bảng/hình truy về artifact và script tương ứng.
- [ ] Nêu validation được dùng chọn checkpoint rồi chọn threshold, nguy cơ overfit, một seed, giới hạn split, heuristic/annotation và license; kết luận trả lời đúng RQ1.

**Xác minh:** đối chiếu từng số quan trọng với T5, đối chiếu đủ mục B/C/D của đề; đọc PDF thật để kiểm tra font/bảng/hình/trích dẫn. **Phụ thuộc:** outline có thể làm từ T1, chốt sau T5–T6. **Quy mô:** vừa; khoảng 3 file nguồn/nội dung cùng PDF sinh ra.

### T8 — Đóng gói tái lập và hồ sơ nộp

**Đầu ra:** README Final, package/manifest kiểm chứng, cập nhật thông tin nhóm.

- [ ] Có hướng dẫn setup → restore model → reproduce validation → tune/lock → test inference → regenerate report; tách rõ việc tái sinh bảng từ saved probabilities để tránh chạy lại test ngoài ý muốn.
- [ ] Có report PDF, code/notebook chạy được, split IDs, configs, probabilities + ba decoder outputs, bảng/hình, hardware/versions/seeds/checkpoint revision. Slides bổ sung nếu yêu cầu.
- [ ] Không kèm raw restricted text, vocabulary hoặc checkpoint private trong bản public; giải thích cách người có quyền truy cập dữ liệu tái lập.

**Xác minh:** chạy lại đường sinh bảng/hình từ saved probabilities trong thư mục output mới; chạy các kiểm thử liên quan rồi regression suite; kiểm tra nội dung archive, link và thông tin nhóm. Không tuyên bố đã chạy các lệnh nếu chưa chạy thực tế. **Phụ thuộc:** T5–T7. **Quy mô:** vừa; khoảng 3–4 file nguồn/tài liệu.

### T9 — Chuẩn bị oral defence

**Đầu ra:** deck Final, `docs/final_defence_notes.md`, bảng Q&A và bằng chứng luyện thử.

- [ ] Slide kể được mạch: bài toán → baseline → RQ1 → controlled protocol → ba decoder → test results → tier/cardinality → error cases → limitations → kết luận. Chuẩn bị phần trình bày chính gọn và appendix; điều chỉnh số slide/thời gian khi có quy định.
- [ ] Mỗi thành viên giải thích được phần mình làm, pipeline chung và các câu hỏi dưới đây; có bảng/hình/artifact để chứng minh, không chỉ học thuộc số.
- [ ] Ít nhất hai lượt mock defence: một lượt bấm giờ sửa mạch nói, một lượt thành viên khác hỏi phản biện/ngắt giữa chừng; sửa câu trả lời thiếu bằng chứng.

**Xác minh:** tự trình bày không phụ thuộc notes; trả lời được metric/threshold/leakage/overfitting và chỉ đúng file evidence; mở được PDF và backup offline. **Phụ thuộc:** ôn nền tảng từ T1, chốt số và rehearsal sau T5–T7. **Quy mô:** tách buổi làm deck, notes và rehearsal; mỗi buổi 1–2 đầu ra.

**Checkpoint 3:** report/package kiểm chứng, mọi người bảo vệ được quyết định và kết quả, sẵn sàng nộp trước 27/11 nếu deadline cuối tháng được giữ.

## Những câu phải chuẩn bị cho oral

1. **Vì sao multi-label, sigmoid+BCE, không softmax?** Một comment có nhiều cảm xúc; học từng output nhị phân, nhưng điều đó không khẳng định các cảm xúc độc lập trong dữ liệu.
2. **Vì sao chọn PhoBERT 27 và epoch 14?** Dẫn baseline cùng 27 cảm xúc; checkpoint theo validation ở 0,5, không chọn bằng test hay bằng decoder đã tuned.
3. **Global và per-label tối ưu gì?** Trình bày công thức/thuật toán, grid, tie-break, train/validation/test dùng cho bước nào.
4. **Vì sao per-label có thể tăng validation nhưng giảm test?** Nhiều tham số hơn (27 so với 1), ít positive ở một số nhãn; đọc gap và support thực tế. Không bảo đảm tăng test.
5. **Macro khác Micro thế nào?** Macro trung bình F1 của từng nhãn; Micro gộp TP/FP/FN. Đổi ngưỡng để tối đa Macro có thể đánh đổi Micro/Hamming.
6. **Vì sao Hamming thấp vẫn có thể bỏ sót nhiều cảm xúc?** Có nhiều label-instance negatives; cần đọc cùng recall/F1 và FP/FN.
7. **Đổi threshold có đổi LRAP không?** Không nếu vẫn đánh giá cùng continuous scores; ranking không đổi.
8. **Threshold tuning có phải probability calibration không?** Không; đây là chọn cutoff theo objective, không chứng minh sigmoid score là xác suất đã calibrated.
9. **Vì sao không retrain backbone?** Để giữ representation/model scores cố định và đo riêng tác động decoder theo đề.
10. **Chia tier và cardinality ra sao?** Train-only tiers đã khóa, cardinality từ 27 emotion targets; giải thích nhóm 0 và phân biệt cardinality với số nhãn model dự đoán.
11. **Neutral được xử lý thế nào?** Neutral-only → emotion vector zero ở model 27, mixed giữ emotion; không gán empty prediction = Neutral. Không so trực tiếp Macro-F1 27 với 28 như cùng không gian nhãn.
12. **Có test leakage không?** Chỉ rõ split IDs, sample train overlap đã loại, train-only preprocessing/tier/co-occurrence và thời điểm khóa thresholds. Làm rõ official split chưa chứng minh thuật toán stratification.
13. **Một FN/FP điển hình chứng minh được điều gì?** Đọc true/pred/score/threshold trước, sau đó nhận xét có giới hạn; ví dụ chọn có chủ đích không đại diện tỷ lệ toàn tập.
14. **Nếu tuned không hơn 0,5 thì nghiên cứu có thất bại?** Không; câu trả lời là kết quả trên checkpoint/split này và mức generalization quan sát được, không nói 0,5 tối ưu cho mọi model/dataset.
15. **Đóng góp từng người và tái lập thế nào?** Mở được README/config/probabilities/script; khai báo AI assistance và phần người đã xác nhận đúng thực tế.

## Lịch dự kiến theo mốc cuối tháng

| Mốc nội bộ | Công việc chính | Điều kiện chuyển bước |
| --- | --- | --- |
| 11–13/11 | T1, bắt đầu T2; bổ sung thông tin nhóm, ôn nền tảng | Protocol rõ, assets/môi trường khôi phục được |
| 14–16/11 | T2–T3 | Checkpoint 1: validation khớp, thresholds khóa |
| 17–20/11 | T4–T6 | Test đánh giá, đủ breakdown và audit người thật |
| 21–24/11 | Chốt T7, làm T8 và deck T9 | Report và artifact thống nhất |
| 25–27/11 | Chốt T8, hai lượt mock defence T9 | Checkpoint 3: sẵn sàng nộp và bảo vệ |
| 28–30/11 | Dự phòng lỗi package/PDF, cập nhật format/lịch giảng viên | Giữ nguyên protocol đã chấm test |

Đây là lịch đề xuất theo mốc chưa chính thức, không phải deadline của học phần. Có thể làm song song theo người phụ trách: kỹ thuật T2–T5, nội dung report từ T1, review dữ liệu T6 khi đã có predictions; mỗi thành viên vẫn cần hiểu pipeline chung. Không giả định số thành viên.

## Phạm vi bổ sung sau phần bắt buộc

- **Q1 Gratitude–Disapproval:** giữ A làm appendix nếu liên quan. B cần hai người annotation độc lập và adjudication thật cho 71 mục; C có 48 slot đề xuất nhưng chưa có paraphrase/verification/inference. Không ghi là kết quả hoàn tất. Việc này không đáp ứng thay cho audit 20 FN/20 FP của Final.
- **Learning-rate sweep:** chưa có kết quả và không phải yêu cầu RQ1; không đưa vào đường găng hoặc retrain chỉ để đổi thresholds.
- **Dependency models, Focal/Class-Balanced Loss, robustness:** extensions theo đề; chỉ làm sau khi RQ1 hoàn tất và được kiểm chứng.
- **Ba seed:** bắt buộc với paper-track, không phải yêu cầu mặc định của RQ1. Nếu nhóm được chọn paper-track cần lập kế hoạch riêng; hiện ghi giới hạn một seed.

## Rủi ro và thông tin còn chờ

| Vấn đề | Cách xử lý |
| --- | --- |
| Môi trường mới không tái tạo đúng scores | Ưu tiên T2 sớm; phục hồi môi trường/segmentation, giữ checkpoint và sai số cố định |
| Validation đã dùng để chọn checkpoint và thăm dò threshold | Nêu rõ trong report; khóa trước test, không diễn giải validation gain là test gain |
| Per-label overfit hoặc subgroup quá nhỏ | Báo support/gap và đánh đổi, không chọn lọc kết quả |
| Audit người thật bị dồn cuối | Chốt người/cách chọn từ T1, làm ngay sau T4 |
| Official split chưa xác minh stratification | Giữ split hiện tại, nêu lịch sử minh bạch; xác nhận với giảng viên trước test, không tự chia lại để có điểm tốt |
| Chưa có deadline, thời lượng/ngôn ngữ oral, template và feedback giảng viên | Xác nhận khi có thông báo; lịch cuối tháng và deck tiếng Anh theo midterm chỉ là giả định làm việc |

Kế hoạch kết thúc khi RQ1 có kết quả test đúng protocol, report/package tái lập được và nhóm đã luyện defence. Không mở rộng thêm để thay đổi điểm sau khi xem test.
