# Evidence Final28 cho oral defence

Final đã hoàn thành trên PhoBERT28, checkpoint epoch15. Validation tái tạo max error=0. Global cutoff=0.4; per-label có 28 cutoffs, Neutral=0.75. Test all28 Macro-F1: fixed0.589964, global0.591254, per-label0.585415. Xem [paper](../final_report.pdf) hoặc [bộ ôn protocol/kết quả](../../docs/oral_defence/02_protocol_results_vi.md).

- [Reproduction](validation_reproduction.json), [thresholds](thresholds.json), [pre-test lock](pre_test_lock.json), [completed attempt](test_attempt.json), [import verification](execution_verification.json).
- [Tables](tables/): validation/test comparison, per-label, tiers, cardinality, Neutral, global sweep và gap.
- [Predictions](predictions/): scores trước decoding, IDs/targets và ba pred matrices; không có raw comments.
- [Figures](figures/): đồ thị dùng trong paper và notes.
- [Case packet](human_audit/README.md): phân tích định tính, chưa có independent adjudication.

Cùng score matrix được dùng cho ba decoder và hai scope. All28 là chính; emotion27 là phép chiếu không tune lại. Tiers chỉ gồm 27 emotions; cardinality chính gồm Neutral. Các config/protocol status trước execution được giữ nguyên để bảo toàn source hashes.

Nhánh ôn giữ artifacts để đọc, không thêm lần test inference. Các snapshot implementation/pending và bảng cũ đã bỏ; workflow/rebuilders đầy đủ ở `feature/final28-report`, commit `289f2fb`.
