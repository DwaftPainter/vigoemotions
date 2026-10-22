# ViGoEmotions — Lời dẫn pre-midterm 23/10/2026

12 slide • 8–10 phút • snapshot 22/10/2026. Điểm trong slide là baseline lịch sử, không phải fresh run.

## Slide 1. ViGoEmotions: tiến độ pre-midterm

- Nhận diện 27 cảm xúc trong bình luận tiếng Việt; một mẫu có thể mang nhiều cảm xúc.
- Đã có pipeline dữ liệu, EDA và kết quả ba baseline được lưu.
- Đang chạy lại từ đầu trên Colab; kết quả mới chưa được nhập.
- Trình bày: 23/10/2026 • Snapshot: 22/10/2026.

Minh chứng: README_PRE_MIDTERM.md; reports/README.md

Lời dẫn: Mở đầu: nhóm báo cáo việc đã làm và chỉ rõ minh chứng. Nói ngay rằng điểm được trình bày là bộ baseline lịch sử, chưa phải lượt chạy mới.

## Slide 2. Nhóm đã hoàn thành những gì?

- Kiểm tra dữ liệu, chuẩn hóa nhãn và lưu split ID.
- EDA: tần suất, cardinality, độ dài, quan hệ nhãn và văn phong.
- Ba baseline; metric, xác suất validation, run metadata và phân tích lỗi ban đầu.
- Còn thiếu: kết quả sweep mới, đọc lỗi neural và làm rõ yêu cầu stratification.

Minh chứng: docs/project_audit.md; reports/tables/model_comparison.csv

Lời dẫn: Cho thấy phần đã hoàn thành có file cụ thể. Không nói tất cả yêu cầu Midterm đã xong.

## Slide 3. Dữ liệu và kiểm tra chất lượng

- 20.664 mẫu nguồn: train 16.531 / val 2.066 / test 2.067.
- Phát hiện một text trùng train–test; loại bản train → 16.530 mẫu train.
- 27 đầu ra; neutral-only → vector 0; mixed-neutral giữ các cảm xúc.
- Mapping đã khớp ảnh nguồn. Paper nêu 8:1:1; chưa xác nhận stratification.

Minh chứng: docs/label_mapping.md; notebooks/01_dataset_validation.ipynb; splits/

Lời dẫn: Mapping đã khớp ảnh dataset card. Paper mục 5.1, trang PDF 6 (2810), chỉ nêu 8:1:1; không suy ra stratification. Mở docs/label_mapping.md để chỉ nguồn. Mở notebook 01 hoặc dataset_check và chỉ vào quy tắc xử lý. Test không được dùng để tối ưu model; kiểm tra trùng là audit dữ liệu.

## Slide 4. EDA: đặc điểm dữ liệu dẫn đến phương pháp

- Train trung bình 1.859 nhãn dương/mẫu → bài toán multilabel.
- Tần suất nhãn không đều → đọc Macro-F1 và F1 từng nhãn.
- Text phi chuẩn → phân tích riêng short, emoji, teencode.

Minh chứng: notebooks/02_exploratory_analysis.ipynb; reports/tables/dataset_summary.csv

Lời dẫn: Chỉ vào biểu đồ tần suất để giải thích imbalance. Nhiều nhãn trong một mẫu là lý do dùng sigmoid/BCE.

## Slide 5. EDA: các cảm xúc có quan hệ

- disappointment–sadness: 904 mẫu train; phi ≈ 0,362.
- Co-occurrence/correlation chỉ tính từ train.
- Đếm cùng xuất hiện cao chưa chứng minh quan hệ nhân quả.

Minh chứng: reports/tables/train_top_pairs.csv; reports/figures/train_cooccurrence.png

Lời dẫn: Mở bảng train_top_pairs, chỉ vào hàng đầu. Đây là cơ sở hỏi liệu model có giữ được các cặp cảm xúc không.

## Slide 6. Ba baseline và fine-tuning

- TF-IDF unigram/bigram + OvR Logistic Regression.
- BiGRU: học từ đầu, 10 epoch, batch 64, learning rate 0,001.
- PhoBERT: fine-tune toàn bộ backbone + head, 4 epoch, batch 16, learning rate 2e-5.
- BCE + AdamW; chọn checkpoint theo validation Macro-F1 tại 0,5.

Minh chứng: configs/midterm.json; scripts/train_neural.py; notebooks/04_colab_neural_baselines.ipynb

Lời dẫn: Giải thích backbone dùng trọng số pretrained, head mới học nhiệm vụ 27 nhãn. Không có trọng số riêng theo mẫu hoặc class weighting trong BCE hiện tại.

## Slide 7. Kết quả baseline đã lưu

- Bộ lịch sử 14/10; audit 22/10. Validation, threshold 0,5.
- Bảng là kết quả đã lưu; lượt chạy Colab mới chưa có xác nhận hoàn tất.

Minh chứng: reports/tables/model_comparison.csv; reports/gru_run.json; reports/phobert_run.json

Lời dẫn: Chỉ lần lượt ba hàng Macro-F1. PhoBERT hơn BiGRU 3.33 điểm phần trăm. Không nhận đây là điểm của fresh run, không gọi là kết quả test.

## Slide 8. Insight: văn phong và threshold

- Teencode: PhoBERT hơn BiGRU 9.41 điểm phần trăm Macro-F1; 307 mẫu.
- TF-IDF: threshold 0.15 → Macro-F1 0.4844; tại 0,5 → 0.2388.
- Đổi ngưỡng có trade-off: Hamming Loss tăng lên 0.0867.
- Đây là insight baseline/validation; chưa có kết luận về learning-rate sweep mới.

Minh chứng: reports/tables/neural_style_comparison.csv; reports/tables/tfidf_lr_thresholds_val.csv

Lời dẫn: Chỉ hàng teencode và threshold 0,15 trong CSV. Chưa chứng minh tác động riêng của pretraining hoặc khả năng tổng quát hóa trên test.

## Slide 9. Phân tích lỗi: đã có và còn thiếu

- TF-IDF tại 0,5 chỉ giữ đủ disappointment–sadness ở 2/122 ca validation có cặp.
- Đã có 20 FN / 20 FP ứng viên; đọc thủ công 10 bình luận TF-IDF khác nhau.
- Chưa hoàn tất đọc thủ công các ca lỗi BiGRU/PhoBERT.
- Danh sách lỗi tự sinh cần được xem và diễn giải trước khi kết luận.

Minh chứng: reports/tables/tfidf_lr_pair_preservation_val.csv; reports/tables/*_error_cases_val.csv

Lời dẫn: Mở một bảng lỗi bằng ID và nhãn; không trình chiếu raw comments. Phân biệt số ca ứng viên với số bình luận đã audit thực sự.

## Slide 10. Tái lập và lượt chạy mới hiện tại

- Lưu config, seed 42, data checksum, package versions và GPU.
- Notebook 04 ghi log, xuất history, kết quả và checkpoint riêng.
- Đã sửa lỗi mock subprocess trên Colab; 34 tests pass ở lần kiểm tra gần nhất.
- Log 22/10 06:39 UTC: PhoBERT khởi động trên CUDA; chờ epoch và Exit code 0.

Minh chứng: notebook 04; reports/*_run.json; scripts/audit_reports.py; commit 5f141de

Lời dẫn: Chỉ file run JSON: config, history, best_epoch, device. Với fresh run, startup log chỉ cho biết đã khởi động, không xác nhận đã train xong.

## Slide 11. Kế hoạch tiếp theo trước Midterm

- Hoàn tất 04, tải ZIP kết quả và checkpoint; kiểm chứng trước khi cập nhật điểm.
- Chạy mới 05: learning rate 1e-5, 2e-5, 3e-5; chỉ thay learning rate.
- Đủ artifacts và môi trường nhất quán mới summarize/chọn winner trên validation.
- Đọc lỗi neural; làm rõ yêu cầu stratification; bổ sung thông tin và phân công nhóm.

Minh chứng: notebook 05; scripts/tune_phobert.py; docs/phobert_tuning.md

Lời dẫn: Không nhập lại study cũ vào lượt fresh run. Resume là tính năng an toàn cho lần gián đoạn sau này; không phải kết quả của study. Không hứa sweep xong trước buổi báo cáo nếu chưa có artifacts.

## Slide 12. Hướng Final và kết luận tiến độ

- RQ1: threshold chung 0,5 có phù hợp với cả 27 nhãn không?
- So fixed / global tuned / per-label tuned; giữ nguyên backbone.
- Tune trên validation, khóa model/ngưỡng rồi đánh giá test đúng một lần.
- Đã có nền tảng và bằng chứng baseline; đang hoàn thiện thí nghiệm mới và audit lỗi.

Minh chứng: docs/02-vigoemotions-vietnamese-emotion.pdf: Final RQ1; README_PRE_MIDTERM.md

Lời dẫn: Kết: nhóm có pipeline kiểm chứng được, kết quả bước đầu và kế hoạch cụ thể. Xin góp ý về hướng phân tích lỗi và threshold; không tuyên bố đã hoàn thành Final.
