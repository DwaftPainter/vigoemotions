# ViGoEmotions — Báo cáo tiến độ pre-midterm

Buổi báo cáo: 23/10/2026 • Snapshot: 22/10/2026

## Báo cáo tiến độ pre-midterm

ViGoEmotions • ATI • Buổi trình bày dự kiến: 23/10/2026
Chốt trạng thái tài liệu: 22/10/2026. Thời lượng đề xuất: 8–10 phút.

### Thông điệp mở đầu

Nhóm đã xây dựng pipeline kiểm tra dữ liệu, EDA, ba baseline và phân tích validation. Đã có một bộ kết quả baseline được lưu và kiểm chứng; hiện đang chạy lại từ đầu trên Colab. Thử nghiệm learning rate mới chưa có kết quả được nhập vào repository.

| Hạng mục | Trạng thái có bằng chứng |
| --- | --- |
| Kiểm tra dữ liệu và EDA | Đã có bảng, hình, split ID và quy ước nhãn. |
| TF-IDF, BiGRU, PhoBERT | Có kết quả lịch sử; không dùng làm kết quả của lượt chạy mới. |
| Lượt chạy mới notebook 04 | Log người dùng gửi xác nhận PhoBERT đã khởi động trên CUDA lúc 06:39 UTC, 22/10; chưa xác nhận hoàn tất. |
| Learning-rate study mới | Code/config/notebook đã chuẩn bị; chưa có kết quả mới để so sánh. |
| Error analysis và Final | Có audit thủ công TF-IDF; neural còn thiếu. Chưa đánh giá metric test. |

### Phạm vi trình bày

Báo cáo việc đã làm, phát hiện chính, bằng chứng có thể mở trực tiếp và công việc còn thiếu. Các số liệu baseline trong tài liệu đều thuộc bộ kết quả đã lưu trước lượt chạy mới.

Minh chứng: README_PRE_MIDTERM.md; docs/project_audit.md; reports/phobert_tuning/study.json; log Colab người dùng cung cấp ngày 22/10/2026.

## Dữ liệu: đã kiểm tra và khám phá gì?

Bài toán: nhận diện nhiều cảm xúc trong cùng một bình luận tiếng Việt. Đầu ra gồm 27 nhãn độc lập; neutral-only được biểu diễn bằng vector toàn 0.

| Đã làm | Phát hiện / ý nghĩa |
| --- | --- |
| Kiểm tra split và trùng lặp | Nguồn: train 16.531, validation 2.066, test 2.067. Loại một mẫu train trùng text với test; train thực nghiệm còn 16.530. |
| Chuẩn hóa nhãn | Loại neutral khỏi mẫu mixed-neutral, giữ các cảm xúc; khử nhãn lặp. |
| Tần suất và cardinality | Train có trung bình 1.859 nhãn dương/mẫu. Nhãn không cân bằng → cần Macro-F1 và per-label F1. |
| Độ dài và văn phong | Đã có phân bố token khoảng trắng/ký tự và các nhóm short, emoji-heavy, teencode. Các nhóm style dựa trên heuristic. |
| Quan hệ giữa nhãn | disappointment–sadness: 904 mẫu train; phi ≈ 0,362. Tính co-occurrence/correlation chỉ từ train. |

### Quyết định phương pháp từ EDA

Nhiều cảm xúc/mẫu → sigmoid và BCE thay cho softmax. Nhãn thưa và mất cân bằng → không đọc riêng Hamming Loss. Văn bản phi chuẩn → phân tích riêng theo phong cách và xem ca lỗi.

### Mapping đã xác minh; split còn giới hạn

Mapping khớp đầy đủ ảnh bảng nhãn Hugging Face người dùng cung cấp ngày 22/10: ID 0–26 là cảm xúc, ID 27 là neutral. Paper mục 5.1, trang PDF 6 (2810), nêu tỷ lệ 8:1:1 nhưng không mô tả thuật toán chia tập. Giữ split được cấp; audit phân bố không chứng minh stratification.

Minh chứng: docs/label_mapping.md; notebooks/02_exploratory_analysis.ipynb; reports/tables/dataset_summary.csv; reports/tables/train_top_pairs.csv; reports/figures/label_frequency.png; splits/train.csv.

## Mô hình và cách fine-tune đã triển khai

| Mô hình | Cấu hình / cách học |
| --- | --- |
| TF-IDF + OvR Logistic Regression | Word unigram/bigram, tối đa 40.000 đặc trưng; fit đặc trưng trên train; 27 bộ phân loại OvR. |
| BiGRU | Embedding 128; hidden 128 mỗi chiều; dropout 0,3; max length 128; batch 64; learning rate 0,001; 10 epoch. |
| PhoBERT | vinai/phobert-base-v2; VnCoreNLP + BPE; max length 256; batch 16; learning rate 2e-5; weight decay 0,01; 4 epoch. |

### PhoBERT học những trọng số nào?

Fine-tune toàn bộ backbone pretrained và classification head mới. Dùng AdamW, BCEWithLogitsLoss và gradient clipping 1. Sau mỗi epoch, đo validation tại threshold 0,5; lưu checkpoint có Macro-F1 cao nhất, hòa thì giữ epoch sớm hơn.

Hiện không có cơ chế gán trọng số riêng cho từng hình/mẫu. Dữ liệu là văn bản. BCE hiện tại không thêm class weights. Checkpoint được chọn theo validation sau epoch; đây là bước khác với tìm learning rate tốt nhất.

### Hyperparameter tuning được làm ở đâu?

Notebook 05 thử learning rate 1e-5, 2e-5, 3e-5; giữ các cấu hình khác cố định. Chọn bằng validation, kiểm tra dữ liệu/môi trường/phần cứng trước khi so sánh. Không dùng test để điều chỉnh hyperparameter.

Minh chứng: configs/midterm.json; scripts/train_neural.py; scripts/tune_phobert.py; notebooks/04_colab_neural_baselines.ipynb; notebooks/05_colab_phobert_tuning.ipynb; docs/phobert_tuning.md.

## Kết quả đã lưu và insight ban đầu

BỘ BASELINE LỊCH SỬ — được nhập/kiểm chứng ngày 14/10 và audit lại ngày 22/10/2026. Validation n = 2.066; threshold = 0,5. Chưa phải điểm của lượt chạy Colab mới.

| Baseline | Macro-F1 ↑ | Micro-F1 ↑ | Hamming ↓ | LRAP ↑ |
| --- | --- | --- | --- | --- |
| TF-IDF + OvR LR | 0.2388 | 0.2712 | 0.0606 | 0.6407 |
| BiGRU | 0.4060 | 0.4526 | 0.0604 | 0.6322 |
| PhoBERT | 0.4393 | 0.5160 | 0.0533 | 0.7158 |

PhoBERT cao hơn BiGRU 3.33 điểm phần trăm Macro-F1 trong lượt chạy này. Đây là so sánh hai pipeline; chưa tách riêng ảnh hưởng của pretraining, kiến trúc hay preprocessing.

### Ba điều nhóm rút ra từ dữ liệu đã đo

1. Theo văn phong: chênh lệch PhoBERT − BiGRU ở nhóm teencode là 9.41 điểm phần trăm Macro-F1 (307 mẫu), lớn nhất trong ba nhóm style đã khảo sát. Nhóm có thể chồng lấp; chưa có kiểm định ý nghĩa thống kê.

2. Threshold rất quan trọng: TF-IDF trên lưới validation đạt Macro-F1 0.4844 tại threshold 0.15, so với 0.2388 ở 0,5. Hamming Loss tăng lên 0.0867. Đây là khảo sát validation, chưa chứng minh tổng quát hóa trên test.

3. Phụ thuộc nhãn chưa được giữ tốt: TF-IDF tại 0,5 dự đoán đúng đồng thời disappointment–sadness ở 2/122 mẫu validation có cặp đó. Co-occurrence cao không tự bảo đảm mô hình dự đoán đủ cả hai nhãn.

Minh chứng: reports/tables/model_comparison.csv; reports/tables/neural_style_comparison.csv; reports/tables/tfidf_lr_thresholds_val.csv; reports/tables/tfidf_lr_pair_preservation_val.csv. Không so tốc độ tuyệt đối từ thời gian CPU và GPU khác phạm vi đo.

## Minh chứng, tái lập và phân tích lỗi

| Muốn chứng minh việc gì? | Mở file nào và chỉ vào đâu? |
| --- | --- |
| Thực sự đã có kết quả | reports/tables/model_comparison.csv: mỗi dòng là một baseline; đọc kèm <model>_run.json. |
| Có train và chọn epoch | reports/gru_run.json, reports/phobert_run.json: config, history, best_epoch, device, environment. |
| Lưu xác suất trước threshold | reports/predictions/: NPZ validation có ID, ground truth, probability và danh sách nhãn; không có raw text. |
| Có EDA và split cố định | reports/figures/: tần suất/phân bố/quan hệ; splits/: ID từng tập. |
| Có kiểm tra tính nhất quán | scripts/audit_reports.py và docs/project_audit.md: đối chiếu nguồn dữ liệu, NPZ, metric và bảng phân tích. |
| Có quy trình Colab mới | Notebook 04: streaming log, summary/history và xuất ZIP kết quả + ZIP checkpoint riêng. |

### Phân tích lỗi đã thực hiện đến đâu?

Đã xuất ứng viên 20 false negatives và 20 false positives; phần diễn giải thủ công hiện có là 10 bình luận TF-IDF khác nhau của bộ baseline lịch sử. Chưa hoàn tất audit thủ công các lỗi BiGRU/PhoBERT. File error_cases được xuất tự động không đồng nghĩa nhóm đã đọc hết các ca lỗi.

### Độ tin cậy của quy trình

Seed 42; lưu cấu hình, package versions, checksum dữ liệu và GPU. Resume tuning kiểm tra từng run, không coi thư mục tồn tại là đã xong, không ghi đè partial output. Bộ kiểm thử gần nhất: 34 tests pass; bản sửa test Colab nằm ở commit 5f141de.

Minh chứng: tests/test_phobert_tuning.py; reports/tables/*_error_cases_val.csv; reports/*_run.json; scripts/audit_reports.py. Không trình chiếu bình luận thô hoặc checkpoint riêng tư.

## Việc tiếp theo và lời kết để trình bày

| Thứ tự | Cần hoàn thành / tiêu chí xong |
| --- | --- |
| 1. Hoàn tất chạy mới 04 | Đủ ba baseline; các lệnh Exit code 0; lưu run JSON, metric, NPZ, logs; tải ZIP kết quả và checkpoint. Chưa có bằng chứng hoàn tất toàn bộ lượt mới. |
| 2. Chạy mới 05 | Ba learning rate hoàn tất, đủ epoch và artifact hợp lệ; môi trường/dữ liệu/phần cứng nhất quán; mới summarize và chọn winner. Không dùng kết quả study cũ. |
| 3. Hoàn thiện Midterm | Cập nhật bảng/insight từ kết quả mới; đọc lỗi neural; làm rõ yêu cầu stratification; điền thành viên/phân công. |
| 4. Thực hiện Final | Chọn neural architecture; giữ nguyên backbone, tune global/per-label threshold bằng validation; khóa quyết định rồi đánh giá test đúng một lần. |

### Trước buổi báo cáo ngày 23/10

Mở sẵn PDF slide, bảng so sánh baseline, notebook 04 và run JSON. Nếu kết quả mới chưa hoàn tất, dùng bảng lịch sử có ghi mốc rõ ràng; trình bày lượt mới là đang chạy. Nếu đã hoàn tất, kiểm chứng rồi mới thay điểm và trạng thái trong tài liệu.

### Lời kết đề xuất

“Nhóm đã hoàn thành nền tảng dữ liệu, EDA và ba baseline có đầu ra kiểm chứng được. Kết quả ban đầu cho thấy PhoBERT tốt nhất tại ngưỡng 0,5, nhưng quyết định ngưỡng ảnh hưởng mạnh đến F1. Hiện nhóm đang chạy lại pipeline và chuẩn bị so sánh learning rate trên validation. Phần còn lại là hoàn tất thí nghiệm mới, đọc lỗi neural và làm rõ yêu cầu stratification trước Midterm.”

### Nếu giảng viên hỏi

Đã fine-tune chưa? Có, backbone và head; sweep hyperparameter mới chưa có kết luận. Test dùng để tune chưa? Chưa; test chỉ được kiểm tra schema/phân bố/trùng lặp trong audit dữ liệu. Kết quả mới đâu? Chỉ xác nhận sau khi có artifacts đầy đủ; log khởi động không chứng minh train xong.

Minh chứng: Đề: docs/02-vigoemotions-vietnamese-emotion.pdf, mục Midterm A–E và Final RQ1. Tài liệu này là snapshot tiến độ ngày 22/10, không tự cập nhật khi Colab hoàn tất.
