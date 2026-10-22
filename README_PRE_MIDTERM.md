# ViGoEmotions — README cho buổi pre-midterm

**Mục đích:** báo cáo tiến độ đã làm và mở minh chứng ngay tại từng phần. Nội dung dựa trên [yêu cầu Midterm A–E](docs/02-vigoemotions-vietnamese-emotion.pdf), [student guide](docs/student-guide.pdf) và hướng dẫn bổ sung trong ảnh người dùng cung cấp: **tối đa 20 slide; khám phá dữ liệu; trình bày model/phương pháp/cách fine-tune; insight; viết report**.

**Thông điệp chính:** Nhóm đã kiểm tra dữ liệu, làm EDA, xây và chạy đủ ba baseline, có kết quả trên validation và một số phân tích lỗi. Trước khi nộp Midterm, nhóm còn phải xác minh một số giả định về dữ liệu, đọc thêm lỗi của mô hình neural và hoàn thiện thông tin nhóm.

## Đối chiếu năm ý giảng viên yêu cầu

| Nội dung | Đã có và còn cần bổ sung | Minh chứng |
| --- | --- | --- |
| Tối đa 20 slide | Bộ hiện tại được cập nhật thành **18 slide**; script build chặn nếu vượt 20 | [Slide PDF](reports/midterm_slides.pdf) |
| Khám phá được gì từ dữ liệu? | Đã có: nhãn dương thưa, trung bình 1,859 cảm xúc/mẫu, median 11 token khoảng trắng; cặp disappointment–sadness có 904 mẫu train; phát hiện một text trùng train/test | [Báo cáo mục 2](reports/midterm_report.md#2-dữ-liệu-quyền-truy-cập-và-kiểm-tra-chất-lượng), [notebook 02](notebooks/02_exploratory_analysis.ipynb) |
| Model, phương pháp, fine-tune/điều chỉnh | Đã fine-tune PhoBERT toàn bộ backbone và head: BCE, AdamW, lr 2e-5, batch 16, 4 epoch. **Chưa chạy sweep GPU**; đã bổ sung code/notebook thử 1e-5, 2e-5, 3e-5 | [Cấu hình baseline](configs/midterm.json), [hướng dẫn tuning](docs/phobert_tuning.md), [notebook 05](notebooks/05_colab_phobert_tuning.ipynb) |
| Insight từ fine-tune | Đã có so sánh baseline: PhoBERT hơn BiGRU 3,33 điểm phần trăm Macro-F1; nhóm teencode có delta lớn nhất trong ba nhóm style. Chưa tách tác động pretraining/kiến trúc; **chưa có insight về thay đổi learning rate** trước khi sweep chạy | [Bảng neural theo style](reports/tables/neural_style_comparison.csv), [báo cáo mục 5.6](reports/midterm_report.md#56-fine-tuning-và-thử-nghiệm-learning-rate-phobert) |
| Viết report | Đã có bản PDF/Markdown; bổ sung mục 5.6 mô tả fine-tuning, thiết kế sweep và trạng thái. Còn điền kết quả mới, audit lỗi và thông tin nhóm | [Report PDF](reports/midterm_report.pdf), [bản sửa được](reports/midterm_report.md) |

Từ đặc điểm dữ liệu đến phương pháp: nhiều cảm xúc/mẫu dẫn đến 27 đầu ra sigmoid + BCE; nhãn dương thưa cần đọc cả Macro-F1 và recall, không chỉ Hamming Loss; văn bản phi chuẩn dẫn đến phân tích riêng teencode/emoji. Đây là lý do lựa chọn phương pháp; kiểm chứng ảnh hưởng của mỗi yếu tố cần thí nghiệm đối chứng.

## Mở sẵn trước khi trình bày

1. [Slide Midterm (PDF)](reports/midterm_slides.pdf) — dùng để dẫn câu chuyện. [Bản Markdown](reports/midterm_slides.md) có ghi chú thuyết trình từng slide.
2. [Bảng so sánh ba mô hình](reports/tables/model_comparison.csv) — mở khi nêu kết quả thực nghiệm.
3. [Báo cáo Midterm](reports/midterm_report.pdf) — mở các bảng và diễn giải chi tiết khi được hỏi.
4. [Nhật ký công việc](docs/midterm_worklog.md) — tra cứu thứ tự công việc, bằng chứng và phần còn thiếu.

Không cần chạy lại mô hình trong lúc thuyết trình. Các notebook cho thấy quy trình có thể tái lập; các bảng và run metadata là minh chứng cho lượt chạy đã thực hiện. Không chiếu bình luận gốc từ `data/` hoặc đưa CSV nguồn vào slide.

## Kịch bản báo cáo tiến độ

| Phần nói | Nội dung ngắn để trình bày | Mở minh chứng ở đâu |
| --- | --- | --- |
| 1. Bài toán | Một bình luận có thể mang nhiều cảm xúc; đầu ra là 27 nhãn cảm xúc. Neutral được mã hóa bằng vector 27 chiều toàn 0. | [Slide 1](reports/midterm_slides.pdf), [báo cáo mục 1](reports/midterm_report.md#1-mục-tiêu-và-phạm-vi) |
| 2. Dữ liệu và kiểm tra chất lượng | Tổng 20.664 mẫu: train 16.531, validation 2.066, test 2.067. Phát hiện một text trùng giữa train và test; loại bản train khỏi huấn luyện nên còn 16.530 mẫu. Đã xử lý nhãn neutral, mixed-neutral và nhãn lặp theo quy ước thống nhất. | [Slide 2](reports/midterm_slides.pdf), [notebook 01](notebooks/01_dataset_validation.ipynb), [dataset check](docs/dataset_check.md) |
| 3. EDA và chia tập | Đã thống kê đủ 27 nhãn, số nhãn trên mỗi mẫu, độ dài văn bản, đồng xuất hiện và tương quan nhãn, cùng các nhóm short/emoji/teencode. Giữ split được cấp; fit đặc trưng trên train, so sánh mô hình trên validation. | [Slide 3–6](reports/midterm_slides.pdf), [notebook 02](notebooks/02_exploratory_analysis.ipynb), [hình tần suất nhãn](reports/figures/label_frequency.png), [bảng cặp nhãn train](reports/tables/train_top_pairs.csv), [ID split](splits/train.csv) |
| 4. Ba baseline | Đã chạy TF-IDF + One-vs-Rest Logistic Regression, BiGRU và PhoBERT. BiGRU/PhoBERT chạy trên Colab Tesla T4; cả ba dùng 27 đầu ra và cùng validation để so sánh. | [Slide 7](reports/midterm_slides.pdf), [notebook 03](notebooks/03_baseline_comparison.ipynb), [notebook 04](notebooks/04_colab_neural_baselines.ipynb), [cấu hình](configs/midterm.json) |
| 5. Kết quả | Tại ngưỡng 0,5, Macro-F1 validation: TF-IDF **0,2388**, BiGRU **0,4060**, PhoBERT **0,4393**. Bảng cũng có Micro-F1, Hamming Loss, LRAP, số tham số và thời gian train. PhoBERT cao nhất trong ba baseline theo Macro-F1 của lượt chạy này. | [Slide 8](reports/midterm_slides.pdf), [bảng so sánh](reports/tables/model_comparison.csv), [kiểm tra kết quả Colab](docs/colab_results_check.md) |
| 6. Đã rút ra gì từ kết quả? | TF-IDF tại ngưỡng 0,5 nhận diện gratitude tốt nhưng bỏ sót nhiều nhãn; cặp disappointment–sadness chỉ dự đoán đúng đủ hai nhãn ở 2/122 ca validation có cặp này. Khảo sát ngưỡng 0,15 trên validation nâng Macro-F1 của TF-IDF lên 0,4844 nhưng Hamming Loss tăng. PhoBERT hơn BiGRU nhiều nhất ở nhóm teencode trong ba nhóm phong cách đã khảo sát. | [Slide 9–12](reports/midterm_slides.pdf), [báo cáo mục 5](reports/midterm_report.md#5-kết-quả-thực-nghiệm-trên-validation), [ca lỗi TF-IDF](reports/tables/tfidf_lr_error_cases_val.csv) |
| 7. Fine-tune và điều chỉnh | Trình bày cấu hình PhoBERT đã chạy; sweep learning rate đang chờ GPU. Sau khi có kết quả sẽ đối chiếu đường học, per-label/style delta và ca lỗi. | [Slide 13–15](reports/midterm_slides.pdf), [thiết kế sweep](docs/phobert_tuning.md) |
| 8. Hướng tiếp theo | Final sẽ kiểm tra liệu dùng một ngưỡng 0,5 cho cả 27 nhãn có hợp lý không: so sánh ngưỡng cố định, ngưỡng chung được chọn và ngưỡng riêng từng nhãn. Chọn ngưỡng trên validation, khóa lại rồi mới đánh giá test. | [Slide 16–17](reports/midterm_slides.pdf), [báo cáo mục 7](reports/midterm_report.md#7-câu-hỏi-nghiên-cứu-cho-final-và-thiết-kế-kiểm-chứng) |

**Cách nói về kết quả:** Các điểm số trên là **validation**, chưa phải kết quả test. Ngưỡng 0,15 là kết quả khảo sát trên validation của **TF-IDF**, không phải điểm đã kiểm chứng cho PhoBERT hay Final. Thời gian TF-IDF đo trên CPU, neural đo trên GPU và bao gồm các bước khác nhau, nên không so tốc độ tuyệt đối từ ba con số này.

## Phần còn thiếu cần nói thẳng

- Chạy notebook 05 trên Colab GPU để hoàn tất sweep PhoBERT; nhập ZIP kết quả rồi cập nhật insight từ số liệu mới. Chi tiết tại [phobert_tuning.md](docs/phobert_tuning.md). Không đợi test để chỉnh hyperparameter.
- Xác minh mapping ID–tên nhãn với metadata của bộ dữ liệu được cấp quyền.
- Xác nhận nguồn tạo split gốc có đáp ứng yêu cầu phân tầng đa nhãn trong đề hay không; hiện mới giữ split và audit phân bố, chưa chứng minh thuật toán tạo split.
- Đọc thủ công các ca lỗi BiGRU/PhoBERT và bổ sung diễn giải. Các CSV hiện tại là danh sách ca lỗi để xem xét; phần đọc thủ công trong báo cáo mới dành cho TF-IDF.
- Lưu riêng checkpoint và vocabulary Colab nếu cần tái sử dụng cho Final; ZIP kết quả đã nhập không chứa chúng.
- Bổ sung tên, MSSV, phân công nhóm và rà lại báo cáo/slide trước khi nộp.

Nguồn trạng thái: [midterm worklog](docs/midterm_worklog.md#10-kiểm-chứng-và-phần-còn-phải-hoàn-tất) và [Colab results check](docs/colab_results_check.md#việc-còn-lại). Rà lại các mục này nếu nhóm chạy thêm thí nghiệm trước buổi báo cáo.

## Nếu giảng viên hỏi sâu

- **Vì sao dùng Macro-F1?** Nó cho mỗi nhãn trọng số ngang nhau, giúp thấy vấn đề ở các nhãn khó; Micro-F1 gộp mọi quyết định nhãn. [Báo cáo mục 4.1](reports/midterm_report.md#41-chỉ-số).
- **Test đã dùng chưa?** Chỉ kiểm tra schema/phân bố và dò trùng với train; chưa dùng để chọn mô hình, epoch hay ngưỡng, chưa báo cáo metric test. [Báo cáo mục 3](reports/midterm_report.md#3-split-và-pipeline-tiền-xử-lý).
- **Có phân tích lỗi chưa?** Đã xuất 20 FN và 20 FP tự tin nhất của TF-IDF và đọc thủ công 10 bình luận khác nhau; còn cần đọc ca lỗi neural. [Báo cáo mục 6](reports/midterm_report.md#6-phân-tích-lỗi-và-trường-hợp-biên).
- **Có thí nghiệm bổ sung không?** Có ablation trên TF-IDF về cách xử lý mixed-neutral. Loại 251 mẫu mixed-neutral khỏi train làm Macro-F1 validation tăng **0,00159** tại ngưỡng 0,5, còn LRAP giảm nhẹ. Đây là kết quả của một split, một seed và một baseline; trình bày khi còn thời gian hoặc khi được hỏi. [Tóm tắt thí nghiệm](reports/ablation_mixed_neutral/experiment_summary.md), [báo cáo mục 5.5](reports/midterm_report.md#55-ablation-xử-lý-nhãn-mixed-neutral). Thí nghiệm này có trong báo cáo hiện tại nhưng chưa có trong bộ slide PDF hiện tại.

Đề Midterm yêu cầu đủ các mục A–E; [báo cáo mục 8](reports/midterm_report.md#8-tái-lập-giới-hạn-và-đối-chiếu-yêu-cầu) đối chiếu từng mục với minh chứng. [README chính](README.md) hướng dẫn thiết lập và chạy lại project.
