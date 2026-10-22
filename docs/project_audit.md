# Audit project ViGoEmotions — 22/10/2026

Phạm vi: toàn bộ script, năm notebook, hợp đồng dữ liệu/split, các artifact đang có trong `reports/`, quy trình xuất Colab và tài liệu. Không chạy lại huấn luyện neural hoặc inference từ checkpoint trong lượt audit này.

## Kết quả đối chiếu dữ liệu và số liệu

- SHA-256 ba CSV nguồn khớp `eda_environment.json`, run JSON của đủ ba baseline và metadata ablation.
- Train thực nghiệm 16.530, validation 2.066, test 2.067; train bị loại ID `1698`. Các ID trong `splits/` khớp dữ liệu sau audit trùng lặp.
- Tạo lại bảng EDA trong thư mục tạm và đối chiếu CSV đã lưu. Ma trận quan hệ nhãn chỉ dùng train. Các cặp cùng count có thể đổi thứ tự giữa phiên bản pandas; audit so theo ID cặp, không coi thứ tự hòa là khác dữ liệu.
- NPZ baseline khớp đúng ID/thứ tự hàng, 27 nhãn, ground truth và cardinality. Tính lại toàn bộ bảy loại bảng evaluator cho từng mô hình từ xác suất, gồm metric, per-label, style, cardinality, threshold, pair preservation và error cases.
- Các analysis JSON, bảng ba mô hình và bảng so sánh neural theo phong cách khớp các nguồn trên. Checkpoint epoch trong history khớp quy tắc chọn Macro-F1, hòa chọn epoch sớm.
- Ablation A/B có đúng ID train và cùng validation; metric tổng thể, delta, precision/recall/F1 từng nhãn, tier và nhóm mixed-neutral khớp NPZ.
- Sweep PhoBERT đang `prepared`: ba config chỉ khác learning rate. Chưa có run sweep hoàn chỉnh hay metric sweep mới; không có file xác suất test trong snapshot.

Số liệu baseline tại threshold 0,5 được giữ nguyên:

| Mô hình | Macro-F1 | Micro-F1 | Hamming Loss | LRAP |
| --- | --- | --- | --- | --- |
| TF-IDF + OvR LR | 0,2388 | 0,2712 | 0,0606 | 0,6407 |
| BiGRU | 0,4060 | 0,4526 | 0,0604 | 0,6322 |
| PhoBERT | 0,4393 | 0,5160 | 0,0533 | 0,7158 |

## Các vấn đề phát hiện và xử lý

| Mức độ | Vấn đề | Xử lý |
| --- | --- | --- |
| Cần sửa | Một số câu tự sinh luôn nói PhoBERT thắng, cả hai chọn epoch cuối, teencode có delta lớn nhất hoặc dùng điểm TF-IDF cố định. Sau lượt chạy mới, câu có thể không khớp bảng. | Report/slide hiện lấy người thắng, epoch, nhãn tốt/kém, pair recall, style delta và độ nhạy threshold từ artifact. Có regression test với BiGRU thắng và epoch tốt nhất ở giữa. |
| Cần sửa | LRAP float32 tính lại ở local khác Colab khoảng `4,63e-7` với BiGRU và `7,48e-7` với PhoBERT. Ngưỡng kiểm tra chặt có thể từ chối bộ kết quả hợp lệ khi nhập giữa môi trường. | Tái hiện với môi trường local sklearn 1.9.1 và metadata Colab sklearn 1.6.1; chuyển đầu vào LRAP sang float64 khôi phục đúng giá trị đã lưu. Giữ NPZ, thứ hạng nhãn và độ chặt của kiểm tra; thêm regression test. |
| Cần làm rõ | Audit thủ công TF-IDF là nhận xét từ baseline 14/10; xuất CSV mới không có nghĩa đã đọc lại lỗi. | Thêm mốc và phạm vi của nhận xét thủ công vào report/slide; nhóm cần đối chiếu khi huấn luyện lại. |
| Cần sửa tài liệu | Thiếu mục lục file; `train_top_pairs.csv` thực tế có 351 cặp; tên cột index ma trận có thể bị hiểu là nhãn thứ 28; nguồn PDF có chỗ ghi sai thư mục root. | Thêm README chính trong reports và README từng nhóm; sửa đường dẫn nguồn và giải thích các quy ước. |

Không thay điểm số thủ công hoặc dựng kết quả tuning. Thay đổi LRAP chỉ chuẩn hóa độ chính xác khi tính metric, không thay mô hình hay xác suất dự đoán.

## Tổ chức reports sau audit

[reports/README.md](../reports/README.md) là điểm vào chính. Có hướng dẫn riêng cho [tables](../reports/tables/README.md), [figures](../reports/figures/README.md), [predictions](../reports/predictions/README.md), [ablation](../reports/ablation_mixed_neutral/README.md) và [tuning](../reports/phobert_tuning/README.md).

Giữ các đường dẫn artifact hiện tại vì scripts, notebook và report generator đang dùng chúng. Phân nhóm bằng mục lục và bảng tra cứu đủ để đọc thuận tiện, tránh phải sửa đồng loạt đường dẫn hoặc tạo hai bản kết quả trùng nhau. Tất cả 67 artifact có trước khi thêm README đều có liên kết trực tiếp từ bộ hướng dẫn.

Các README phân biệt file hiện hữu với file chỉ được tạo sau lượt chạy mới. Không tạo CSV/history/metric giả để lấp chỗ trống.

## Kiểm chứng

Từ project root, dùng môi trường local đã cài dependency và dữ liệu được cấp quyền:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/audit_reports.py
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/package_colab.py --include-results
```

- **19 kiểm thử qua**, gồm hai regression test report và một test LRAP mới.
- `audit_reports.py` đối chiếu EDA/split, ba baseline, ablation và trạng thái tuning. Bảng/hình tái tạo chỉ ghi trong thư mục tạm; nếu sweep completed, nó kiểm tra một bản sao tạm để giữ reports gốc nguyên vẹn.
- Đã kiểm tra schema và cú pháp code của cả năm notebook, cú pháp các script, liên kết nội bộ và report/slide PDF. Slide vẫn 18 trang, không vượt 20.
- ZIP Colab được tạo lại để chứa hướng dẫn và code đã sửa; dữ liệu nguồn, checkpoint và vocabulary nằm ngoài ZIP code/kết quả công khai.

## Giới hạn còn lại

- Chưa xác minh mapping ID–nhãn với metadata dataset được cấp quyền; hiện dùng mapping kế thừa từ notebook gốc.
- Chưa xác minh thuật toán tạo split gốc có multilabel stratification hay không; audit prevalence không chứng minh điều đó.
- Chưa đọc thủ công ca lỗi neural, chưa chạy sweep learning rate và chưa có thí nghiệm Final/test.
- Local không có checkpoint baseline hoặc PyTorch/GPU; chỉ kiểm chứng đầu ra đã lưu, không xác nhận suy luận lại từ trọng số hoặc tái đo thời gian train.
- Một seed, heuristic style và validation đã dùng chọn checkpoint/ngưỡng hạn chế khả năng khái quát. Đây không phải kiểm định ý nghĩa thống kê hoặc chứng nhận các nhãn gốc đều đúng.
- Report có phần mô tả thiết kế cố định của project và phần audit viết tay. Số liệu tự sinh đã được cập nhật theo artifact; nhóm vẫn cần rà diễn giải sau khi đổi giao thức, dữ liệu hoặc config.

## Bổ sung nguồn ngày 22/10 sau audit

Mapping ID–nhãn đã khớp ảnh dataset card người dùng cung cấp; hạng mục mapping ở snapshot audit phía trên được giải quyết trong phạm vi nguồn này. Paper mục 5.1, trang PDF 6 (2810), xác nhận 8:1:1 nhưng không mô tả thuật toán chia tập. Chưa có bằng chứng stratification. Xem [bảng mapping và đối chiếu paper](label_mapping.md). Không thay config, split, mô hình hay kết quả đo.
