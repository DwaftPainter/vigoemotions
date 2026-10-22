# ViGoEmotions — ATI Midterm 2026

Nội dung giữa kỳ bám `02-vigoemotions-vietnamese-emotion.pdf` và `student-guide.pdf`.
Báo cáo/slide viết tiếng Việt; kết quả đo trên **validation**, test giữ cho Final.

## Mở nội dung

- [Mục lục reports và ý nghĩa từng file](reports/README.md) — thứ tự đọc, quy ước tên, metric và trạng thái thí nghiệm.
- [Kết quả audit toàn project](docs/project_audit.md) — đã kiểm tra gì, lỗi đã sửa và giới hạn còn lại.
- [README cho buổi pre-midterm](README_PRE_MIDTERM.md) — kịch bản báo cáo tiến độ và đường dẫn đến từng minh chứng.
- [Các công việc từ bắt đầu đến midterm](docs/midterm_worklog.md) — thứ tự thực hiện, lý do, bằng chứng và việc còn lại.
- [Báo cáo PDF](reports/midterm_report.pdf) / [Markdown có thể sửa](reports/midterm_report.md).
- [18 slide PDF (tối đa 20)](reports/midterm_slides.pdf) / [Nội dung và lời dẫn](reports/midterm_slides.md).
- [Notebook Colab GPU](notebooks/04_colab_neural_baselines.ipynb).
- [Bảng ba mô hình](reports/tables/model_comparison.csv).
- [Điều chỉnh learning rate PhoBERT](docs/phobert_tuning.md) / [Notebook Colab 05](notebooks/05_colab_phobert_tuning.ipynb) — đã chuẩn bị, chưa chạy sweep GPU.

**Trạng thái (14/10/2026):** Đã nhập và kiểm chứng kết quả Colab của đủ ba mô hình.
BiGRU chạy 10 epoch (37,50 giây); PhoBERT chạy 4 epoch (3.047,11 giây), trên Tesla T4.
Macro-F1 validation tại 0,5 lần lượt là 0,2388 / 0,4060 / 0,4393 cho TF-IDF / BiGRU / PhoBERT.
Xem [kiểm tra kết quả Colab](docs/colab_results_check.md). Còn cần xác minh mapping/split, đọc lỗi neural
và bổ sung tên, MSSV, phân công nhóm trước khi nộp.

**Bổ sung cho pre-midterm:** đã fine-tune PhoBERT với cấu hình baseline; sweep hyperparameter là bước tiếp theo.
Đã thêm quy trình so sánh learning rate `1e-5 / 2e-5 / 3e-5`, chạy lại `2e-5` trong cùng runtime làm đối chứng.
Mỗi lượt lưu kết quả/checkpoint riêng; giữ bảng ba baseline ban đầu. Chọn cấu hình bằng validation tại 0,5.
**Chưa có số liệu sweep mới**; report/slide ghi rõ trạng thái chờ GPU.

Để chạy: tạo ZIP bằng `.venv/bin/python scripts/package_colab.py --include-results`, mở notebook 05 trên Colab T4,
upload ZIP và ba CSV riêng rồi chạy các cell. Notebook xuất ZIP kết quả và tải riêng checkpoint được chọn.
Xem [hướng dẫn tuning](docs/phobert_tuning.md) để nhập kết quả, audit lỗi và tạo lại report/slide.

## Dữ liệu và quyền sử dụng

Xin quyền tại https://huggingface.co/datasets/uitnlp/vigoemotions và tuân thủ giấy phép nghiên cứu/học thuật
phi thương mại theo đề. Đặt ba CSV được cấp quyền tại `data/train.csv`, `data/val.csv`, `data/test.csv`;
cột theo thứ tự `id,text,labels`. `labels` là chuỗi danh sách ID, ví dụ `[2, 11]` (ví dụ nhãn, không phải comment).
Không commit/upload CSV nguồn, comment gốc, vocabulary hoặc checkpoint có thể chứa dữ liệu nhạy cảm lên repo công khai.
`data/` đã được ignore. Báo cáo, biểu đồ, case audit chỉ chứa tổng hợp/ID và mô tả trừu tượng.

Tổng 20.664 mẫu; train 16.531, val 2.066, test 2.067. Loại train ID `1698` trùng text với test
`red000094`; train dùng thực nghiệm còn 16.530. Neutral ID 27 → không thêm output;
mixed-neutral giữ nhãn cảm xúc; gộp ID lặp. Chi tiết: [dataset check](docs/dataset_check.md).
Mapping nhãn kế thừa notebook gốc, cần người có quyền truy cập xác minh với metadata dataset.

Giữ split được cung cấp và lưu `splits/*.csv`. Chưa chứng minh thuật toán split gốc được multilabel
stratified: audit prevalence không thay thế bằng chứng đó. Nếu giảng viên yêu cầu tự stratify, phải tạo
split mới trước huấn luyện và chạy lại toàn bộ mô hình; không đổi test sau khi đã xem kết quả.

## Chạy local: EDA, TF-IDF, báo cáo

Môi trường đã chạy: macOS Intel, Python 3.14; các phiên bản chính được khóa ở `requirements.txt`.
Colab dùng bộ dependency riêng tương thích Python 3.11/3.12, không cài file lock của macOS vào Colab.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name vigoemotions --display-name "Python (ViGoEmotions)"
python -m unittest discover -s tests -v
python scripts/analyze_data.py
python scripts/train_tfidf.py
python scripts/build_report.py
```

Chạy từ project root. Các notebook 01–03 cũng chạy được từ root hoặc `notebooks/`.
Trong giao diện notebook, chọn kernel **Python (ViGoEmotions)**, rồi **Restart Kernel → Run All**.
Nếu trình soạn thảo chọn interpreter theo đường dẫn, dùng `ViGoEmotions/.venv/bin/python`.
Không chọn nhầm `ATI/.venv/bin/python`: môi trường đó đã ghi nhận NumPy 1.26.4 với SciPy 1.18.1
(SciPy yêu cầu NumPy ≥2.0), gây `AttributeError: module 'numpy' has no attribute 'long'`.
Cell đầu kiểm tra import của môi trường đang chạy và in interpreter/versions để phát hiện vấn đề sớm.
Không cần thêm alias `np.long` vào code; phải dùng bộ dependency tương thích.
Notebook 01 kiểm tra schema; 02 tái tạo EDA; 03 chạy baseline và xem bảng/error analysis.
Chạy lại baseline ghi đè artifact của lần trước. Thời gian train có thể thay đổi; giữ `*_run.json` cùng kết quả.

TF-IDF hiện tại: Macro-F1 0,2388; Micro-F1 0,2712; Hamming Loss 0,0606; LRAP 0,6407 trên validation tại 0,5.
Khảo sát ngưỡng validation tốt nhất trong lưới 0,05–0,80 là 0,15 (Macro-F1 0,4844).
Không diễn giải điểm đã tune này là điểm test hoặc kết quả Final.

## Chạy Colab GPU: BiGRU và PhoBERT

1. Tạo gói code bằng `python scripts/package_colab.py` → `vigoemotions_colab.zip` (không có raw data).
2. Upload `notebooks/04_colab_neural_baselines.ipynb` vào Colab; chọn runtime GPU (T4 trở lên).
3. Chạy cell đầu và upload gói code ZIP. Chạy cell setup (Python dependencies + Java cho VnCoreNLP).
4. Cell dữ liệu: upload **riêng** ba CSV có quyền sử dụng; không xuất bản notebook/runtime có dữ liệu nguồn.
5. Chạy lần lượt EDA, TF-IDF, BiGRU và PhoBERT. PhoBERT tải checkpoint công khai và VnCoreNLP từ mạng.
6. Chạy cell tạo báo cáo, tải `vigoemotions_results.zip`; giải nén kết quả vào project để cập nhật `reports/`, `splits/`.
7. Đọc lại phần Q4, error analysis và giới hạn trước khi nộp. Kết quả error analysis thủ công hiện có chỉ dành cho TF-IDF;
   các file neural error cases là danh sách ứng viên audit, không phải bằng chứng đã đọc thủ công.

Không dùng `test.csv` để chọn epoch, hyperparameter hay threshold. Mã chỉ suy luận validation trong midterm.
Colab cần kết nối mạng khi cài dependencies/tải pretrained resources; các bước chạy script kiểm tra exit code để dừng khi lỗi.
Đã có kết quả chạy end-to-end trên Colab Tesla T4; run JSON ghi Python 3.13.15 và package versions thực tế.

Sau khi chạy notebook 04 mới, output được ghi vào `reports/logs/`; có bảng history đánh dấu checkpoint
được chọn, `reports/tables/{gru,phobert}_history.csv` và `reports/figures/neural_learning_curves.png`.
Cuối notebook tải **hai ZIP**: `vigoemotions_results.zip` chứa kết quả/log và
`vigoemotions_baseline_private.zip` chứa checkpoint, vocabulary và config/run metadata.
Lưu ZIP riêng tư trước khi đóng runtime; checkpoint chỉ có trọng số, không có optimizer state để resume giữa chừng.

Nếu runtime hết RAM/VRAM, giảm batch trong `configs/midterm.json`, giữ nguyên các lựa chọn khác,
ghi nhận thay đổi và chạy lại mô hình từ đầu. Không hứa thời gian chạy neural khi chưa có phép đo.
Checkpoint ở `data/checkpoints/`, vocab GRU ở `data/derived/`; lưu vào Drive **riêng** trước khi runtime hết hạn
nếu cần tái dùng backbone cho Final. ZIP kết quả không mang checkpoint/vocab/raw text.

## Thiết kế thí nghiệm

- Seed 42; config chung ở `configs/midterm.json`.
- TF-IDF: unigram/bigram, min_df 2, max_features 40.000, sublinear TF; LR C=1, liblinear, max_iter 1000.
- GRU: embedding 128, hidden 128/hướng, dropout 0,3, max_length 128, batch 64, 10 epoch, AdamW lr=0,001.
- PhoBERT: `vinai/phobert-base-v2`, revision `86cd7fd4c148980922ac11a2cf5e257f2ba639e1`,
  VnCoreNLP word segmentation, 256 BPE tokens, batch 16, 4 epoch, AdamW lr=2e-5, weight_decay=0,01.
- Neural: 27 logits, BCEWithLogitsLoss, sigmoid; chọn epoch bằng validation Macro-F1 tại 0,5, hòa chọn epoch sớm.
- Tier: sắp train frequency giảm dần, hòa theo ID; head 9 nhãn đầu, mid 9 nhãn tiếp, tail 9 nhãn còn lại.
- Tất cả ma trận co-occurrence/correlation chỉ dùng train. Phân tích cặp dự đoán dùng các cặp chọn từ train,
  so với ground truth validation, không dùng test.
- Mỗi model xuất Macro/Micro-F1, Hamming, LRAP, per-label F1, số tham số, thời gian, breakdown style/cardinality,
  threshold sweep và 20 FN/20 FP tự tin nhất. Error selection không là mẫu ngẫu nhiên.
- LRAP dùng xác suất; mẫu all-zero được thư viện chấm 1. Hamming thấp không đủ chứng minh mô hình tốt.

## Artifact và tái lập

```text
configs/midterm.json          seed, hyperparameters, checkpoint revision, protocol
scripts/common.py            parse labels, normalization, leakage handling, metrics
scripts/analyze_data.py      EDA, train-only dependency matrices, fixed IDs
scripts/train_tfidf.py       CPU baseline
scripts/train_neural.py      BiGRU / PhoBERT trên GPU
scripts/evaluate.py          evaluator dùng chung và lưu xác suất
scripts/build_report.py      tạo báo cáo/slide từ kết quả hiện có
notebooks/01..05             validation → EDA → baseline → Colab GPU → tuning
scripts/audit_reports.py     kiểm tra EDA/split/metric từ dữ liệu và xác suất, không huấn luyện
reports/README.md            mục lục và hướng dẫn đọc từng nhóm kết quả
splits/*.csv                fixed IDs + excluded train ID
reports/tables/*.csv         thống kê và kết quả thật
reports/figures/*.png        hình sinh bằng mã
reports/predictions/*.npz    sample_id, true_label_vector, probability_vector, labels, cardinality
reports/*_run.json           config, hardware, version, runtime, input SHA-256
tests/                      invariants neutral/labels/leakage và metric biết trước
```

Probabilities được lưu trước threshold. NPZ chỉ chứa ID, vector nhãn và xác suất, không chứa text.
Đọc bằng `np.load(path, allow_pickle=False)`. Khi chuyển sang Final, đóng băng neural checkpoint,
tune global/per-label thresholds trên validation, khóa decoder rồi inference test một lần.
Chưa có implementation/chạy thí nghiệm Final trong phạm vi gói midterm này.

Sau khi nhập kết quả Colab, kiểm tra tính nhất quán mà không huấn luyện lại:

```bash
.venv/bin/python scripts/audit_reports.py
```

Lệnh cần ba CSV nguồn được cấp quyền và đủ các artifact của project hiện tại; tạo bảng/hình đối chiếu trong
thư mục tạm, không ghi đè `reports/`. Xem [audit](docs/project_audit.md) để biết phạm vi kiểm tra.

`scripts/build_report.py` tự thêm các dòng neural và bảng Q4 khi có file kết quả; nội dung phân tích TF-IDF
giữ rõ mô hình và ngưỡng. Sau khi chạy neural cần người thực hiện đọc thêm các ca lỗi và viết diễn giải
cụ thể, không coi tự sinh bảng là đã hoàn thành audit thủ công.

## Nguồn

- [Đề project](docs/02-vigoemotions-vietnamese-emotion.pdf) và [student guide](docs/student-guide.pdf) trong `docs/`.
- [ViGoEmotions, EACL 2026](https://aclanthology.org/2026.eacl-long.129/).
- [PhoBERT paper](https://aclanthology.org/2020.findings-emnlp.92/),
  [model card](https://huggingface.co/vinai/phobert-base-v2).
- [scikit-learn multilabel](https://scikit-learn.org/stable/modules/multiclass.html),
  [Transformers 4.57.3](https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/auto),
  [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP).

Mã/tài liệu được hỗ trợ bởi AI; nhóm cần tự xác nhận, hiểu và bảo vệ các quyết định, đồng thời tuân thủ
quy định của học phần về khai báo hỗ trợ AI. Không có thí nghiệm hoặc điểm số được dựng lên.
