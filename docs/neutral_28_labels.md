# Thí nghiệm độc lập 28 nhãn — notebook 06

**Cập nhật 04/11/2026 — đã có kết quả chạy thật.** Đã nhận `vigoemotions_neutral28_private_results.zip` và notebook Kaggle có output; tính lại metric từ ba NPZ khớp CSV (sai số dưới 1e-12). [Bảng nguồn và giới hạn](../reports/midterm_2026-11-04/evidence/README.md), [phân tích đưa vào báo cáo](../reports/midterm_2026-11-04/neutral28_results.md), [lời dẫn thuyết trình/Q&A](midterm_presentation_2026-11-04.md).

| Mô hình | Macro-F1 28 | Macro-F1 chiếu 27 | Neutral F1 |
| --- | --- | --- | --- |
| TF-IDF + LR | 0,231040 | 0,238825 | 0,020833 |
| BiGRU | 0,399227 | 0,410764 | 0,087719 |
| PhoBERT | 0,574653 | 0,579640 | 0,440000 |

Tất cả là validation (2.066 mẫu), seed 42, ngưỡng 0,5; test chưa chấm. Neutral support 94 = 75 only + 19 mixed. BiGRU chọn epoch 10/10, PhoBERT 11/12 trên Tesla T4; config thật nằm trong ZIP. Baseline gốc chỉ chạy PhoBERT 4 epoch nên chưa tách tác động Neutral khỏi epoch/runtime. ZIP chứa checkpoint/tokenizer/vocab; không cần train lại để đọc kết quả.

Cell tìm ví dụ validation (cell index 24, execution_count 16) lỗi JVM đã chạy trước đó; không có `validation_prediction_cases.csv` trong ZIP. Training, metric và inference bốn câu tự viết có output; metric được kiểm chứng riêng. Notebook thực thi có output comment nguồn, giữ riêng tư. Các hướng dẫn bên dưới dùng để tái chạy, không ngụ ý toàn notebook đã chạy không lỗi.

Không sửa pipeline 27 nhãn. Notebook mới gọi `scripts/neutral_28_labels.py`, dùng namespace riêng
`reports/neutral_28_labels/` cho CSV, model, tokenizer, vocabulary, config và báo cáo Markdown.
Checkpoint, vocabulary và kết quả riêng của study được Git ignore; chỉ code/notebook/test/tài liệu được đưa lên repo.

## Thiết kế

- IDs 0–26 giữ thứ tự `common.LABELS`; ID 27 là `neutral`.
- Annotation được giữ nguyên: `[27]` là vector có duy nhất Neutral dương; `[2,27]` có cả hai phần tử dương.
- Guideline người dùng cung cấp nói Neutral không đi cùng cảm xúc, nhưng CSV có 725 Neutral-only và 306 mixed.
  Đếm riêng theo split được ghi lại trước loại overlap. Không tự quyết nhãn nào sai, không thêm decoder loại trừ Neutral.
- Dùng lại `common.parse_labels`, `normalize`, `scores`, `write_json`; loader có data path riêng và giữ đúng
  kiểm tra/split/leakage rule của `common.load_data`, vì loader cũ không nhận tham số path.
- TF-IDF giữ cách fit và OvR LR của `train_tfidf.py`. Không refactor file gốc để tránh đổi kết quả cũ.
- BiGRU dùng lại class/forward từ `train_neural.py`, thay đúng linear head sang 28; vocabulary chỉ xây từ train.
- PhoBERT giữ checkpoint/revision, VnCoreNLP, tokenizer và fine-tune toàn bộ backbone/head. Dùng lại
  `segmented_texts`, `logits`, `predict`; BCE, AdamW và gradient clipping như baseline.
- Config 28 được sao chép riêng từ `configs/midterm.json` đi kèm ZIP; kiểm tra epoch trong config
  in ra sau init. Không lấy epoch từ notebook khác hay suy ra từ reports cũ. Muốn đổi phải tạo study mới
  và nêu rõ khác biệt với baseline.
- Chọn checkpoint bằng validation Macro-F1 trên 28 nhãn tại 0.5 (khác tập nhãn chọn epoch của baseline 27).
  Không đổi threshold; không tuning trên test. Test tùy chọn sau khi hoàn tất cả ba model và khóa checksum.
- Study từ chối ghi lên các thư mục gốc; model hoàn tất được giữ, model dở dang không bị xóa hoặc tự train đè.

## Chạy Google Colab

1. Trên máy chạy `python scripts/package_neutral28.py` để tạo ZIP mới theo code/config hiện tại.
2. Mở `notebooks/06_colab_neutral_28_labels.ipynb` bằng Colab. Chọn Runtime > Change runtime type > GPU.
3. Chạy cell upload, chọn **vigoemotions_neutral28_colab.zip** mới, không chọn ZIP notebook 04 cũ.
4. Chạy cell nối workspace, dependency, kiểm tra GPU/Java. Upload riêng ba CSV được cấp quyền.
5. Init study, chạy smoke test riêng cho 28 nhãn. Chạy lần lượt hoặc riêng từng model sau setup.
6. Chạy report, inference với comment mới, tìm các kiểu dự đoán thực tế trên validation.
7. Chỉ bật `RUN_FINAL_TEST=True` sau khi đã chốt mọi lựa chọn; mặc định giữ test cho Final như baseline.
8. Chạy cell ZIP cuối và tải `vigoemotions_neutral28_private_results.zip` trước khi mất runtime.

Nếu kernel restart nhưng filesystem còn, chạy cell nối workspace, dependency nếu thiếu, helper/import,
và init (idempotent khi config/data/môi trường khớp). Không extract ZIP đè lên kết quả đang có.
Model đã hoàn tất có thể reload để inference mà không train lại. PhoBERT inference cần Java/VnCoreNLP;
tokenizer và model backbone/head được nạp từ checkpoint local.

## Chạy Kaggle GPU — notebook 06

1. Từ project root chạy `python3 scripts/prepare_kaggle.py --neutral28` hoặc
   `python3 scripts/package_neutral28.py`. Cả hai đều tạo lại
   `notebooks/06_kaggle_neutral_28_labels.ipynb` và `vigoemotions_neutral28_colab.zip`.
   Tên ZIP được giữ để dùng chung Colab/Kaggle; không dùng `vigoemotions_colab.zip` của 04/05.
2. Tạo hai Dataset **Private**: một chứa ZIP code vừa tạo, một chứa ba file
   `train.csv`, `val.csv`, `test.csv` có cột đúng thứ tự `id,text,labels`.
3. Import **06_kaggle_neutral_28_labels.ipynb** vào Kaggle, giữ notebook Private.
   Trong Input > Add Input, attach cả hai Dataset. Chỉ attach một bộ CSV/code để tránh tên trùng.
4. Trong Settings/Session options, bật **GPU + Internet**. Không dùng TPU.
   Nếu có hai GPU, code hiện tại chỉ dùng GPU thứ nhất.
5. Chạy từ trên xuống: nạp project → nối workspace → dependency → GPU/Java → CSV → init
   → smoke tests → TF-IDF → BiGRU → PhoBERT → report/inference → ZIP. Dừng khi cell lỗi.
   Phải thấy `CUDA available: True` trước neural. Internet cần cho pip, PhoBERT và VnCoreNLP.
6. Sau init, đọc config được in ra: epoch thực dùng lấy từ `configs/midterm.json` trong ZIP.
   Muốn chỉnh hyperparameters thì chỉnh file trước init; study đã init phải dùng OUT mới.
   Giữ `RUN_FINAL_TEST = False` nếu chưa khóa các lựa chọn hoặc vẫn giữ test cho Final.
7. Chạy cell ZIP cuối; tải **/kaggle/working/vigoemotions_neutral28_private_results.zip**
   từ file browser/Output trước khi kết thúc session. ZIP gồm toàn bộ study/checkpoint, giữ Private.
   Save Version > Save & Run All chạy lại từ đầu trong session mới; tải bản sao trước khi chọn.

Code tự tìm ZIP hoặc project đã được Kaggle giải nén bên trong `/kaggle/input`, rồi copy sang
`/kaggle/working/ViGoEmotions28`. CSV cũng được copy sang `data/` trong project.
Artifact nằm trong `reports/neutral_28_labels/` của project mới. Không cần sửa đường dẫn slug Dataset.
Cell nạp CSV bỏ qua `splits/*.csv` có duy nhất cột `id` trong Dataset code đã giải nén;
chỉ chọn CSV nguồn có header `id,text,labels`. Nếu có hai CSV nguồn cùng tên, cell vẫn dừng
để bạn kiểm tra Dataset, không tự lấy file đầu tiên. Với notebook cũ báo trùng `train.csv`,
chỉ thay cell nạp CSV bằng cell tương ứng trong notebook mới rồi chạy lại cell đó;
không cần upload lại dữ liệu hay giải nén đè project đang chạy.

Nếu project đã tồn tại trong cùng session, bỏ qua cell giải nén. Sau restart kernel mà filesystem còn,
chạy lại cell nối workspace, import/setup cần thiết, CSV và helper/init. Model hoàn tất được giữ;
model dở dang không hỗ trợ resume giữa epoch, hãy giữ bản sao và dùng study mới để chạy lại.
Đừng dùng `--resume` của notebook 05 cho study 28 nhãn.

Đã có bằng chứng train GPU đủ ba model trên Kaggle ngày 04/11; cell tìm ví dụ validation vẫn lỗi JVM như ghi ở đầu tài liệu.
Nguồn giao diện/runtime: https://www.kaggle.com/docs/notebooks.

## CLI local / cloud

Chạy từ project root, trong môi trường có các dependency của notebook:

```text
python scripts/neutral_28_labels.py init --data-dir /path/to/csvs
python scripts/neutral_28_labels.py train --model tfidf_lr --data-dir /path/to/csvs
python scripts/neutral_28_labels.py train --model gru --data-dir /path/to/csvs
python scripts/neutral_28_labels.py train --model phobert --data-dir /path/to/csvs
python scripts/neutral_28_labels.py report
python scripts/neutral_28_labels.py examples --data-dir /path/to/csvs
```

GPU không có sẽ dừng neural; `--allow-cpu` là lựa chọn rõ ràng nếu chấp nhận chạy chậm.
Để tạo study khác, thêm `--output reports/neutral_28_labels/ten_study` vào mọi lệnh.
`--base-config` được chỉ định lúc init và chỉ đọc. `--baseline-reports` có thể chọn nguồn 27 nhãn khác.
Không trỏ output sang reports cũ hoặc data/checkpoints cũ.

Test cuối cùng (không bắt buộc cho validation comparison):

```text
python scripts/neutral_28_labels.py test --data-dir /path/to/csvs
```

Lệnh khóa mô hình trước scoring; sau đó không train tiếp trong study, không dùng test chọn cấu hình.
Các kết quả test đã có được giữ lại. Không tạo số baseline test khi repo chỉ có baseline validation.

## Output

```text
reports/neutral_28_labels/
  study.json, config.json, annotation_counts.csv, splits/
  metrics_summary.csv                 metric 28 và emotion27_* (val, test nếu đã chạy)
  per_class_metrics.csv               đủ 28 nhãn cho mỗi model/split hoàn tất
  neutral_analysis.csv                precision/recall/F1/support/predicted_positive của Neutral
  comparison_27_vs_28.csv              6 dòng baseline thật + 28 nhãn, nguồn và conditions
  emotion27_per_class_comparison.csv   F1 delta chỉ khi ID/target/metric baseline được xác minh
  prediction_examples.csv             dự đoán comment ví dụ do người dùng nhập
  prediction_example_probabilities.csv
  validation_prediction_cases.csv     tìm 4 kiểu dự đoán thật, ID-only, báo 0 nếu không có
  experiment_report.md                tự cập nhật từ số đo thật; chưa chạy ghi rõ chưa chạy
  models/tfidf_lr/model.joblib         cả vectorizer và 28 classifier
  models/gru/best.pt, vocab.json       trọng số + vocabulary riêng
  models/phobert/pretrained/           model/head/config local
  models/phobert/tokenizer/            tokenizer local
  models/*/run.json, history.csv, val_predictions.npz, val_metrics.csv, val_per_class.csv
  resources/vncorenlp/                 tài nguyên tải phục vụ model 28
  test_lock.json                      chỉ xuất hiện khi chấm test cuối
```

Study ngày 04/11 đã có số neural đầy đủ trong ZIP riêng tư và bản evidence tổng hợp. Khi tạo study mới, chỉ ghi kết quả sau khi chạy GPU; tiny RoBERTa smoke test không phải kết quả pretrained PhoBERT.
Chỉ dùng `package_neutral28.py` để đóng gói code study này: nó loại trừ model/vocabulary mới.
Script cũ `package_colab.py --include-results` thu thập toàn bộ reports nên có thể mang cả artifact private
của study 28 nhãn; không dùng gói đó làm gói code công khai.
Checkpoint không có optimizer state để resume giữa epoch. Giữ toàn bộ study private vì chứa vocabulary,
checkpoint và ví dụ người dùng nhập; dữ liệu nguồn không được copy vào các CSV báo cáo.

## Đọc kết quả đúng cách

So sánh metric toàn bộ 27 với toàn bộ 28 không cùng tập nhãn. Đọc emotion27_* để biết tác động lên
27 cảm xúc gốc; Neutral cần precision/recall/F1 và số dự đoán dương. Không gán Neutral F1=0 cho baseline
không có đầu ra Neutral. LRAP của all-zero 27-label target và Neutral-only 28-label target cũng khác nghĩa.
Các điều kiện khác epoch, môi trường/checksum và việc checkpoint chọn bằng tập nhãn khác đều được ghi rõ.
Nếu không xác minh được ID/targets/metrics, không xuất delta từng cảm xúc như một so sánh paired.

## Kiểm thử

```text
python -m unittest discover -s tests -p test_neutral_28_labels.py -v
```

Smoke test gồm target Neutral-only/mixed, không đổi encoding 27 nhãn, output guard, dữ liệu/split,
metric 28 chiều, top-5, TF-IDF fit/reload, BiGRU train/reload và RoBERTa tiny train/reload offline.
Tiny RoBERTa xác minh backbone có thay trọng số và head có 28 đầu ra; không tải pretrained checkpoint.

Nguồn giao diện/runtime Colab: https://research.google.com/colaboratory/faq.html.
