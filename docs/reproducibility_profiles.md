# Cấu hình và lệnh tái lập từng lượt

Cập nhật 05/11/2026. Mỗi bảng kết quả đi cùng một profile rõ ràng; không sửa `configs/midterm.json` để chuyển qua lại giữa các lượt. File đó giữ giá trị legacy PhoBERT 12 epoch để tương thích gói cũ, không còn là mặc định train neural 27 nhãn.

| Lượt | Profile | GRU / PhoBERT epoch | Bảng nguồn |
| --- | --- | --- | --- |
| Baseline gốc 27 nhãn | `configs/baseline27_4ep.json` | 10 / 4 | `reports/tables/*_metrics.csv`, `reports/*_run.json` |
| Study 28 nhãn (config base trước thêm Neutral) | `configs/neutral28_12ep_base.json` | 10 / 12 | `reports/midterm_2026-11-04/evidence/` |
| Cấu hình Kaggle 27 nhãn 25/20 trước đây | `configs/kaggle27_25_20.json` | 25 / 20 | Profile giữ riêng; không phải nguồn của bảng baseline gốc |

Hai profile đầu được sao chép trực tiếp từ run/study metadata đã kiểm chứng. Profile Neutral base có `label_count=27`; lệnh `init` thêm Neutral, đặt 28 đầu ra và quy tắc chọn checkpoint bằng validation Macro-F1 28 nhãn tại 0,5. Không suy ra rằng baseline gốc chạy 12 epoch từ config base của study mới.

## Kiểm chứng số liệu đã lưu, không train lại

```bash
.venv/bin/python scripts/audit_reports.py
```

Lệnh kiểm tra baseline/EDA/ablation bằng xác suất đã lưu và chỉ sinh file tạm. Xác suất study 28 nhãn được cung cấp riêng tại `reports/midterm_2026-11-04/evidence/models/{tfidf_lr,gru,phobert}/val_predictions.npz`; file chỉ có ID, nhãn và xác suất. `provenance.json` ghi hash các thành phần lấy nguyên từ ZIP nguồn. Notebook 07 tái chạy audit split và ghi vào `reports/split_audit/`.

## Huấn luyện lại baseline 27 nhãn khi cần

Thực hiện trong môi trường GPU có dependencies tại `requirements-colab.txt` và CSV được cấp quyền. Đây là lệnh tái lập điều kiện thí nghiệm, không cam kết bitwise-identical giữa runtime/GPU/package khác nhau. Không cần huấn luyện lại để dùng bộ midterm hiện có.

```bash
python scripts/train_neural.py --model gru --config configs/baseline27_4ep.json --output-dir reports/baseline27_rerun/gru --checkpoint data/checkpoints/baseline27_rerun/gru.pt
python scripts/train_neural.py --model phobert --config configs/baseline27_4ep.json --output-dir reports/baseline27_rerun/phobert --checkpoint data/checkpoints/baseline27_rerun/phobert.pt
```

Mỗi output/checkpoint phải chưa tồn tại. `train_tfidf.py` dùng profile baseline gốc; fit lại ghi vào artifact chính nên lưu bản sao trước nếu chỉ cần thí nghiệm mới. Notebook 04 Colab/Kaggle giờ truyền profile baseline 10/4 tường minh. Muốn chạy profile 25/20, đổi đối số `--config` cho cả hai model, lưu và trình bày thành lượt khác; không sửa đè config chung hoặc trộn bảng kết quả.

## Huấn luyện study Neutral 28 nhãn mới khi cần

```bash
python scripts/neutral_28_labels.py init --output reports/neutral_28_labels/rerun_01 --base-config configs/neutral28_12ep_base.json
python scripts/neutral_28_labels.py train --model tfidf_lr --output reports/neutral_28_labels/rerun_01
python scripts/neutral_28_labels.py train --model gru --output reports/neutral_28_labels/rerun_01
python scripts/neutral_28_labels.py train --model phobert --output reports/neutral_28_labels/rerun_01
python scripts/neutral_28_labels.py report --output reports/neutral_28_labels/rerun_01
```

Notebook 06 dùng cùng profile 12 epoch. Study đóng băng config/data khi init; không thay config giữa lượt. Test vẫn mặc định chưa chấm. Sweep learning rate notebook 05 dùng profile baseline 4 epoch và chưa có kết quả GPU; không phải điều kiện bắt buộc trước khi nộp midterm.

## Gói chia sẻ

```bash
python scripts/package_colab.py --include-results
```

Gói `vigoemotions_colab.zip` gồm source, docs/PDF, config, fixed split IDs, bảng/hình và saved probabilities được phép chia sẻ; notebook trong ZIP được xóa output/execution metadata. Script lọc private study, checkpoint, vocabulary/tokenizer/resources và CSV có cột văn bản nguồn. Gói riêng `package_neutral28.py` cũng làm sạch notebook.

Không kèm `data/`, `.git/`, backup thực thi riêng tư hoặc ZIP checkpoint. Các commit cũ từng có output dữ liệu nguồn: làm sạch bản hiện tại không xóa lịch sử; dùng ZIP chia sẻ đã kiểm tra, không phân phối lịch sử Git cũ. Không rewrite lịch sử trong lần hoàn thiện này.
